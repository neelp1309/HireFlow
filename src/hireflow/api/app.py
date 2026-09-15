"""FastAPI application for production-style HireFlow serving."""
from __future__ import annotations

import hmac
import io
import json
import logging
import tempfile
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

from hireflow import __version__
from hireflow.api.schemas import (
    HealthResponse,
    ProblemDetail,
    ReadyResponse,
    SearchApiResponse,
    StructuredSearchRequest,
    bundle_to_api_response,
)
from hireflow.config import Settings, get_settings
from hireflow.exceptions import HireFlowError
from hireflow.logging_config import configure_logging
from hireflow.observability import get_request_id, set_request_id
from hireflow.observability.metrics import HTTP_LATENCY, HTTP_REQUESTS
from hireflow.product import HireFlowProductService, ProductSearchConfig

logger = logging.getLogger(__name__)
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_ZIP_UNCOMPRESSED_BYTES = 250 * 1024 * 1024


def _valid_request_id(value: str | None) -> str:
    if value and len(value) <= 64 and all(ch.isalnum() or ch in "-_" for ch in value):
        return value
    return uuid.uuid4().hex


def _problem(status: int, error: str, message: str) -> JSONResponse:
    payload = ProblemDetail(error=error, message=message, request_id=get_request_id())
    return JSONResponse(status_code=status, content=payload.model_dump(mode="json"))


def _read_upload(upload: UploadFile, max_bytes: int) -> bytes:
    data = upload.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"{Path(upload.filename or 'upload').name} exceeds the allowed size")
    return data


def _stage_pdf(name: str, data: bytes, directory: Path) -> Path:
    safe_name = Path(name).name
    if not safe_name.lower().endswith(".pdf"):
        raise ValueError(f"Only PDF files are accepted; received {safe_name}")
    target = directory / safe_name
    target.write_bytes(data)
    return target


def _extract_zip(data: bytes, directory: Path, max_candidates: int, start_index: int = 0) -> list[Path]:
    output: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        total_size = sum(info.file_size for info in archive.infolist())
        if total_size > MAX_ZIP_UNCOMPRESSED_BYTES:
            raise ValueError("Resume ZIP exceeds the 250 MB uncompressed safety limit")
        for info in archive.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/"):
                continue
            if Path(info.filename).suffix.casefold() != ".pdf":
                continue
            if len(output) >= max_candidates:
                raise ValueError(f"At most {max_candidates} resumes are allowed per request")
            payload = archive.read(info)
            if len(payload) > MAX_PDF_BYTES:
                raise ValueError(f"{Path(info.filename).name} exceeds the 10 MB PDF limit")
            unique_name = f"{start_index + len(output) + 1:04d}_{Path(info.filename).name}"
            output.append(_stage_pdf(unique_name, payload, directory))
    return output


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(resolved.log_level, resolved.log_dir)
        app.state.settings = resolved
        app.state.service = HireFlowProductService(settings=resolved)
        logger.info("api_started", extra={"environment": resolved.app_env, "version": __version__})
        yield
        logger.info("api_stopped")

    app = FastAPI(
        title="HireFlow API",
        version=__version__,
        description="Candidate retrieval, hybrid ranking and grounded evaluation API.",
        lifespan=lifespan,
        docs_url="/docs" if resolved.app_env != "production" else None,
        redoc_url="/redoc" if resolved.app_env != "production" else None,
    )

    @app.middleware("http")
    async def production_middleware(request: Request, call_next):
        request_id = _valid_request_id(request.headers.get("X-Request-ID"))
        set_request_id(request_id)
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > resolved.api_max_request_mb * 1024 * 1024:
                    response = _problem(413, "payload_too_large", "Request body exceeds configured limit")
                    response.headers["X-Request-ID"] = request_id
                    return response
            except ValueError:
                pass
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled_request_error")
            response = _problem(500, "internal_error", "An unexpected server error occurred")
        elapsed = time.perf_counter() - started
        path = request.url.path
        HTTP_REQUESTS.labels(method=request.method, path=path, status=str(response.status_code)).inc()
        HTTP_LATENCY.labels(method=request.method, path=path).observe(elapsed)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        logger.info(
            "http_request",
            extra={
                "method": request.method,
                "path": path,
                "status": response.status_code,
                "duration_ms": round(elapsed * 1000.0, 2),
            },
        )
        return response

    @app.exception_handler(HireFlowError)
    async def hireflow_error_handler(_: Request, exc: HireFlowError):
        return _problem(422, "hireflow_error", str(exc))

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError):
        return _problem(400, "invalid_request", str(exc))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError):
        # Do not echo the entire request body, which may contain resume content.
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", []))
        message = first.get("msg", "Request validation failed")
        return _problem(422, "validation_error", f"{location}: {message}".strip(": "))

    def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        expected = resolved.api_access_key
        if expected is None:
            return
        if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.get("/health/live", response_model=HealthResponse, tags=["health"])
    async def live() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=resolved.app_name,
            version=__version__,
            environment=resolved.app_env,
        )

    @app.get("/health/ready", response_model=ReadyResponse, tags=["health"])
    async def ready(request: Request) -> ReadyResponse:
        checks: dict[str, str] = {}
        taxonomy = Path(__file__).resolve().parents[3] / "configs" / "matching_taxonomy.yaml"
        checks["taxonomy"] = "ok" if taxonomy.exists() else "missing"
        try:
            resolved.artifacts_dir.mkdir(parents=True, exist_ok=True)
            probe = resolved.artifacts_dir / ".ready"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            checks["artifacts"] = "ok"
        except OSError:
            checks["artifacts"] = "unwritable"
        checks["service"] = "ok" if hasattr(request.app.state, "service") else "unavailable"
        status = "ready" if all(value == "ok" for value in checks.values()) else "not_ready"
        if status != "ready":
            raise HTTPException(status_code=503, detail=checks)
        return ReadyResponse(
            status=status,
            service=resolved.app_name,
            version=__version__,
            environment=resolved.app_env,
            checks=checks,
        )

    @app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        if not resolved.metrics_enabled:
            raise HTTPException(status_code=404, detail="Metrics disabled")
        return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/v1/search/structured",
        response_model=SearchApiResponse,
        dependencies=[Depends(require_api_key)],
        tags=["search"],
    )
    async def structured_search(payload: StructuredSearchRequest, request: Request) -> SearchApiResponse:
        if len(payload.candidates) > resolved.max_candidates_per_request:
            raise ValueError(
                f"At most {resolved.max_candidates_per_request} candidates are allowed per request"
            )
        bundle = await run_in_threadpool(
            request.app.state.service.search_structured,
            payload.job,
            payload.candidates,
            payload.config,
        )
        return bundle_to_api_response(bundle, get_request_id())

    @app.post(
        "/v1/search/files",
        response_model=SearchApiResponse,
        dependencies=[Depends(require_api_key)],
        tags=["search"],
    )
    async def file_search(
        request: Request,
        job_pdf: UploadFile = File(...),
        resume_pdfs: list[UploadFile] = File(default=[]),
        resume_zip: UploadFile | None = File(default=None),
        config_json: str = Form(default="{}"),
    ) -> SearchApiResponse:
        try:
            cfg = ProductSearchConfig.model_validate(json.loads(config_json))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid config_json: {exc}") from exc
        if len(resume_pdfs) > resolved.max_candidates_per_request:
            raise ValueError(
                f"At most {resolved.max_candidates_per_request} resumes are allowed per request"
            )
        with tempfile.TemporaryDirectory(prefix="hireflow_api_") as tmp:
            workspace = Path(tmp)
            job_path = _stage_pdf(
                job_pdf.filename or "job.pdf",
                _read_upload(job_pdf, MAX_PDF_BYTES),
                workspace,
            )
            resume_dir = workspace / "resumes"
            resume_dir.mkdir()
            paths: list[Path] = []
            for idx, upload in enumerate(resume_pdfs, start=1):
                data = _read_upload(upload, MAX_PDF_BYTES)
                paths.append(
                    _stage_pdf(f"{idx:04d}_{Path(upload.filename or 'resume.pdf').name}", data, resume_dir)
                )
            if resume_zip is not None:
                zip_bytes = _read_upload(resume_zip, MAX_ZIP_UNCOMPRESSED_BYTES)
                remaining = resolved.max_candidates_per_request - len(paths)
                paths.extend(_extract_zip(zip_bytes, resume_dir, remaining, start_index=len(paths)))
            if not paths:
                raise ValueError("Upload at least one resume PDF or a resume ZIP")
            bundle = await run_in_threadpool(
                request.app.state.service.search_paths,
                job_path,
                paths,
                cfg,
            )
        return bundle_to_api_response(bundle, get_request_id())

    return app


app = create_app()

__all__ = ["app", "create_app"]
