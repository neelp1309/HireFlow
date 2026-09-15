"""HireFlow recruiter-facing Streamlit application."""
from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hireflow import __version__  # noqa: E402
from hireflow.product import (  # noqa: E402
    HireFlowProductService,
    ProductSearchConfig,
    filter_results,
    results_to_csv,
    results_to_json,
)
from hireflow.models import Recommendation  # noqa: E402


MAX_RESUMES = 200
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_ZIP_UNCOMPRESSED_BYTES = 250 * 1024 * 1024


st.set_page_config(
    page_title="HireFlow · Candidate Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1500px;}
    [data-testid="stSidebar"] {border-right: 1px solid #e5e7eb;}
    .hf-kicker {font-size: .78rem; font-weight: 700; letter-spacing: .12em; color: #2563eb; text-transform: uppercase;}
    .hf-title {font-size: 2.4rem; font-weight: 760; line-height: 1.08; margin: .25rem 0 .35rem;}
    .hf-subtitle {font-size: 1rem; color: #64748b; max-width: 900px; margin-bottom: 1.2rem;}
    .hf-badge {display: inline-block; border-radius: 999px; padding: .22rem .58rem; font-size: .74rem; font-weight: 650; background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe;}
    .hf-note {border-left: 3px solid #2563eb; padding: .65rem .9rem; background: #f8fafc; border-radius: .2rem .6rem .6rem .2rem; color:#475569;}
    .hf-small {font-size:.82rem;color:#64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _recommendation_label(value: Recommendation) -> str:
    return {
        Recommendation.HIGHLY_RECOMMENDED: "Highly recommended",
        Recommendation.RECOMMENDED: "Recommended",
        Recommendation.CONSIDER: "Consider",
        Recommendation.NOT_RECOMMENDED: "Not recommended",
    }[value]


def _safe_write(uploaded, target: Path) -> Path:
    data = uploaded.getvalue()
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f"{uploaded.name} exceeds the 10 MB per-PDF limit")
    target.write_bytes(data)
    return target


def _stage_resume_uploads(pdf_uploads, zip_upload, directory: Path) -> list[Path]:
    paths: list[Path] = []
    used_names: set[str] = set()

    def add_pdf(name: str, data: bytes) -> None:
        if len(paths) >= MAX_RESUMES:
            raise ValueError(f"HireFlow accepts at most {MAX_RESUMES} resumes per interactive run")
        safe_name = Path(name).name
        if not safe_name.lower().endswith(".pdf"):
            return
        if len(data) > MAX_PDF_BYTES:
            raise ValueError(f"{safe_name} exceeds the 10 MB per-PDF limit")
        base = safe_name
        counter = 2
        while base.casefold() in used_names:
            stem = Path(safe_name).stem
            base = f"{stem}_{counter}.pdf"
            counter += 1
        used_names.add(base.casefold())
        target = directory / base
        target.write_bytes(data)
        paths.append(target)

    for uploaded in pdf_uploads or []:
        add_pdf(uploaded.name, uploaded.getvalue())

    if zip_upload is not None:
        with zipfile.ZipFile(io.BytesIO(zip_upload.getvalue())) as archive:
            total_size = sum(item.file_size for item in archive.infolist())
            if total_size > MAX_ZIP_UNCOMPRESSED_BYTES:
                raise ValueError("Resume ZIP exceeds the 250 MB uncompressed safety limit")
            for info in archive.infolist():
                if info.is_dir() or info.filename.startswith("__MACOSX/"):
                    continue
                if Path(info.filename).suffix.casefold() != ".pdf":
                    continue
                add_pdf(Path(info.filename).name, archive.read(info))

    return sorted(paths, key=lambda path: path.name.casefold())


@st.cache_resource
def _get_product_service() -> HireFlowProductService:
    # One process-local service instance allows the bounded TTL search cache to
    # survive Streamlit reruns without persisting resume data to disk.
    return HireFlowProductService(PROJECT_ROOT / "configs" / "matching_taxonomy.yaml")


def _run_search(job_upload, resume_uploads, zip_upload, config: ProductSearchConfig):
    with tempfile.TemporaryDirectory(prefix="hireflow_ui_") as tmp:
        workspace = Path(tmp)
        job_path = _safe_write(job_upload, workspace / "job.pdf")
        resume_dir = workspace / "resumes"
        resume_dir.mkdir()
        resume_paths = _stage_resume_uploads(resume_uploads, zip_upload, resume_dir)
        if not resume_paths:
            raise ValueError("Upload at least one resume PDF or a ZIP containing PDFs")
        service = _get_product_service()
        return service.search_paths(job_path, resume_paths, config)


def _component_frame(result) -> pd.DataFrame:
    comp = result.match.component_scores
    if comp is None:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Component": ["Semantic", "Requirements", "Experience", "Role", "Preferred"],
            "Score": [comp.semantic, comp.required_skills, comp.experience, comp.role, comp.preferred],
        }
    ).set_index("Component")


def _results_frame(results) -> pd.DataFrame:
    rows = []
    for item in results:
        rows.append(
            {
                "Rank": item.match.rank,
                "Candidate": item.profile.full_name or item.profile.candidate_id,
                "Current role": item.profile.headline or "—",
                "Experience": item.profile.total_experience_years,
                "Fit score": item.match.final_score,
                "Recommendation": _recommendation_label(item.evaluation.recommendation),
            }
        )
    return pd.DataFrame(rows)


def _render_candidate(result) -> None:
    profile = result.profile
    match = result.match
    evaluation = result.evaluation
    name = profile.full_name or profile.candidate_id
    score = float(match.final_score or 0.0)

    with st.container(border=True):
        left, center, right = st.columns([4.5, 2, 1.7])
        with left:
            st.markdown(f"### #{match.rank} · {name}")
            st.caption(
                f"{profile.headline or 'Role not detected'} · "
                f"{profile.total_experience_years:g} years experience"
                if profile.total_experience_years is not None
                else f"{profile.headline or 'Role not detected'} · Experience not detected"
            )
        with center:
            st.markdown(f"**{_recommendation_label(evaluation.recommendation)}**")
            st.progress(min(1.0, max(0.0, score / 100.0)), text=f"Overall fit {score:.1f}/100")
        with right:
            st.metric("Confidence", f"{evaluation.confidence * 100:.0f}%")

        matches, gaps = st.columns(2)
        with matches:
            st.markdown("**Strong alignment**")
            for value in evaluation.strengths[:4]:
                st.markdown(f"- {value}")
        with gaps:
            st.markdown("**Potential gaps**")
            if evaluation.gaps:
                for value in evaluation.gaps[:4]:
                    st.markdown(f"- {value}")
            else:
                st.markdown("- No material evidence gaps identified in the current rubric.")


def _render_candidate_detail(result) -> None:
    profile = result.profile
    match = result.match
    evaluation = result.evaluation
    name = profile.full_name or profile.candidate_id
    st.markdown(f"## {name}")
    st.caption(
        f"{profile.headline or 'Role not detected'} · Candidate ID {profile.candidate_id} · "
        "identity is displayed for recruiter use but excluded from scoring and LLM evidence"
    )

    a, b, c, d = st.columns(4)
    a.metric("Overall fit", f"{float(match.final_score or 0):.1f}/100")
    b.metric("Recommendation", _recommendation_label(evaluation.recommendation))
    b.caption("Decision support only")
    c.metric(
        "Experience",
        f"{profile.total_experience_years:g} years" if profile.total_experience_years is not None else "Unknown",
    )
    d.metric("Evidence confidence", f"{evaluation.confidence * 100:.0f}%")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("### Score breakdown")
        frame = _component_frame(result)
        if not frame.empty:
            st.bar_chart(frame, height=300)
        st.markdown("### Why this candidate")
        st.write(evaluation.reasoning)
    with right:
        st.markdown("### Profile snapshot")
        if profile.skills:
            st.markdown("**Skills:** " + ", ".join(profile.skills[:12]))
        if profile.software:
            st.markdown("**Software:** " + ", ".join(profile.software[:12]))
        if profile.education:
            edu = profile.education[0]
            degree = edu.degree + (f" in {edu.field_of_study}" if edu.field_of_study else "")
            st.markdown(f"**Education:** {degree}")
        if profile.certifications:
            st.markdown("**Certifications:** " + ", ".join(item.name for item in profile.certifications))
        with st.expander("Recruiter contact details", expanded=False):
            st.caption("These fields are never used for retrieval, ranking, benchmark labels, or Gemini prompts.")
            if profile.contact and profile.contact.email:
                st.write(f"Email: {profile.contact.email}")
            if profile.contact and profile.contact.phone:
                st.write(f"Phone: {profile.contact.phone}")
            if profile.contact and profile.contact.linkedin:
                st.write(f"LinkedIn: {profile.contact.linkedin}")
            if profile.location:
                st.write(f"Location: {profile.location}")

    st.markdown("### Evidence-backed assessment")
    strengths, gaps = st.columns(2)
    with strengths:
        st.markdown("**Strengths**")
        for item in evaluation.strengths:
            st.markdown(f"- {item}")
    with gaps:
        st.markdown("**Evidence gaps**")
        for item in evaluation.gaps:
            st.markdown(f"- {item}")
        if not evaluation.gaps:
            st.write("No current evidence gaps.")

    st.markdown("### Supporting resume evidence")
    if evaluation.evidence:
        evidence_df = pd.DataFrame(
            [
                {
                    "Requirement": item.requirement,
                    "Evidence": item.evidence,
                    "Resume section": item.source_section,
                }
                for item in evaluation.evidence
            ]
        )
        st.dataframe(evidence_df, hide_index=True, width="stretch")
    else:
        st.info("No structured evidence records were produced for this candidate.")


with st.sidebar:
    st.markdown("## HireFlow")
    st.caption(f"Candidate Intelligence · v{__version__}")
    st.divider()
    st.markdown("### Pipeline settings")
    document_mode = st.selectbox(
        "Resume representation",
        ["full", "section"],
        format_func=lambda x: "Full resume (Phase 4 best NDCG@10)" if x == "full" else "Section-aware",
    )
    embedding_provider = st.selectbox(
        "Embedding model",
        ["tfidf", "gemini"],
        format_func=lambda x: "TF-IDF · reproducible offline baseline" if x == "tfidf" else "Gemini dense embeddings",
    )
    evaluator = st.selectbox(
        "Explanation layer",
        ["heuristic", "gemini"],
        format_func=lambda x: "Grounded deterministic" if x == "heuristic" else "Grounded Gemini",
    )
    top_k_retrieval = st.slider("Retrieve", 5, 50, 20, 5)
    top_k_rerank = st.slider("Rerank", 5, min(20, top_k_retrieval), min(10, top_k_retrieval), 1)
    final_top_k = st.slider("Show final candidates", 1, top_k_rerank, min(5, top_k_rerank), 1)
    st.divider()
    st.markdown(
        '<div class="hf-small">Ranking uses job-related evidence only. Name, contact, location, and protected attributes are excluded from retrieval and scoring.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="hf-kicker">AI-assisted recruiting · explainable by design</div>', unsafe_allow_html=True)
st.markdown('<div class="hf-title">Find the right candidates, then show the evidence.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hf-subtitle">HireFlow combines semantic retrieval, deterministic job-fit reranking, and grounded explanations. It is designed as recruiter decision support—not an autonomous hiring decision maker.</div>',
    unsafe_allow_html=True,
)

search_tab, detail_tab, evaluation_tab = st.tabs(["Candidate search", "Candidate detail", "Model & evaluation"])

with search_tab:
    st.markdown("### 1 · Provide the role and candidate pool")
    upload_left, upload_right = st.columns(2)
    with upload_left:
        job_upload = st.file_uploader("Job description PDF", type=["pdf"], key="job_pdf")
        st.caption("Phase 5 uses the deterministic parser built for the supplied project document structure.")
    with upload_right:
        resume_uploads = st.file_uploader(
            "Resume PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="resume_pdfs",
        )
        zip_upload = st.file_uploader("…or one ZIP of resume PDFs", type=["zip"], key="resume_zip")

    config = ProductSearchConfig(
        document_mode=document_mode,
        embedding_provider=embedding_provider,
        evaluator=evaluator,
        vector_backend="auto",
        top_k_retrieval=top_k_retrieval,
        top_k_rerank=top_k_rerank,
        final_top_k=final_top_k,
    )
    can_run = job_upload is not None and bool(resume_uploads or zip_upload)
    if st.button("Analyze candidates", type="primary", disabled=not can_run, width="content"):
        try:
            with st.spinner("Parsing resumes, retrieving candidates, reranking, and building grounded explanations…"):
                st.session_state["hireflow_bundle"] = _run_search(job_upload, resume_uploads, zip_upload, config)
            st.success("Candidate analysis complete.")
        except Exception as exc:
            st.error(f"HireFlow could not complete this run: {exc}")

    bundle = st.session_state.get("hireflow_bundle")
    if bundle is None:
        st.markdown(
            '<div class="hf-note">Upload the supplied Senior Accountant JD and resume PDFs (or the provided resume ZIP), then run the analysis. Raw uploads are staged in a temporary workspace for the interactive run rather than written into the repository.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.divider()
        st.markdown("### 2 · Ranked shortlist")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Candidates analyzed", bundle.candidate_count)
        m2.metric("Shortlist", len(bundle.results))
        m3.metric("Processing time", f"{bundle.processing_time_ms / 1000:.2f}s")
        m4.metric("Vector backend", bundle.vector_backend)
        m5.metric("Cache", "Hit" if bundle.cache_hit else "Miss")
        st.caption(
            f"{bundle.job.title} · {bundle.embedding_model} · {bundle.evaluator_used} · "
            f"{bundle.config.document_mode}-resume retrieval"
        )
        with st.expander("Pipeline latency breakdown", expanded=False):
            timing_rows = [
                {"Stage": stage.replace("_", " ").title(), "Latency (ms)": latency}
                for stage, latency in bundle.stage_timings_ms.items()
            ]
            st.dataframe(pd.DataFrame(timing_rows), width="stretch", hide_index=True)

        with st.expander("Filter shortlist", expanded=False):
            fc1, fc2, fc3 = st.columns(3)
            min_score = fc1.slider("Minimum fit score", 0, 100, 0)
            role_query = fc2.text_input("Role contains", "")
            recommendation_options = list(Recommendation)
            selected_recommendations = fc3.multiselect(
                "Recommendations",
                recommendation_options,
                default=recommendation_options,
                format_func=_recommendation_label,
            )
            ex1, ex2 = st.columns(2)
            min_exp = ex1.number_input("Minimum experience", min_value=0.0, max_value=40.0, value=0.0, step=1.0)
            max_exp = ex2.number_input("Maximum experience", min_value=0.0, max_value=40.0, value=40.0, step=1.0)

        filtered = filter_results(
            bundle.results,
            min_score=float(min_score),
            recommendations=set(selected_recommendations),
            role_query=role_query,
            min_experience=float(min_exp),
            max_experience=float(max_exp),
        )
        if filtered:
            st.dataframe(
                _results_frame(filtered),
                hide_index=True,
                width="stretch",
                column_config={"Fit score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f")},
            )
            for result in filtered:
                _render_candidate(result)
        else:
            st.warning("No shortlisted candidate matches the current filters.")

        download_a, download_b, _ = st.columns([1, 1, 3])
        download_a.download_button(
            "Download CSV",
            results_to_csv(bundle),
            file_name="hireflow_candidate_shortlist.csv",
            mime="text/csv",
            width="stretch",
        )
        download_b.download_button(
            "Download JSON",
            results_to_json(bundle),
            file_name="hireflow_candidate_shortlist.json",
            mime="application/json",
            width="stretch",
        )

with detail_tab:
    bundle = st.session_state.get("hireflow_bundle")
    if bundle is None:
        st.info("Run a candidate search first. The detailed evidence view will appear here.")
    else:
        options = {
            f"#{item.match.rank} · {item.profile.full_name or item.profile.candidate_id} · {float(item.match.final_score or 0):.1f}": item
            for item in bundle.results
        }
        selected = st.selectbox("Candidate", list(options))
        _render_candidate_detail(options[selected])

with evaluation_tab:
    st.markdown("## Model card & offline evaluation")
    st.write(
        "HireFlow separates retrieval, deterministic reranking, and explanation. "
        "The numeric fit score is not produced by Gemini."
    )
    st.markdown(
        '<div class="hf-note"><b>Benchmark limitation:</b> the supplied dataset contains one Senior Accountant JD and no recruiter labels. Phase 4 therefore uses a project-authored silver benchmark. The metrics below describe this project dataset and should not be presented as evidence of cross-role production performance.</div>',
        unsafe_allow_html=True,
    )
    metrics_path = PROJECT_ROOT / "artifacts" / "evaluation" / "phase4_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.markdown("### Phase 4 experiment comparison")
        st.dataframe(metrics, hide_index=True, width="stretch")
    else:
        st.info("Run `python scripts/evaluate_phase4.py` to regenerate the experiment table.")

    st.markdown("### Responsible-use controls")
    st.markdown(
        """
        - Candidate identity/contact fields are retained only for recruiter display and are excluded from matching text.
        - The scoring layer uses job-related role, experience, requirements, software, education and preferred-criteria signals.
        - Gemini receives an identity-free evidence catalog and cannot change the deterministic numeric score.
        - Missing evidence is reported as **not found in the resume**, not as proof that a candidate lacks a capability.
        - HireFlow is decision support. A human recruiter remains responsible for review and hiring decisions.
        """
    )
