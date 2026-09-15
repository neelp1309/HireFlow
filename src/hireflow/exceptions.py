"""Project-specific exception hierarchy."""


class HireFlowError(Exception):
    """Base exception for all HireFlow errors."""


class ConfigurationError(HireFlowError):
    """Raised when application configuration is invalid."""


class DocumentParsingError(HireFlowError):
    """Raised when a resume or job document cannot be parsed."""


class EmbeddingError(HireFlowError):
    """Raised when embedding generation fails."""


class RetrievalError(HireFlowError):
    """Raised when vector retrieval fails."""


class EvaluationError(HireFlowError):
    """Raised when candidate evaluation fails."""
