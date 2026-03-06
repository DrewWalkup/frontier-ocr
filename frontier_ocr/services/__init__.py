from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from frontier_ocr.models.ocr_models import OcrResponse

BackendName = Literal["auto", "paddle", "deepseek"]


class OcrBackendError(RuntimeError):
    """Base error class for backend selection and availability failures."""


class UnsupportedBackendError(OcrBackendError):
    """Raised when a backend identifier is unknown."""


class BackendUnavailableError(OcrBackendError):
    """Raised when a known backend cannot be used in the current environment."""


class BackendNotEnabledError(OcrBackendError):
    """Raised when a known backend is not enabled in configuration."""


@runtime_checkable
class OcrBackend(Protocol):
    """Protocol for OCR backend implementations."""

    backend_name: BackendName

    def is_model_loaded(self) -> bool:
        """Return backend model readiness status."""
        ...

    def shutdown(self) -> None:
        """Release backend resources and stop background work."""
        ...

    def is_available(self) -> bool:
        """Return whether runtime dependencies/configuration are available."""
        ...

    def extract_from_path(
        self,
        *,
        document_path: Path,
        original_filename: str,
        include_structured_result: bool,
    ) -> OcrResponse:
        """Extract OCR output from a local file path."""
        ...


def parse_backend_name(value: str) -> BackendName:
    """Parse a backend string into a typed backend identifier."""
    normalized = value.strip().lower()
    if normalized in {"auto", "paddle", "deepseek"}:
        return cast(BackendName, normalized)

    raise UnsupportedBackendError(
        f"Unsupported backend {value!r}. Supported values: auto, paddle, deepseek."
    )


__all__ = [
    "BackendName",
    "BackendNotEnabledError",
    "BackendUnavailableError",
    "OcrBackend",
    "OcrBackendError",
    "UnsupportedBackendError",
    "parse_backend_name",
]
