from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from frontier_ocr.core.config import Settings
from frontier_ocr.services import (
    BackendName,
    BackendNotEnabledError,
    BackendUnavailableError,
    OcrBackend,
    UnsupportedBackendError,
    parse_backend_name,
)
from frontier_ocr.services.paddleocr_vl_service import PaddleOcrVlService

AUTO_BACKEND_PRIORITY: tuple[BackendName, ...] = ("paddle", "deepseek")


@dataclass(frozen=True)
class BackendStatus:
    name: BackendName
    enabled: bool
    available: bool
    loaded: bool


class OcrBackendRegistry:
    """Resolves OCR backends based on configuration and runtime availability."""

    def __init__(
        self,
        *,
        backends: dict[BackendName, OcrBackend],
        enabled_backends: Iterable[BackendName],
        default_backend: BackendName,
    ):
        self._backends = backends
        self._enabled_backends = tuple(enabled_backends)
        self._default_backend = default_backend

    @classmethod
    def from_settings(cls, settings: Settings) -> OcrBackendRegistry:
        enabled_backends = cls._parse_enabled_backends(settings.enabled_backends)
        default_backend = parse_backend_name(settings.default_backend)
        backends: dict[BackendName, OcrBackend] = {
            "paddle": PaddleOcrVlService.from_settings(settings),
        }
        return cls(
            backends=backends,
            enabled_backends=enabled_backends,
            default_backend=default_backend,
        )

    @property
    def default_backend(self) -> BackendName:
        return self._default_backend

    def resolve(self, requested_backend: BackendName) -> OcrBackend:
        backend_name = (
            self._resolve_auto_backend()
            if requested_backend == "auto"
            else requested_backend
        )

        if backend_name not in self._enabled_backends:
            raise BackendNotEnabledError(
                f"Backend '{backend_name}' is not enabled. Enabled backends: {', '.join(self._enabled_backends)}."
            )

        backend = self._backends.get(backend_name)
        if backend is None:
            raise BackendUnavailableError(
                f"Backend '{backend_name}' is not available in this build."
            )

        if not backend.is_available():
            raise BackendUnavailableError(
                f"Backend '{backend_name}' is enabled but its runtime dependencies are not installed. Install the matching package extra first."
            )

        return backend

    def status(self) -> list[BackendStatus]:
        statuses: list[BackendStatus] = []
        for backend_name in AUTO_BACKEND_PRIORITY:
            backend = self._backends.get(backend_name)
            statuses.append(
                BackendStatus(
                    name=backend_name,
                    enabled=backend_name in self._enabled_backends,
                    available=backend.is_available() if backend is not None else False,
                    loaded=backend.is_model_loaded() if backend is not None else False,
                )
            )
        return statuses

    def shutdown(self) -> None:
        for backend in self._backends.values():
            backend.shutdown()

    def _resolve_auto_backend(self) -> BackendName:
        preferred_backends = list(AUTO_BACKEND_PRIORITY)
        if (
            self._default_backend != "auto"
            and self._default_backend in preferred_backends
        ):
            preferred_backends.remove(self._default_backend)
            preferred_backends.insert(0, self._default_backend)

        for backend_name in preferred_backends:
            if backend_name not in self._enabled_backends:
                continue

            backend = self._backends.get(backend_name)
            if backend is not None and backend.is_available():
                return backend_name

        if self._enabled_backends:
            first_enabled = self._enabled_backends[0]
            raise BackendUnavailableError(
                f"No enabled backends are available. First configured backend: '{first_enabled}'."
            )

        raise BackendUnavailableError("No OCR backends are enabled.")

    @staticmethod
    def _parse_enabled_backends(raw_value: str) -> tuple[BackendName, ...]:
        values = [value.strip() for value in raw_value.split(",") if value.strip()]
        if not values:
            return ("paddle",)

        parsed = [parse_backend_name(value) for value in values if value != "auto"]
        deduped: list[BackendName] = []
        for backend_name in parsed:
            if backend_name == "auto":
                raise UnsupportedBackendError(
                    "'auto' cannot be used in OCR_ENABLED_BACKENDS."
                )
            if backend_name not in deduped:
                deduped.append(backend_name)
        return tuple(deduped)
