from __future__ import annotations

from pathlib import Path

import pytest

from frontier_ocr.models.ocr_models import OcrResponse
from frontier_ocr.services import BackendNotEnabledError, BackendUnavailableError
from frontier_ocr.services.backend_registry import OcrBackendRegistry


class FakeBackend:
    def __init__(self, *, name: str, available: bool, loaded: bool = False):
        self.backend_name = name
        self._available = available
        self._loaded = loaded

    def is_model_loaded(self) -> bool:
        return self._loaded

    def is_available(self) -> bool:
        return self._available

    def shutdown(self) -> None:
        return None

    def extract_from_path(
        self,
        *,
        document_path: Path,
        original_filename: str,
        include_structured_result: bool,
    ) -> OcrResponse:
        raise NotImplementedError


def test_auto_prefers_configured_default_when_available() -> None:
    paddle = FakeBackend(name="paddle", available=True)
    deepseek = FakeBackend(name="deepseek", available=True)
    registry = OcrBackendRegistry(
        backends={"paddle": paddle, "deepseek": deepseek},
        enabled_backends=("paddle", "deepseek"),
        default_backend="deepseek",
    )

    assert registry.resolve("auto") is deepseek


def test_auto_falls_back_to_next_available_backend() -> None:
    paddle = FakeBackend(name="paddle", available=True)
    deepseek = FakeBackend(name="deepseek", available=False)
    registry = OcrBackendRegistry(
        backends={"paddle": paddle, "deepseek": deepseek},
        enabled_backends=("paddle", "deepseek"),
        default_backend="deepseek",
    )

    assert registry.resolve("auto") is paddle


def test_resolve_raises_for_disabled_backend() -> None:
    registry = OcrBackendRegistry(
        backends={"paddle": FakeBackend(name="paddle", available=True)},
        enabled_backends=("paddle",),
        default_backend="auto",
    )

    with pytest.raises(BackendNotEnabledError):
        registry.resolve("deepseek")


def test_resolve_raises_for_unavailable_backend() -> None:
    registry = OcrBackendRegistry(
        backends={"paddle": FakeBackend(name="paddle", available=False)},
        enabled_backends=("paddle",),
        default_backend="auto",
    )

    with pytest.raises(BackendUnavailableError):
        registry.resolve("paddle")
