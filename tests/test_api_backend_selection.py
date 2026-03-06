from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from frontier_ocr.api.ocr_routes import get_ocr_registry
from frontier_ocr.main import app
from frontier_ocr.models.ocr_models import OcrPageResult, OcrResponse
from frontier_ocr.services import BackendNotEnabledError, BackendUnavailableError


class FakeBackend:
    backend_name = "paddle"

    def is_model_loaded(self) -> bool:
        return False

    def is_available(self) -> bool:
        return True

    def shutdown(self) -> None:
        return None

    def extract_from_path(
        self,
        *,
        document_path: Path,
        original_filename: str,
        include_structured_result: bool,
    ) -> OcrResponse:
        return OcrResponse(
            filename=original_filename,
            total_pages=1,
            backend_used="paddle",
            pages=[
                OcrPageResult(
                    page_number=1,
                    markdown="hello",
                    text="hello",
                    structured_result=None,
                )
            ],
            combined_markdown="hello",
            combined_text="hello",
        )


class FakeRegistry:
    def __init__(self, *, mode: str = "success"):
        self.mode = mode
        self.requested_backend: str | None = None
        self.default_backend = "auto"

    def resolve(self, requested_backend: str):
        self.requested_backend = requested_backend
        if self.mode == "disabled":
            raise BackendNotEnabledError("Backend 'deepseek' is not enabled.")
        if self.mode == "unavailable":
            raise BackendUnavailableError("Backend 'deepseek' is unavailable.")
        return FakeBackend()

    def status(self):
        return [
            type(
                "BackendStatus",
                (),
                {
                    "name": "paddle",
                    "enabled": True,
                    "available": True,
                    "loaded": False,
                },
            )(),
            type(
                "BackendStatus",
                (),
                {
                    "name": "deepseek",
                    "enabled": False,
                    "available": False,
                    "loaded": False,
                },
            )(),
        ]


def test_extract_supports_backend_auto_override() -> None:
    registry = FakeRegistry()
    app.dependency_overrides[get_ocr_registry] = lambda: registry

    with TestClient(app) as client:
        response = client.post(
            "/v1/ocr/extract?backend=auto",
            files={"file": ("demo.png", b"fake-image", "image/png")},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["backend_used"] == "paddle"
    assert registry.requested_backend == "auto"


def test_extract_rejects_invalid_backend_value() -> None:
    registry = FakeRegistry()
    app.dependency_overrides[get_ocr_registry] = lambda: registry

    with TestClient(app) as client:
        response = client.post(
            "/v1/ocr/extract?backend=bogus",
            files={"file": ("demo.png", b"fake-image", "image/png")},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 400


def test_extract_returns_conflict_for_disabled_backend() -> None:
    registry = FakeRegistry(mode="disabled")
    app.dependency_overrides[get_ocr_registry] = lambda: registry

    with TestClient(app) as client:
        response = client.post(
            "/v1/ocr/extract?backend=deepseek",
            files={"file": ("demo.png", b"fake-image", "image/png")},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 409


def test_extract_returns_unprocessable_for_unavailable_backend() -> None:
    registry = FakeRegistry(mode="unavailable")
    app.dependency_overrides[get_ocr_registry] = lambda: registry

    with TestClient(app) as client:
        response = client.post(
            "/v1/ocr/extract?backend=deepseek",
            files={"file": ("demo.png", b"fake-image", "image/png")},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 422


def test_health_reports_backend_statuses() -> None:
    registry = FakeRegistry()
    app.dependency_overrides[get_ocr_registry] = lambda: registry

    with TestClient(app) as client:
        response = client.get("/health")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "idle"
    assert payload["backends"][0]["name"] == "paddle"
