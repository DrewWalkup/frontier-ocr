from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from frontier_ocr.core.config import Settings
from frontier_ocr.services.paddleocr_vl_service import PaddleOcrVlService


class FakePipeline:
    """Minimal stand-in for PaddleOCRVL."""

    def predict_iter(self, *, input: str, use_queues: bool):
        result = MagicMock()
        result.markdown = {"markdown_texts": "hello"}
        result.json = None
        return [result]

    @staticmethod
    def concatenate_markdown_pages(markdown_info_list):
        return "hello"


def _make_service(*, max_requests: int = 0) -> PaddleOcrVlService:
    settings = Settings(paddle_device="cpu")
    service = PaddleOcrVlService(settings)
    service._max_requests_before_reload = max_requests
    return service


def _run_extraction(service: PaddleOcrVlService) -> None:
    """Run one extraction with a fake pipeline."""
    with patch.object(service, "_load_pipeline_if_needed"):
        service._pipeline = FakePipeline()
        service.extract_from_path(
            document_path=Path("/fake/test.png"),
            original_filename="test.png",
            include_structured_result=False,
        )


def test_request_counter_increments() -> None:
    service = _make_service(max_requests=0)
    assert service._request_count == 0

    _run_extraction(service)
    assert service._request_count == 1

    _run_extraction(service)
    assert service._request_count == 2

    service.shutdown()


def test_forced_reload_triggers_at_threshold() -> None:
    service = _make_service(max_requests=3)

    _run_extraction(service)
    _run_extraction(service)
    assert service._request_count == 2
    assert service._pipeline is not None

    _run_extraction(service)
    # Counter should have reset after hitting threshold
    assert service._request_count == 0
    # Pipeline should have been unloaded
    assert service._pipeline is None

    service.shutdown()


def test_no_forced_reload_when_disabled() -> None:
    service = _make_service(max_requests=0)

    for _ in range(10):
        _run_extraction(service)

    assert service._request_count == 10
    assert service._pipeline is not None

    service.shutdown()


def test_counter_resets_on_manual_unload() -> None:
    service = _make_service(max_requests=0)

    _run_extraction(service)
    _run_extraction(service)
    assert service._request_count == 2

    service._unload_pipeline()
    assert service._request_count == 0

    service.shutdown()
