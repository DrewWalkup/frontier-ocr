from __future__ import annotations

import logging
import threading
import time
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from frontier_ocr.models.ocr_models import OcrPageResult, OcrResponse
from frontier_ocr.services import BackendUnavailableError, OcrBackend
from frontier_ocr.utils.text_utils import markdown_to_plain_text

if TYPE_CHECKING:
    from frontier_ocr.core.config import Settings

logger = logging.getLogger(__name__)


class PaddleOcrVlService(OcrBackend):
    """
    Wraps PaddleOCRVL and converts its outputs into our API response models.

    Notes:
    - PaddleOCRVL supports image/PDF paths as input.
    - For large PDFs, PaddleOCRVL also provides predict_iter() for memory efficiency.
    - Implements lazy loading: Model is loaded on first request.
    - Implements auto-unloading: Model is unloaded after 5 minutes of inactivity.
    """

    backend_name = "paddle"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._pipeline: Any | None = None
        self._inference_lock = threading.Lock()
        self._loading_lock = threading.Lock()
        self._last_accessed_time: float = 0.0
        self._unload_timeout: float = 300.0  # 5 minutes
        self._request_count: int = 0
        self._max_requests_before_reload: int = 0  # 0 = disabled

        # Start background thread for auto-unloading
        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(
            target=self._monitor_inactivity, daemon=True
        )
        self._monitor_thread.start()

    @classmethod
    def from_settings(cls, settings: Settings) -> PaddleOcrVlService:
        """Create a service instance from application settings."""
        service = cls(settings=settings)
        service._max_requests_before_reload = (
            settings.paddle_max_requests_before_reload
        )
        return service

    def is_available(self) -> bool:
        """Return True when Paddle runtime dependencies are importable."""
        return find_spec("paddleocr") is not None

    def _load_pipeline_class(self) -> type[Any]:
        if not self.is_available():
            raise BackendUnavailableError(
                "Backend 'paddle' requires optional dependencies. Install `frontier-ocr[paddle]` and the matching Paddle runtime."
            )

        try:
            from paddleocr import PaddleOCRVL
        except ImportError as error:
            raise BackendUnavailableError(
                "Backend 'paddle' could not import PaddleOCR runtime dependencies."
            ) from error

        return PaddleOCRVL

    def is_model_loaded(self) -> bool:
        """Return True when the OCR pipeline is currently loaded."""
        with self._loading_lock:
            return self._pipeline is not None

    def shutdown(self) -> None:
        """Stop background work and release model resources."""
        self._stop_event.set()
        self._monitor_thread.join(timeout=1)
        with self._inference_lock:
            self._unload_pipeline()

    def _resolve_device(self) -> str:
        """Resolve the configured device into a concrete Paddle device string."""
        configured_device = (
            (self._settings.resolved_paddle_device or "auto").strip().lower()
        )

        if configured_device in {"", "auto"}:
            if self._is_cuda_available():
                return "gpu:0"
            if self.is_available():
                return "cpu"
            try:
                self._load_pipeline_class()
            except BackendUnavailableError:
                logger.warning(
                    "Failed to probe CUDA availability; falling back to CPU",
                )
            return "cpu"

        supported_prefixes = (
            "cpu",
            "gpu",
            "xpu",
            "dcu",
            "mlu",
            "npu",
            "gcu",
            "sdaa",
            "iluvatar_gpu",
        )
        if configured_device.startswith(supported_prefixes):
            return configured_device

        raise ValueError(
            "Invalid OCR_DEVICE value "
            f"{self._settings.resolved_paddle_device!r}. Use 'auto', 'cpu', 'gpu', 'gpu:0', or another Paddle device string."
        )

    def _load_pipeline_if_needed(self) -> None:
        """Ensures the pipeline is loaded. Thread-safe."""
        with self._loading_lock:
            self._last_accessed_time = time.time()
            if self._pipeline is not None:
                return

            resolved_device = self._resolve_device()
            logger.info("Loading PaddleOCR-VL model...")
            logger.info(
                "Resolved OCR device: configured=%r, using=%s",
                self._settings.resolved_paddle_device,
                resolved_device,
            )
            pipeline_class = self._load_pipeline_class()
            self._pipeline = pipeline_class(
                device=resolved_device,
                vl_rec_model_dir=self._settings.resolved_paddle_vl_rec_model_dir,
                vl_rec_backend=self._settings.resolved_paddle_vl_rec_backend,
                vl_rec_server_url=self._settings.resolved_paddle_vl_rec_server_url,
            )
            logger.info("PaddleOCR-VL model loaded.")

    def _unload_pipeline(self) -> None:
        """Unloads the pipeline to free resources."""
        with self._loading_lock:
            if self._pipeline is None:
                return

            logger.info("Unloading PaddleOCR-VL model...")
            self._pipeline = None
            self._request_count = 0

            # Force garbage collection and clear GPU memory
            import gc

            gc.collect()

            try:
                paddle = import_module("paddle")
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.cuda.empty_cache()
            except Exception:
                logger.debug("GPU cache clearing skipped or failed", exc_info=True)

            logger.info("PaddleOCR-VL model unloaded.")

    def _monitor_inactivity(self) -> None:
        """Background loop to check for inactivity."""
        while not self._stop_event.is_set():
            time.sleep(10)  # Check every 10 seconds

            # Check safely without lock first to avoid blocking needlessly
            if self._pipeline is None:
                continue

            if time.time() - self._last_accessed_time > self._unload_timeout:
                if self._inference_lock.acquire(blocking=False):
                    try:
                        if (
                            time.time() - self._last_accessed_time
                            > self._unload_timeout
                        ):
                            self._unload_pipeline()
                    finally:
                        self._inference_lock.release()

    def extract_from_path(
        self,
        *,
        document_path: Path,
        original_filename: str,
        include_structured_result: bool,
    ) -> OcrResponse:
        """
        Run PaddleOCR-VL on a local file path and return a normalized response.
        """
        start_time = time.perf_counter()

        # Ensure model is loaded
        self._load_pipeline_if_needed()

        with self._inference_lock:
            self._last_accessed_time = time.time()

            if self._pipeline is None:
                self._load_pipeline_if_needed()

            pipeline = self._pipeline
            if pipeline is None:
                raise RuntimeError("OCR pipeline was not available after loading")

            try:
                results_iterator = self._get_results_iterator(document_path)

                page_results: list[OcrPageResult] = []
                markdown_info_list: list[dict[str, Any]] = []

                for index, result in enumerate(results_iterator):
                    self._last_accessed_time = time.time()
                    page_number = index + 1

                    markdown_info = getattr(result, "markdown", None) or {}
                    markdown_text = str(markdown_info.get("markdown_texts", "") or "")
                    plain_text = markdown_to_plain_text(markdown_text)

                    structured_result = (
                        getattr(result, "json", None)
                        if include_structured_result
                        else None
                    )

                    page_results.append(
                        OcrPageResult(
                            page_number=page_number,
                            markdown=markdown_text,
                            text=plain_text,
                            structured_result=structured_result,
                        )
                    )
                    markdown_info_list.append(markdown_info)

                combined_markdown = pipeline.concatenate_markdown_pages(
                    markdown_info_list
                )
                combined_text = markdown_to_plain_text(combined_markdown)
            except Exception:
                # Unload model on failure to reset state for next request attempt
                logger.warning("Inference failed, unloading model to reset state")
                self._unload_pipeline()
                raise

            self._request_count += 1

            if (
                self._max_requests_before_reload > 0
                and self._request_count >= self._max_requests_before_reload
            ):
                logger.info(
                    "Reached %d requests, scheduling pipeline reload",
                    self._request_count,
                )
                self._unload_pipeline()

        elapsed_seconds = time.perf_counter() - start_time
        logger.info(
            "OCR completed for %s (%d pages) in %.2fs [request %d since reload]",
            original_filename,
            len(page_results),
            elapsed_seconds,
            self._request_count,
        )

        return OcrResponse(
            filename=original_filename,
            total_pages=len(page_results),
            backend_used=self.backend_name,
            pages=page_results,
            combined_markdown=combined_markdown,
            combined_text=combined_text,
        )

    def _get_results_iterator(self, document_path: Path) -> Iterator[Any]:
        """
        Return an iterator of PaddleOCR-VL results.

        Uses predict_iter() when available for memory efficiency on large docs.
        """
        pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError("OCR pipeline is not loaded")

        if hasattr(pipeline, "predict_iter"):
            return iter(
                pipeline.predict_iter(
                    input=str(document_path),
                    use_queues=False,
                )
            )
        return iter(pipeline.predict(input=str(document_path), use_queues=False))

    def _is_cuda_available(self) -> bool:
        try:
            paddle = import_module("paddle")
        except Exception:
            return False

        try:
            return bool(paddle.device.is_compiled_with_cuda())
        except Exception:
            return False
