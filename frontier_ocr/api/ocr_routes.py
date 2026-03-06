from __future__ import annotations

import asyncio
import logging
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from starlette.requests import Request

from frontier_ocr.core.config import settings
from frontier_ocr.models.ocr_models import OcrResponse
from frontier_ocr.services import (
    BackendNotEnabledError,
    BackendUnavailableError,
    OcrBackend,
    UnsupportedBackendError,
    parse_backend_name,
)
from frontier_ocr.services.backend_registry import OcrBackendRegistry
from frontier_ocr.utils.file_validation import (
    SUPPORTED_EXTENSIONS_DISPLAY,
    get_file_extension,
    is_pdf_filename,
    is_supported_extension,
)
from frontier_ocr.utils.pdf_utils import count_pdf_pages, extract_single_page
from frontier_ocr.utils.temp_storage import (
    UploadTooLargeError,
    build_safe_temp_filename,
    save_upload_to_temp_file,
)

router = APIRouter(tags=["ocr"])
logger = logging.getLogger(__name__)


def get_ocr_registry(request: Request) -> OcrBackendRegistry:
    """Dependency to retrieve the OCR backend registry from app state."""
    return request.app.state.ocr_registry


def _backend_error_to_http(error: Exception) -> HTTPException:
    if isinstance(error, UnsupportedBackendError):
        return HTTPException(status_code=400, detail=str(error))
    if isinstance(error, BackendNotEnabledError):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, BackendUnavailableError):
        return HTTPException(status_code=422, detail=str(error))
    return HTTPException(status_code=500, detail="OCR backend resolution failed.")


@router.get("/health")
def health_check(
    registry: OcrBackendRegistry = Depends(get_ocr_registry),
) -> dict[str, Any]:
    """Return registry-level backend health information."""
    backends = registry.status()
    status = "loaded" if any(backend.loaded for backend in backends) else "idle"
    return {
        "status": status,
        "default_backend": registry.default_backend,
        "backends": [
            {
                "name": backend.name,
                "enabled": backend.enabled,
                "available": backend.available,
                "loaded": backend.loaded,
            }
            for backend in backends
        ],
    }


def _prepare_pdf_for_ocr(
    pdf_path: Path,
    temp_dir: Path,
    requested_page: int | None,
) -> Path:
    """
    Validate a PDF and optionally extract a single page for OCR.

    Args:
        pdf_path: Path to the uploaded PDF.
        temp_dir: Temporary directory for extracted pages.
        requested_page: 1-indexed page to extract, or None for all pages.

    Returns:
        Path to use for OCR (original PDF or extracted single-page PDF).

    Raises:
        HTTPException: If PDF is unreadable, page is out of range, or exceeds limits.
    """
    try:
        page_count = count_pdf_pages(pdf_path)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Could not read PDF.") from error

    if requested_page is not None:
        if requested_page > page_count:
            raise HTTPException(
                status_code=400,
                detail=f"Page {requested_page} out of range (PDF has {page_count} pages).",
            )
        page_index = requested_page - 1  # Convert 1-indexed (user) to 0-indexed (pypdf)
        single_page_path = temp_dir / f"page_{requested_page}.pdf"
        extract_single_page(pdf_path, page_index, single_page_path)
        return single_page_path

    if page_count > settings.max_pdf_pages:
        raise HTTPException(
            status_code=413,
            detail=f"PDF has {page_count} pages; max allowed is {settings.max_pdf_pages}.",
        )

    return pdf_path


@router.post("/v1/ocr/extract", response_model=OcrResponse)
async def extract_document(
    file: UploadFile = File(...),
    backend: str = Query(
        "auto",
        description="OCR backend to use: auto, paddle, or deepseek.",
    ),
    page: int | None = Query(
        None,
        ge=1,
        description="Extract only this page (1-indexed). If omitted, processes all pages.",
    ),
    include_structured_result: bool = Query(
        False,
        description="Include PaddleOCR-VL structured JSON output per page.",
    ),
    registry: OcrBackendRegistry = Depends(get_ocr_registry),
) -> OcrResponse:
    """
    Extract text from a PDF or image using the selected OCR backend.

    Returns markdown and plain text for each page, plus combined output.
    """
    if not file.filename or not is_supported_extension(file.filename):
        extension = get_file_extension(file.filename or "")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{extension}'. Upload a {SUPPORTED_EXTENSIONS_DISPLAY}.",
        )

    with TemporaryDirectory(prefix="paddleocr_vl_") as temp_dir:
        safe_filename = build_safe_temp_filename(file.filename)
        temp_path = Path(temp_dir) / safe_filename

        try:
            await save_upload_to_temp_file(
                upload_file=file,
                destination_path=temp_path,
                chunk_bytes=settings.upload_chunk_bytes,
                max_bytes=settings.max_upload_bytes,
            )
        except UploadTooLargeError as error:
            raise HTTPException(status_code=413, detail=str(error)) from error

        ocr_input_path = (
            _prepare_pdf_for_ocr(
                pdf_path=temp_path,
                temp_dir=Path(temp_dir),
                requested_page=page,
            )
            if is_pdf_filename(file.filename)
            else temp_path
        )

        try:
            backend_name = parse_backend_name(backend)
            ocr_backend: OcrBackend = registry.resolve(backend_name)
        except Exception as error:
            raise _backend_error_to_http(error) from error

        try:
            response: OcrResponse = await asyncio.to_thread(
                partial(
                    ocr_backend.extract_from_path,
                    document_path=ocr_input_path,
                    original_filename=file.filename,
                    include_structured_result=include_structured_result,
                )
            )
            return response
        except Exception as error:
            logger.exception("OCR processing failed")
            raise HTTPException(status_code=500, detail="OCR failed.") from error
