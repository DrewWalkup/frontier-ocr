from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OcrPageResult(BaseModel):
    """OCR result for a single page."""

    page_number: int = Field(..., ge=1)
    markdown: str
    text: str
    structured_result: dict[str, Any] | None = None


class OcrResponse(BaseModel):
    """Complete OCR response for a document."""

    filename: str
    total_pages: int
    backend_used: str
    pages: list[OcrPageResult]

    combined_markdown: str
    combined_text: str
