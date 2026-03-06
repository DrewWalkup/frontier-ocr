from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter


def count_pdf_pages(pdf_path: Path) -> int:
    """
    Count PDF pages quickly so we can reject huge PDFs before model inference.
    """
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def extract_single_page(pdf_path: Path, page_index: int, output_path: Path) -> None:
    """
    Extract a single page from a PDF to a new file.

    Args:
        pdf_path: Source PDF file path.
        page_index: 0-indexed page number to extract.
        output_path: Destination path for single-page PDF.

    Raises:
        IndexError: If page_index is out of range.
    """
    reader = PdfReader(str(pdf_path))
    if page_index < 0 or page_index >= len(reader.pages):
        raise IndexError(
            f"Page index {page_index} out of range (PDF has {len(reader.pages)} pages)"
        )
    writer = PdfWriter()
    writer.add_page(reader.pages[page_index])
    writer.write(str(output_path))
