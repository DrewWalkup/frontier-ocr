from __future__ import annotations

from pathlib import Path

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTENSION = ".pdf"
SUPPORTED_IMAGE_EXTENSIONS_DISPLAY = ", ".join(
    extension.removeprefix(".") for extension in sorted(ALLOWED_IMAGE_EXTENSIONS)
)

SUPPORTED_EXTENSIONS_DISPLAY = f"PDF or image ({SUPPORTED_IMAGE_EXTENSIONS_DISPLAY})"


def get_file_extension(filename: str) -> str:
    """Returns the lowercase file extension including the dot."""
    return Path(filename).suffix.lower()


def is_supported_extension(filename: str) -> bool:
    """Returns True if the file extension is a supported image or PDF format."""
    extension = get_file_extension(filename)
    return extension == PDF_EXTENSION or extension in ALLOWED_IMAGE_EXTENSIONS


def is_pdf_filename(filename: str) -> bool:
    """Returns True if the filename has a PDF extension."""
    return get_file_extension(filename) == PDF_EXTENSION
