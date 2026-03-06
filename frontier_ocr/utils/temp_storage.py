from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import UploadFile


class UploadTooLargeError(Exception):
    """Raised when an upload exceeds the maximum allowed size."""

    pass


async def save_upload_to_temp_file(
    *,
    upload_file: UploadFile,
    destination_path: Path,
    chunk_bytes: int,
    max_bytes: int,
) -> int:
    """
    Save an UploadFile to disk in chunks and return the number of bytes written.

    Raises UploadTooLargeError if the file exceeds max_bytes.
    """
    bytes_written = 0

    try:
        with destination_path.open("wb") as output_file:
            while True:
                chunk = await upload_file.read(chunk_bytes)
                if not chunk:
                    break

                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise UploadTooLargeError(
                        f"Upload exceeds max size of {max_bytes} bytes."
                    )

                output_file.write(chunk)

        return bytes_written
    finally:
        await upload_file.close()


def build_safe_temp_filename(original_filename: str) -> str:
    """
    Return a safe filename for storage on disk (no user-provided path segments).
    """
    suffix = Path(original_filename).suffix.lower()
    random_id = uuid.uuid4().hex
    return f"upload_{random_id}{suffix}"
