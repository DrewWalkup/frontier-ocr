from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OCR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = "Frontier OCR API"

    # Logging
    log_level: str = "INFO"

    # Upload controls
    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MB
    upload_chunk_bytes: int = 1 * 1024 * 1024  # 1 MB
    max_pdf_pages: int = 50

    # Backend selection controls
    default_backend: str = "auto"
    enabled_backends: str = "paddle"

    # Legacy PaddleOCR-VL pipeline controls (kept for compatibility)
    device: str = "auto"  # "auto", "cpu", "gpu", "gpu:0", "xpu", "dcu", etc.
    vl_rec_model_dir: str | None = (
        None  # e.g. "/opt/models/paddleocr-vl" (None = auto-download from CDN)
    )

    # Optional: point VL recognition to a remote inference service (vLLM/SGLang)
    vl_rec_backend: str | None = None  # e.g. "vllm-server"
    vl_rec_server_url: str | None = None  # e.g. "http://127.0.0.1:8118/v1"

    # Namespaced Paddle settings for multi-backend compatibility
    paddle_device: str | None = Field(default=None)
    paddle_vl_rec_model_dir: str | None = Field(default=None)
    paddle_vl_rec_backend: str | None = Field(default=None)
    paddle_vl_rec_server_url: str | None = Field(default=None)
    paddle_max_requests_before_reload: int = Field(default=0)

    @staticmethod
    def _normalize_optional_string(value: str | None) -> str | None:
        if value is None:
            return None

        normalized = value.strip()
        if not normalized:
            return None

        return normalized

    @property
    def resolved_paddle_device(self) -> str:
        return self.paddle_device or self.device

    @property
    def resolved_paddle_vl_rec_model_dir(self) -> str | None:
        return self._normalize_optional_string(
            self.paddle_vl_rec_model_dir
        ) or self._normalize_optional_string(self.vl_rec_model_dir)

    @property
    def resolved_paddle_vl_rec_backend(self) -> str | None:
        return self._normalize_optional_string(
            self.paddle_vl_rec_backend
        ) or self._normalize_optional_string(self.vl_rec_backend)

    @property
    def resolved_paddle_vl_rec_server_url(self) -> str | None:
        return self._normalize_optional_string(
            self.paddle_vl_rec_server_url
        ) or self._normalize_optional_string(self.vl_rec_server_url)


settings = Settings()
