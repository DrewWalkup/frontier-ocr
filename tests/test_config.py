from __future__ import annotations

from frontier_ocr.core.config import Settings


def test_resolved_paddle_vl_rec_backend_treats_blank_as_none() -> None:
    settings = Settings(
        vl_rec_backend="",
        paddle_vl_rec_backend="   ",
    )

    assert settings.resolved_paddle_vl_rec_backend is None


def test_resolved_paddle_vl_rec_backend_prefers_non_blank_namespaced_value() -> None:
    settings = Settings(
        vl_rec_backend="vllm-server",
        paddle_vl_rec_backend="native",
    )

    assert settings.resolved_paddle_vl_rec_backend == "native"


def test_resolved_paddle_vl_rec_backend_falls_back_to_legacy_value() -> None:
    settings = Settings(
        vl_rec_backend="native",
        paddle_vl_rec_backend="",
    )

    assert settings.resolved_paddle_vl_rec_backend == "native"
