# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Install dependencies (use uv, not pip)
uv sync --extra dev --extra paddle

# Run the server (dev mode with auto-reload)
uv run frontier-ocr --reload --host 0.0.0.0 --port 8000

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_backend_registry.py::test_auto_prefers_configured_default_when_available

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Check syntax (prefer this over `python -c "..."`)
uv run python -m py_compile frontier_ocr/main.py
```

## Architecture

**Frontier OCR** is a FastAPI service that provides a backend-agnostic OCR API. It currently supports PaddleOCR-VL with a pluggable architecture for adding more backends (e.g., DeepSeek).

### Key Concepts

- **Backend registry pattern**: `OcrBackendRegistry` (in `services/backend_registry.py`) resolves which OCR backend to use based on config and runtime availability. It supports `auto` resolution which picks the first available backend from a priority list.
- **`OcrBackend` protocol**: Defined in `services/__init__.py`. Any new backend must implement this protocol (`is_available`, `is_model_loaded`, `extract_from_path`, `shutdown`).
- **Lazy model loading**: The PaddleOCR model loads on first request, not at startup. It auto-unloads after 5 minutes of inactivity to free GPU memory.
- **Thread-safe inference**: `PaddleOcrVlService` uses separate `_loading_lock` and `_inference_lock` to allow concurrent request queuing while keeping inference single-threaded.

### Request Flow

1. `POST /v1/ocr/extract` → `api/ocr_routes.py`
2. File validation and chunked upload to temp dir
3. PDF preparation (page count validation, optional single-page extraction)
4. Backend resolution via `OcrBackendRegistry.resolve()`
5. OCR runs in a thread (`asyncio.to_thread`) to avoid blocking the event loop
6. Response includes per-page markdown/text plus combined output

### Configuration

All settings use the `OCR_` env prefix via pydantic-settings (`core/config.py`). Paddle-specific settings have both legacy (`OCR_DEVICE`) and namespaced (`OCR_PADDLE_DEVICE`) forms; namespaced takes precedence via `resolved_*` properties.

### Install Extras

The package uses optional dependency groups: `paddle`, `deepseek` (placeholder), `all`, `dev`. Backends gracefully degrade when their extras aren't installed (`is_available()` checks for importable dependencies).

## Code Style

- Clean, readable code for junior developers — descriptive variable names (e.g., `response` not `r`)
- Functions should do one thing well
- Use `from __future__ import annotations` in all modules
- Use `uv run` to execute project commands
