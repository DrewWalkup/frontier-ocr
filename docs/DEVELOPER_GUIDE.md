# Frontier OCR — Developer Guide

A FastAPI service for modern OCR backends, with a core package plus optional provider extras.

---

## Architecture Overview

```
frontier_ocr/
├── main.py                 # FastAPI app entrypoint, lifespan management
├── api/
│   └── ocr_routes.py       # HTTP endpoints (/health, /v1/ocr/extract)
├── core/
│   └── config.py           # Environment-based settings (pydantic-settings)
├── models/
│   └── ocr_models.py       # Pydantic response schemas
├── services/
│   └── paddleocr_vl_service.py  # OCR model wrapper
└── utils/
    ├── file_validation.py  # File extension checks
    ├── pdf_utils.py        # PDF page counting and extraction
    ├── temp_storage.py     # Chunked file uploads
    └── text_utils.py       # Markdown → plain text conversion
```

### Request Flow

```
Client uploads file
       ↓
ocr_routes.py validates extension + size
       ↓
File saved to temp directory (chunked)
       ↓
PDF page count checked (if PDF)
       ↓
Backend registry resolves the requested backend
       ↓
Selected backend extract_from_path() runs inference
       ↓
Response returned with markdown + plain text
       ↓
Temp file deleted automatically
```

---

## Local Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- GPU with CUDA (for production) or CPU (for testing)

### Install Dependencies

```bash
uv sync --extra dev --extra paddle

# Install PaddlePaddle 3.0+ (REQUIRED - must use official index)

# First, check your CUDA version:
nvidia-smi  # Look for "CUDA Version: X.Y" in the output

# Then choose ONE based on your CUDA version:

# GPU with CUDA 11.8 (driver ≥450.80.02)
uv pip install paddlepaddle-gpu>=3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# GPU with CUDA 12.6 (driver ≥550.54.14)
uv pip install paddlepaddle-gpu>=3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# GPU with CUDA 12.9
uv pip install paddlepaddle-gpu>=3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

# CPU only
uv pip install paddlepaddle>=3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# The `paddle` extra installs `paddleocr` and `paddlex[ocr]`

# Verify installation
python -c "import paddle; print(paddle.__version__)"  # Should print: 3.x.x
```

> [!CAUTION] > **PaddleOCR 3.3+ requires PaddlePaddle 3.0+.** The `paddlepaddle-gpu` package on PyPI is version 2.x and will NOT work. You must install from PaddlePaddle's official package index as shown above.

### Run the Server

```bash
uv run frontier-ocr --reload --host 0.0.0.0 --port 8000
```

### Test the API

```bash
# Health check (returns "idle" before first OCR request, then "loaded")
curl http://localhost:8000/health

# Extract text from all pages
curl -X POST "http://localhost:8000/v1/ocr/extract?backend=auto" \
  -F "file=@/path/to/document.pdf"

# Extract text from a specific page (1-indexed)
curl -X POST "http://localhost:8000/v1/ocr/extract?page=1&backend=paddle" \
  -F "file=@/path/to/document.pdf"
```

### Interactive Docs

Open http://localhost:8000/docs for Swagger UI.

---

## Configuration

All settings use the `OCR_` prefix and can be set via environment variables.

| Variable                | Default            | Description                                             |
| ----------------------- | ------------------ | ------------------------------------------------------- |
| `OCR_LOG_LEVEL`                | `INFO`             | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `OCR_MAX_UPLOAD_BYTES`         | `52428800` (50 MB) | Max file upload size                                    |
| `OCR_MAX_PDF_PAGES`            | `50`               | Max pages per PDF                                       |
| `OCR_DEFAULT_BACKEND`          | `auto`             | Preferred backend for `backend=auto`                    |
| `OCR_ENABLED_BACKENDS`         | `paddle`           | Comma-separated enabled backends                        |
| `OCR_PADDLE_DEVICE`            | `None`             | Paddle device override                                  |
| `OCR_PADDLE_VL_REC_MODEL_DIR`  | `None`             | Path to pre-downloaded Paddle VL models                 |
| `OCR_PADDLE_VL_REC_BACKEND`    | `None`             | Remote Paddle VLM backend                               |
| `OCR_PADDLE_VL_REC_SERVER_URL` | `None`             | URL for remote Paddle VLM server                        |

### Example: Production with Pre-Downloaded Models

```bash
export OCR_DEFAULT_BACKEND="auto"
export OCR_ENABLED_BACKENDS="paddle"
export OCR_PADDLE_DEVICE="gpu:0"
export OCR_PADDLE_VL_REC_MODEL_DIR="/opt/models/paddleocr-vl"
export OCR_LOG_LEVEL="WARNING"

frontier-ocr --host 0.0.0.0 --port 8000
```

---

## Key Components

### `OcrBackendRegistry`

The app now resolves OCR backends through a registry. Key points:

- `backend=auto` selects the best installed backend in priority order
- known-but-disabled backends return `409`
- enabled-but-unavailable backends return `422`

### `PaddleOcrVlService`

The core OCR wrapper. Key points:

- **Thread-safe**: Uses `_inference_lock` to prevent concurrent model access
- **Memory efficient**: Uses `predict_iter()` for large PDFs when available
- **Logs timing**: Every extraction logs filename, page count, and duration

```python
# Usage (internal)
service = PaddleOcrVlService.from_settings(settings)
result = service.extract_from_path(
    document_path=Path("/tmp/doc.pdf"),
    original_filename="invoice.pdf",
    include_structured_result=False,
)
```

### Response Models

```python
class OcrPageResult:
    page_number: int      # 1-indexed
    markdown: str         # Formatted text with tables, headers
    text: str             # Plain text version
    structured_result: dict | None  # Raw PaddleOCR output (optional)

class OcrResponse:
    filename: str
    total_pages: int
    backend_used: str
    pages: list[OcrPageResult]
    combined_markdown: str
    combined_text: str
```

### File Validation

Supported formats:

- **PDF**: `.pdf`
- **Images**: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.webp`

---

## Adding New Features

### Adding a New Endpoint

1. Add route in `frontier_ocr/api/ocr_routes.py`
2. Add any new response models in `frontier_ocr/models/ocr_models.py`
3. Add business logic in `frontier_ocr/services/` (keep routes thin)

### Adding a New OCR Backend

The service is designed to be extensible. To add another backend (e.g., DeepSeek OCR):

1. Create `frontier_ocr/services/deepseek_ocr_service.py` with the same method signature:

    ```python
    def extract_from_path(self, *, document_path, original_filename, include_structured_result) -> OcrResponse
    ```

2. Add the backend name to the registry and config surface in `config.py` and `backend_registry.py`:

    ```python
    default_backend: str = "auto"
    enabled_backends: str = "paddle,deepseek"
    ```

3. Register the backend implementation in `frontier_ocr/services/backend_registry.py`.

---

## Troubleshooting

### Model Download Slow on First Run

PaddleOCR downloads ~2GB of models on first startup. Set `OCR_VL_REC_MODEL_DIR` to use pre-downloaded models for faster cold starts.

### Out of Memory Errors

- Reduce `OCR_MAX_PDF_PAGES`
- Use a GPU with more VRAM
- Set `OCR_VL_REC_BACKEND` to offload VLM to a separate server

### Slow Response Times

- Ensure GPU is being used (`OCR_DEVICE=gpu:0`)
- For high throughput, consider running multiple instances with a load balancer

---

## V2 Roadmap

Potential enhancements for future versions:

### Performance

- [ ] **Request queuing** — Queue long-running OCR jobs with status polling
- [ ] **Batch processing** — Accept multiple files in one request
- [ ] **Streaming results** — Return pages as they complete (SSE or WebSocket)

### Features

- [ ] **Multiple OCR backends** — Add DeepSeek, Tesseract, or cloud APIs as alternatives
- [ ] **Language detection** — Auto-detect document language
- [ ] **Confidence scores** — Return OCR confidence per word/line
- [ ] **Bounding boxes** — Return text positions for UI highlighting

### Operations

- [ ] **Prometheus metrics** — Request count, latency histograms, error rates
- [ ] **API authentication** — API keys or JWT for production
- [ ] **Rate limiting** — Prevent abuse on public deployments
- [ ] **Async job API** — Submit job → poll status → retrieve result

### Developer Experience

- [ ] **Docker image** — Pre-built image with models included
- [ ] **OpenAPI client generation** — Auto-generate Python/TypeScript clients
- [ ] **Integration tests** — Test suite with sample documents
