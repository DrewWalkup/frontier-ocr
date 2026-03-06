# Frontier OCR

A high-performance FastAPI service for modern OCR backends, starting with [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR) and designed to expand to additional providers over time.

---

## Features

- **Multi-format support** — PDFs and images (PNG, JPG, TIFF, WebP, BMP)
- **Dual output** — Markdown formatting (tables, headers) and plain text
- **GPU-accelerated** — Optimized for NVIDIA GPUs with CPU fallback
- **Memory efficient** — Chunked uploads and streaming for large documents
- **Production ready** — Thread-safe, configurable limits, structured logging

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- GPU with CUDA (recommended) or CPU

### Installation

#### Choose an install profile

```bash
pip install frontier-ocr
pip install "frontier-ocr[paddle]"
pip install "frontier-ocr[deepseek]"
pip install "frontier-ocr[all]"
```

- `frontier-ocr` installs the core API and backend-selection framework.
- `frontier-ocr[paddle]` adds the PaddleOCR backend.
- `frontier-ocr[deepseek]` reserves the future DeepSeek install path.
- `frontier-ocr[all]` installs every supported backend extra.

#### Paddle backend setup

Install PaddlePaddle 3.0+ (REQUIRED - must use official index)

##### First, check your CUDA version:

```bash
nvidia-smi  # Look for "CUDA Version: X.Y" in the output
```

##### Then choose ONE based on your CUDA version:

**GPU with CUDA 11.8 (driver ≥450.80.02)**
`uv pip install paddlepaddle-gpu>=3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/`

**GPU with CUDA 12.6 (driver ≥550.54.14)**
`uv pip install paddlepaddle-gpu>=3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/`

**GPU with CUDA 12.9**
`uv pip install paddlepaddle-gpu>=3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/`

**CPU only**
`uv pip install paddlepaddle>=3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/`

The `frontier-ocr[paddle]` extra installs `paddleocr` and `paddlex[ocr]`, but PaddlePaddle itself must still be installed from the official Paddle index as shown above.

##### Verify installation

```bash
python -c "import paddle; print(paddle.__version__)"  # Should print: 3.x.x
```

> [!CAUTION] > **PaddleOCR 3.3+ requires PaddlePaddle 3.0+.** The `paddlepaddle-gpu` package on PyPI is version 2.x and will NOT work. You must install from PaddlePaddle's official package index as shown above.

### Run the Server

```bash
frontier-ocr --reload --host 0.0.0.0 --port 8000
```

### Test It

```bash
# Health check
curl http://localhost:8000/health

# Extract text from a document with automatic backend selection
curl -X POST "http://localhost:8000/v1/ocr/extract?backend=auto" \
  -F "file=@/path/to/document.pdf"

# Force the Paddle backend explicitly
curl -X POST "http://localhost:8000/v1/ocr/extract?backend=paddle" \
  -F "file=@/path/to/document.pdf"
```

> [!NOTE]
> The OCR model loads lazily on the first extraction request and may download ~2GB of models then. Set `OCR_VL_REC_MODEL_DIR` to use pre-downloaded models for faster cold starts.

### Install from Source

```bash
git clone https://github.com/your-org/frontier-ocr.git
cd frontier-ocr
pip install ".[paddle]"
```

---

## API Usage

### Extract Text

```
POST /v1/ocr/extract
```

| Parameter                   | Type    | Required | Description                                                           |
| --------------------------- | ------- | -------- | --------------------------------------------------------------------- |
| `file`                      | File    | Yes      | PDF or image file                                                     |
| `backend`                   | String  | No       | Backend to use: `auto`, `paddle`, or `deepseek`                       |
| `include_structured_result` | Boolean | No       | Include raw backend structured output when supported (default: false) |

### Example Response

```json
{
	"filename": "invoice.pdf",
	"total_pages": 2,
	"backend_used": "paddle",
	"pages": [
		{
			"page_number": 1,
			"markdown": "# Invoice\n\n**Date:** 2024-01-15...",
			"text": "Invoice\n\nDate: 2024-01-15...",
			"structured_result": null
		}
	],
	"combined_markdown": "# Invoice\n\n**Date:** 2024-01-15...",
	"combined_text": "Invoice\n\nDate: 2024-01-15..."
}
```

### Python Example

```python
import requests

def extract_text(file_path: str, base_url: str, backend: str = "auto") -> dict:
    """Extract text from a PDF or image using the OCR API."""
    url = f"{base_url}/v1/ocr/extract?backend={backend}"

    with open(file_path, "rb") as file:
        response = requests.post(url, files={"file": file})

    response.raise_for_status()
    return response.json()


result = extract_text("./invoice.pdf", "http://localhost:8000", backend="paddle")
print(result["combined_text"])
```

### JavaScript Example

```javascript
async function extractText(file, baseUrl) {
	const formData = new FormData();
	formData.append("file", file);

	const response = await fetch(`${baseUrl}/v1/ocr/extract`, {
		method: "POST",
		body: formData,
	});

	if (!response.ok) {
		throw new Error(`OCR failed: ${response.statusText}`);
	}

	return response.json();
}
```

> [!TIP]
> For complete API documentation with more examples, see [docs/API_USAGE_GUIDE.md](docs/API_USAGE_GUIDE.md).

---

## Configuration

All settings use the `OCR_` prefix. The app reads values from real environment variables and from a local `.env` file.

Environment variables take precedence over values in `.env`.

| Variable                 | Default            | Description                                   |
| ------------------------ | ------------------ | --------------------------------------------- |
| `OCR_PROJECT_NAME`             | `Frontier OCR API` | FastAPI application title                                     |
| `OCR_LOG_LEVEL`                | `INFO`             | Logging verbosity                                             |
| `OCR_MAX_UPLOAD_BYTES`         | `52428800`         | Max file size in bytes (50 MB)                                |
| `OCR_UPLOAD_CHUNK_BYTES`       | `1048576`          | Upload stream chunk size in bytes (1 MB)                      |
| `OCR_MAX_PDF_PAGES`            | `50`               | Max pages per PDF                                             |
| `OCR_DEFAULT_BACKEND`          | `auto`             | Preferred backend for `backend=auto`                          |
| `OCR_ENABLED_BACKENDS`         | `paddle`           | Comma-separated enabled backends                              |
| `OCR_PADDLE_DEVICE`            | unset              | Preferred Paddle device setting                               |
| `OCR_PADDLE_VL_REC_MODEL_DIR`  | unset              | Path to pre-downloaded Paddle VL models                       |
| `OCR_PADDLE_VL_REC_BACKEND`    | unset              | Optional remote Paddle inference backend name                 |
| `OCR_PADDLE_VL_REC_SERVER_URL` | unset              | Optional remote Paddle inference server base URL              |
| `OCR_DEVICE`                   | `auto`             | Legacy Paddle device alias retained for compatibility         |
| `OCR_VL_REC_MODEL_DIR`         | auto-download      | Legacy Paddle model dir alias retained for compatibility      |
| `OCR_VL_REC_BACKEND`           | unset              | Legacy Paddle remote backend alias retained for compatibility |
| `OCR_VL_REC_SERVER_URL`        | unset              | Legacy Paddle server URL alias retained for compatibility     |

### `.env` Example

```dotenv
OCR_DEVICE=auto
OCR_VL_REC_MODEL_DIR=/opt/models/paddleocr-vl
OCR_DEFAULT_BACKEND=auto
OCR_ENABLED_BACKENDS=paddle
OCR_LOG_LEVEL=WARNING
OCR_MAX_PDF_PAGES=25
```

### Production Example

```bash
export OCR_DEFAULT_BACKEND="auto"
export OCR_ENABLED_BACKENDS="paddle"
export OCR_PADDLE_DEVICE="auto"
export OCR_PADDLE_VL_REC_MODEL_DIR="/opt/models/paddleocr-vl"
export OCR_LOG_LEVEL="WARNING"

frontier-ocr --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
frontier_ocr/
├── main.py                      # FastAPI entrypoint, lifespan management
├── api/
│   └── ocr_routes.py            # HTTP endpoints (/health, /v1/ocr/extract)
├── core/
│   └── config.py                # Environment-based settings
├── models/
│   └── ocr_models.py            # Pydantic response schemas
├── services/
│   └── paddleocr_vl_service.py  # OCR model wrapper (thread-safe)
└── utils/
    ├── file_validation.py       # File extension checks
    ├── pdf_utils.py             # PDF page counting
    ├── temp_storage.py          # Chunked file uploads
    └── text_utils.py            # Markdown → plain text conversion
```

---

## Development

### Running Locally

```bash
# Install dependencies
uv sync --extra dev --extra paddle

# Start with auto-reload
uv run frontier-ocr --reload --host 0.0.0.0 --port 8000
```

### Interactive API Docs

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Check Syntax

```bash
uv run python -m py_compile frontier_ocr/main.py
```

> [!TIP]
> For the full developer guide including architecture details and extending the service, see [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md).

---

## Troubleshooting

| Issue               | Solution                                               |
| ------------------- | ------------------------------------------------------ |
| Slow first startup  | Set `OCR_VL_REC_MODEL_DIR` to pre-downloaded models    |
| Out of memory       | Reduce `OCR_MAX_PDF_PAGES` or use a GPU with more VRAM |
| Slow response times | Ensure GPU is enabled (`OCR_DEVICE=gpu:0`)             |
| `status: not_ready` | Model is still loading — wait a few seconds and retry  |

---

## Error Handling

| Status | Meaning        | Example                     |
| ------ | -------------- | --------------------------- |
| `200`  | Success        | Text extracted successfully                    |
| `400`  | Bad request    | Unsupported file type or backend value         |
| `409`  | Conflict       | Requested backend is known but not enabled     |
| `413`  | File too large | Exceeds size or page limit                     |
| `422`  | Unprocessable  | Requested backend is enabled but unavailable   |
| `500`  | Server error   | OCR processing failed after backend resolution |

All errors return a JSON response:

```json
{
	"detail": "Unsupported file type: '.doc'. Upload a PDF or image."
}
```

---

## License

MIT
