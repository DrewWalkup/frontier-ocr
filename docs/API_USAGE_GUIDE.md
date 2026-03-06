# Frontier OCR API Usage Guide

This guide explains how to use the OCR API to extract text from PDFs and images.

---

## Quick Start

### Base URL

```
https://your-ocr-server.example.com
```

### Health Check

Verify the server is running and inspect backend availability:

```bash
curl https://your-ocr-server.example.com/health
```

**Response:**

```json
{
  "status": "idle",
  "default_backend": "auto",
  "backends": [
    { "name": "paddle", "enabled": true, "available": true, "loaded": false },
    { "name": "deepseek", "enabled": false, "available": false, "loaded": false }
  ]
}
```

After the first successful OCR request, the matching backend reports `"loaded": true` and overall `status` becomes `"loaded"`.

---

## Extract Text from a Document

### Endpoint

```
POST /v1/ocr/extract
```

### Request

| Parameter                   | Type    | Required | Description                                                          |
| --------------------------- | ------- | -------- | -------------------------------------------------------------------- |
| `file`                      | File    | Yes      | PDF or image file to process                                         |
| `backend`                   | String  | No       | `auto`, `paddle`, or `deepseek`                                      |
| `page`                      | Integer | No       | Extract only this page (1-indexed). If omitted, processes all pages. |
| `include_structured_result` | Boolean | No       | Include raw backend structured output when available (default: `false`) |

**Supported file types:**

- PDF (`.pdf`)
- Images: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.webp`

### Example: cURL

```bash
# OCR first page with automatic backend selection
curl -X POST "http://localhost:8000/v1/ocr/extract?page=1&backend=auto" \
  -F "file=@document.pdf"

# OCR fifth page with Paddle forced
curl -X POST "http://localhost:8000/v1/ocr/extract?page=5&backend=paddle" \
  -F "file=@document.pdf"

# OCR all pages (existing behavior, unchanged)
curl -X POST "http://localhost:8000/v1/ocr/extract?backend=auto" \
  -F "file=@document.pdf"
```

### Example: Python

```python
import requests

def extract_text(file_path: str, base_url: str, backend: str = "auto") -> dict:
    """
    Extract text from a PDF or image using the OCR API.

    Args:
        file_path: Path to the file to process
        base_url: OCR server URL (e.g., "https://ocr.example.com")

    Returns:
        API response with extracted text
    """
    url = f"{base_url}/v1/ocr/extract?backend={backend}"

    with open(file_path, "rb") as file:
        response = requests.post(url, files={"file": file})

    response.raise_for_status()
    return response.json()


# Usage
result = extract_text("./invoice.pdf", "https://your-ocr-server.example.com", backend="paddle")
print(result["combined_text"])
```

### Example: JavaScript/TypeScript

```typescript
async function extractText(file: File, baseUrl: string): Promise<OcrResponse> {
	const formData = new FormData();
	formData.append("file", file);

	const response = await fetch(`${baseUrl}/v1/ocr/extract?backend=auto`, {
		method: "POST",
		body: formData,
	});

	if (!response.ok) {
		throw new Error(`OCR failed: ${response.statusText}`);
	}

	return response.json();
}

// Usage
const fileInput = document.querySelector<HTMLInputElement>("#file-upload");
const file = fileInput?.files?.[0];

if (file) {
	const result = await extractText(
		file,
		"https://your-ocr-server.example.com",
	);
	console.log(result.combined_text);
}
```

---

## Response Format

```json
{
	"filename": "document.pdf",
	"total_pages": 3,
	"backend_used": "paddle",
	"pages": [
		{
			"page_number": 1,
			"markdown": "# Invoice\n\n**Date:** 2024-01-15\n...",
			"text": "Invoice\n\nDate: 2024-01-15\n...",
			"structured_result": null
		},
		{
			"page_number": 2,
			"markdown": "## Line Items\n\n| Item | Qty | Price |\n...",
			"text": "Line Items\n\nItem  Qty  Price\n...",
			"structured_result": null
		}
	],
	"combined_markdown": "# Invoice\n\n**Date:** 2024-01-15\n...\n\n## Line Items\n...",
	"combined_text": "Invoice\n\nDate: 2024-01-15\n...\n\nLine Items\n..."
}
```

### Field Descriptions

| Field                       | Type           | Description                                                     |
| --------------------------- | -------------- | --------------------------------------------------------------- |
| `filename`                  | string         | Original filename you uploaded                                  |
| `total_pages`               | integer        | Number of pages processed                                       |
| `backend_used`              | string         | Backend that actually handled the OCR request                   |
| `pages`                     | array          | Per-page results                                                |
| `pages[].page_number`       | integer        | 1-indexed page number                                           |
| `pages[].markdown`          | string         | Extracted text with markdown formatting (tables, headers, etc.) |
| `pages[].text`              | string         | Plain text version (no markdown)                                |
| `pages[].structured_result` | object or null | Raw PaddleOCR output (only if `include_structured_result=true`) |
| `combined_markdown`         | string         | All pages combined as markdown                                  |
| `combined_text`             | string         | All pages combined as plain text                                |

---

## Error Handling

The API returns standard HTTP status codes:

| Status | Meaning        | Example                               |
| ------ | -------------- | ------------------------------------- |
| `200`  | Success        | Text extracted                        |
| `400`  | Bad request    | Unsupported file type, unreadable PDF, invalid backend |
| `409`  | Conflict       | Requested backend is known but disabled |
| `413`  | File too large | Exceeds size limit or page limit      |
| `422`  | Unprocessable  | Requested backend is enabled but unavailable |
| `500`  | Server error   | OCR processing failed                 |

### Error Response Format

```json
{
	"detail": "Unsupported file type: '.doc'. Upload a PDF or image (png, jpg, bmp, tiff, webp)."
}
```

### Handling Errors in Python

```python
import requests

def extract_text_safely(file_path: str, base_url: str) -> dict | None:
    """Extract text with proper error handling."""
    url = f"{base_url}/v1/ocr/extract"

    try:
        with open(file_path, "rb") as file:
            response = requests.post(url, files={"file": file}, timeout=120)

        if response.status_code == 400:
            print(f"Invalid file: {response.json()['detail']}")
            return None

        if response.status_code == 413:
            print(f"File too large: {response.json()['detail']}")
            return None

        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        print("Request timed out — the document may be too large")
        return None

    except requests.exceptions.RequestException as error:
        print(f"Request failed: {error}")
        return None
```

---

## Limits

| Limit         | Default | Environment Variable   |
| ------------- | ------- | ---------------------- |
| Max file size | 50 MB   | `OCR_MAX_UPLOAD_BYTES` |
| Max PDF pages | 50      | `OCR_MAX_PDF_PAGES`    |

> [!NOTE]
> Large PDFs take longer to process. Expect ~2-5 seconds per page depending on complexity and GPU performance.

---

## Tips for Best Results

1. **Use high-quality scans** — 300 DPI works well for most documents
2. **Prefer PDF over images** — multi-page PDFs process more efficiently than separate images
3. **Set appropriate timeouts** — large documents can take 30+ seconds
4. **Use `combined_text` for simple cases** — only iterate `pages` if you need per-page data
5. **Cache results** — OCR is expensive, store results if you'll need them again

---

## Interactive API Docs

The server includes auto-generated API documentation:

- **Swagger UI:** `https://your-ocr-server.example.com/docs`
- **ReDoc:** `https://your-ocr-server.example.com/redoc`

You can test the API directly from the Swagger UI interface.
