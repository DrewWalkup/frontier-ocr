from __future__ import annotations

import argparse

import uvicorn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frontier-ocr",
        description="Run the Frontier OCR API server.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    uvicorn.run(
        "frontier_ocr.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
