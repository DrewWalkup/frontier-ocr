from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from frontier_ocr.api.ocr_routes import router as ocr_router
from frontier_ocr.core.config import settings
from frontier_ocr.services.backend_registry import OcrBackendRegistry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle while keeping OCR model loading lazy."""
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app.state.ocr_registry = OcrBackendRegistry.from_settings(settings)
    logger.info("OCR backend registry initialized")

    yield

    logger.info("Shutting down OCR backends")
    app.state.ocr_registry.shutdown()


app = FastAPI(title=settings.project_name, lifespan=lifespan)
app.include_router(ocr_router)
