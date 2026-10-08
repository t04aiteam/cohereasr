"""
Cohere Transcribe API
Wraps CohereLabs/cohere-transcribe-03-2026 — a 2B-parameter ASR model
supporting 14 languages. Audio is accepted as file upload or base64 payload.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load .env before importing modules that read env vars at import time
# (models.loader reads HF_TOKEN, middleware.auth reads API_KEY).
from dotenv import load_dotenv
load_dotenv()

from models.loader import load_model
from routes.transcribe import router as transcribe_router

# DEVICE and PORT come from the environment (real env first, then .env).
# Unset keeps box 222: CPU (its GPU is shared) on 6221. c09 starts with DEVICE=cuda PORT=8000.
DEVICE = os.environ.get("DEVICE", "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release on shutdown."""
    load_model(DEVICE)
    yield


app = FastAPI(
    title="Cohere Transcribe API",
    description=(
        "REST API for CohereLabs/cohere-transcribe-03-2026 — "
        "a 2B-parameter Conformer ASR model supporting 14 languages."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcribe_router)


@app.get("/health", tags=["Health"])
async def health():
    """Returns service status and the loaded model identifier."""
    return {
        "status": "ok",
        "model": "CohereLabs/cohere-transcribe-03-2026",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all handler so unhandled errors return structured JSON."""
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "6221")))
