"""
routes/transcribe.py
Provides two transcription endpoints:

  POST /transcribe/file   — multipart file upload  (recommended for large audio)
  POST /transcribe/base64 — JSON body with base64-encoded audio

Both endpoints run inference via model.transcribe() on a thread-pool executor
so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import tempfile
from typing import Annotated, Optional

import numpy as np
import soundfile as sf
import librosa
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from middleware.auth import verify_api_key
from models.loader import get_model, get_processor
from schemas.request import SUPPORTED_LANGUAGES, TranscribeJsonRequest, TranscribeResponse

MODEL_NAME = os.getenv("MODEL_ID", "CohereLabs/cohere-transcribe-03-2026")

router = APIRouter(prefix="", tags=["Transcription"])


# ---------------------------------------------------------------------------
# Shared inference helper
# ---------------------------------------------------------------------------

def _run_transcription(
    audio_array: np.ndarray,
    sample_rate: int,
    language: str,
    punctuation: bool,
    batch_size: Optional[int],
) -> str:
    """
    Synchronous wrapper around model.transcribe().
    Called via run_in_executor to avoid blocking the async event loop.
    """
    model = get_model()
    processor = get_processor()

    kwargs: dict = {
        "processor": processor,
        "audio_arrays": [audio_array],
        "sample_rates": [sample_rate],
        "language": language,
        "punctuation": punctuation,
    }
    if batch_size is not None:
        kwargs["batch_size"] = batch_size

    results = model.transcribe(**kwargs)
    return results[0]


async def _transcribe_array(
    audio_array: np.ndarray,
    sample_rate: int,
    language: str,
    punctuation: bool,
    batch_size: Optional[int],
) -> TranscribeResponse:
    """Async wrapper: runs blocking inference on the default thread-pool."""
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None,
        _run_transcription,
        audio_array,
        sample_rate,
        language,
        punctuation,
        batch_size,
    )
    return TranscribeResponse(transcription=text, language=language, model=MODEL_NAME)


# ---------------------------------------------------------------------------
# Endpoint 1: multipart file upload
# ---------------------------------------------------------------------------

@router.post(
    "/transcribe/file",
    response_model=TranscribeResponse,
    summary="Transcribe an uploaded audio file",
    description=(
        "Upload an audio file (wav, mp3, flac, ogg, …) and receive its transcription. "
        "Audio longer than 35 s is automatically chunked. "
        "Requires `X-Api-Key` header when `API_KEY` env var is set."
    ),
    dependencies=[Depends(verify_api_key)],
)
async def transcribe_file(
    file: UploadFile = File(..., description="Audio file to transcribe."),
    language: str = Form("en", description=f"ISO 639-1 language code. Supported: {sorted(SUPPORTED_LANGUAGES)}"),
    punctuation: bool = Form(True, description="Include punctuation."),
    batch_size: Optional[int] = Form(None, description="GPU batch size (optional)."),
):
    """
    Transcribe audio submitted as a multipart file upload.

    - **file**: any audio format supported by soundfile / librosa
    - **language**: one of the 14 supported ISO 639-1 codes
    - **punctuation**: set to false to strip punctuation from output
    - **batch_size**: override the default GPU batch size
    """
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}",
        )

    audio_bytes = await file.read()

    try:
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        # Fallback to librosa for mp3 and other compressed formats
        try:
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename or ".tmp")[1] or ".tmp", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            audio_array, sample_rate = librosa.load(tmp_path, sr=None, mono=True)
            os.unlink(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")

    # Ensure mono float32
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    audio_array = audio_array.astype(np.float32)

    return await _transcribe_array(audio_array, sample_rate, language, punctuation, batch_size)


# ---------------------------------------------------------------------------
# Endpoint 2: base64 JSON body
# ---------------------------------------------------------------------------

@router.post(
    "/transcribe/base64",
    response_model=TranscribeResponse,
    summary="Transcribe base64-encoded audio",
    description=(
        "Submit audio as a base64-encoded string inside a JSON body. "
        "Useful for programmatic clients that cannot send multipart forms. "
        "Requires `X-Api-Key` header when `API_KEY` env var is set."
    ),
    dependencies=[Depends(verify_api_key)],
)
async def transcribe_base64(body: TranscribeJsonRequest):
    """
    Transcribe audio provided as base64.

    - **audio_base64**: standard base64 encoding of the raw audio file bytes
    - **language**: ISO 639-1 code (e.g. `"en"`, `"ja"`, `"fr"`)
    - **punctuation**: include punctuation in output
    - **batch_size**: optional GPU batch size override
    """
    try:
        audio_bytes = base64.b64decode(body.audio_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")

    try:
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        try:
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            audio_array, sample_rate = librosa.load(tmp_path, sr=None, mono=True)
            os.unlink(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")

    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    audio_array = audio_array.astype(np.float32)

    return await _transcribe_array(
        audio_array, sample_rate, body.language, body.punctuation, body.batch_size
    )