"""
schemas/request.py
Pydantic models for the /transcribe endpoints.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

# Supported ISO 639-1 codes for this model
SUPPORTED_LANGUAGES = {
    "en", "de", "fr", "it", "es", "pt", "el",
    "nl", "pl", "ar", "vi", "zh", "ja", "ko",
}


class TranscribeJsonRequest(BaseModel):
    """
    Request body for POST /transcribe/base64.
    Send a single audio file encoded as a base64 string.
    """

    audio_base64: str = Field(
        ...,
        description="Raw audio file bytes encoded as base64. Most formats supported (wav, mp3, flac, ogg, …).",
    )
    language: str = Field(
        "en",
        description="ISO 639-1 language code. Must be one of the 14 supported languages.",
        examples=["en", "ja", "fr"],
    )
    punctuation: bool = Field(True, description="Include punctuation in transcription output.")
    batch_size: Optional[int] = Field(None, description="GPU batch size. Defaults to model config value.")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{v}'. Supported codes: {sorted(SUPPORTED_LANGUAGES)}"
            )
        return v


class TranscribeResponse(BaseModel):
    """Transcription result for a single audio input."""

    transcription: str = Field(..., description="Transcribed text.")
    language: str = Field(..., description="Language code used for transcription.")
    model: str = Field(..., description="Model identifier.")