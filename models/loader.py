"""
models/loader.py
Loads the CohereLabs/cohere-transcribe-03-2026 processor and model exactly
once at startup. All inference routes call get_processor() / get_model().
"""

import os
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

MODEL_ID = os.getenv("MODEL_ID", "CohereLabs/cohere-transcribe-03-2026")
HF_TOKEN = os.getenv("HF_TOKEN")  # required — model is gated
print(f"[loader] HF_TOKEN set: {bool(HF_TOKEN)}")
DEVICE = "cuda:0" if torch.cuda.is_available() and os.getenv("DEVICE", "auto") != "cpu" else "cpu"

_processor = None
_model = None


def load_model():
    """
    Download (or load from cache) the processor and model.
    Called once during application startup via the FastAPI lifespan hook.
    This is a 2B parameter model — first load will take several minutes
    and ~4 GB of disk/VRAM.
    """
    global _processor, _model

    kwargs = {"trust_remote_code": True}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    print(f"[loader] Loading processor for {MODEL_ID} …")
    _processor = AutoProcessor.from_pretrained(MODEL_ID, **kwargs)

    print(f"[loader] Loading model for {MODEL_ID} on {DEVICE} …")
    _model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, **kwargs)
    _model.to(DEVICE)
    _model.eval()

    print("[loader] Model ready.")


def get_processor() -> AutoProcessor:
    if _processor is None:
        raise RuntimeError("Processor not initialised — call load_model() first.")
    return _processor


def get_model() -> AutoModelForSpeechSeq2Seq:
    if _model is None:
        raise RuntimeError("Model not initialised — call load_model() first.")
    return _model