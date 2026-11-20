"""
models/loader.py
Loads the CohereLabs/cohere-transcribe-03-2026 processor and model exactly
at startup and reloads it on the next inference after an idle unload.
"""

import gc
import os
from ctypes import CDLL
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from models.idle import IdleModel

MODEL_ID = os.getenv("MODEL_ID", "CohereLabs/cohere-transcribe-03-2026")
HF_TOKEN = os.getenv("HF_TOKEN")  # required — model is gated
print(f"[loader] HF_TOKEN set: {bool(HF_TOKEN)}")
DEVICE = "cuda:0" if torch.cuda.is_available() and os.getenv("DEVICE", "auto") != "cpu" else "cpu"

_processor = None
_model = None


def load_model(device=None):
    """
    Download (or load from cache) the processor and model.
    Called once during application startup via the FastAPI lifespan hook.
    This is a 2B parameter model — first load will take several minutes
    and ~4 GB of disk/VRAM.
    """
    global _processor, _model

    device = device or DEVICE

    kwargs = {"trust_remote_code": True}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    print(f"[loader] Loading processor for {MODEL_ID} …")
    processor = AutoProcessor.from_pretrained(MODEL_ID, **kwargs)

    print(f"[loader] Loading model for {MODEL_ID} on {device} …")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, **kwargs)
    model.to(device)
    model.eval()
    _model, _processor = model, processor

    print("[loader] Model ready.")


def unload_model():
    global _model, _processor
    _model = _processor = None
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    # glibc may retain several GiB of free pages after model loading.
    trim = getattr(CDLL(None), "malloc_trim", None)
    if trim is not None:
        trim(0)


idle_model = IdleModel(
    lambda: load_model(os.environ.get("DEVICE", "cpu")),
    unload_model,
    int(os.getenv("MODEL_IDLE_SECONDS", "0")),
)


def get_processor() -> AutoProcessor:
    if _processor is None:
        raise RuntimeError("Processor not initialised — call load_model() first.")
    return _processor


def get_model() -> AutoModelForSpeechSeq2Seq:
    if _model is None:
        raise RuntimeError("Model not initialised — call load_model() first.")
    return _model