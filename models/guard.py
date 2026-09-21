"""
models/guard.py
Hallucination guard around model.transcribe().

The model invents text on audio that holds no speech (instrumental intro, silence):
greedy decoding runs to max_new_tokens and loops. This module transcribes chunk by
chunk and drops chunks whose text is a repetition loop or whose audio is silent.

Optional Silero VAD pre-segmentation: set USE_SILERO_VAD=1. Off by default because
Silero rates singing as non-speech and would empty a song's transcript.
"""

import os
import sys
import zlib

import librosa
import numpy as np
import torch

SR = 16000
CR_MAX = float(
    os.getenv("GUARD_CR_MAX", "2.4")
)  # Whisper's compression_ratio_threshold
MIN_LOOP_WORDS = int(
    os.getenv("GUARD_MIN_LOOP_WORDS", "50")
)  # measured loops: 71+ words
RMS_MIN_DB = float(os.getenv("GUARD_RMS_MIN_DB", "-50"))
USE_SILERO_VAD = os.getenv("USE_SILERO_VAD", "0") == "1"
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_MAX_GROUP_S = 30.0
VAD_PAD_S = 0.2

_vad_model = None


def hallucination_reason(wave, text):
    """Decide whether a chunk's text should be dropped.

    Args:
        wave: Chunk waveform.
        text: Text the model produced for it.

    Returns:
        Reason string when the chunk is dropped, else None.
    """
    rms_db = (
        20 * np.log10(np.sqrt(np.mean(np.square(wave))) + 1e-9) if len(wave) else -180.0
    )
    if rms_db < RMS_MIN_DB:
        return f"silence {rms_db:.0f} dBFS"
    raw = text.encode()
    cr = len(raw) / len(zlib.compress(raw)) if raw else 0.0
    # ponytail: word floor protects short sung refrains ("nhà em ở nơi đó" x4 = cr 2.6).
    # A 35 s chunk that really repeats one line 10+ times is still dropped; upgrade path
    # is a per-token logprob check once the decode loop is ours.
    if cr > CR_MAX and len(text.split()) >= MIN_LOOP_WORDS:
        return f"loop cr={cr:.1f}"
    return None


def pack_regions(regions, total_s, max_s=VAD_MAX_GROUP_S, pad_s=VAD_PAD_S):
    """Merge speech regions into padded groups no longer than max_s.

    Args:
        regions: List of {"start", "end"} dicts in seconds, sorted.
        total_s: Audio length in seconds.
        max_s: Longest group before padding.
        pad_s: Context kept on both sides of a group.

    Returns:
        List of (start_s, end_s) tuples.
    """
    groups = []
    for r in regions:
        if groups and r["end"] - groups[-1][0] <= max_s:
            groups[-1][1] = r["end"]
        else:
            groups.append([r["start"], r["end"]])
    return [(max(0.0, s - pad_s), min(total_s, e + pad_s)) for s, e in groups]


def vad_chunks(wav):
    """Cut a 16 kHz waveform down to its Silero speech regions.

    Args:
        wav: Mono 16 kHz float32 waveform.

    Returns:
        List of waveforms, empty when no speech was found.
    """
    global _vad_model
    threads = torch.get_num_threads()
    from silero_vad import (
        get_speech_timestamps,
        load_silero_vad,
    )  # lazy: optional dependency

    if _vad_model is None:
        _vad_model = load_silero_vad()
    regions = get_speech_timestamps(
        torch.from_numpy(wav),
        _vad_model,
        sampling_rate=SR,
        threshold=VAD_THRESHOLD,
        return_seconds=True,
    )
    # importing silero_vad calls torch.set_num_threads(1) process-wide, which would
    # put the ASR model on one core
    torch.set_num_threads(threads)
    return [
        wav[int(s * SR) : int(e * SR)] for s, e in pack_regions(regions, len(wav) / SR)
    ]


def transcribe_guarded(
    model, processor, audio_array, sample_rate, language, punctuation, batch_size=None
):
    """Transcribe one waveform chunk by chunk and drop hallucinated chunks.

    Args:
        model: Loaded ASR model.
        processor: Its processor.
        audio_array: Mono float32 waveform.
        sample_rate: Its sample rate.
        language: ISO 639-1 code.
        punctuation: Whether to keep punctuation.
        batch_size: Optional batch size override.

    Returns:
        The joined text of the chunks that were kept.
    """
    wav = audio_array
    if sample_rate != SR:
        wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=SR)
    pieces = vad_chunks(wav) if USE_SILERO_VAD else [wav]
    if USE_SILERO_VAD:
        print(
            f"[guard] silero kept {sum(len(p) for p in pieces) / SR:.1f}s of {len(wav) / SR:.1f}s"
        )

    cfg = model.config
    module = sys.modules[type(model).__module__]
    split = module.split_audio_chunks_energy
    chunks = []
    # same split model.transcribe() does, so chunks are not re-split
    for piece in pieces:
        chunks += split(
            piece,
            SR,
            float(cfg.max_audio_clip_s),
            float(cfg.overlap_chunk_second),
            int(cfg.min_energy_window_samples),
        )
    if not chunks:
        return ""

    kwargs = {
        "processor": processor,
        "audio_arrays": chunks,
        "sample_rates": [SR] * len(chunks),
        "language": language,
        "punctuation": punctuation,
    }
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    texts = model.transcribe(**kwargs)

    kept = []
    for i, (wave, text) in enumerate(zip(chunks, texts)):
        why = hallucination_reason(wave, text)
        if why:
            print(f"[guard] dropped chunk {i + 1}/{len(chunks)} ({why}): {text[:80]!r}")
        else:
            kept.append(text)
    return module.join_chunk_texts(kept, module.get_chunk_separator(language))
