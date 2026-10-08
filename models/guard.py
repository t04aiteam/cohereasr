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
import re
import sys
import threading

import numpy as np
import soxr

SR = 16000
RUN_MAX = 4  # back-to-back repeats a real refrain reaches; measured loops: 11+
MIN_LOOP_SPAN = 30  # tokens the repeats must cover, keeps "la la la la la la"
RMS_MIN_DB = float(os.getenv("GUARD_RMS_MIN_DB", "-50"))
USE_SILERO_VAD = os.getenv("USE_SILERO_VAD", "0") == "1"
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_MAX_GROUP_S = 30.0
VAD_PAD_S = 0.2

_vad = None
_vad_lock = threading.Lock()
# One model is shared by every request thread; overlapping generate calls
# corrupt its cache (IndexError in index_copy_). One at a time.
_model_lock = threading.Lock()


def find_loop(text, by_char=False):
    """Find a back-to-back repetition long enough to be a decoder loop.

    Args:
        text: Chunk text or whole transcript.
        by_char: Compare characters instead of words (languages without spaces).

    Returns:
        Tuple of (repeat count, phrase) for the loop covering the most tokens, or
        None when nothing repeats more than RUN_MAX times over MIN_LOOP_SPAN tokens.
    """
    tokens = re.findall(r"\w" if by_char else r"\w+", text.lower())
    best = None
    # ponytail: exact repeats with a period up to 8 words / 32 characters. A loop with
    # a longer period or small variations passes; upgrade path is a per-token logprob
    # check once the decode loop is ours.
    for p in range(1, (32 if by_char else 8) + 1):
        matches = (
            0  # consecutive positions where the next p tokens equal the previous p
        )
        for i in range(p, len(tokens) - p + 1):
            matches = matches + 1 if tokens[i : i + p] == tokens[i - p : i] else 0
            # k back-to-back copies of a p-token phrase give k*p - 2p + 1 matches
            reps = 2 + (matches - 1) // p if matches else 1
            if (
                reps > RUN_MAX
                and reps * p >= MIN_LOOP_SPAN
                and (not best or reps * p > best[0] * best[2])
            ):
                best = (reps, ("" if by_char else " ").join(tokens[i : i + p]), p)
    return best[:2] if best else None


def hallucination_reason(wave, text, by_char=False):
    """Decide whether a chunk's text should be dropped.

    Args:
        wave: Chunk waveform at 16 kHz.
        text: Text the model produced for it.
        by_char: Whether the language is written without spaces.

    Returns:
        Reason string when the chunk is dropped, else None.
    """
    if not len(wave):
        return "silence (empty)"
    # loudest 1 s window, not the chunk mean: one quiet sentence in a silent chunk stays
    n = len(wave) // SR
    frames = wave[: n * SR].reshape(n, SR) if n else wave[None, :]
    rms_db = 20 * np.log10(np.sqrt(np.mean(np.square(frames), axis=1)).max() + 1e-9)
    if rms_db < RMS_MIN_DB:
        return f"silence {rms_db:.0f} dBFS"
    loop = find_loop(text, by_char)
    if loop:
        return f"loop {loop[1]!r} x{loop[0]}"
    return None


def pack_regions(regions, total_s, max_s=VAD_MAX_GROUP_S, pad_s=VAD_PAD_S):
    """Pack padded speech regions into groups of at most max_s of audio.

    Args:
        regions: List of {"start", "end"} dicts in seconds, sorted.
        total_s: Audio length in seconds.
        max_s: Most audio in one group. One longer region still makes one group.
        pad_s: Context kept on both sides of a region.

    Returns:
        List of groups, each a list of (start_s, end_s) tuples. The audio between
        the regions of a group is not part of it.
    """
    groups = []
    for r in regions:
        start, end = max(0.0, r["start"] - pad_s), min(total_s, r["end"] + pad_s)
        if groups and start <= groups[-1][-1][1]:  # padding overlap: no audio twice
            groups[-1][-1] = (groups[-1][-1][0], end)
        elif groups and sum(e - s for s, e in groups[-1]) + end - start <= max_s:
            groups[-1].append((start, end))
        else:
            groups.append([(start, end)])
    return groups


def _load_vad():
    """Import silero_vad and load its model, keeping torch's thread count."""
    global _vad
    import torch

    # importing silero_vad calls torch.set_num_threads(1) process-wide, which would
    # put the ASR model on one core
    threads = torch.get_num_threads()
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    finally:
        torch.set_num_threads(threads)
    _vad = (get_speech_timestamps, load_silero_vad())


if USE_SILERO_VAD:
    # at startup, not per request: a missing dependency stops the service start, and
    # the thread count is back before the first request
    _load_vad()


def vad_chunks(wav):
    """Cut a 16 kHz waveform down to its Silero speech regions.

    Args:
        wav: Mono 16 kHz float32 waveform.

    Returns:
        List of waveforms, empty when no speech was found.
    """
    import torch  # lazy: 1.5 s import the gate tests must not pay

    with _vad_lock:  # the Silero model keeps state between frames
        if _vad is None:
            _load_vad()
        get_speech_timestamps, vad_model = _vad
        regions = get_speech_timestamps(
            torch.from_numpy(wav),
            vad_model,
            sampling_rate=SR,
            threshold=VAD_THRESHOLD,
            return_seconds=True,
        )
    return [
        np.concatenate([wav[int(s * SR) : int(e * SR)] for s, e in group])
        for group in pack_regions(regions, len(wav) / SR)
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
    if not len(wav):
        return ""
    if sample_rate != SR:
        wav = soxr.resample(wav, sample_rate, SR)  # what librosa.resample runs, minus its 1.7 s import
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
    with _model_lock:
        texts = model.transcribe(**kwargs)

    separator = module.get_chunk_separator(language)
    kept = []
    for i, (wave, text) in enumerate(zip(chunks, texts)):
        why = hallucination_reason(wave, text, by_char=separator == "")
        if why:
            print(f"[guard] dropped chunk {i + 1}/{len(chunks)} ({why}): {text[:80]!r}")
        else:
            kept.append(text)
    return module.join_chunk_texts(kept, separator)
