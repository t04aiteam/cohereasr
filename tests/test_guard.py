"""Gate tests for models/guard.py. No model and no torch import.

SLOW_TESTS=1 adds the real Silero model (imports torch, not a gate test).

Run: cohere_env/bin/python -m unittest discover -s tests -v
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import guard  # noqa: E402

LOOP = "Một số người nghiện rõ ràng là " + "một người nghiện rõ ràng là " * 40
LYRIC = "nhà em ở lưng đồi nơi chim rừng thanh thót bầu trời xanh dịu ngọt gió tràn về mênh mang"
LOUD = np.full(guard.SR, 0.1, np.float32)


def split_audio_chunks_energy(wav, sr, max_s, overlap_s, min_window):
    """Stand in for the model's splitter: fixed 1 s chunks.

    Args:
        wav: Waveform.
        sr: Sample rate.
        max_s: Unused.
        overlap_s: Unused.
        min_window: Unused.

    Returns:
        List of 1 s waveforms.
    """
    return [wav[i : i + sr] for i in range(0, len(wav), sr)]


def join_chunk_texts(texts, separator=" "):
    """Stand in for the model's joiner.

    Args:
        texts: Chunk texts.
        separator: Join string.

    Returns:
        Joined text.
    """
    return separator.join(t.strip() for t in texts if t.strip())


def get_chunk_separator(language):
    """Stand in for the model's separator lookup.

    Args:
        language: ISO 639-1 code.

    Returns:
        Empty string for no-space languages, else a space.
    """
    return "" if language in ("zh", "ja") else " "


class FakeConfig:
    """Chunking config the guard reads."""

    max_audio_clip_s = 35
    """Longest chunk in seconds."""
    overlap_chunk_second = 5
    """Split search window."""
    min_energy_window_samples = 1600
    """Energy window."""


class FakeModel:
    """Model whose remote-code module is this test module."""

    config = FakeConfig()
    """Config stub."""

    def __init__(self, texts):
        self.texts = texts
        self.calls = []

    def transcribe(self, **kwargs):
        """Record the call and return one canned text per chunk.

        Args:
            **kwargs: What the guard passed.

        Returns:
            Canned texts, one per chunk.
        """
        self.calls.append(kwargs)
        return self.texts[: len(kwargs["audio_arrays"])]


class HallucinationReasonTest(unittest.TestCase):
    """Drop rules."""

    def test_loop_is_dropped(self):
        """The tester's loop text trips the compression-ratio rule."""
        self.assertIn("loop", guard.hallucination_reason(LOUD, LOOP))

    def test_silence_is_dropped_whatever_the_text(self):
        """Digital silence is dropped even with plausible text."""
        self.assertIn(
            "silence", guard.hallucination_reason(np.zeros(guard.SR, np.float32), LYRIC)
        )

    def test_real_lyrics_and_refrain_are_kept(self):
        """A sung refrain repeated four times stays under the threshold."""
        self.assertIsNone(guard.hallucination_reason(LOUD, LYRIC))
        self.assertIsNone(guard.hallucination_reason(LOUD, "nhà em ở nơi đó " * 4))

    def test_legit_repetition_is_kept(self):
        """Review M1: the zlib rule dropped all of these."""
        chorus = LYRIC + " "
        counting = " ".join(f"số {i}" for i in range(1, 61))
        for text in (chorus * 3, chorus * 4, counting, "la " * 7, "na na na " * 9):
            self.assertIsNone(guard.hallucination_reason(LOUD, text), text[:30])

    def test_batched_loop_from_the_service_is_dropped(self):
        """The loop the live service returned for the intro of NhaEmOLungDoi.mp3."""
        text = "Một số người nghiện rằng " + "là người nghiện rằng mình " * 11
        self.assertIn("x11", guard.hallucination_reason(LOUD, text))

    def test_loop_boundaries(self):
        """More than RUN_MAX repeats and at least MIN_LOOP_SPAN tokens."""
        self.assertIsNone(guard.find_loop("a b c d e f g h " * 4))  # 32 tokens, x4
        self.assertIsNone(guard.find_loop("a b c d e " * 5 + "a b c d"))  # x5, 25 tokens
        self.assertEqual(guard.find_loop("a b c d e " * 6), (6, "a b c d e"))

    def test_no_space_language_loop(self):
        """Review M2: zh/ja text has no spaces, the word rule never fired."""
        self.assertIn("loop", guard.hallucination_reason(LOUD, "明显是一个人" * 60, True))
        self.assertIn(
            "loop", guard.hallucination_reason(LOUD, "これは明らかに一人の人です" * 30, True)
        )
        self.assertIsNone(guard.hallucination_reason(LOUD, "你好世界，今天天气很好。哈哈哈哈哈", True))

    def test_quiet_sentence_in_silent_chunk_is_kept(self):
        """Review M3: 2 s at -40 dBFS in 35 s of silence averaged to -52 dBFS."""
        wave = np.zeros(35 * guard.SR, np.float32)
        wave[: 2 * guard.SR] = 0.01
        self.assertIsNone(guard.hallucination_reason(wave, LYRIC))

    def test_empty_inputs(self):
        """Empty text is kept, empty audio counts as silence."""
        self.assertIsNone(guard.hallucination_reason(LOUD, ""))
        self.assertIn(
            "silence", guard.hallucination_reason(np.zeros(0, np.float32), "x")
        )


class PackRegionsTest(unittest.TestCase):
    """Silero region packing."""

    def test_groups_cap_and_padding(self):
        """Regions pad by 0.2 s, clamp to the audio and pack up to 30 s of audio."""
        regions = [
            {"start": 0.1, "end": 10},
            {"start": 12, "end": 29},
            {"start": 40, "end": 50},
        ]
        self.assertEqual(
            guard.pack_regions(regions, 50.1),
            [[(0.0, 10.2), (11.8, 29.2)], [(39.8, 50.1)]],
        )

    def test_gap_is_left_out_and_overlap_is_merged(self):
        """Review L4: music between two regions is not transcribed, no audio twice."""
        regions = [{"start": 0, "end": 2}, {"start": 28, "end": 30}, {"start": 30.3, "end": 31}]
        self.assertEqual(guard.pack_regions(regions, 60), [[(0.0, 2.2), (27.8, 31.2)]])

    def test_no_speech(self):
        """No regions gives no groups."""
        self.assertEqual(guard.pack_regions([], 10), [])


@unittest.skipUnless(
    os.getenv("SLOW_TESTS") and importlib.util.find_spec("silero_vad"),
    "set SLOW_TESTS=1 (needs silero-vad, imports torch)",
)
class SileroTest(unittest.TestCase):
    """Real Silero model, no ASR model."""

    def test_noise_has_no_speech_and_threads_survive(self):
        """Importing silero_vad sets torch to 1 thread; vad_chunks must undo that."""
        import torch

        torch.set_num_threads(4)
        noise = (np.random.default_rng(0).standard_normal(2 * guard.SR) * 0.05).astype(
            np.float32
        )
        self.assertEqual(guard.vad_chunks(noise), [])
        self.assertEqual(torch.get_num_threads(), 4)


class TranscribeGuardedTest(unittest.TestCase):
    """End-to-end with a fake model."""

    def test_drops_loop_chunk_and_keeps_the_rest(self):
        """Regression for NhaEmOLungDoi.mp3: intro loop removed, lyrics kept."""
        model = FakeModel([LOOP, LYRIC, "em về nơi lưng đồi"])
        text = guard.transcribe_guarded(
            model, "proc", np.tile(LOUD, 3), guard.SR, "vi", True
        )
        self.assertEqual(text, LYRIC + " em về nơi lưng đồi")
        self.assertEqual(len(model.calls[0]["audio_arrays"]), 3)
        self.assertNotIn("batch_size", model.calls[0])

    def test_resamples_and_passes_batch_size(self):
        """A 32 kHz input reaches the model as 16 kHz chunks."""
        model = FakeModel(["a b c", "d e f"])
        guard.transcribe_guarded(
            model, "proc", np.tile(LOUD, 4), 32000, "vi", True, batch_size=2
        )
        call = model.calls[0]
        self.assertEqual(
            (len(call["audio_arrays"]), call["sample_rates"], call["batch_size"]),
            (2, [16000] * 2, 2),
        )

    def test_no_space_language_join(self):
        """Chinese chunks join without a space, like model.transcribe does."""
        model = FakeModel(["你好", "世界"])
        self.assertEqual(
            guard.transcribe_guarded(
                model, "proc", np.tile(LOUD, 2), guard.SR, "zh", True
            ),
            "你好世界",
        )

    def test_empty_audio(self):
        """Zero-length audio returns an empty string without calling the model."""
        model = FakeModel([])
        self.assertEqual(
            guard.transcribe_guarded(
                model, "proc", np.zeros(0, np.float32), guard.SR, "vi", True
            ),
            "",
        )
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
