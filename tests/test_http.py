"""HTTP behavior of the ASR app with a fake model: auth and overlapping requests."""

import asyncio
import base64
import io
import os
import unittest
import wave
from unittest import mock

import httpx

os.environ["API_KEY"] = "test-key"
os.environ["MODEL_IDLE_SECONDS"] = "0"

import main  # noqa: E402
from models import loader  # noqa: E402
from routes import transcribe  # noqa: E402
from tests.test_guard import OverlapModel  # noqa: E402

KEY = {"X-Api-Key": "test-key"}


def wav_base64(seconds=1.0, rate=16000):
    """Encode a loud constant tone as a base64 WAV.

    Args:
        seconds: Length of the audio.
        rate: Sample rate.

    Returns:
        The base64 text.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            int(0.1 * 32767).to_bytes(2, "little", signed=True) * int(seconds * rate)
        )
    return base64.b64encode(buf.getvalue()).decode()


class TranscribeHttpTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.model = OverlapModel(["xin chao"])
        self.enterContext(mock.patch.object(loader, "load_model", lambda _device: None))
        self.enterContext(
            mock.patch.object(transcribe, "get_model", lambda: self.model)
        )
        self.enterContext(
            mock.patch.object(transcribe, "get_processor", lambda: "proc")
        )
        self.lifespan = main.app.router.lifespan_context(main.app)
        await self.lifespan.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://asr"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)

    async def post(self, headers=KEY):
        """Send one base64 transcribe request.

        Args:
            headers: Request headers.

        Returns:
            The response.
        """
        body = {"audio_base64": wav_base64(), "language": "vi"}
        return await self.client.post("/transcribe/base64", json=body, headers=headers)

    async def test_missing_or_wrong_api_key_is_refused(self):
        self.assertEqual((await self.post(headers={})).status_code, 401)
        self.assertEqual(
            (await self.post(headers={"X-Api-Key": "wrong"})).status_code, 401
        )
        self.assertEqual(self.model.calls, [])

    async def test_overlapping_requests_all_succeed_one_transcribe_at_a_time(self):
        """23b0918: overlapping requests crashed the shared model with HTTP 500."""
        responses = await asyncio.gather(*(self.post() for _ in range(4)))
        self.assertEqual([r.status_code for r in responses], [200] * 4)
        self.assertEqual({r.json()["transcription"] for r in responses}, {"xin chao"})
        self.assertEqual(self.model.most, 1)


if __name__ == "__main__":
    unittest.main()
