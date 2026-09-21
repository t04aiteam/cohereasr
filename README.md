# Cohere Transcribe API

A production-ready FastAPI service wrapping
[CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) —
a **2B-parameter Conformer ASR model** that transcribes audio in **14 languages** with best-in-class accuracy.

---

## Supported Languages

`en` `de` `fr` `it` `es` `pt` `el` `nl` `pl` `ar` `vi` `zh` `ja` `ko`

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | |
| ~4 GB disk | Model weights download on first startup |
| ~4 GB VRAM (GPU) | Or ~8 GB RAM for CPU inference |
| HF_TOKEN | Gated model — accept terms then create a token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens) |
| ffmpeg | Required for mp3 / ogg decoding (`apt install ffmpeg` / `brew install ffmpeg`) |

---

## Setup

### 1. Clone & install

```bash
git clone <this-repo>
cd cohere_asr

# (recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set HF_TOKEN (required) and API_KEY
```

### 3. Run

```bash
uvicorn main:app --host 0.0.0.0 --port 6221
# or equivalently:
python main.py
```

The `.env` file is loaded automatically at startup. The model (~4 GB) is
downloaded from the Hub on the first startup and cached under
`~/.cache/huggingface`; subsequent starts use the local cache.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | **Yes** | — | Hugging Face access token for the gated model |
| `API_KEY` | No | *(empty)* | Protects endpoints via `X-Api-Key` header. Leave empty to disable auth |
| `MODEL_ID` | No | `CohereLabs/cohere-transcribe-03-2026` | HuggingFace model ID |
| `DEVICE` | No | auto | Set to `cpu` to force CPU inference. Ignored while the `DEVICE` constant in `main.py` is not `None` (currently `"cpu"`: the GPU on the deploy box is shared) |
| `GUARD_RMS_MIN_DB` | No | `-50` | Hallucination guard: a chunk whose loudest 1 s is quieter than this (dBFS) is dropped as silence |
| `USE_SILERO_VAD` | No | `0` | `1` = transcribe only the regions Silero VAD marks as speech. For spoken audio only: Silero rates singing as non-speech and returns an empty transcript for songs. Needs `requirements-vad.txt`, the service does not start without it |
| `VAD_THRESHOLD` | No | `0.5` | Silero speech probability threshold |

### Hallucination guard

The model invents text on audio without speech (instrumental intro, silence) and can loop until the token limit. `models/guard.py` transcribes chunk by chunk and drops chunks that are a loop (one phrase repeated back-to-back more than 4 times over 30+ words, or characters for zh/ja) or silent, each drop is logged as `[guard] dropped chunk ...`. Known limits: invented text that does not loop passes through, a loop with a phrase longer than 8 words or with small variations passes, a real chant of one short phrase 7+ times in a row is dropped, and a drop removes the whole chunk (up to 35 s).

- Tests (no model, no torch): `cohere_env/bin/python -m unittest discover -s tests -v`. `SLOW_TESTS=1` adds the real Silero model.
- Eval (needs the service up, ~100 s per song): `cohere_env/bin/python evals/eval_hallucination.py song.mp3 --must "chim rừng" --forbid "nghiện"`

---

## API Reference

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | None | Service liveness check |
| `POST` | `/transcribe/file` | `X-Api-Key` | Transcribe a multipart audio file upload |
| `POST` | `/transcribe/base64` | `X-Api-Key` | Transcribe base64-encoded audio in JSON |

Interactive docs available at **http://localhost:6221/docs** once the server is running.

---

## Example Requests

### Health check

```bash
curl http://localhost:6221/health
```

```json
{"status": "ok", "model": "CohereLabs/cohere-transcribe-03-2026"}
```

---

### Transcribe a file (multipart upload)

```bash
curl -X POST http://localhost:6221/transcribe/file \
  -H "X-Api-Key: changeme" \
  -F "file=@/path/to/audio.wav" \
  -F "language=en" \
  -F "punctuation=true"
```

```json
{
  "transcription": "Hello, this is a test of the Cohere Transcribe API.",
  "language": "en",
  "model": "CohereLabs/cohere-transcribe-03-2026"
}
```

---

### Transcribe via base64 JSON

```bash
# Encode audio to base64
B64=$(base64 -w 0 /path/to/audio.mp3)

curl -X POST http://localhost:6221/transcribe/base64 \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: changeme" \
  -d "{
    \"audio_base64\": \"$B64\",
    \"language\": \"ja\",
    \"punctuation\": true
  }"
```

```json
{
  "transcription": "こんにちは、これはテストです。",
  "language": "ja",
  "model": "CohereLabs/cohere-transcribe-03-2026"
}
```

---

### Transcribe French audio

```bash
curl -X POST http://localhost:6221/transcribe/file \
  -H "X-Api-Key: changeme" \
  -F "file=@interview.flac" \
  -F "language=fr"
```

---

## Notes & Tips

- **Long audio**: The model automatically chunks audio longer than 35 seconds — no special configuration needed.
- **Large model**: First startup downloads ~4 GB to `~/.cache/huggingface`. Subsequent starts load from the cache.
- **CPU inference**: Transcribing a 1-minute clip takes ~3–5 minutes on CPU. A GPU is strongly recommended for production use.
- **GPU acceleration**: `main.py` pins `DEVICE = "cpu"`. Set it to `None` to auto-select CUDA, after which the `DEVICE=cpu` env var applies again.
- **Batch size**: Both endpoints accept an optional `batch_size` parameter to override the default GPU batch size for chunked long-audio inference.
- **Noise gate**: The model transcribes non-speech sounds. Pre-processing with a VAD (e.g. Silero VAD) is recommended for noisy environments.
- **No language detection**: You must specify the `language` parameter — the model does not auto-detect it.
- **Concurrency**: The API uses a single-process, single-model setup. For high-concurrency workloads, consider the vLLM serving integration documented on the [model card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).