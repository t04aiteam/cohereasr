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
cd cohere-transcribe-api
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set HF_TOKEN (required) and API_KEY
```

### 3. Run locally

```bash
uvicorn main:app --reload --port 8000
```

The model (~4 GB) is downloaded from the Hub on the first startup. Subsequent starts use the local cache.

### 4. Run with Docker

```bash
docker-compose up --build
```

For GPU inference, uncomment the `deploy` block in `docker-compose.yml` and ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | **Yes** | — | Hugging Face access token for the gated model |
| `API_KEY` | No | *(empty)* | Protects endpoints via `X-Api-Key` header. Leave empty to disable auth |
| `MODEL_ID` | No | `CohereLabs/cohere-transcribe-03-2026` | HuggingFace model ID |
| `DEVICE` | No | auto | Set to `cpu` to force CPU inference |

---

## API Reference

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | None | Service liveness check |
| `POST` | `/transcribe/file` | `X-Api-Key` | Transcribe a multipart audio file upload |
| `POST` | `/transcribe/base64` | `X-Api-Key` | Transcribe base64-encoded audio in JSON |

Interactive docs available at **http://localhost:8000/docs** once the server is running.

---

## Example Requests

### Health check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model": "CohereLabs/cohere-transcribe-03-2026"}
```

---

### Transcribe a file (multipart upload)

```bash
curl -X POST http://localhost:8000/transcribe/file \
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

curl -X POST http://localhost:8000/transcribe/base64 \
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
curl -X POST http://localhost:8000/transcribe/file \
  -H "X-Api-Key: changeme" \
  -F "file=@interview.flac" \
  -F "language=fr"
```

---

## Notes & Tips

- **Long audio**: The model automatically chunks audio longer than 35 seconds — no special configuration needed.
- **Large model**: First startup downloads ~4 GB. Mount `~/.cache/huggingface` as a Docker volume to persist across rebuilds (see commented block in `docker-compose.yml`).
- **CPU inference**: Transcribing a 1-minute clip takes ~3–5 minutes on CPU. A GPU is strongly recommended for production use.
- **GPU acceleration**: Set `DEVICE=cuda` (or leave as `auto`) and uncomment the `deploy` block in `docker-compose.yml`.
- **Noise gate**: The model transcribes non-speech sounds. Pre-processing with a VAD (e.g. Silero VAD) is recommended for noisy environments.
- **No language detection**: You must specify the `language` parameter — the model does not auto-detect it.
- **Concurrency**: The API uses a single-process, single-model setup. For high-concurrency workloads, consider the vLLM serving integration documented on the [model card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).