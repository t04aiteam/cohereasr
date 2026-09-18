# Cohere Transcribe API — Usage (Postman)

Transcribe speech to text using `CohereLabs/cohere-transcribe-03-2026`
(a 2B-parameter ASR model). Audio longer than 35 s is chunked automatically.

## Main endpoint

```
POST /transcribe/file
```

Base URL (default): `http://localhost:6221`

Full URL: `http://localhost:6221/transcribe/file`

---

## Authentication

- If the server's `API_KEY` environment variable is **set**, every request must
  include the header `X-Api-Key: <your key>`.
- If `API_KEY` is **unset/empty** (development mode), no header is needed.

In Postman, add the header under the **Headers** tab when required:

| Key         | Value          |
|-------------|----------------|
| `X-Api-Key` | `<your key>`   |

---

## Postman setup

1. **Method**: `POST`
2. **URL**: `http://localhost:6221/transcribe/file`
3. **Headers**: add `X-Api-Key` only if the server requires it (see above).
4. **Body** tab → select **form-data**.
5. Add the fields:

   | Key           | Type | Required | Value / Notes                                              |
   |---------------|------|----------|------------------------------------------------------------|
   | `file`        | File | Yes      | The audio file to transcribe (wav, mp3, flac, ogg, …).     |
   | `language`    | Text | No       | ISO 639-1 code. Default `en`.                              |
   | `punctuation` | Text | No       | `true` (default) or `false`.                               |
   | `batch_size`  | Text | No       | Integer GPU batch size. Leave blank for the model default. |

   > For the `file` key, use the dropdown next to the key name to switch its
   > type from *Text* to *File*.

6. Do **not** set `Content-Type` manually — Postman handles the multipart boundary.
7. Click **Send**.

### Supported languages (14)
`en`, `de`, `fr`, `it`, `es`, `pt`, `el`, `nl`, `pl`, `ar`, `vi`, `zh`, `ja`, `ko`

---

## Response

- **Status**: `200 OK`
- **Content-Type**: `application/json`

```json
{
  "transcription": "Hello, this is a test recording.",
  "language": "en",
  "model": "CohereLabs/cohere-transcribe-03-2026"
}
```

---

## Error responses

| Status | Meaning                                                  |
|--------|----------------------------------------------------------|
| `400`  | Audio could not be decoded.                              |
| `401`  | Invalid or missing `X-Api-Key` (when auth is enabled).   |
| `422`  | Unsupported `language` code.                             |
| `500`  | Inference / unexpected server error.                     |

---

## Alternative: base64 JSON (programmatic clients)

If you cannot send a multipart form, `POST /transcribe/base64` accepts a JSON body.
Set **Body → raw → JSON**:

```json
{
  "audio_base64": "<base64-encoded audio file bytes>",
  "language": "en",
  "punctuation": true
}
```

---

## Quick checks

- **Health probe**: `GET http://localhost:6221/health`
  ```json
  { "status": "ok", "model": "CohereLabs/cohere-transcribe-03-2026" }
  ```
- **Interactive docs**: open `http://localhost:6221/docs` in a browser.
