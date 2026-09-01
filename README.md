# Simple n8n Multilingual Video Dubbing

## What it demonstrates

The n8n canvas shows the complete project as seven clear stages:

```text
1 Upload video
      ↓
2 Extract audio (FFmpeg)
      ↓
3 Speech to text (faster-whisper)
      ↓
4 Translate (NLLB-200)
      ↓
5 Generate voice (Google Colab TTS API, or local MMS fallback)
      ↓
6 Mix and synchronize (FFmpeg)
      ↓
7 Render video (FFmpeg)
```

There are only two local services:

- **n8n**: visual orchestration, execution history, and errors.
- **FastAPI**: the seven media/AI operations and the static demo frontend.

The frontend is one HTML file with no React, Node.js, npm, or separate container. It
uploads the video to FastAPI, starts the job through an n8n webhook, polls the job
manifest, and displays the final video and subtitle downloads.

Google Colab is an optional external AI service for stage 5. The app accepts 50 target
languages and sends the selected ISO-639-1 code to Colab's `/synthesize` endpoint.

Jobs are ordinary folders in a Docker volume. n8n uses its built-in SQLite database.

## Intentionally removed

- React/Node frontend build chain (replaced by one static HTML page)
- PostgreSQL job registry
- MinIO object storage
- Demucs separation
- multi-speaker diarization
- complex model routing, retries, benchmarks, and GPU Docker variants
- the translation adaptation loop and detailed progress UI

The result is a **voice-over**: the original soundtrack is lowered to 25%, then one
translated TTS voice is placed at the original Whisper timestamps. Original dialogue may
remain faintly audible. This is the main quality tradeoff that keeps the project small.

## Google Colab TTS

Open `colab/tts_service.ipynb` in Google Colab and run all cells. Copy the tunnel
values printed by the last cell into `.env`:

```env
COLAB_TTS_URL=https://your-tunnel.trycloudflare.com
COLAB_TTS_TOKEN=your-token
```

The Colab service has one responsibility: run multilingual MMS-TTS on its GPU. It
supports the same 50 languages shown in the frontend and keeps only the current
language model in memory. If `COLAB_TTS_URL` is empty, FastAPI runs the same MMS
model locally.

## Run it

Docker Desktop must be running. The first build and first model run are large and slow,
so use a 10–20 second video for the first demonstration.

```bash
# Only when .env does not already exist:
cp .env.example .env

make check
make start
```

If `.env` already exists, do not overwrite it. Add the `COLAB_TTS_URL`,
`COLAB_TTS_TOKEN`, and `COLAB_TTS_TIMEOUT` values shown in `.env.example`.

Then:

1. Open <http://localhost:5678>.
2. Open **Simple Multilingual Dubbing**.
3. Save it and switch it to **Active**.
4. Open the demo app at <http://localhost:8000>.
5. Upload a short video and choose the target language.
6. Keep n8n open in another tab to show the execution moving across the pipeline.
7. Preview or download the dubbed video from the demo app.

API documentation is at <http://localhost:8000/docs>.

## Files

```text
MultilingualVideoDubbingSystem/
├── README.md
├── Makefile
├── docker-compose.yml
├── .env.example
├── frontend/
│   └── index.html          # complete demo UI; no build step
├── colab/                  # optional GPU TTS microservice
│   ├── tts_service.ipynb
│   ├── server.py
│   └── requirements.txt
├── ai-service/
│   ├── app.py              # all seven stages
│   ├── requirements.txt
│   └── Dockerfile
└── n8n/workflows/
    └── simple-dubbing.json
```

## Useful commands

```bash
make status
make logs
make import   # re-import after changing workflow JSON
make stop
```
