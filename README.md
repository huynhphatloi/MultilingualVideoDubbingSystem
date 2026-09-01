# Multilingual Video Dubbing with n8n

This is an application demo with one visible n8n pipeline:

```text
1 Upload video
      ↓
2 Extract audio — local FFmpeg
      ↓
3 Speech to text — Whisper on Google Colab
      ↓
4 Translate — NLLB-200 on Google Colab
      ↓
5 Generate voice — MMS-TTS on Google Colab
      ↓
6 Mix and synchronize — local FFmpeg
      ↓
7 Render video — local FFmpeg
```

No AI model runs or downloads on the local machine. FastAPI only stores jobs, runs
FFmpeg, calls Colab, and serves the single-file frontend. If Colab is stopped,
unreachable, or not configured, the active AI stage fails and n8n stops the workflow.
There is no local fallback.

## Start Colab first

1. Open `colab/ai_service.ipynb` in Google Colab.
2. Select a GPU runtime and run all cells.
3. Copy the three values printed by the final cell into `.env`:

```env
COLAB_API_URL=https://your-tunnel.trycloudflare.com
COLAB_API_TOKEN=your-token
COLAB_API_TIMEOUT=1800
```

The temporary tunnel URL changes whenever the Colab session restarts. Update `.env`
and restart the local stack after every new session.

## Run the application

Docker Desktop must be running:

```bash
make check
make start
```

Then:

1. Open <http://localhost:5678>.
2. Open **Simple Multilingual Dubbing**, save it, and switch it to **Active**.
3. Open the demo frontend at <http://localhost:8000>.
4. Upload a short video and watch the seven nodes in n8n.

FastAPI documentation is available at <http://localhost:8000/docs>.

## Project files

```text
MultilingualVideoDubbingSystem/
├── README.md
├── Makefile
├── docker-compose.yml
├── .env.example
├── frontend/
│   └── index.html
├── ai-service/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── colab/
│   ├── ai_service.ipynb
│   ├── server.py
│   ├── requirements.txt
│   └── README.md
└── n8n/workflows/
    └── simple-dubbing.json
```

Generated job data stays in a Docker volume. n8n uses its built-in SQLite database.

## Commands

```bash
make status
make logs
make import
make stop
```
