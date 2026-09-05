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

## Prerequisites

- **Docker Desktop**, running. It supplies n8n and FFmpeg; nothing else is
  installed locally.
- **`python3`** on the host, used only by `make check`.
- **A Google account with Colab GPU access.** This is a hard blocker, not an
  optional step: every AI stage runs on Colab and there is no local fallback.

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

With Docker Desktop running:

```bash
make start
```

The first run creates `.env` from `.env.example`, builds the images, imports the
workflow, activates it, and restarts n8n so the webhook is registered. Paste the
Colab values into the new `.env`, then apply them:

```bash
make restart
```

Open the demo app at <http://localhost:8000>, upload a short video, and watch the
seven nodes run at <http://localhost:5678>. FastAPI documentation is at
<http://localhost:8000/docs>.

Activation is applied to the workflow id pinned in
`n8n/workflows/simple-dubbing.json`, so re-importing updates that one workflow
in place instead of creating duplicates. If `make start` reports that
auto-activation is unavailable, open <http://localhost:5678>, open **Simple
Multilingual Dubbing**, save it, and switch it to **Active** by hand.

n8n only registers webhooks at process start, so anything that changes the
active flag needs an n8n restart — `make activate` and `make import` both do
this for you.

`make check` runs the offline syntax and configuration checks without starting
anything.

## Using Colab on its own

The Colab server also runs the whole pipeline itself, so another application
can dub a video with nothing but this URL - no Docker, no n8n, and no files
accumulating on the caller's machine.

```bash
# 1. start a job (returns immediately)
curl -X POST "$COLAB_API_URL/jobs" -H "Authorization: Bearer $COLAB_API_TOKEN" \
     -F video=@clip.mp4 -F target_language=vi -F tts_engine=mms

# 2. poll it
curl "$COLAB_API_URL/jobs/<job_id>" -H "Authorization: Bearer $COLAB_API_TOKEN"

# 3. collect the result
curl -O -J "$COLAB_API_URL/jobs/<job_id>/download" -H "Authorization: Bearer $COLAB_API_TOKEN"
```

`GET /jobs` lists them, `DELETE /jobs/{id}` removes one, and the server keeps
only the most recent `JOBS_LIMIT` (20) jobs. One worker thread drains the queue
so jobs run in order rather than competing for the GPU.

Two things to know before relying on this. **Colab storage is ephemeral**: a
dropped session loses queued and running jobs, so download results promptly.
And **the video crosses the tunnel twice** - up as the source, down as the
result - where the n8n route sends only a 16 kHz mono track. Prefer this route
when you want one API to call; prefer the n8n route when the file is large or
you want the seven stages visible.

## Notebook backends

`AI_BACKENDS` lists the notebook sessions to try, most preferred first. Each
name `NAME` reads `NAME_API_URL` and `NAME_API_TOKEN`, so the default
`colab,kaggle` picks up the existing `COLAB_*` pair unchanged:

```env
AI_BACKENDS=colab,kaggle
COLAB_API_URL=https://your-tunnel.trycloudflare.com
COLAB_API_TOKEN=your-token
KAGGLE_API_URL=
KAGGLE_API_TOKEN=
```

The notebook's last cell prints a ready-to-run command, so a new session costs
one paste rather than editing this file:

```bash
make colab URL=https://your-tunnel.trycloudflare.com TOKEN=your-token
```

That writes `.env`, restarts the API, and reports which backends are live.
`make kaggle` does the same for the second slot. Other settings and comments in
`.env` are left alone, and a URL that does not start with `http` is rejected
rather than saved — swapping the URL and the token is the usual mistake.

Before each call the service probes `/health` on each backend in order and uses
the first that answers, caching the result for `BACKEND_PROBE_TTL` seconds. If
that backend dies mid-job it retries on the next one, except for uploads — a
consumed file stream cannot be replayed, so those fail with a message instead
of silently sending an empty body.

```bash
make backends
```

reports which sessions are alive and, when none are, why.

**A notebook session cannot be started from here.** Neither Colab nor Kaggle
exposes an API to bring a runtime up; both need a person to press Run. So this
fails over between sessions that are *already* running — it is not a way to
keep a backend permanently available. For that you need something that stays
up on its own.

## Engines

Every stage picks its model per request, from the form. The base Colab install
covers all of column one; the rest need their own `pip install` in the notebook,
which the first cell has flags for.

| Stage | Choices | Extra install |
|---|---|---|
| Speech to text | 14 Whisper checkpoints, `tiny` to `large-v3` | none |
| Translate | `nllb`, `seamless` | none |
| Voice | `mms` | none |
| Voice | `edge` | `edge-tts` |
| Voice | `piper` | `piper-tts` |
| Voice | `xtts_v2`, `vixtts` | `coqui-tts` |
| Voice | `f5_vi`, `f5_base` | `f5-tts` |

`GET /health` on the Colab URL reports which voices that session can actually
see, so check there before blaming a failed run on the model.

The last four clone a speaker rather than using a stock voice. For those the
local API cuts a 12-second sample from the original audio, starting at the first
transcribed segment so the clip is known to contain speech, uploads it once per
job to `POST /reference`, and passes the returned id with each segment. The
dubbed video then keeps the original speaker's voice.

Language coverage is not uniform: XTTS v2 has no Vietnamese, `f5_base` is
English and Chinese only, and `vixtts` and `f5_vi` are Vietnamese only. Picking
an engine that cannot speak the target language fails that stage with a message
naming a working alternative.

The form, the local API, and the Colab server each hold their own list of these
names. `make check` runs `scripts/check_contract.py`, which fails when the three
drift apart — an earlier version offered seven voices while Colab implemented
one, and the extra choices were silently discarded.

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
│   └── requirements.txt
├── scripts/
│   └── check_contract.py
└── n8n/workflows/
    └── simple-dubbing.json
```

Generated job data stays in a Docker volume. n8n uses its built-in SQLite database.

## Commands

```bash
make status    # service status
make logs      # follow both service logs
make restart   # apply .env or app.py changes
make backends  # which notebook backends are alive
make colab URL=.. TOKEN=..  # point at a new Colab session
make import    # re-import and re-activate the workflow after editing its JSON
make stop      # stop containers, keep jobs and n8n data
```

`frontend/index.html` and `ai-service/app.py` are bind-mounted into the
container. Frontend edits show up on a browser refresh; `app.py` edits need
`make restart`. Only dependency changes require a rebuild via `make start`.
