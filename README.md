# Multilingual Video Dubbing with n8n

A model-agnostic dubbing pipeline. Every AI step is a **provider** behind a
capability, so ASR, translation, TTS, diarization, source separation and lip
sync can be swapped from the frontend or the n8n form without touching the
pipeline. The Colab notebook is the reference execution environment; nothing is
downloaded onto the local machine.

```text
 UPLOAD → EXTRACT AUDIO → DIARIZATION* → SPEECH RECOGNITION → MERGE SPEAKERS
        → TRANSLATE → PER-SPEAKER REFERENCES* → GENERATE VOICE → ALIGN*
        → SOURCE SEPARATION* → MIX → LIP SYNC* → RENDER → RESULT
                                                       (* optional, skips itself)
```

## Architecture

Three layers, and nothing reaches across them:

| Layer | Where | Knows about |
|---|---|---|
| HTTP API | `colab/server.py`, `ai-service/app.py` | requests, auth, files |
| Pipeline | `colab/pipeline/*` | stage order, job state |
| Providers | `colab/providers/<task>/*` | one model family each |

* **`dubflow_core/`** is pure standard library and is shared by both services:
  the language table, the canonical segment schema, and the alignment maths.
  It is the reason the two implementations of the pipeline cannot disagree
  about what a segment is.
* **`colab/providers/<task>/registry.py`** holds metadata only - no torch, no
  transformers - so `GET /capabilities` answers correctly in a session where
  nothing optional is installed, reporting each engine as `available: false`
  instead of failing to start.
* **`colab/providers/<task>/<name>.py`** is imported the first time a model is
  actually run.
* **`colab/pipeline/<stage>.py`** depends on capabilities (`ASRProvider`,
  `TTSProvider`, ...), never on Whisper or F5 directly.

```text
dubflow_core/          languages.py  segments.py  alignment.py
colab/
  server.py            HTTP only
  jobs.py              job store + single-worker queue
  core/                errors, runtime (device, model slots), media, config
  providers/
    base.py            ModelSpec + Registry
    asr/               base, registry, faster_whisper, seamless, mms,
                       sensevoice, parakeet, windows
    translation/       base, registry, nllb, seamless
    tts/               base, registry, mms, edge, piper, xtts, f5, chatterbox,
                       kokoro, openvoice, cosyvoice, vieneu
    diarization/       base, registry, pyannote
    separation/        base, registry, demucs
    lipsync/           base, registry (no provider ships - see the file)
  pipeline/            extract, diarize, transcribe, merge_segments, translate,
                       voice_reference, synthesize, align, separate, mix,
                       lipsync, render
```

### One source of truth

`GET /capabilities` on the AI service is the catalogue. The local API proxies
it, the frontend renders it, and `scripts/check_contract.py` fails the build if
anything grows a second copy of a model list. The n8n form is the one place
that cannot fetch - it is a static form - so the contract check verifies that
every id it offers exists in the registry.

## Prerequisites

- **Docker Desktop**, running. It supplies n8n and FFmpeg.
- **`python3`** on the host, for `make check` and the tests.
- **A Google account with Colab GPU access.** Every AI stage runs there and
  there is no local fallback.

## Start Colab first

1. Open `colab/ai_service.ipynb` in Google Colab, select a GPU runtime.
2. In the first cell, turn on only the engines you intend to test (see
   [Dependency conflicts](#dependency-conflicts)) and paste an `HF_TOKEN` if you
   want diarization.
3. Run all cells. The last one prints a ready-to-run command:

```bash
make colab URL=https://your-tunnel.trycloudflare.com TOKEN=your-token
```

The tunnel URL changes whenever the session restarts.

## Run the application

```bash
make start
```

Then open <http://localhost:8000> for the demo app and <http://localhost:5678>
for the n8n graph. `make check` runs the offline checks and the test suite;
neither downloads a model.

## Models

Everything below was verified against the model's own card or source. A model is
listed under a language **only** if its own documentation says so.

### Speech recognition

| Model | Provider | Languages | Timestamps | Detects language | Licence | Install |
|---|---|---|---|---|---|---|
| `tiny` … `large-v3-turbo` (7 multilingual) | faster_whisper | all 50 | yes | yes | MIT | base |
| `tiny.en` … `medium.en`, `distil-*` (7 English-only) | faster_whisper | English | yes | no | MIT | base |
| `seamless_asr` | seamless | 48 (no Malay, Sinhala) | no, windowed | no | CC-BY-NC-4.0 | base |
| `mms_asr` | mms | 49 (no Sinhala) | no, windowed | no | CC-BY-NC-4.0 | base |
| `sensevoice_small` | sensevoice | zh, en, ja, ko | no, windowed | yes | FunASR model licence | `funasr` |
| `parakeet_tdt_0.6b_v3` | parakeet | 21 European | yes | no¹ | CC-BY-4.0 | `nemo_toolkit[asr]` |
| `parakeet_tdt_0.6b_v2` | parakeet | English | yes | no | CC-BY-4.0 | `nemo_toolkit[asr]` |

¹ Parakeet v3 transcribes without being told the language but never reports
which one it heard, and the translation stage needs that answer - so the source
language still has to be named.

A recogniser without timestamps is not excluded: the pipeline gives it windows,
taken from the diarization turns when that stage ran and from a voice-activity
pass otherwise, and it produces the same canonical segments as Whisper.

### Translation

| Model | Languages | Licence | Install |
|---|---|---|---|
| `nllb` (600M, default) | all 50 | CC-BY-NC-4.0 | base |
| `nllb_1.3b` | all 50 | CC-BY-NC-4.0 | base |
| `seamless` | 49 (no Sinhala) | CC-BY-NC-4.0 | base |

### Speech generation

| Model | Languages | Clones | Reference | Licence | Install |
|---|---|---|---|---|---|
| `mms` | 34 | no | – | CC-BY-NC-4.0 | base |
| `edge` | 49 (no Armenian) | no | – | Microsoft service terms | `edge-tts` |
| `piper` | 41 | no | – | MIT | `piper-tts` |
| `xtts_v2` | 17 (**no Vietnamese**) | yes | required | CPML (non-commercial) | `coqui-tts` |
| `vixtts` | Vietnamese | yes | required | CPML (non-commercial) | `coqui-tts` |
| `f5_base` | en, zh | yes | required | CC-BY-NC-4.0 | `f5-tts` |
| `f5_vi` | Vietnamese | yes | required | CC-BY-NC-4.0 | `f5-tts` |
| `chatterbox` | 23 (**no Vietnamese**) | yes | optional | MIT | `chatterbox-tts` |
| `chatterbox_en` | English | yes | optional | MIT | `chatterbox-tts` |
| `kokoro` | en, es, fr, hi, it, ja, pt, zh | no | – | Apache-2.0 | `kokoro` + espeak-ng |
| `openvoice_v2` | en, es, fr, zh, ja, ko | yes | required | MIT | OpenVoice + MeloTTS (git) |
| `cosyvoice2` | zh, en, ja, ko, de, es, fr, it, ru | yes | required + transcript | Apache-2.0 | git clone |
| `vieneu` | Vietnamese | yes | required | Apache-2.0 | `vieneu` |

`mms` covers 34 of the application's 50 languages: **Meta publishes no MMS-TTS
checkpoint** for Japanese, Chinese, Italian, Czech, Danish, Norwegian, Urdu,
Croatian, Serbian, Slovak, Slovenian, Armenian, Georgian, Nepali, Sinhala or
Mongolian. Those combinations are now refused with a clear message instead of
failing on a 404 from the Hub.

### Diarization, separation, lip sync

| Task | Model | Licence | Install |
|---|---|---|---|
| diarization | `pyannote_3_1` | MIT (gated weights) | `pyannote.audio` + `HF_TOKEN` |
| separation | `htdemucs`, `htdemucs_ft` | MIT | `demucs` |
| lip sync | none | – | see `colab/providers/lipsync/registry.py` |

## Optional installation flags

The notebook's first cell has one flag per optional package. The base install
covers Whisper, SeamlessM4T and MMS recognition, both translation engines, and
the `mms` voice - nothing else is installed unless asked for.

```python
INSTALL_EDGE = True          INSTALL_SENSEVOICE = False
INSTALL_PIPER = False        INSTALL_PARAKEET = False
INSTALL_COQUI = False        INSTALL_DIARIZATION = False
INSTALL_F5 = False           INSTALL_DEMUCS = False
INSTALL_CHATTERBOX = False   INSTALL_LIPSYNC = False
INSTALL_KOKORO = False
INSTALL_OPENVOICE = False
INSTALL_COSYVOICE = False
INSTALL_VIENEU = False
```

### Dependency conflicts

Several of these packages pin their own `torch` or `transformers`. Enabling two
that disagree is the usual cause of a Colab runtime that has to be restarted.

| Do not combine | Reason |
|---|---|
| `INSTALL_COQUI` + `INSTALL_F5` | both pin `torch`/`transformers`, in different directions |
| `INSTALL_COQUI` + `INSTALL_PARAKEET` | NeMo pins `transformers` against coqui-tts |
| `INSTALL_CHATTERBOX` + anything else heavy | pins `transformers`; keep it with the base install |
| `INSTALL_SENSEVOICE` + anything else heavy | `funasr` pins `torch` |
| `INSTALL_VIENEU` + anything else heavy | pins its own `transformers` |

`INSTALL_EDGE`, `INSTALL_PIPER`, `INSTALL_KOKORO`, `INSTALL_DIARIZATION` and
`INSTALL_DEMUCS` are safe alongside the base install. `INSTALL_OPENVOICE` and
`INSTALL_COSYVOICE` install from git and are the most fragile of the set.

Base Colab stays reproducible: `colab/requirements.txt` pins seven packages and
never touches torch, which Colab ships matched to its own CUDA.

## Memory

Nothing is preloaded. Each task owns a slot that holds exactly one model;
asking for a different one releases the previous one, runs `gc.collect()` and
empties the CUDA cache. `POST /unload` drops everything, and `GET /health`
reports what is resident.

## API

```bash
# What can this session actually run?
curl "$COLAB_API_URL/capabilities" -H "Authorization: Bearer $COLAB_API_TOKEN"

# Check a configuration without creating a job.
curl -X POST "$COLAB_API_URL/validate" -H "Authorization: Bearer $COLAB_API_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"target_language":"vi","tts_model":"xtts_v2"}'
# 400: 'XTTS v2' (xtts_v2) does not support target language 'vi'. ...

# Dub a video end to end.
curl -X POST "$COLAB_API_URL/jobs" -H "Authorization: Bearer $COLAB_API_TOKEN" \
     -F video=@clip.mp4 \
     -F target_language=vi \
     -F asr_provider=faster_whisper -F asr_model=large-v3-turbo \
     -F translation_provider=nllb \
     -F tts_provider=f5 -F tts_model=f5_vi \
     -F enable_diarization=true -F enable_voice_cloning=true \
     -F enable_alignment=true

curl "$COLAB_API_URL/jobs/<job_id>"           # status, config, alignment summary
curl "$COLAB_API_URL/jobs/<job_id>/segments"  # canonical segments and turns
curl -OJ "$COLAB_API_URL/jobs/<job_id>/download"
```

Stage endpoints are reusable on their own - a system that only needs
translation and speech never touches ASR or video:

```text
POST /diarize     audio            → speaker turns
POST /transcribe  audio [+turns]   → canonical segments
POST /translate   texts            → translations
POST /synthesize  text             → WAV
POST /align       segments         → per-segment speed plan
POST /separate    audio            → one stem
POST /reference   audio [+text]    → reference_id
```

### Canonical segment

```json
{
  "id": 0,
  "speaker_id": "SPEAKER_00",
  "start": 1.23, "end": 4.85, "duration": 3.62,
  "source_text": "Hello everyone",
  "translated_text": "Xin chào mọi người",
  "tts_file": "tts/0000.wav",
  "source_duration": 3.62, "tts_duration_raw": 4.4,
  "alignment_speed": 1.25, "tts_duration_final": 3.52,
  "alignment_status": "aligned"
}
```

`alignment_status` is one of `fits`, `aligned`, `clamped` (the speed limit was
reached and the line still overruns), `stretched` or `unmeasured`. Limits are
`min_speed` / `max_speed` per request, defaulting to 0.75 and 1.35.

Speaker labels are diarization labels. `SPEAKER_00` is a cluster, not a person.

## Migration from the previous API

Nothing was removed. The old parameters still work and map onto the new ones:

| Old | New | Notes |
|---|---|---|
| `model=large-v3` | `asr_model=large-v3` | `model` still accepted |
| `translation_engine=nllb` | `translation_model=nllb` | still accepted |
| `tts_engine=mms` | `tts_model=mms` | still accepted |
| `GET /health` `whisper_models`, `tts_engines` | `GET /capabilities` | old keys still published |
| `POST /jobs` 6-stage progress | 12 stages, optional ones skip | `progress` counts planned stages only |

Behaviour that changed on purpose:

* **Alignment is on by default.** Generated speech is fitted to its window; the
  old pipeline measured the duration and ignored it.
* **Language support is enforced.** Combinations that used to fail deep inside
  a stage (`mms` with Japanese, `edge` with Norwegian or Tagalog) are refused at
  job creation with a message naming a working alternative.
* **Voice references are per speaker** when diarization is on. With it off, the
  behaviour is the old one: one reference for the whole video.

## Testing

```bash
make test    # or: python3 -m pytest tests -q
make check   # tests + syntax + JSON + contract + compose config
```

The suite needs no GPU, no model download and no torch: the registry, the API,
the configuration rules and both pipelines (Colab and local) are exercised with
stand-in models and real FFmpeg.

## Commands

```bash
make status    # service status
make logs      # follow both service logs
make restart   # apply .env or app.py changes
make backends  # which notebook backends are alive
make colab URL=.. TOKEN=..  # point at a new Colab session
make import    # re-import and re-activate the workflow after editing its JSON
make test      # run the test suite
make check     # offline checks
make stop      # stop containers, keep jobs and n8n data
```

## Notebook backends

`AI_BACKENDS` lists the notebook sessions to try, most preferred first. Each
name `NAME` reads `NAME_API_URL` and `NAME_API_TOKEN`:

```env
AI_BACKENDS=colab,kaggle
COLAB_API_URL=https://your-tunnel.trycloudflare.com
COLAB_API_TOKEN=your-token
KAGGLE_API_URL=
KAGGLE_API_TOKEN=
```

Before each call the service probes `/health` in order and uses the first that
answers, caching for `BACKEND_PROBE_TTL` seconds. **A notebook session cannot be
started from here** - neither Colab nor Kaggle exposes an API for that - so this
fails over between sessions that are already running.

## Known limitations

* No lip-sync provider ships; the stage and the flag exist, the registry is
  empty on purpose.
* Source separation sends the full soundtrack to the notebook and downloads a
  stem, which is the largest transfer in the pipeline.
* Segments are never split at a speaker change: an utterance where two people
  overlap is credited to whoever holds most of it, recorded in
  `speaker_confidence`.
* Colab storage is ephemeral. A dropped session loses queued and running jobs.
