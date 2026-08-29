# Master prompt — redesign the web UI

Copy everything between the two `═══` rules into the design AI (v0, Lovable, Claude,
ChatGPT, Cursor…). It is written to be pasted as a single message.

Two knobs to set before you paste — they are marked `[CHOOSE]` in the prompt:

1. **Stack**: keep the current one (React 18 + Vite + one plain CSS file, zero runtime
   deps) or allow Tailwind + shadcn/ui. Tools like v0/Lovable assume Tailwind; a plain-CSS
   answer drops straight into this repo with no build changes.
2. **Language**: UI copy in English or Vietnamese.

═══════════════════════════════════════════════════════════════════════════════

## Role

You are a senior product designer *and* front-end engineer. I need you to redesign the
operator console of a working system, then hand me production React code — not a mockup.

## The product

A **multilingual video dubbing pipeline**. A user uploads a video; the system transcribes
it, identifies who speaks when, translates each line, re-voices it in the target language
with a cloned voice, places every take back at its original timestamp, mixes it under the
preserved original music, and re-renders the video.

The web UI is the **operator console** for that pipeline. It is not a consumer product.

### Who uses it, and what they actually need

There are two users, and today's UI serves neither well:

1. **The operator** (me) — runs jobs, and when something breaks needs to answer, fast:
   *which stage failed, why, and what do I do about it?* A dubbing job has 13 stages and
   takes 5–40 minutes; several stages legitimately run for minutes with no visible change,
   so **"is it working or is it stuck?" is the single most important question the UI must
   answer at a glance.**
2. **A reviewer** — reads the machine translation of every line and edits the bad ones
   before the final render. This is a text-editing task over 20–200 rows, each with a
   speaker, a timestamp, the source line, and a translation, plus a signal for whether the
   translated line will *fit* in the time available.

## What exists today (and what is wrong with it)

Current implementation: React 18 + Vite, five components, one 170-line stylesheet, dark
theme, three-column-ish layout (left sidebar = upload + job list, right = tabs).

Honest assessment of the current UI:

- It is a wall of undifferentiated dark grey panels. Nothing has visual hierarchy, so the
  eye has no idea where to look first.
- The 13-stage list is a flat `<ol>` of near-identical rows. The one thing that matters —
  where the pipeline is *right now* — has the same weight as the twelve rows that do not.
- Job status is a small text pill. When a job stalls, the UI says "waiting" in 11px grey.
- The transcript editor is a stack of raw `<textarea>`s. Editing 60 lines in it is
  miserable; there is no keyboard flow, no sense of position, no way to see only the
  problem rows.
- Errors dump raw stderr into a `<details>`. Useful, but it looks like a crash log.
- Nothing is responsive; nothing has focus states; no empty states worth the name.

**Do not preserve the current visual design. Do preserve the information architecture and
every state listed below** — they were derived from real failures and each one exists
because its absence caused a real support problem.

## The real data

These are the actual API payloads. Design against these fields; do not invent new ones,
and do not drop ones you find inconvenient.

### `GET /api/jobs/{id}`

```json
{
  "job_id": "235f2161fa5c4877",
  "status": "created | running | awaiting_review | completed | failed | cancelled",
  "source_filename": "avengers_60s.mp4",
  "source_language": "en",
  "detected_language": "en",
  "language_confidence": 0.97,
  "target_language": "vi",
  "duration_seconds": 60.02,
  "speaker_count": 3,
  "segment_count": 14,
  "error": { "code": "separation_failed", "message": "Demucs failed on mps and on cpu" },
  "metrics": { "ratio_stats": { "mean": 1.08, "min": 0.71, "max": 1.44 },
               "tts_models": { "xtts_v2": 11, "mms_tts": 3 } },
  "progress": {
    "completed": 3, "total": 13, "percent": 23.1,
    "current_stage": "separate_sources",
    "running_seconds": 184.2,
    "idle_seconds": 6.0,
    "waiting_for_trigger": false
  },
  "stages": [
    { "name": "extract_audio", "status": "completed", "duration_ms": 1200, "attempt": 1 },
    { "name": "separate_sources", "status": "failed", "attempt": 2,
      "error_code": "separation_failed", "message": "Demucs failed on mps and on cpu",
      "output": { "details": { "device": "mps", "stderr_cpu": "<40 lines of traceback>",
                               "hint": "Set SEPARATION_DEVICE=cpu, or DEMUCS_ENABLED=false" } } }
  ]
}
```

The 13 stages, in fixed order:
`create_job · store_video · extract_audio · separate_sources · transcribe · diarize ·
merge_segments · translate · synthesize · synchronize · mix_audio · generate_subtitles ·
render_video`

Stage status ∈ `pending | running | completed | failed | skipped`.

### `GET /api/jobs/{id}/segments`

```json
{
  "source_language": "en", "target_language": "vi",
  "turn_source": "exclusive",
  "speakers": [ { "speaker_id": "SPEAKER_00", "segment_count": 8, "total_seconds": 31.4,
                  "voice_reference": "jobs/…/speakers/speaker_00.wav" } ],
  "segments": [
    { "segment_id": 12, "speaker_id": "SPEAKER_01",
      "start": 10.2, "end": 12.7,
      "source_text": "We need to solve this problem.",
      "translated_text": "Chúng ta cần giải quyết vấn đề này.",
      "duration_ratio": 1.24,
      "sync_action": "accept | stretch | retranslate | failed",
      "tts_model": "xtts_v2", "attempts": 1, "edited_by_user": false }
  ]
}
```

`duration_ratio` = spoken length ÷ the original line's length. **This is the domain's key
number.** 0.90–1.10 is good; 0.80–1.20 is fixable with light time-stretching; outside that
the line must be rewritten shorter or longer. The UI has to make a row's ratio readable at
a glance across 200 rows.

### Other endpoints already available

`GET /api/languages` (50 languages, each `{code, name, voice_cloning: bool}`) ·
`GET /api/health/ready` (per-dependency ok/error) · `GET /api/jobs` ·
`POST /api/jobs` · `POST /api/jobs/{id}/upload` (multipart, with upload progress) ·
`POST /api/jobs/{id}/review` (`{edits: [{segment_id, translated_text}]}`) ·
`GET /api/jobs/{id}/artifacts` · `GET /api/jobs/{id}/download/{video|subtitle|audio}` ·
`POST /webhook/dubbing/start` (starts the n8n workflow).

## States the design must handle explicitly

Each of these is a real, frequent situation. A design that only draws the happy path is
not acceptable.

| State | What the user must instantly understand |
|---|---|
| **Working** | A model is running. Which stage, for how long, and that minutes are normal here. Needs *motion* — a frozen screen is indistinguishable from a crash. |
| **Stalled** (`waiting_for_trigger: true`) | Nothing is running and nothing will happen without action. This is NOT the same as "running", and today it looks identical. Must name the likely cause and the exact fix. |
| **Failed** | Which stage, the stable error code, a human explanation, the suggested fix, and the raw stderr available but not shouting. |
| **Degraded but succeeding** | e.g. speaker diarization fell back to one speaker (no HF token), or separation fell back to "voice-over" mode where the original dialogue stays audible. The job completes, but the result is *worse than it looks* and the UI must say so. |
| **Off-target lines** | N segments whose `duration_ratio` is out of tolerance. The reviewer needs to jump straight to them. |
| **Empty** | No jobs yet. First-run guidance. |
| **Upload in progress** | A 130 MB file over a slow connection; distinct from "pipeline running". |

## Design goals, in priority order

1. **Answer "working or stuck?" in under one second**, from across the room. This is a
   system that gets demoed on a projector.
2. **Make the failing stage the loudest thing on screen** when there is one.
3. **Make reviewing 200 translated lines pleasant** — keyboard navigation, filter to
   problem rows, clear per-row fit signal, obvious unsaved-changes state.
4. **Progressive disclosure.** Stack traces, artifact lists and object keys stay available
   but must not compete with the primary answer.
5. **Legible on a projector**: generous type, real contrast, no 11px grey on dark grey.

## Constraints

- `[CHOOSE]` **Stack**: React 18 function components + Vite, **plain CSS in a single
  stylesheet, no runtime dependencies beyond react/react-dom**.
  *(Alternative, if you prefer: Tailwind CSS + shadcn/ui. Say which you used.)*
- Data arrives by polling every 4 s. Elapsed timers must tick locally between polls, or
  the UI looks frozen.
- Dark theme is the default and is not negotiable (it is demoed in dim rooms). A light
  theme is optional.
- WCAG AA contrast. Visible keyboard focus. The whole review flow must be usable from the
  keyboard.
- `[CHOOSE]` UI copy in **English** *(or Vietnamese — say which)*.
- Must work from 1280×720 up to 2560×1440. Mobile is out of scope.
- No fake data in the delivered code: components take the real payloads above as props.

## What I want back

1. **A short design rationale** — the layout you chose and *why*, in terms of the priority
   list above. Half a page, not an essay.
2. **A component inventory** — every component, its props, and which states it renders.
3. **Complete, working code**: `App.jsx`, the components, and the stylesheet. It must run
   as-is against the payloads above.
4. **A states walkthrough** — for each row of the states table, tell me what the screen
   looks like.
5. **What you deliberately did not do**, and why.

Ask me clarifying questions before writing code if anything above is ambiguous.

═══════════════════════════════════════════════════════════════════════════════

## Notes for me (not part of the prompt)

- If the tool returns Tailwind, adding it to this Vite project is `npm i -D tailwindcss
  @tailwindcss/postcss postcss` plus a postcss config — cheap, but it does change the
  Docker build, so decide before merging.
- Keep `src/lib/api.js` as-is. It already handles upload progress, the 409 re-upload
  protocol and n8n's error bodies. A redesign should not touch the transport layer.
- The payloads above are real. If the design AI invents a field, that field does not
  exist and the component will render `undefined`.
