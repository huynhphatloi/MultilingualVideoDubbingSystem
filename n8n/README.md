# n8n workflow

## Import

```bash
make import-workflow
# or
docker compose exec n8n n8n import:workflow \
  --input=/workflows/multilingual-dubbing-pipeline.json
```

Then open <http://localhost:47678> → **Multilingual Video Translation & Dubbing Pipeline**.

## Layout

The canvas is grouped by six sticky notes that mirror the pipeline design:

| Group | Nodes |
|---|---|
| 1 — Input & media preprocessing | Manual / Form / Webhook triggers → Create Processing Job → Upload Video to Storage → Extract Audio → Separate Speech / Background |
| 2 — Speech understanding | Speech-to-Text → Speaker Diarization (`community-1`) → Merge Transcript + Speaker + Time |
| 3 — Multilingual translation | Translate Segments → *(error)* Fallback: SeamlessM4T → Translation Ready |
| 4 — Speech generation | Generate Speech → Duration Within Tolerance? |
| 5 — Temporal synchronization | Adapt Translation → Re-generate Speech ⟲ · Place Audio on Original Timeline |
| 6 — Mixing & rendering | Mix Dub + Background → Generate Subtitle → FFmpeg Render → Final Dubbed Video |

## Three ways to start a run

1. **Manual Trigger (Demo)** — set `job_id` in *Demo Parameters* and hit *Test workflow*.
   Useful when you want to re-run the pipeline on a job that already has its video uploaded.
2. **Form: Upload Video** — n8n's built-in form. Click the node → *Form URL* to get a
   public upload page. No frontend required; good for a live demo.
3. **Webhook: Start Dubbing** — `POST http://localhost:47678/webhook/dubbing/start`
   with `{ "job_id": "...", "target_language": "vi" }`. This is what the React UI calls.

## Conventions every node follows

* **No hard-coded paths.** URLs come from `{{ $env.AI_SERVICE_BASE_URL }}`.
* **Retries.** Each HTTP node retries 3× with a 5 s backoff.
* **Fail gracefully.** Error outputs converge on *Report Pipeline Failure*, which writes
  the error code into Postgres so the UI can show which stage went red — the execution
  then stops with a readable message instead of a stack trace.
* **No binary in the graph.** Only *Upload Video to Storage* touches the file itself.

## The retry loop

`Duration Within Tolerance?` returns `false` while `needs_adaptation` is non-empty, which
sends those segment ids to *Adapt Translation* → *Re-generate Speech* → back into the IF
node. The loop terminates because the AI service caps rewrites per segment
(`SYNC_MAX_RETRANSLATE_ATTEMPTS`, default 2) and then downgrades the segment to
`stretch`, which empties `needs_adaptation`.

## Timeouts

Stages that run heavy models have generous node timeouts (Demucs and TTS up to 2–3 h on
CPU). If you see `ETIMEDOUT`, raise the *Options → Timeout* value on that node rather
than lowering the model size — or switch to the GPU compose overlay.
