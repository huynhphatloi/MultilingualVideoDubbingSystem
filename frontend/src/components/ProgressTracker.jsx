import React, { useEffect, useState } from 'react';

const LABELS = {
  create_job: 'Create job',
  store_video: 'Store video',
  extract_audio: 'Extract audio',
  separate_sources: 'Separate speech / background',
  transcribe: 'Speech-to-text',
  diarize: 'Speaker diarization',
  merge_segments: 'Merge transcript + speaker',
  translate: 'Translation',
  synthesize: 'Speech generation',
  synchronize: 'Timeline synchronization',
  mix_audio: 'Mix with background',
  generate_subtitles: 'Subtitles',
  render_video: 'FFmpeg render',
};

const ICON = { completed: '✓', running: '●', failed: '✕', skipped: '–', pending: '' };

// Stages that legitimately take minutes; used only to word the hint.
const SLOW = new Set(['separate_sources', 'transcribe', 'synthesize', 'render_video']);

function human(seconds) {
  if (seconds == null) return '';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${String(s).padStart(2, '0')}s`;
}

export default function ProgressTracker({ job }) {
  // Local ticker so elapsed times move between polls - a frozen number is
  // exactly what makes "is it stuck?" impossible to answer.
  const [, tick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);

  if (!job) return null;

  const stages = job.stages || [];
  const p = job.progress || {};
  const pct = p.percent ?? 0;

  // Seconds since the API last touched this job, corrected by how long ago we
  // fetched it.
  const fetchedAgo = job.updated_at
    ? Math.max(0, (Date.now() - new Date(job.updated_at).getTime()) / 1000)
    : null;
  const idle = fetchedAgo ?? p.idle_seconds;

  const terminal = ['completed', 'failed', 'cancelled'].includes(job.status);
  const waiting = p.waiting_for_trigger && !terminal;
  const runningStage = p.current_stage;

  let state = job.status;
  if (waiting) state = 'waiting';

  return (
    <section className="card">
      <div className="row-between">
        <h2>Pipeline progress</h2>
        <span className={`pill ${state}`}>{state}</span>
      </div>

      <div className="bar">
        <div className={`bar-fill ${runningStage ? 'active' : ''}`} style={{ width: `${pct}%` }} />
      </div>

      <p className="muted small">
        {p.completed}/{p.total} stages · {pct}%
        {runningStage && ` · running ${LABELS[runningStage] || runningStage} for ${human(p.running_seconds)}`}
        {idle != null && ` · last update ${human(idle)} ago`}
      </p>

      {waiting && (
        <div className="callout warn">
          <strong>Nothing is running right now.</strong>
          <span>
            {p.completed} stage(s) finished and no stage has started since
            {' '}{human(idle)} ago. The pipeline only advances when something calls
            the API — normally the n8n workflow.
          </span>
          {/* Stalling at exactly store_video is not a generic stall: the upload
              worked (the AI service handled it directly) and then the handoff to
              n8n never happened. In practice that is always an inactive
              workflow — an inactive workflow has no production webhook, so
              POST /webhook/dubbing/start answers 404 and no stage 3 ever runs. */}
          {p.completed <= 2 ? (
            <>
              <span>
                It stopped right after the upload, which means the n8n workflow never
                picked the job up. Almost always: <strong>the workflow is not Active</strong>,
                so its <code>/webhook/dubbing/start</code> URL does not exist yet.
              </span>
              <span className="muted small">
                Fix it in one command: <code>./scripts/activate-workflow.sh</code>
                {' '}— or in the n8n editor, open the workflow, press Save, then flip
                the Active toggle. Re-upload afterwards.
              </span>
            </>
          ) : (
            <span className="muted small">
              Check n8n → Executions, or drive it by hand:
              {' '}<code>./scripts/smoke-test.sh</code>
            </span>
          )}
        </div>
      )}

      {runningStage && (
        <div className="callout ok">
          <strong>Working: {LABELS[runningStage] || runningStage}</strong>
          <span>
            {human(p.running_seconds)} elapsed.
            {SLOW.has(runningStage)
              ? ' This stage is model-heavy — minutes are normal, and the first run also downloads weights.'
              : ' Should finish shortly.'}
          </span>
        </div>
      )}

      <ol className="stages">
        {stages.map((s) => (
          <li key={s.name} className={s.status}>
            <span className="icon">{ICON[s.status]}</span>
            <span className="name">{LABELS[s.name] || s.name}</span>
            <span className="meta">
              {s.status === 'running'
                ? human(p.running_seconds)
                : s.duration_ms
                  ? `${(s.duration_ms / 1000).toFixed(1)}s`
                  : ''}
              {s.attempt > 1 ? ` · attempt ${s.attempt}` : ''}
            </span>
            {s.error_code && <span className="err">{s.error_code}: {s.message}</span>}
            {s.error_code && s.output?.details && (
              <details className="stage-details">
                <summary>diagnostics</summary>
                {Object.entries(s.output.details).map(([k, v]) => (
                  <div key={k}>
                    <span className="dk">{k}</span>
                    <pre>{typeof v === 'string' ? v : JSON.stringify(v, null, 2)}</pre>
                  </div>
                ))}
              </details>
            )}
          </li>
        ))}
      </ol>

      {job.error && (
        <p className="error"><strong>{job.error.code}</strong> — {job.error.message}</p>
      )}
    </section>
  );
}
