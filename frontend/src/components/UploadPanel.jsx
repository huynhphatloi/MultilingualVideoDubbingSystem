import React, { useEffect, useRef, useState } from 'react';
import { api } from '../lib/api.js';

export default function UploadPanel({ languages, onJobStarted }) {
  const [file, setFile] = useState(null);
  const [target, setTarget] = useState('vi');
  const [source, setSource] = useState('auto');
  const [burn, setBurn] = useState(false);
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [warning, setWarning] = useState(null);
  const [models, setModels] = useState([]);
  const [ttsModel, setTtsModel] = useState('auto');
  const inputRef = useRef();

  // Which engines can speak a language is a backend question - XTTS covers 17
  // of them, MMS needs a per-language checkpoint to exist, and a remote
  // endpoint serves whatever the notebook loaded. So re-ask on every change
  // rather than hard-coding a table here that goes stale the moment someone
  // edits REMOTE_TTS_ENGINES.
  useEffect(() => {
    let live = true;
    api.ttsModels(target)
      .then((res) => {
        if (!live) return;
        setModels(res.models || []);
        // The engine picked for the previous language may not speak this one.
        // Silently falling back to 'auto' beats submitting a job that fails at
        // stage 7 with "XTTS does not support 'vi'".
        setTtsModel((current) => {
          const still = (res.models || []).find((m) => m.id === current);
          return still && still.available ? current : 'auto';
        });
      })
      .catch(() => { if (live) setModels([]); });
    return () => { live = false; };
  }, [target]);

  const pick = (e) => {
    const f = e.target.files?.[0];
    if (f) { setFile(f); setError(null); }
  };

  const drop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) { setFile(f); setError(null); }
  };

  const submit = async () => {
    if (!file) return setError('Choose a video file first.');
    setBusy(true); setError(null); setWarning(null); setProgress(0);
    try {
      const job = await api.createJob({
        source_filename: file.name,
        target_language: target,
        source_language: source === 'auto' ? null : source,
        // The choice rides on the JOB, so every driver honours it - n8n, the
        // smoke test, a hand-rolled curl - without any of them passing it on.
        options: { burn_subtitles: burn, tts_model: ttsModel, origin: 'web-ui' },
      });
      await api.uploadVideo(job.job_id, file, target, setProgress);

      // The job exists and the video is stored from here on. Select it before
      // triggering n8n so a webhook problem does not lose the upload.
      onJobStarted(job.job_id);
      setFile(null);
      if (inputRef.current) inputRef.current.value = '';

      try {
        await api.startPipeline({
          job_id: job.job_id,
          target_language: target,
          source_language: source,
          burn_subtitles: burn,
          tts_model: ttsModel,
        });
      } catch (hookErr) {
        // The upload itself succeeded (the AI service took it directly); only the
        // handoff to n8n failed, so the job is real and resumable - say so, and
        // give the exact command rather than "something went wrong".
        setWarning(
          `${hookErr.message} The video is safe (job ${job.job_id}) — ` +
          `run ./scripts/activate-workflow.sh, then upload again. ` +
          `To finish this job without n8n: ./scripts/smoke-test.sh`,
        );
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const selected = models.find((m) => m.id === ttsModel);
  const usable = models.filter((m) => m.available);
  // 'auto' is always usable, so anything beyond it is a real alternative.
  const noAlternative = models.length > 0 && usable.length <= 2;

  return (
    <section className="card">
      <h2>New dubbing job</h2>

      <div
        className={`dropzone ${file ? 'filled' : ''}`}
        onDragOver={(e) => e.preventDefault()}
        onDrop={drop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,video/x-matroska,video/quicktime,video/webm"
          onChange={pick}
          hidden
        />
        {file ? (
          <>
            <strong>{file.name}</strong>
            <span className="muted">{(file.size / 1_048_576).toFixed(1)} MB</span>
          </>
        ) : (
          <>
            <strong>Drop an MP4 here</strong>
            <span className="muted">or click to browse — mp4, mkv, mov, webm</span>
          </>
        )}
      </div>

      <div className="grid-2">
        <label>
          Source language
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="auto">Auto-detect</option>
            {languages.map((l) => (
              <option key={l.code} value={l.code}>{l.name}</option>
            ))}
          </select>
        </label>

        <label>
          Target language
          <select value={target} onChange={(e) => setTarget(e.target.value)}>
            {languages.map((l) => (
              <option key={l.code} value={l.code}>
                {l.name}{l.voice_cloning ? ' — voice cloning' : ''}
              </option>
            ))}
          </select>
        </label>
      </div>

      <label>
        Voice / TTS model
        <select
          value={ttsModel}
          onChange={(e) => setTtsModel(e.target.value)}
          disabled={models.length === 0}
        >
          {models.length === 0 && <option value="auto">Loading…</option>}
          {models.map((m) => (
            // Unavailable engines stay in the list, disabled, with the reason
            // in the label. Hiding them makes the pipeline look like it never
            // had another option; showing them explains why it does not.
            <option key={m.id} value={m.id} disabled={!m.available}>
              {m.label}{m.available ? '' : ` — ${m.reason}`}
            </option>
          ))}
        </select>
      </label>

      {selected?.blurb && (
        <p className={`hint ${selected.voice_cloning === false ? 'warn' : 'ok'}`}>
          {selected.blurb}
        </p>
      )}

      {noAlternative && (
        <p className="hint warn">
          Only one engine can speak this language locally. Start the Colab
          notebook (<code>colab/tts_lab.ipynb</code>) and set{' '}
          <code>REMOTE_TTS_URL</code> + <code>REMOTE_TTS_ENGINES</code> to get
          voice cloning here.
        </p>
      )}

      <label className="checkbox">
        <input type="checkbox" checked={burn} onChange={(e) => setBurn(e.target.checked)} />
        Burn subtitles into the video
      </label>

      {progress > 0 && progress < 100 && (
        <div className="bar"><div className="bar-fill" style={{ width: `${progress}%` }} /></div>
      )}

      {warning && (
        <div className="callout warn">
          <strong>Uploaded, but the pipeline was not started.</strong>
          <span>{warning}</span>
        </div>
      )}
      {error && <p className="error">{error}</p>}

      <button className="primary" disabled={busy || !file} onClick={submit}>
        {busy ? 'Starting…' : 'Upload & run pipeline'}
      </button>
    </section>
  );
}
