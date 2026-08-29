import React, { useEffect, useMemo, useState } from 'react';
import { api } from '../lib/api.js';

const ts = (s) => {
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1).padStart(4, '0');
  return `${String(m).padStart(2, '0')}:${sec}`;
};

const ratioClass = (r) => {
  if (r == null) return '';
  if (r >= 0.9 && r <= 1.1) return 'ok';
  if (r >= 0.8 && r <= 1.2) return 'warn';
  return 'bad';
};

export default function TranscriptEditor({ jobId }) {
  const [doc, setDoc] = useState(null);
  const [edits, setEdits] = useState({});
  const [status, setStatus] = useState(null);
  const [filter, setFilter] = useState('all');

  const load = async () => {
    try {
      setDoc(await api.segments(jobId));
      setEdits({});
    } catch (err) {
      setDoc(null);
      setStatus(err.message);
    }
  };

  useEffect(() => { if (jobId) load(); }, [jobId]);

  const speakers = useMemo(() => {
    const map = {};
    (doc?.speakers || []).forEach((s, i) => { map[s.speaker_id] = i % 6; });
    return map;
  }, [doc]);

  const segments = useMemo(() => {
    const all = doc?.segments || [];
    if (filter === 'all') return all;
    if (filter === 'off') return all.filter((s) => ratioClass(s.duration_ratio) !== 'ok' && s.duration_ratio != null);
    return all.filter((s) => s.speaker_id === filter);
  }, [doc, filter]);

  if (!doc) {
    return (
      <section className="card">
        <h2>Transcript & translation</h2>
        <p className="muted">
          {status || 'Not available yet — appears once transcription and merging finish.'}
        </p>
        <button onClick={load}>Reload</button>
      </section>
    );
  }

  const save = async () => {
    const payload = Object.entries(edits).map(([id, text]) => ({
      segment_id: Number(id), translated_text: text,
    }));
    if (!payload.length) return setStatus('Nothing changed.');
    try {
      const res = await api.review(jobId, payload);
      setStatus(`Saved ${res.count} edit(s). Re-run "Generate Speech" in n8n to apply them.`);
      await load();
    } catch (err) {
      setStatus(err.message);
    }
  };

  return (
    <section className="card">
      <div className="row-between">
        <h2>Transcript &amp; translation</h2>
        <div className="row-gap">
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="all">All segments ({doc.segments.length})</option>
            <option value="off">Off-target duration</option>
            {(doc.speakers || []).map((s) => (
              <option key={s.speaker_id} value={s.speaker_id}>{s.speaker_id}</option>
            ))}
          </select>
          <button onClick={load}>Reload</button>
          <button className="primary" onClick={save} disabled={!Object.keys(edits).length}>
            Save edits
          </button>
        </div>
      </div>

      <p className="muted small">
        {doc.source_language} → {doc.target_language || '?'} ·
        {' '}{doc.speakers?.length || 0} speaker(s) · engine {doc.translation_engine || '—'}
      </p>
      {status && <p className="hint ok">{status}</p>}

      <div className="segments">
        {segments.map((s) => (
          <div key={s.segment_id} className="segment">
            <div className="seg-head">
              <span className={`speaker c${speakers[s.speaker_id] ?? 0}`}>{s.speaker_id}</span>
              <span className="time">{ts(s.start)} → {ts(s.end)}</span>
              <span className="time">{(s.end - s.start).toFixed(2)}s</span>
              {s.duration_ratio != null && (
                <span className={`ratio ${ratioClass(s.duration_ratio)}`}>
                  ratio {s.duration_ratio.toFixed(2)}
                </span>
              )}
              {s.tts_model && <span className="tag">{s.tts_model}</span>}
              {s.edited_by_user && <span className="tag edited">edited</span>}
            </div>
            <p className="source">{s.source_text}</p>
            <textarea
              value={edits[s.segment_id] ?? s.translated_text ?? ''}
              placeholder="not translated yet"
              onChange={(e) =>
                setEdits((prev) => ({ ...prev, [s.segment_id]: e.target.value }))
              }
              rows={2}
            />
          </div>
        ))}
      </div>
    </section>
  );
}
