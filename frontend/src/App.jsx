import React, { useCallback, useEffect, useState } from 'react';
import { api } from './lib/api.js';
import UploadPanel from './components/UploadPanel.jsx';
import ProgressTracker from './components/ProgressTracker.jsx';
import TranscriptEditor from './components/TranscriptEditor.jsx';
import ResultPanel from './components/ResultPanel.jsx';
import JobList from './components/JobList.jsx';

const POLL_MS = 4000;

// Convenience links in the header. Ports live in .env and are injected at build
// time, so moving them never means editing JSX.
const LINKS = {
  n8n: import.meta.env.VITE_N8N_URL || 'http://localhost:47678',
  api: import.meta.env.VITE_API_URL || 'http://localhost:47800',
  minio: import.meta.env.VITE_MINIO_URL || 'http://localhost:47901',
};

export default function App() {
  const [languages, setLanguages] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [selected, setSelected] = useState(null);
  const [job, setJob] = useState(null);
  const [health, setHealth] = useState(null);
  const [tab, setTab] = useState('progress');

  useEffect(() => {
    api.languages().then((r) => setLanguages(r.languages)).catch(() => {});
    api.health().then(setHealth).catch(() => setHealth({ status: 'unreachable' }));
  }, []);

  const refreshJobs = useCallback(async () => {
    try {
      const r = await api.listJobs();
      setJobs(r.jobs);
      if (!selected && r.jobs.length) setSelected(r.jobs[0].job_id);
    } catch { /* backend not up yet */ }
  }, [selected]);

  const refreshJob = useCallback(async () => {
    if (!selected) return;
    try { setJob(await api.getJob(selected)); } catch { /* deleted */ }
  }, [selected]);

  useEffect(() => { refreshJobs(); }, [refreshJobs]);
  useEffect(() => { refreshJob(); }, [refreshJob]);

  useEffect(() => {
    const id = setInterval(() => { refreshJobs(); refreshJob(); }, POLL_MS);
    return () => clearInterval(id);
  }, [refreshJobs, refreshJob]);

  const remove = async (id) => {
    if (!confirm(`Delete job ${id} and all its artifacts?`)) return;
    await api.deleteJob(id);
    if (selected === id) { setSelected(null); setJob(null); }
    refreshJobs();
  };

  return (
    <div className="app">
      <header>
        <div>
          <h1>Multilingual Video Translation &amp; Dubbing</h1>
          <p className="muted small">
            n8n orchestrates · FastAPI runs the models · MinIO holds the media
          </p>
        </div>
        <nav className="links">
          <a href={LINKS.n8n} target="_blank" rel="noreferrer">n8n</a>
          <a href={`${LINKS.api}/docs`} target="_blank" rel="noreferrer">API docs</a>
          <a href={LINKS.minio} target="_blank" rel="noreferrer">MinIO</a>
          <span className={`pill ${health?.status === 'ready' ? 'completed' : 'failed'}`}>
            {health?.status || '…'}
          </span>
        </nav>
      </header>

      <main>
        <aside>
          <UploadPanel
            languages={languages}
            onJobStarted={(id) => { setSelected(id); setTab('progress'); refreshJobs(); }}
          />
          <JobList jobs={jobs} selected={selected} onSelect={setSelected} onDelete={remove} />
        </aside>

        <div className="content">
          <div className="tabs">
            {['progress', 'transcript', 'result'].map((t) => (
              <button
                key={t}
                className={tab === t ? 'tab active' : 'tab'}
                onClick={() => setTab(t)}
              >
                {t}
              </button>
            ))}
          </div>

          {!selected && <p className="muted card">Select or create a job to get started.</p>}
          {selected && tab === 'progress' && <ProgressTracker job={job} />}
          {selected && tab === 'transcript' && <TranscriptEditor jobId={selected} />}
          {selected && tab === 'result' && <ResultPanel job={job} />}
        </div>
      </main>
    </div>
  );
}
