import React, { useEffect, useState } from 'react';
import { api } from '../lib/api.js';

export default function ResultPanel({ job }) {
  const [artifacts, setArtifacts] = useState([]);

  useEffect(() => {
    if (!job?.job_id) return;
    api.artifacts(job.job_id).then((r) => setArtifacts(r.artifacts)).catch(() => setArtifacts([]));
  }, [job?.job_id, job?.status]);

  if (!job) return null;
  const done = job.status === 'completed' || !!job.output_key;

  return (
    <section className="card">
      <h2>Result</h2>

      {done ? (
        <>
          <video
            key={job.output_key}
            className="player"
            controls
            src={api.downloadUrl(job.job_id, 'video')}
          />
          <div className="row-gap">
            <a className="button primary" href={api.downloadUrl(job.job_id, 'video')} download>
              Download video
            </a>
            <a className="button" href={api.downloadUrl(job.job_id, 'subtitle')} download>
              Download .srt
            </a>
            <a className="button" href={api.downloadUrl(job.job_id, 'subtitle-vtt')} download>
              Download .vtt
            </a>
            <a className="button" href={api.downloadUrl(job.job_id, 'audio')} download>
              Download final audio
            </a>
          </div>
        </>
      ) : (
        <p className="muted">The preview appears here once the render stage completes.</p>
      )}

      {artifacts.length > 0 && (
        <details className="artifacts">
          <summary>{artifacts.length} artifacts in object storage</summary>
          <ul>
            {artifacts.map((a) => (
              <li key={a.key}>
                <a href={`/api/artifacts/${a.key}`} target="_blank" rel="noreferrer">{a.key}</a>
                <span className="muted small"> {(a.size / 1024).toFixed(0)} KB</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
