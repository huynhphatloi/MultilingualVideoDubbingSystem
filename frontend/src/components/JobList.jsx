import React from 'react';

export default function JobList({ jobs, selected, onSelect, onDelete }) {
  return (
    <section className="card jobs">
      <h2>Jobs</h2>
      {!jobs.length && <p className="muted">No jobs yet.</p>}
      <ul>
        {jobs.map((j) => (
          <li
            key={j.job_id}
            className={j.job_id === selected ? 'active' : ''}
            onClick={() => onSelect(j.job_id)}
          >
            <div className="row-between">
              <code>{j.job_id}</code>
              <span className={`pill ${j.progress?.waiting_for_trigger &&
                !['completed', 'failed', 'cancelled'].includes(j.status)
                ? 'waiting' : j.status}`}>
                {j.progress?.waiting_for_trigger &&
                 !['completed', 'failed', 'cancelled'].includes(j.status)
                  ? 'waiting' : j.status}
              </span>
            </div>
            <div className="muted small">
              {j.source_filename || 'unnamed'} · {j.detected_language || j.source_language || '?'}
              {' → '}{j.target_language} · {j.progress?.percent ?? 0}%
            </div>
            <button
              className="link danger"
              onClick={(e) => { e.stopPropagation(); onDelete(j.job_id); }}
            >
              delete
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
