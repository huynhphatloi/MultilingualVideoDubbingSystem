const API = '/api';

async function request(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: options.body instanceof FormData ? undefined : { 'Content-Type': 'application/json' },
    ...options,
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const message = data?.error?.message || data?.detail || res.statusText;
    const err = new Error(message);
    err.code = data?.error?.code;
    err.status = res.status;
    throw err;
  }
  return data;
}

export const api = {
  languages: () => request('/languages'),
  ttsModels: (language) =>
    request(`/speech/tts-models?language=${encodeURIComponent(language)}`),
  stages: () => request('/pipeline/stages'),
  health: () => request('/health/ready'),
  device: () => request('/health/device'),

  listJobs: () => request('/jobs?limit=25'),
  getJob: (id) => request(`/jobs/${id}`),
  deleteJob: (id) => request(`/jobs/${id}`, { method: 'DELETE' }),

  createJob: (payload) =>
    request('/jobs', { method: 'POST', body: JSON.stringify(payload) }),

  uploadVideo: (id, file, targetLanguage, onProgress) =>
    new Promise((resolve, reject) => {
      const form = new FormData();
      form.append('file', file);
      if (targetLanguage) form.append('target_language', targetLanguage);

      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${API}/jobs/${id}/upload`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        const body = xhr.responseText ? JSON.parse(xhr.responseText) : null;
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new Error(body?.error?.message || `Upload failed (${xhr.status})`));
      };
      xhr.onerror = () => reject(new Error('Network error while uploading'));
      xhr.send(form);
    }),

  segments: (id) => request(`/jobs/${id}/segments`),
  review: (id, edits) =>
    request(`/jobs/${id}/review`, { method: 'POST', body: JSON.stringify({ edits }) }),
  artifacts: (id) => request(`/jobs/${id}/artifacts`),

  downloadUrl: (id, kind) => `${API}/jobs/${id}/download/${kind}`,

  startPipeline: async (payload) => {
    const res = await fetch('/webhook/dubbing/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res.ok) return res.json().catch(() => ({}));

    // Show what n8n actually said - "returned 500" on its own is useless.
    const body = await res.text().catch(() => '');
    let detail = body.slice(0, 400);
    try {
      const parsed = JSON.parse(body);
      detail = parsed.message || parsed.error || detail;
    } catch { /* not json, keep the raw text */ }

    if (res.status === 404) {
      throw new Error(
        'n8n webhook not registered. Open the workflow in n8n, click Save, then toggle it Active.',
      );
    }
    throw new Error(`n8n returned ${res.status}${detail ? ` - ${detail}` : ''}`);
  },
};
