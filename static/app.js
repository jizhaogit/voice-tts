/* Voice TTS Studio — vanilla JS frontend */

'use strict';

// ── Routing ────────────────────────────────────────────────────────────────

const pages = document.querySelectorAll('.page');
const navItems = document.querySelectorAll('.nav-item');

navItems.forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.page;
    navItems.forEach(n => n.classList.remove('active'));
    pages.forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`page-${target}`).classList.add('active');
    if (target === 'generate') refreshGeneratePage();
  });
});

// ── Helpers ────────────────────────────────────────────────────────────────

const EXT_ICON = { pdf: '📕', docx: '📘', doc: '📘', txt: '📝', html: '🌐', htm: '🌐' };
const LANG_LABEL = { en: 'EN', zh: '中文', ja: 'JA', auto: 'AUTO' };

function getExt(name) { return (name.split('.').pop() || '').toLowerCase(); }

function fmtBytes(b) {
  if (!b) return '';
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(0) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

function fmtDate(iso) {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' });
}

function show(el) { el.style.display = ''; }
function hide(el) { el.style.display = 'none'; }

function showError(el, msg) { el.textContent = msg; show(el); }
function hideError(el) { hide(el); }

async function apiFetch(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || body.error || `HTTP ${res.status}`);
  }
  return res.json();
}

function setupDropZone(zone, input, onFile) {
  zone.addEventListener('click', () => input.click());
  zone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') input.click(); });
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  });
  input.addEventListener('change', () => {
    if (input.files[0]) onFile(input.files[0]);
  });
}

// ── VOICES PAGE ─────────────────────────────────────────────────────────────

let voiceFile = null;

(function initVoices() {
  const form        = document.getElementById('voice-form');
  const nameIn      = document.getElementById('voice-name');
  const langSel     = document.getElementById('voice-language');
  const refText     = document.getElementById('voice-ref-text');
  const descIn      = document.getElementById('voice-desc');
  const dropZone    = document.getElementById('voice-drop');
  const fileInput   = document.getElementById('voice-file');
  const dropPh      = document.getElementById('voice-drop-ph');
  const transBtn    = document.getElementById('transcribe-btn');
  const transStatus = document.getElementById('transcribe-status');
  const progress    = document.getElementById('voice-progress');
  const errEl       = document.getElementById('voice-error');
  const submitBtn   = document.getElementById('voice-submit');

  setupDropZone(dropZone, fileInput, file => {
    voiceFile = file;
    dropPh.innerHTML = `
      <div class="drop-icon">${file.name.match(/\.wav$/i) ? '🔊' : '🎤'}</div>
      <div>${file.name}</div>
      <div class="muted small">${fmtBytes(file.size)}</div>`;
    transBtn.disabled = false;
  });

  // Auto-transcribe reference audio
  transBtn.addEventListener('click', async () => {
    if (!voiceFile) return;
    transBtn.disabled = true;
    transStatus.textContent = '🔍 Transcribing with Whisper (may download ~150 MB model on first use)…';
    show(transStatus);

    const fd = new FormData();
    fd.append('file', voiceFile);
    fd.append('language', langSel.value === 'auto' ? 'auto' : langSel.value);

    try {
      const { text } = await apiFetch('/api/voices/transcribe', { method: 'POST', body: fd });
      refText.value = text;
      transStatus.textContent = '✓ Transcription complete. Review and correct if needed.';
    } catch (err) {
      transStatus.textContent = `⚠ Transcription failed: ${err.message}`;
    } finally {
      transBtn.disabled = false;
    }
  });

  form.addEventListener('submit', async e => {
    e.preventDefault();
    hideError(errEl);

    if (!voiceFile) { showError(errEl, 'Please select a reference audio file.'); return; }
    if (!refText.value.trim()) { showError(errEl, 'Reference text is required.'); return; }

    submitBtn.disabled = true;
    show(progress);

    const fd = new FormData();
    fd.append('name', nameIn.value.trim());
    fd.append('language', langSel.value);
    fd.append('ref_text', refText.value.trim());
    fd.append('description', descIn.value.trim());
    fd.append('file', voiceFile);

    try {
      await apiFetch('/api/voices/', { method: 'POST', body: fd });
      form.reset();
      voiceFile = null;
      dropPh.innerHTML = `
        <div class="drop-icon">🎤</div>
        <div>Click or drag &amp; drop audio file here</div>
        <div class="muted small">MP3 · WAV · M4A · OGG · FLAC · AAC</div>`;
      transBtn.disabled = true;
      hide(transStatus);
      await loadVoices();
    } catch (err) {
      showError(errEl, err.message);
    } finally {
      submitBtn.disabled = false;
      hide(progress);
    }
  });

  loadVoices();
})();

async function loadVoices() {
  const list  = document.getElementById('voice-list');
  const badge = document.getElementById('voice-count');

  try {
    const voices = await apiFetch('/api/voices/');
    badge.textContent = voices.length;

    if (!voices.length) {
      list.innerHTML = '<p class="muted empty-state">No voices yet. Create your first voice above.</p>';
      return;
    }

    list.innerHTML = '<div class="voice-list-inner"></div>';
    const inner = list.querySelector('.voice-list-inner');

    voices.forEach(v => {
      const card = document.createElement('div');
      card.className = 'voice-card';
      card.innerHTML = `
        <div class="voice-avatar">🎤</div>
        <div class="voice-info">
          <div class="voice-name">
            ${v.name}
            <span class="voice-lang">${LANG_LABEL[v.language] || v.language}</span>
          </div>
          ${v.description ? `<div class="voice-desc muted">${v.description}</div>` : ''}
          <div class="voice-ref muted">"${v.ref_text.slice(0, 80)}${v.ref_text.length > 80 ? '…' : ''}"</div>
          <div class="voice-meta muted small">${fmtBytes(v.size_bytes)} · ${fmtDate(v.created_at)}</div>
        </div>
        <div class="voice-actions">
          <audio id="vaud-${v.id}" src="/api/voices/${v.id}/audio" style="display:none"></audio>
          <button class="btn btn-ghost btn-sm" data-play="${v.id}">▶ Play</button>
          <button class="btn btn-ghost btn-sm btn-danger" data-del-voice="${v.id}">Delete</button>
        </div>`;
      inner.appendChild(card);
    });

    // Play / Stop buttons
    inner.querySelectorAll('[data-play]').forEach(btn => {
      const id = btn.dataset.play;
      const audio = document.getElementById(`vaud-${id}`);
      btn.addEventListener('click', () => {
        if (audio.paused) { audio.play(); btn.textContent = '■ Stop'; }
        else { audio.pause(); audio.currentTime = 0; btn.textContent = '▶ Play'; }
        audio.addEventListener('ended', () => { btn.textContent = '▶ Play'; }, { once: true });
      });
    });

    // Delete buttons
    inner.querySelectorAll('[data-del-voice]').forEach(btn => {
      btn.addEventListener('click', async () => {
        if (!confirm('Delete this voice?')) return;
        try {
          await apiFetch(`/api/voices/${btn.dataset.delVoice}`, { method: 'DELETE' });
          await loadVoices();
        } catch (err) { alert(err.message); }
      });
    });

  } catch (err) {
    list.innerHTML = `<div class="alert alert-error">${err.message}</div>`;
  }
}

// ── DOCUMENTS PAGE ──────────────────────────────────────────────────────────

let docFile = null;

(function initDocuments() {
  const form      = document.getElementById('doc-form');
  const dropZone  = document.getElementById('doc-drop');
  const fileInput = document.getElementById('doc-file');
  const dropPh    = document.getElementById('doc-drop-ph');
  const errEl     = document.getElementById('doc-error');
  const submitBtn = document.getElementById('doc-submit');
  const closeBtn  = document.getElementById('preview-close');

  setupDropZone(dropZone, fileInput, file => {
    docFile = file;
    const icon = EXT_ICON[getExt(file.name)] || '📄';
    dropPh.innerHTML = `
      <div class="drop-icon">${icon}</div>
      <div>${file.name}</div>
      <div class="muted small">${fmtBytes(file.size)}</div>`;
  });

  form.addEventListener('submit', async e => {
    e.preventDefault();
    hideError(errEl);
    if (!docFile) { showError(errEl, 'Please select a document file.'); return; }

    submitBtn.disabled = true;
    submitBtn.textContent = 'Extracting text…';

    const fd = new FormData();
    fd.append('file', docFile);

    try {
      await apiFetch('/api/documents/', { method: 'POST', body: fd });
      docFile = null;
      fileInput.value = '';
      dropPh.innerHTML = `
        <div class="drop-icon">📄</div>
        <div>Click or drag &amp; drop a document here</div>
        <div class="muted small">TXT · PDF · DOCX · HTML — up to 50 MB</div>`;
      await loadDocuments();
    } catch (err) {
      showError(errEl, err.message);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = '📤 Upload & Extract Text';
    }
  });

  closeBtn.addEventListener('click', () => hide(document.getElementById('doc-preview-panel')));

  loadDocuments();
})();

async function loadDocuments() {
  const list  = document.getElementById('doc-list');
  const badge = document.getElementById('doc-count');

  try {
    const docs = await apiFetch('/api/documents/');
    badge.textContent = docs.length;

    if (!docs.length) {
      list.innerHTML = '<p class="muted empty-state">No documents yet. Upload one above.</p>';
      return;
    }

    list.innerHTML = '<div class="doc-list-inner"></div>';
    const inner = list.querySelector('.doc-list-inner');

    docs.forEach(doc => {
      const row = document.createElement('div');
      row.className = 'doc-row';
      const icon = EXT_ICON[getExt(doc.original_name)] || '📄';
      row.innerHTML = `
        <div class="doc-icon">${icon}</div>
        <div class="doc-info">
          <div class="doc-name">${doc.original_name}</div>
          <div class="doc-meta muted small">
            ${doc.word_count?.toLocaleString()} words · ${fmtBytes(doc.size_bytes)} · ${fmtDate(doc.uploaded_at)}
          </div>
        </div>
        <div class="doc-actions">
          <button class="btn btn-ghost btn-sm" data-preview="${doc.id}" data-name="${doc.original_name}">Preview</button>
          <button class="btn btn-ghost btn-sm btn-danger" data-del-doc="${doc.id}">Delete</button>
        </div>`;
      inner.appendChild(row);
    });

    inner.querySelectorAll('[data-preview]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const panel = document.getElementById('doc-preview-panel');
        const titleEl = document.getElementById('preview-title');
        const textEl  = document.getElementById('preview-text');
        titleEl.textContent = btn.dataset.name;
        textEl.textContent  = 'Loading…';
        show(panel);
        try {
          const { text } = await apiFetch(`/api/documents/${btn.dataset.preview}/text`);
          textEl.textContent = text;
        } catch (err) {
          textEl.textContent = `Error: ${err.message}`;
        }
      });
    });

    inner.querySelectorAll('[data-del-doc]').forEach(btn => {
      btn.addEventListener('click', async () => {
        if (!confirm('Delete this document?')) return;
        try {
          await apiFetch(`/api/documents/${btn.dataset.delDoc}`, { method: 'DELETE' });
          hide(document.getElementById('doc-preview-panel'));
          await loadDocuments();
        } catch (err) { alert(err.message); }
      });
    });

  } catch (err) {
    list.innerHTML = `<div class="alert alert-error">${err.message}</div>`;
  }
}

// ── GENERATE PAGE ───────────────────────────────────────────────────────────

const _pollers = {};

async function refreshGeneratePage() {
  await Promise.all([populateSelects(), loadJobs()]);
}

async function populateSelects() {
  const voiceSel = document.getElementById('gen-voice');
  const docSel   = document.getElementById('gen-doc');
  const warnings = document.getElementById('gen-warnings');

  const [voices, docs] = await Promise.all([
    apiFetch('/api/voices/').catch(() => []),
    apiFetch('/api/documents/').catch(() => []),
  ]);

  const warns = [];
  if (!voices.length) warns.push('⚠ No voices found. Go to the <strong>Voices</strong> page to create one.');
  if (!docs.length)   warns.push('⚠ No documents found. Go to the <strong>Documents</strong> page to upload one.');
  warnings.innerHTML = warns.map(w => `<div class="alert alert-warning" style="margin-bottom:12px">${w}</div>`).join('');

  voiceSel.innerHTML = '<option value="">Select a voice…</option>' +
    voices.map(v => `<option value="${v.id}">${v.name} (${LANG_LABEL[v.language] || v.language})</option>`).join('');

  docSel.innerHTML = '<option value="">Select a document…</option>' +
    docs.map(d => `<option value="${d.id}">${d.original_name} (${d.word_count?.toLocaleString()} words)</option>`).join('');
}

(function initGenerate() {
  const form      = document.getElementById('gen-form');
  const speedIn   = document.getElementById('gen-speed');
  const speedLbl  = document.getElementById('speed-label');
  const errEl     = document.getElementById('gen-error');
  const submitBtn = document.getElementById('gen-submit');
  const noteEl    = document.getElementById('model-note');

  speedIn.addEventListener('input', () => {
    speedLbl.textContent = parseFloat(speedIn.value).toFixed(2) + '×';
  });

  form.addEventListener('submit', async e => {
    e.preventDefault();
    hideError(errEl);

    const voiceId = document.getElementById('gen-voice').value;
    const docId   = document.getElementById('gen-doc').value;
    if (!voiceId || !docId) { showError(errEl, 'Select both a voice and a document.'); return; }

    submitBtn.disabled = true;
    submitBtn.textContent = 'Starting…';
    hide(noteEl);

    try {
      const { job_id } = await apiFetch('/api/generate/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice_id: voiceId, document_id: docId, speed: parseFloat(speedIn.value) }),
      });
      await loadJobs();
      startPolling(job_id);
    } catch (err) {
      showError(errEl, err.message);
      show(noteEl);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = '🎵 Generate Audio';
    }
  });
})();

function startPolling(jobId) {
  if (_pollers[jobId]) return;
  _pollers[jobId] = setInterval(async () => {
    try {
      const job = await apiFetch(`/api/generate/${jobId}`);
      updateJobCard(job);
      if (job.status === 'ready' || job.status === 'failed') {
        clearInterval(_pollers[jobId]);
        delete _pollers[jobId];
      }
    } catch {
      clearInterval(_pollers[jobId]);
      delete _pollers[jobId];
    }
  }, 2000);
}

async function loadJobs() {
  const list  = document.getElementById('job-list');
  const badge = document.getElementById('job-count');

  try {
    const jobs = await apiFetch('/api/generate/');
    badge.textContent = jobs.length;

    if (!jobs.length) {
      list.innerHTML = '<p class="muted empty-state">No generations yet.</p>';
      return;
    }

    list.innerHTML = '<div class="job-list-inner"></div>';
    jobs.forEach(job => {
      const inner = list.querySelector('.job-list-inner');
      const card = createJobCard(job);
      inner.appendChild(card);
      if (job.status === 'pending' || job.status === 'processing') {
        startPolling(job.id);
      }
    });

  } catch (err) {
    list.innerHTML = `<div class="alert alert-error">${err.message}</div>`;
  }
}

function createJobCard(job) {
  const card = document.createElement('div');
  card.className = 'job-card';
  card.id = `job-${job.id}`;
  card.appendChild(buildJobContent(job));

  card.querySelector('[data-del-job]')?.addEventListener('click', async () => {
    if (!confirm('Delete this job and audio file?')) return;
    clearInterval(_pollers[job.id]);
    delete _pollers[job.id];
    try {
      await apiFetch(`/api/generate/${job.id}`, { method: 'DELETE' });
      card.remove();
      const badge = document.getElementById('job-count');
      badge.textContent = parseInt(badge.textContent) - 1;
    } catch (err) { alert(err.message); }
  });

  return card;
}

function updateJobCard(job) {
  const card = document.getElementById(`job-${job.id}`);
  if (!card) return;
  const old = card.firstElementChild;
  const newContent = buildJobContent(job);
  card.replaceChild(newContent, old);

  card.querySelector('[data-del-job]')?.addEventListener('click', async () => {
    if (!confirm('Delete this job and audio file?')) return;
    clearInterval(_pollers[job.id]);
    delete _pollers[job.id];
    try {
      await apiFetch(`/api/generate/${job.id}`, { method: 'DELETE' });
      card.remove();
    } catch (err) { alert(err.message); }
  });
}

function buildJobContent(job) {
  const wrap = document.createElement('div');

  const pct = job.total_chunks > 0
    ? Math.round(job.processed_chunks / job.total_chunks * 100)
    : 20;

  const progressHtml = (job.status === 'processing' || job.status === 'pending') ? `
    <div class="progress-wrap">
      <div class="progress-bar">
        <div class="progress-fill" style="width:${job.status === 'pending' ? 5 : pct}%"></div>
      </div>
      <div class="progress-label muted small">
        ${job.status === 'pending'
          ? 'Waiting to start… (F5-TTS model may download ~1.2 GB on first run)'
          : job.total_chunks > 0
            ? `Processing chunk ${job.processed_chunks} of ${job.total_chunks}…`
            : 'Initialising F5-TTS…'}
      </div>
    </div>` : '';

  const audioHtml = job.status === 'ready' ? `
    <div class="job-audio">
      <audio controls class="audio-player" src="/api/generate/${job.id}/audio?inline=1"></audio>
      <a class="btn btn-ghost btn-sm" href="/api/generate/${job.id}/audio" download>⬇ Download</a>
    </div>` : '';

  const errorHtml = job.status === 'failed' ? `
    <div class="alert alert-error">${job.error || 'Unknown error'}</div>` : '';

  const sizeHtml = job.status === 'ready' && job.file_size_bytes
    ? ` · ${fmtBytes(job.file_size_bytes)}`
    : '';

  wrap.innerHTML = `
    <div class="job-top">
      <div class="job-names">🎤 ${job.voice_name} <span class="muted">×</span> 📄 ${job.document_name}</div>
      <span class="status-badge status-${job.status}">${job.status}${job.status === 'processing' && job.total_chunks > 0 ? ` ${job.processed_chunks}/${job.total_chunks}` : ''}</span>
    </div>
    <div class="job-meta muted small">Speed ${job.speed}× · ${fmtDate(job.created_at)}${sizeHtml}</div>
    ${progressHtml}
    ${audioHtml}
    ${errorHtml}
    <div><button class="btn btn-ghost btn-sm btn-danger" data-del-job="${job.id}">Delete</button></div>`;

  return wrap;
}

// Inline audio (for the audio tag src, we need the server to stream it)
// Patch the audio src to not trigger a download
document.addEventListener('click', e => {
  const audio = e.target.closest('audio');
  // nothing needed — browser handles it
});

// Initial load
loadVoices();
loadDocuments();
