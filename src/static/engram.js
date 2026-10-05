// Every button and form on the admin pages talks to the JSON API through this file.
//
//   <button data-api="POST /api/cleanup/merge" data-body='{"keep":"x"}' data-then="reload">
//   <form data-api="PUT /api/settings/models" data-then="toast:Saved">
//
// data-collect="#selector"  merge that form's fields into the body
// data-confirm="Question?"  ask first (destructive actions)
// data-then: reload | remove:<selector> | redirect:<url> | redirect-to:<response field> | toast:<message> | event
// Failures show the API's error text; nothing fails silently.
(function () {
  const toasts = () => document.getElementById('toasts');

  function toast(msg, kind) {
    const el = document.createElement('div');
    el.className = 'toast' + (kind === 'error' ? ' error' : '');
    el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
    el.textContent = msg;
    toasts().appendChild(el);
    setTimeout(() => el.remove(), kind === 'error' ? 8000 : 3500);
  }

  async function api(method, url, body) {
    const opts = { method, headers: { Accept: 'application/json' } };
    if (body !== undefined && method !== 'GET') {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const r = await fetch(url, opts);
    const text = await r.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (_) { data = text; }
    if (!r.ok) {
      const d = data && data.detail;
      const msg = typeof d === 'string' ? d : d && d.error ? `${d.error} ${d.fix || ''}`
        : Array.isArray(d) ? d.map((x) => `${(x.loc || []).slice(-1)[0]}: ${x.msg}`).join('; ')
        : (data && data.error) || `${r.status} ${r.statusText}`;
      throw new Error(msg);
    }
    return data;
  }

  function collect(form) {
    const out = {};
    for (const el of form.querySelectorAll('[name]')) {
      if (el.disabled || el.name === '') continue;
      const name = el.name, type = el.dataset.type;
      let v;
      if (el.type === 'checkbox') {
        if (el.dataset.multi !== undefined) { if (!el.checked) continue; v = el.value; }
        else v = el.checked;
      } else if (el.type === 'radio') { if (!el.checked) continue; v = el.value; }
      else if (el.tagName === 'SELECT' && el.multiple) v = [...el.selectedOptions].map((o) => o.value);
      else v = el.value;
      if (type === 'int') v = v === '' ? null : parseInt(v, 10);
      if (type === 'float') v = v === '' ? null : parseFloat(v);
      if (type === 'json') v = v ? JSON.parse(v) : null;
      if (type === 'lines') v = v.split('\n').map((s) => s.trim()).filter(Boolean);
      if (type === 'null-empty' && v === '') v = null;
      if (el.dataset.multi !== undefined && v === '') { out[name] = out[name] || []; continue; }
      if (type === 'step') { const [provider, ...m] = v.split('|'); v = { provider, model: m.join('|') || null }; }
      if (el.dataset.multi !== undefined) (out[name] = out[name] || []).push(v);
      else out[name] = v;
    }
    return out;
  }

  function then(el, result) {
    const spec = el.dataset.then || 'reload';
    el.dispatchEvent(new CustomEvent('engram:done', { bubbles: true, detail: result }));
    const leaves = /(^|\|)(reload|redirect)/.test(spec);
    for (const step of spec.split('|')) {
      const [what, ...rest] = step.split(':');
      const arg = rest.join(':');
      if (what === 'reload') location.reload();
      else if (what === 'remove') document.querySelectorAll(arg).forEach((n) => n.remove());
      else if (what === 'redirect') location.href = arg;
      else if (what === 'redirect-to') location.href = result[arg];
      else if (what === 'toast') {
        // the page is about to reload or move: show the message on the next page instead
        if (leaves) { try { sessionStorage.setItem('engram-toast', arg || 'Done'); } catch (_) { /* private mode */ } }
        else toast(arg || 'Done');
      }
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    try {
      const msg = sessionStorage.getItem('engram-toast');
      if (msg) { sessionStorage.removeItem('engram-toast'); toast(msg); }
    } catch (_) { /* storage unavailable */ }
  });

  async function run(el, body) {
    const [method, url] = el.dataset.api.trim().split(/\s+/);
    if (el.dataset.confirm && !confirm(el.dataset.confirm)) return;
    const btns = el.tagName === 'FORM' ? el.querySelectorAll('button[type=submit], button:not([type])') : [el];
    btns.forEach((b) => { b.disabled = true; b.setAttribute('aria-busy', 'true'); });
    try {
      then(el, await api(method, url, body));
    } catch (err) {
      toast(err.message, 'error');
    } finally {
      btns.forEach((b) => { b.disabled = false; b.removeAttribute('aria-busy'); });
    }
  }

  document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-api]:not(form)');
    if (!el) return;
    e.preventDefault();
    let body = el.dataset.body ? JSON.parse(el.dataset.body) : {};
    if (el.dataset.collect) body = { ...body, ...collect(document.querySelector(el.dataset.collect)) };
    run(el, body);
  });

  document.addEventListener('submit', (e) => {
    const form = e.target.closest('form[data-api]');
    if (!form) return;
    e.preventDefault();
    const extra = form.dataset.body ? JSON.parse(form.dataset.body) : {};
    run(form, { ...extra, ...collect(form) });
  });

  // <button data-open="#dialog-id"> opens a <dialog>; [data-close] inside closes it.
  document.addEventListener('click', (e) => {
    const opener = e.target.closest('[data-open]');
    if (opener) { e.preventDefault(); document.querySelector(opener.dataset.open).showModal(); }
    const closer = e.target.closest('[data-close]');
    if (closer) { e.preventDefault(); closer.closest('dialog').close(); }
  });

  // ⌘K / Ctrl+K focuses global search.
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      const s = document.getElementById('global-search');
      if (s) { e.preventDefault(); s.focus(); }
    }
  });

  window.Engram = { api, toast, collect };
})();
