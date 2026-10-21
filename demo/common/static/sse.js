// sse.js: tiny shared helpers for the demo dashboards (EventSource + DOM), no framework.
(function (g) {
  function stream(onEvent, path) {
    let es;
    const open = () => {
      es = new EventSource(path || "/api/stream");
      es.onmessage = (m) => { try { onEvent(JSON.parse(m.data)); } catch (e) {} };
      es.onerror = () => { es.close(); setTimeout(open, 1500); };  // reconnect if the app restarts
    };
    open();
  }
  const el = (sel, root) => (root || document).querySelector(sel);
  const els = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  function node(tag, attrs, kids) {
    const n = document.createElement(tag);
    for (const k in (attrs || {})) {
      if (k === "class") n.className = attrs[k];
      else if (k === "html") n.innerHTML = attrs[k];
      else if (k === "text") n.textContent = attrs[k];
      else n.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach((c) => n.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return n;
  }
  async function post(path, body) {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}) });
    return r.json().catch(() => ({}));
  }
  const fmt = (n) => (n == null ? "–" : Number(n).toLocaleString());
  const pct = (n) => (n == null ? "–" : (Math.round(n * 10) / 10) + "%");
  const clock = (t) => new Date((t || Date.now() / 1000) * 1000).toLocaleTimeString();
  g.SSE = { stream, el, els, node, post, fmt, pct, clock };
})(window);
