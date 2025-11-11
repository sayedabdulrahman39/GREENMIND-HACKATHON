

(() => {
  // Config
  let API_BASE = "http://127.0.0.1:8000";
  const POLL_MS = 2000;
  let pollTimer = null;

  // Elements
  const el = id => document.getElementById(id);
  const peopleEl = el("people");
  const brightEl = el("bright");
  const lightEl = el("light");
  const statusEl = el("status");
  const snapEl = el("snap");
  const alertUrlEl = el("alertUrl");
  const errorEl = el("error");
  const btnChangeApi = el("btnChangeApi");
  const btnSnapshot = el("btnSnapshot");
  const btnRefresh = el("btnRefresh");

  // Helpers
  function setStatusOk(text = "✅ All good") {
    statusEl.textContent = text;
    statusEl.className = "status ok";
  }
  function setStatusWarn(text = "⚠️ You forgot something!") {
    statusEl.textContent = text;
    statusEl.className = "status warn";
  }
  function setDisconnected() {
    statusEl.textContent = "⚠️ Disconnected";
    statusEl.className = "status";
  }
  function safeText(v) { return (v === null || v === undefined) ? "-" : v; }

  // Main poll function
  async function pollStatusOnce() {
    errorEl.textContent = "";
    try {
      const res = await fetch(API_BASE + "/status", { cache: "no-store" , method: "GET" });
      if (!res.ok) throw new Error("Status fetch failed: " + res.status);
      const j = await res.json();

      // Update basic stats
      peopleEl.textContent = safeText(j.people_count);
      brightEl.textContent = safeText(j.brightness);
      lightEl.textContent = j.light_on ? "ON" : "OFF";

      // Alert logic display
      const alertCondition = (j.alert_ready || (j.light_on && j.people_count === 0));
      if (alertCondition) {
        setStatusWarn("⚠️ You forgot something!");
      } else {
        setStatusOk("✅ All good");
      }

      // last alert image link
      if (j.last_alert_image_url) {
        alertUrlEl.textContent = "view";
        alertUrlEl.href = j.last_alert_image_url;
        alertUrlEl.target = "_blank";
      } else {
        alertUrlEl.textContent = "none";
        alertUrlEl.href = "#";
        alertUrlEl.removeAttribute("target");
      }

      // update snapshot image (do this AFTER status so UI appears responsive)
      refreshSnapshot();

    } catch (err) {
      // show error but keep trying
      console.warn("poll error", err);
      errorEl.textContent = "Connection error: " + (err.message || err);
      setDisconnected();
    }
  }

  // Refresh snapshot image with cache-buster
  function refreshSnapshot() {
    try {
      // create a new object URL by fetching the blob, to avoid caching issues on some browsers
      // But keep it simple: set image src with timestamp
      snapEl.src = API_BASE + "/snapshot?t=" + Date.now();
    } catch (e) {
      console.warn("snapshot refresh failed", e);
    }
  }

  // Start/stop poll
  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollStatusOnce();
    pollTimer = setInterval(pollStatusOnce, POLL_MS);
  }
  function stopPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
  }

  // Button handlers
  if (btnChangeApi) {
    btnChangeApi.addEventListener("click", () => {
      const v = prompt("Backend base URL (example: http://127.0.0.1:8000):", API_BASE);
      if (!v) return;
      API_BASE = v.replace(/\/+$/, ""); // strip trailing slash
      // quick immediate poll
      startPolling();
      alert("API_BASE set to: " + API_BASE);
    });
  }

  if (btnSnapshot) {
    btnSnapshot.addEventListener("click", async () => {
      try {
        // fetch snapshot and show it (use blob to avoid CORS caching issues)
        const r = await fetch(API_BASE + "/snapshot?t=" + Date.now(), { cache: "no-store" });
        if (!r.ok) throw new Error("Snapshot failed: " + r.status);
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        snapEl.src = url;
        // revoke after a short time to avoid memory leak
        setTimeout(() => URL.revokeObjectURL(url), 30000);
      } catch (e) {
        console.warn("manual snapshot failed", e);
        errorEl.textContent = "Snapshot failed: " + (e.message || e);
      }
    });
  }

  if (btnRefresh) {
    btnRefresh.addEventListener("click", () => pollStatusOnce());
  }

  // initialize
  startPolling();

  // expose for debugging
  window.__GreenMind = {
    setApi: (u) => { API_BASE = u; startPolling(); },
    stop: stopPolling,
    start: startPolling,
  };
})();
