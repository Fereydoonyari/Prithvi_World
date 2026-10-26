const state = {
  token: localStorage.getItem("pw_token") || "",
  user: null,
  aois: [],
  selected: null,
  observations: [],
  embeddings: [],
  overlay: null,
  placing: false,
  charts: {},
  googleBasemap: null,
  selectedExp: null,
  lastOverlay: null,
  lastVeg: null,
  lastDecode: null,
  lastMetrics: null,
  explainHistory: [],
  explainLlm: false,
};

let jobPollTimer = null;

const $ = (id) => document.getElementById(id);

if (window.Chart) {
  Chart.defaults.color = "#8a8a96";
  Chart.defaults.borderColor = "#22222c";
  Chart.defaults.backgroundColor = "#101014";
}

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (opts.body && !(opts.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.body);
  }
  const res = await fetch(path, { ...opts, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail || JSON.stringify(j);
    } catch (_) {}
    if (typeof detail !== "string") detail = JSON.stringify(detail);
    const authCall = path.includes("/auth/login") || path.includes("/auth/register");
    if (res.status === 401 && !authCall) logout();
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  if (res.headers.get("content-type")?.includes("application/json")) return res.json();
  return res;
}

function logout() {
  const token = state.token;
  state.token = "";
  state.user = null;
  localStorage.removeItem("pw_token");
  if (jobPollTimer) {
    clearTimeout(jobPollTimer);
    jobPollTimer = null;
  }
  if (token) {
    fetch("/api/auth/logout", { method: "POST", headers: { Authorization: `Bearer ${token}` } }).catch(() => {});
  }
  $("app").hidden = true;
  $("login").hidden = false;
  setAuthMode("signin");
}

let authMode = "signin";
function setAuthMode(mode) {
  authMode = mode;
  const register = mode === "register";
  $("auth-signin").classList.toggle("active", !register);
  $("auth-register").classList.toggle("active", register);
  $("password2-wrap").hidden = !register;
  $("auth-submit").textContent = register ? "Create account" : "Sign in";
  $("auth-hint").textContent = register
    ? "Creates a researcher account. Shared study sites (Italy, …) stay visible to everyone."
    : "Demo: admin@workbench.local / admin · researcher@workbench.local / researcher";
  $("login-error").hidden = true;
  $("password").autocomplete = register ? "new-password" : "current-password";
  if (register) {
    $("email").value = "";
    $("password").value = "";
    $("password2").value = "";
  }
}
$("auth-signin").addEventListener("click", () => setAuthMode("signin"));
$("auth-register").addEventListener("click", () => setAuthMode("register"));

$("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("login-error").hidden = true;
  const email = $("email").value.trim();
  const password = $("password").value;
  try {
    if (authMode === "register") {
      if (password !== $("password2").value) throw new Error("Passwords do not match");
      if (password.length < 6) throw new Error("Password must be at least 6 characters");
    }
    const path = authMode === "register" ? "/api/auth/register" : "/api/auth/login";
    const data = await api(path, { method: "POST", body: { email, password } });
    state.token = data.token;
    state.user = data.user;
    localStorage.setItem("pw_token", data.token);
    await showApp();
  } catch (err) {
    $("login-error").textContent = err.message;
    $("login-error").hidden = false;
  }
});

$("logout").addEventListener("click", logout);

// ---------------------------------------------------------------------------
// Left-rail collapsible panes + right-side task panels.
// Each task (AOI / Overlay / Train / Prediction) is its own panel; only one
// is shown at a time. Collapse hides the body without switching tasks.
// ---------------------------------------------------------------------------
const PANE_STATE_KEY = "pw_panes";
const TASK_KEY = "pw_task";
const paneState = JSON.parse(localStorage.getItem(PANE_STATE_KEY) || "{}");

document.querySelectorAll("details.pane").forEach((pane) => {
  const key = pane.dataset.pane;
  if (key in paneState) pane.open = Boolean(paneState[key]);
  pane.addEventListener("toggle", () => {
    paneState[key] = pane.open;
    localStorage.setItem(PANE_STATE_KEY, JSON.stringify(paneState));
    if (pane.open) resizePaneCharts(pane);
  });
  pane.querySelectorAll("summary button").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
  });
});

function resizePaneCharts(root) {
  root.querySelectorAll("canvas").forEach((c) => {
    const chart = Object.values(state.charts).find((ch) => ch && ch.canvas === c);
    if (chart) chart.resize();
  });
}

function showTask(key, { expand = true } = {}) {
  const allowed = ["aoi", "overlay", "train", "predict", "explain"];
  if (!allowed.includes(key)) {
    const pane = document.querySelector(`details.pane[data-pane="${key}"]`);
    if (pane && !pane.open) pane.open = true;
    return;
  }
  document.querySelectorAll(".task-tabs button").forEach((btn) => {
    btn.setAttribute("aria-selected", btn.dataset.task === key ? "true" : "false");
  });
  document.querySelectorAll(".task-panel").forEach((panel) => {
    const on = panel.dataset.task === key;
    panel.classList.toggle("active", on);
    if (on && expand) {
      panel.classList.remove("collapsed");
      const btn = panel.querySelector(".task-collapse");
      if (btn) {
        btn.textContent = "−";
        btn.setAttribute("aria-expanded", "true");
        btn.title = "Collapse";
      }
      paneState[`task_${key}`] = true;
      requestAnimationFrame(() => resizePaneCharts(panel));
    }
  });
  localStorage.setItem(TASK_KEY, key);
  localStorage.setItem(PANE_STATE_KEY, JSON.stringify(paneState));
}

function setPanelCollapsed(panel, collapsed) {
  panel.classList.toggle("collapsed", collapsed);
  const btn = panel.querySelector(".task-collapse");
  if (btn) {
    btn.textContent = collapsed ? "+" : "−";
    btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    btn.title = collapsed ? "Expand" : "Collapse";
  }
  paneState[`task_${panel.dataset.task}`] = !collapsed;
  localStorage.setItem(PANE_STATE_KEY, JSON.stringify(paneState));
  if (!collapsed) requestAnimationFrame(() => resizePaneCharts(panel));
}

document.querySelectorAll(".task-tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    const panel = document.querySelector(`.task-panel[data-task="${btn.dataset.task}"]`);
    if (panel?.classList.contains("active")) {
      setPanelCollapsed(panel, !panel.classList.contains("collapsed"));
      return;
    }
    showTask(btn.dataset.task);
  });
});
document.querySelectorAll(".task-panel .task-collapse").forEach((btn) => {
  btn.addEventListener("click", () => {
    const panel = btn.closest(".task-panel");
    setPanelCollapsed(panel, !panel.classList.contains("collapsed"));
  });
});
document.querySelectorAll(".task-panel .task-head h2").forEach((h) => {
  h.addEventListener("click", () => {
    const panel = h.closest(".task-panel");
    setPanelCollapsed(panel, !panel.classList.contains("collapsed"));
  });
});

const savedTask = localStorage.getItem(TASK_KEY);
showTask(allowedTask(savedTask) ? savedTask : "aoi", { expand: false });
document.querySelectorAll(".task-panel").forEach((panel) => {
  const key = `task_${panel.dataset.task}`;
  if (key in paneState) setPanelCollapsed(panel, paneState[key] === false);
});

function allowedTask(key) {
  return ["aoi", "overlay", "train", "predict", "explain"].includes(key);
}

function openPane(key) {
  showTask(key);
}

const map = L.map("map", { zoomControl: true }).setView([30, 10], 3);
const streets = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
});
const satellite = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  {
    maxZoom: 19,
    attribution: "Tiles © Esri — Esri, Maxar, Earthstar Geographics",
  }
).addTo(map);
const layerControl = L.control.layers(
  { "Satellite": satellite, "Streets": streets },
  {},
  { position: "topright", collapsed: true }
).addTo(map);

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const el = document.createElement("script");
    el.src = src;
    el.async = true;
    el.onload = () => resolve();
    el.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(el);
  });
}

async function enableGoogleSatellite() {
  if (state.googleBasemap) return;
  const cfg = await api("/api/config");
  const key = (cfg.google_maps_api_key || "").trim();
  if (!key) return;
  await loadScript(`https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(key)}`);
  await loadScript("https://unpkg.com/leaflet.gridlayer.googlemutant@0.14.1/dist/Leaflet.GoogleMutant.js");
  const googleSat = L.gridLayer.googleMutant({ type: "hybrid", maxZoom: 20 });
  layerControl.addBaseLayer(googleSat, "Google satellite");
  map.removeLayer(satellite);
  googleSat.addTo(map);
  state.googleBasemap = googleSat;
}

const markers = L.layerGroup().addTo(map);
let footprint = null;
let draftPin = null;

function setVisibility(shared) {
  const want = shared ? "shared" : "private";
  document.querySelectorAll('input[name="visibility"]').forEach((el) => {
    el.checked = el.value === want;
  });
}

function visibilityShared() {
  return document.querySelector('input[name="visibility"]:checked')?.value === "shared";
}

function clearDraftPin() {
  if (draftPin) {
    map.removeLayer(draftPin);
    draftPin = null;
  }
}

function moveDraftPin(lat, lon) {
  const ll = L.latLng(Number(lat), Number(lon));
  if (!Number.isFinite(ll.lat) || !Number.isFinite(ll.lng)) return;
  if (!draftPin) {
    draftPin = L.marker(ll, {
      draggable: true,
      zIndexOffset: 2000,
      icon: L.divIcon({
        className: "draft-pin",
        html: '<span class="draft-pin-mark"></span>',
        iconSize: [22, 28],
        iconAnchor: [11, 26],
      }),
    }).addTo(map);
    draftPin.on("drag", () => {
      const p = draftPin.getLatLng();
      const form = $("aoi-form");
      form.latitude.value = p.lat.toFixed(4);
      form.longitude.value = p.lng.toFixed(4);
    });
  } else {
    draftPin.setLatLng(ll);
  }
}

function startNewAoi() {
  state.selected = null;
  state.placing = true;
  const form = $("aoi-form");
  form.reset();
  form.name.value = "New AOI";
  setVisibility(false);
  const c = map.getCenter();
  form.latitude.value = c.lat.toFixed(4);
  form.longitude.value = c.lng.toFixed(4);
  moveDraftPin(c.lat, c.lng);
  renderAoIs();
  openPane("aoi");
}

map.on("click", (ev) => {
  const form = $("aoi-form");
  if (!state.placing) startNewAoi();
  form.latitude.value = ev.latlng.lat.toFixed(4);
  form.longitude.value = ev.latlng.lng.toFixed(4);
  if (!form.name.value) form.name.value = "New AOI";
  state.placing = true;
  moveDraftPin(ev.latlng.lat, ev.latlng.lng);
  openPane("aoi");
});

function renderAoIs() {
  markers.clearLayers();
  const ul = $("aoi-list");
  ul.innerHTML = "";
  state.aois.forEach((a) => {
    const li = document.createElement("li");
    li.className = state.selected?.id === a.id ? "active" : "";
    const tag = a.shared
      ? ' <span class="badge">shared</span>'
      : ' <span class="badge private">private</span>';
    li.innerHTML = `<strong>${a.name}</strong>${tag}<span class="sub">${a.latitude.toFixed(2)}, ${a.longitude.toFixed(2)} · ${a.n_observations} scenes</span>`;
    li.onclick = () => selectAoi(a.id);
    ul.appendChild(li);
    const m = L.circleMarker([a.latitude, a.longitude], {
      radius: 7,
      color: "#d4b06a",
      fillColor: "#d4b06a",
      fillOpacity: 0.85,
      weight: 1,
    }).addTo(markers);
    m.bindTooltip(a.name);
    m.on("click", () => selectAoi(a.id));
    if (a.bounds) {
      L.rectangle(a.bounds, { color: "#d4b06a", weight: 1, fillOpacity: 0.05 }).addTo(markers);
    }
  });
}

async function selectAoi(id) {
  state.selected = state.aois.find((a) => a.id === id) || null;
  state.placing = false;
  clearDraftPin();
  renderAoIs();
  if (!state.selected) return;
  const a = state.selected;
  const form = $("aoi-form");
  form.name.value = a.name;
  form.latitude.value = a.latitude;
  form.longitude.value = a.longitude;
  form.biome.value = a.biome || "";
  form.notes.value = a.notes || "";
  setVisibility(!!a.shared);
  map.setView([a.latitude, a.longitude], 11);
  if (footprint) map.removeLayer(footprint);
  if (a.bounds) {
    footprint = L.rectangle(a.bounds, { color: "#8a8a96", weight: 2, fillOpacity: 0.06 }).addTo(map);
  }
  state.selectedExp = null;
  await Promise.all([loadObs(), loadExperiments()]);
}

async function loadAoIs() {
  state.aois = await api("/api/aois");
  renderAoIs();
  if (!state.selected && state.aois.length) await selectAoi(state.aois[0].id);
}

async function loadObs() {
  if (!state.selected) return;
  state.observations = await api(`/api/aois/${state.selected.id}/observations`);
  state.embeddings = await api(`/api/aois/${state.selected.id}/embeddings`);
  const ul = $("obs-list");
  ul.innerHTML = "";
  const encoded = new Set(state.embeddings.map((e) => e.date_key));
  $("obs-count").textContent = state.observations.length
    ? `${state.observations.length} · ${encoded.size} encoded`
    : "";
  state.observations.forEach((o) => {
    const li = document.createElement("li");
    const pct = (o.clear_fraction * 100).toFixed(0);
    const tag = encoded.has(o.date_key) ? "encoded" : "HLS only";
    li.innerHTML = `${o.date_key}<span class="sub">clear ${pct}% · ${tag}</span>`;
    ul.appendChild(li);
  });
  fillDates();
}

function overlayKind() {
  return document.querySelector('input[name="overlay-kind"]:checked')?.value || "token";
}

function fillDates() {
  const keys = [...new Set(state.embeddings.map((e) => e.date_key))].sort();
  for (const id of ["date-a", "date-b"]) {
    const sel = $(id);
    const prev = sel.value;
    sel.innerHTML = keys.map((k) => `<option value="${k}">${k}</option>`).join("");
    if (keys.includes(prev)) sel.value = prev;
  }
  if (keys.length >= 2) {
    if (!$("date-b").value || $("date-b").value === $("date-a").value) {
      $("date-b").selectedIndex = keys.length - 1;
    }
    if ($("date-a").value === $("date-b").value) $("date-a").selectedIndex = 0;
  }
  syncDatePair();
  if (keys.length < 2) {
    $("overlay-msg").textContent = "Need two encoded dates. Run Encode first.";
  }
  if (keys.length) {
    const lo = keys[0];
    const hi = keys[keys.length - 1];
    for (const id of ["veg-from", "veg-to"]) {
      const el = $(id);
      el.min = lo;
      el.max = hi;
      if (!el.value || el.value < lo || el.value > hi) el.value = id === "veg-from" ? lo : hi;
    }
  }
}

function syncDatePair() {
  const a = $("date-a");
  const b = $("date-b");
  [...b.options].forEach((opt) => {
    opt.disabled = opt.value === a.value;
  });
  if (a.value && a.value === b.value && b.options.length > 1) {
    const next = [...b.options].find((opt) => !opt.disabled);
    if (next) b.value = next.value;
  }
}

function syncOverlayLegend() {
  const veg = overlayKind() === "vegetation";
  $("legend-veg").hidden = !veg;
  $("legend-change").hidden = veg;
}

$("date-a").addEventListener("change", syncDatePair);
$("date-b").addEventListener("change", syncDatePair);
document.querySelectorAll('input[name="overlay-kind"]').forEach((el) => {
  el.addEventListener("change", syncOverlayLegend);
});

$("aoi-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const body = {
    name: form.name.value,
    latitude: Number(form.latitude.value),
    longitude: Number(form.longitude.value),
    biome: form.biome.value,
    notes: form.notes.value,
    shared: visibilityShared(),
  };
  if (state.selected && !state.placing) {
    await api(`/api/aois/${state.selected.id}`, { method: "PUT", body });
  } else {
    await api("/api/aois", { method: "POST", body });
  }
  state.placing = false;
  clearDraftPin();
  await loadAoIs();
});

$("aoi-new").addEventListener("click", () => {
  startNewAoi();
});

["latitude", "longitude"].forEach((name) => {
  $("aoi-form")[name].addEventListener("input", () => {
    if (!state.placing) return;
    moveDraftPin($("aoi-form").latitude.value, $("aoi-form").longitude.value);
  });
});

$("aoi-delete").addEventListener("click", async () => {
  if (!state.selected) return;
  if (!confirm(`Delete ${state.selected.name}?`)) return;
  await api(`/api/aois/${state.selected.id}`, { method: "DELETE" });
  state.selected = null;
  clearDraftPin();
  await loadAoIs();
});

async function postJob(kind, extra = {}) {
  if (!state.selected && kind !== "evaluate") throw new Error("Select an AOI");
  return api("/api/jobs", {
    method: "POST",
    body: {
      kind,
      aoi_id: extra.aoi_id ?? state.selected?.id,
      experiment_id: extra.experiment_id,
      payload: extra.payload || {},
    },
  });
}

$("btn-ingest").addEventListener("click", async () => {
  if (!state.selected) {
    alert("Select an AOI first.");
    return;
  }
  const start = $("ingest-start").value || "2022-01-01";
  const end = $("ingest-end").value || "2024-12-31";
  if (start > end) {
    alert("Start date must be before end date.");
    return;
  }
  await postJob("ingest_hls", { payload: { start, end } });
  openPane("jobs");
  pollJobs();
});
$("btn-index").addEventListener("click", async () => {
  await postJob("index_observations", { payload: { min_clear: 0 } });
  openPane("jobs");
  pollJobs();
});
$("btn-clear-jobs").addEventListener("click", async () => {
  const res = await api("/api/jobs", { method: "DELETE" });
  const ul = $("job-list");
  ul.innerHTML = "";
  if (res && typeof res.deleted === "number" && res.deleted === 0) {
    const li = document.createElement("li");
    li.className = "sub";
    li.textContent = "Nothing to clear (queued/running jobs are kept).";
    ul.appendChild(li);
  }
  pollJobs();
});
$("btn-encode").addEventListener("click", async () => {
  await postJob("encode", { payload: { min_clear: 0.9 } });
  openPane("jobs");
  pollJobs();
});

async function pollJobs() {
  if (!state.token) return;
  const jobs = await api("/api/jobs");
  const ul = $("job-list");
  ul.innerHTML = "";
  jobs.slice(0, 12).forEach((j) => {
    const li = document.createElement("li");
    li.innerHTML = `#${j.id} ${j.kind} <span class="status-${j.status}">${j.status}</span>`;
    ul.appendChild(li);
  });
  if (jobs.some((j) => j.status === "queued" || j.status === "running")) {
    jobPollTimer = setTimeout(pollJobs, 2000);
  } else {
    if (state.selected) loadObs();
    loadExperiments();
  }
}

$("btn-overlay").addEventListener("click", async () => {
  if (!state.selected) return;
  $("overlay-msg").textContent = "Composing overlay…";
  try {
    const data = await api(
      `/api/overlays/change?aoi_id=${state.selected.id}&date_a=${$("date-a").value}&date_b=${$("date-b").value}&kind=${overlayKind()}`
    );
    if (state.overlay) map.removeLayer(state.overlay);
    state.overlay = L.imageOverlay(`${data.url}?t=${Date.now()}`, data.bounds, { opacity: Number($("opacity").value) }).addTo(map);
    map.fitBounds(data.bounds);
    state.lastOverlay = {
      kind: data.kind,
      date_a: $("date-a").value,
      date_b: $("date-b").value,
      mean: data.mean,
      p98: data.p98,
      frac_gain: data.frac_gain,
      frac_loss: data.frac_loss,
      snapped: data.snapped,
    };
    const snap = (data.snapped && data.snapped.length) ? `  snapped ${data.snapped.join("; ")}` : "";
    if (data.kind === "vegetation") {
      const gain = ((data.frac_gain || 0) * 100).toFixed(0);
      const loss = ((data.frac_loss || 0) * 100).toFixed(0);
      $("overlay-msg").textContent = `NDVI Δ mean=${data.mean.toFixed(3)}  greening ${gain}%  loss ${loss}%${snap}`;
    } else {
      $("overlay-msg").textContent = `change mean=${data.mean.toFixed(3)}  p98=${data.p98.toFixed(3)}${snap}`;
    }
  } catch (err) {
    $("overlay-msg").textContent = err.message;
  }
});

$("opacity").addEventListener("input", () => {
  if (state.overlay) state.overlay.setOpacity(Number($("opacity").value));
});

$("btn-veg-trend").addEventListener("click", async () => {
  if (!state.selected) {
    $("veg-msg").textContent = "Select an AOI first.";
    return;
  }
  const from = $("veg-from").value;
  const to = $("veg-to").value;
  if (!from || !to) {
    $("veg-msg").textContent = "Pick a start and end date.";
    return;
  }
  $("veg-msg").textContent = "Computing NDVI per scene…";
  try {
    const data = await api(
      `/api/analysis/vegetation?aoi_id=${state.selected.id}&date_from=${from}&date_to=${to}`
    );
    drawVegTrend(data);
    state.lastVeg = {
      date_from: data.date_from,
      date_to: data.date_to,
      count: data.count,
      first: data.first,
      last: data.last,
      delta: data.delta,
    };
    const arrow = data.delta >= 0 ? "▲" : "▼";
    $("veg-msg").textContent =
      `${data.count} scenes · ${data.date_from} → ${data.date_to} · ` +
      `NDVI ${data.first.toFixed(3)} → ${data.last.toFixed(3)} (${arrow} ${data.delta >= 0 ? "+" : ""}${data.delta.toFixed(3)})`;
  } catch (err) {
    $("veg-msg").textContent = err.message;
  }
});

function drawVegTrend(data) {
  const ctx = $("veg-chart").getContext("2d");
  if (state.charts["veg-chart"]) state.charts["veg-chart"].destroy();
  const labels = data.points.map((p) => p.date);
  const values = data.points.map((p) => p.ndvi);
  const valids = data.points.map((p) => p.valid);
  const maxTicks = 8;
  const step = Math.max(1, Math.ceil(labels.length / maxTicks));
  state.charts["veg-chart"] = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "mean NDVI",
          data: values,
          borderColor: "#5bbf72",
          backgroundColor: "rgba(91,191,114,0.15)",
          pointBackgroundColor: "#5bbf72",
          pointRadius: 3,
          tension: 0.25,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (item) => `NDVI ${item.parsed.y.toFixed(3)} · valid ${(valids[item.dataIndex] * 100).toFixed(0)}%`,
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: "#8a8a96",
            maxRotation: 0,
            autoSkip: false,
            callback: (val, idx) => (idx % step === 0 ? labels[idx] : ""),
          },
          grid: { color: "#22222c" },
        },
        y: {
          suggestedMin: 0,
          suggestedMax: 1,
          title: { display: true, text: "NDVI", color: "#8a8a96" },
          ticks: { color: "#8a8a96" },
          grid: { color: "#22222c" },
        },
      },
    },
  });
}

$("exp-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!state.selected) return;
  const form = e.target;
  const models = [...document.querySelectorAll("#model-checks input:checked")].map((el) => el.value);
  if (!models.length) {
    alert("Select at least one model.");
    return;
  }
  const exp = await api("/api/experiments", {
    method: "POST",
    body: {
      name: form.name.value,
      aoi_id: state.selected.id,
      models_csv: models.join(","),
      seeds: Number(form.seeds.value),
      epochs: Number(form.epochs.value),
    },
  });
  await postJob("evaluate", { experiment_id: exp.id, aoi_id: state.selected.id });
  openPane("jobs");
  pollJobs();
});

async function loadExperiments() {
  const rows = await api("/api/experiments");
  const ul = $("exp-list");
  ul.innerHTML = "";
  const mine = state.selected ? rows.filter((e) => e.aoi_id === state.selected.id) : rows;
  if (!mine.length) {
    ul.innerHTML = `<li class="sub">No training runs yet.</li>`;
  }
  mine.slice(0, 12).forEach((e) => {
    const li = document.createElement("li");
    li.className = state.selectedExp?.id === e.id ? "active" : "";

    const info = document.createElement("div");
    info.className = "exp-info";
    info.innerHTML = `${e.name} <span class="sub">${e.status} · ${e.models_csv}</span>`;
    info.onclick = () => selectExperiment(e);

    const del = document.createElement("button");
    del.type = "button";
    del.className = "ghost danger";
    del.textContent = "Clear";
    del.title = "Delete this training run and its logs";
    del.onclick = async (ev) => {
      ev.stopPropagation();
      if (!confirm(`Clear training run "${e.name}" and its logs?`)) return;
      await api(`/api/experiments/${e.id}`, { method: "DELETE" });
      if (state.selectedExp?.id === e.id) resetSelectedExp();
      loadExperiments();
    };

    li.appendChild(info);
    li.appendChild(del);
    ul.appendChild(li);
  });
  if (mine[0] && !state.selectedExp) selectExperiment(mine[0]);
  else if (state.selectedExp) {
    const fresh = mine.find((e) => e.id === state.selectedExp.id);
    if (fresh) {
      fillDecodeModels(fresh);
    } else {
      resetSelectedExp();
    }
  }
}

function resetSelectedExp() {
  state.selectedExp = null;
  $("metrics-run").textContent = "no run selected";
  $("metrics-table").innerHTML = `<p class="muted">Select a run above to see its metrics.</p>`;
  $("decode-model").innerHTML = "";
  $("decode-run").textContent = "";
  setArchDynamic("");
}

$("btn-clear-exps").addEventListener("click", async () => {
  const rows = await api("/api/experiments");
  const mine = state.selected ? rows.filter((e) => e.aoi_id === state.selected.id) : rows;
  if (!mine.length) return;
  if (!confirm(`Clear all ${mine.length} training run(s) for this AOI?`)) return;
  for (const e of mine) {
    await api(`/api/experiments/${e.id}`, { method: "DELETE" });
  }
  resetSelectedExp();
  loadExperiments();
});

const MODEL_LABELS = {
  linear: "Linear + day-of-year",
  token_lstm: "Token LSTM",
  s4d: "S4D",
  convlstm: "ConvLSTM",
};

function setArchDynamic(name) {
  const el = $("arch-dynamic");
  if (!el) return;
  const key = (name || "").trim();
  el.textContent = key ? MODEL_LABELS[key] || key : "— select a model";
}

function fillDecodeModels(exp) {
  const sel = $("decode-model");
  const names = exp.checkpoints?.length ? exp.checkpoints : (exp.models || exp.models_csv.split(","));
  sel.innerHTML = names.map((m) => `<option value="${m.trim()}">${m.trim()}</option>`).join("");
  const trained = exp.checkpoints?.length ? `${exp.checkpoints.length} trained` : "no checkpoints yet";
  $("decode-run").textContent = `Run: ${exp.name} · ${exp.status} · ${trained}`;
  setArchDynamic(sel.value);
}

$("decode-model").addEventListener("change", (e) => setArchDynamic(e.target.value));

function selectExperiment(exp) {
  state.selectedExp = exp;
  $("metrics-run").textContent = `${exp.name} · ${exp.status}`;
  fillDecodeModels(exp);
  loadMetrics(exp.id);
  document.querySelectorAll("#exp-list li").forEach((li) => {
    const info = li.querySelector(".exp-info");
    li.classList.toggle("active", !!info && info.textContent.startsWith(exp.name));
  });
}

// Metrics we summarise, with the direction that counts as "better".
const METRIC_COLS = [
  { key: "residual_mse", label: "resid MSE", better: "low" },
  { key: "aligned_cosine", label: "cosine", better: "high" },
  { key: "change_map_corr", label: "Δ-corr", better: "high" },
];

async function loadMetrics(id) {
  const box = $("metrics-table");
  const rows = await api(`/api/experiments/${id}/results`);
  if (!rows.length) {
    box.innerHTML = `<p class="muted">No results yet — run <b>Train &amp; evaluate</b> to populate this.</p>`;
    state.lastMetrics = null;
    return;
  }

  // Aggregate each metric per model, averaging across seeds.
  const agg = {};
  const order = [];
  const seeds = new Set();
  for (const r of rows) {
    if (r.seed != null) seeds.add(r.seed);
    if (!(r.model_name in agg)) {
      agg[r.model_name] = {};
      order.push(r.model_name);
    }
    const cell = agg[r.model_name][r.metric_name] || { sum: 0, n: 0 };
    cell.sum += r.metric_value;
    cell.n += 1;
    agg[r.model_name][r.metric_name] = cell;
  }
  const mean = (model, key) => {
    const c = agg[model]?.[key];
    return c ? c.sum / c.n : null;
  };

  // Order rows by residual MSE (best first); keep the nuisance floor pinned last.
  const isFloor = (m) => m === "nuisance_floor";
  const models = order.slice().sort((a, b) => {
    if (isFloor(a) !== isFloor(b)) return isFloor(a) ? 1 : -1;
    const av = mean(a, "residual_mse");
    const bv = mean(b, "residual_mse");
    if (av == null) return 1;
    if (bv == null) return -1;
    return av - bv;
  });

  // Best value per metric (competitors only — the floor is a reference, not a model).
  const best = {};
  for (const col of METRIC_COLS) {
    let bv = null;
    for (const m of models) {
      if (isFloor(m)) continue;
      const v = mean(m, col.key);
      if (v == null) continue;
      if (bv == null || (col.better === "low" ? v < bv : v > bv)) bv = v;
    }
    best[col.key] = bv;
  }

  const fmt = (v) => (v == null ? "—" : v.toFixed(3));
  const head = METRIC_COLS.map((c) => `<th title="${c.better === "low" ? "lower is better" : "higher is better"}">${c.label}</th>`).join("");
  let bodyRows = "";
  for (const m of models) {
    const cells = METRIC_COLS.map((c) => {
      const v = mean(m, c.key);
      const isBest = v != null && best[c.key] != null && Math.abs(v - best[c.key]) < 1e-9 && !isFloor(m);
      return `<td class="${isBest ? "best" : ""}">${fmt(v)}</td>`;
    }).join("");
    bodyRows += `<tr class="${isFloor(m) ? "floor" : ""}"><td>${m}</td>${cells}</tr>`;
  }
  const seedNote = seeds.size ? `mean over ${seeds.size} seed${seeds.size > 1 ? "s" : ""}` : "single run";
  state.lastMetrics = {
    experiment_id: id,
    experiment_name: state.selectedExp?.name || null,
    seed_count: seeds.size,
    models: models.map((m) => ({
      name: m,
      residual_mse: mean(m, "residual_mse"),
      aligned_cosine: mean(m, "aligned_cosine"),
      change_map_corr: mean(m, "change_map_corr"),
    })),
  };
  box.innerHTML = `
    <table class="mtable">
      <thead><tr><th>model</th>${head}</tr></thead>
      <tbody>${bodyRows}</tbody>
    </table>
    <p class="hint">${seedNote}. Best per column highlighted; nuisance floor is the noise reference.</p>`;
}

$("btn-decode").addEventListener("click", async () => {
  if (!state.selectedExp) {
    $("decode-msg").textContent = "Train a model first (Train model panel).";
    openPane("train");
    return;
  }
  const model = $("decode-model").value;
  if (!model) {
    $("decode-msg").textContent = "No trained model checkpoint yet.";
    return;
  }
  $("decode-msg").textContent = "Decoding with frozen Prithvi… first load can take a minute.";
  $("decode-strip").innerHTML = "";
  try {
    const data = await api(`/api/experiments/${state.selectedExp.id}/decode?model=${encodeURIComponent(model)}`);
    const bust = `?t=${Date.now()}`;
    const fig = (cap, url) =>
      url
        ? `<figure><img src="${url}${bust}" alt="${cap}" data-cap="${cap}" title="Click to enlarge" /><figcaption>${cap}</figcaption></figure>`
        : `<figure><figcaption>${cap}: not available</figcaption></figure>`;
    const ndviCap = (label, mean, hlsMean) => {
      let text = mean == null ? label : `${label} · decode mean ${Number(mean).toFixed(3)}`;
      if (hlsMean != null) text += ` · HLS ${Number(hlsMean).toFixed(3)}`;
      return text;
    };
    const horizon = data.delta_days != null ? `${data.delta_days}d` : "t+1";
    const err =
      data.ndvi_pred_mean != null && data.ndvi_target_mean != null
        ? data.ndvi_pred_mean - data.ndvi_target_mean
        : null;
    $("decode-strip").innerHTML = `
      <div class="decode-row">
        <h3>Last scene ${data.last_key}</h3>
        ${fig("HLS satellite RGB (t)", data.sat_last_url)}
        ${fig("Decode zₜ (persistence)", data.persist_url)}
      </div>
      <div class="decode-row">
        <h3>Next scene ${data.target_key} (+${horizon})</h3>
        ${fig("HLS satellite RGB (t+1)", data.sat_target_url)}
        ${fig("Decode real zₜ₊₁", data.real_url)}
        ${fig("Decode forecast ẑₜ₊₁", data.pred_url)}
      </div>
      <div class="decode-row">
        <h3>NDVI index (same decode path)</h3>
        ${fig(ndviCap("NDVI at zₜ", data.ndvi_last_mean, data.hls_ndvi_last_mean), data.ndvi_last_url)}
        ${fig(ndviCap(`NDVI at zₜ₊Δ (${horizon})`, data.ndvi_target_mean, data.hls_ndvi_target_mean), data.ndvi_target_url)}
        ${fig(ndviCap("NDVI of forecast ẑₜ₊Δ", data.ndvi_pred_mean, null), data.ndvi_pred_url)}
      </div>
      <p class="hint">All three NDVI maps use Prithvi-decoded B04/B8A so forecast vs real is comparable. HLS means are satellite ground truth (absolute scale can differ from decode).</p>`;
    const errTxt = err == null ? "" : ` · NDVI err ${err >= 0 ? "+" : ""}${err.toFixed(3)}`;
    $("decode-msg").textContent = `${data.last_key} → ${data.target_key}  (${data.model}${data.delta_days != null ? `, Δ=${data.delta_days}d` : ""}${errTxt})`;
    state.lastDecode = {
      model: data.model,
      last_key: data.last_key,
      target_key: data.target_key,
      delta_days: data.delta_days,
      ndvi_last_mean: data.ndvi_last_mean,
      ndvi_target_mean: data.ndvi_target_mean,
      ndvi_pred_mean: data.ndvi_pred_mean,
      hls_ndvi_last_mean: data.hls_ndvi_last_mean,
      hls_ndvi_target_mean: data.hls_ndvi_target_mean,
      ndvi_err: err,
    };
  } catch (err) {
    $("decode-msg").textContent = err.message;
  }
});

// Lightbox: click any decoded/RGB image to view it maximized in a floating tab.
function openLightbox(src, cap) {
  $("lightbox-img").src = src;
  $("lightbox-img").alt = cap || "";
  $("lightbox-cap").textContent = cap || "";
  $("lightbox").hidden = false;
}
function closeLightbox() {
  $("lightbox").hidden = true;
  $("lightbox-img").src = "";
}
$("decode-strip").addEventListener("click", (e) => {
  const img = e.target.closest("img");
  if (!img) return;
  openLightbox(img.src, img.dataset.cap || img.alt);
});
$("lightbox").addEventListener("click", (e) => {
  if (e.target.id === "lightbox" || e.target.id === "lightbox-close") closeLightbox();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("lightbox").hidden) closeLightbox();
});

function explainContext() {
  const a = state.selected;
  return {
    aoi: a
      ? {
          id: a.id,
          name: a.name,
          latitude: a.latitude,
          longitude: a.longitude,
          biome: a.biome || "",
          shared: !!a.shared,
        }
      : null,
    experiment: state.selectedExp
      ? {
          id: state.selectedExp.id,
          name: state.selectedExp.name,
          status: state.selectedExp.status,
          models_csv: state.selectedExp.models_csv,
        }
      : null,
    overlay: state.lastOverlay,
    vegetation_trend: state.lastVeg,
    metrics: state.lastMetrics,
    decode: state.lastDecode,
  };
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function appendExplain(role, text, meta) {
  const log = $("explain-log");
  const div = document.createElement("div");
  div.className = `explain-msg ${role}`;
  const metaEl = meta ? `<span class="meta">${meta}</span>` : "";
  div.innerHTML = `${metaEl}${escapeHtml(text)}`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function setExplainChatEnabled(on) {
  state.explainLlm = !!on;
  $("explain-send").disabled = !on;
  $("explain-input").disabled = !on;
}

async function initExplain() {
  const status = $("explain-status");
  const chips = $("explain-chips");
  try {
    const data = await api("/api/explain/status");
    chips.innerHTML = (data.topics || [])
      .map((t) => `<button type="button" data-topic="${t.id}">${escapeHtml(t.title)}</button>`)
      .join("");
    if (data.openai_configured) {
      status.textContent = `LLM chat ready · ${data.model}`;
      setExplainChatEnabled(true);
    } else {
      status.textContent =
        "Canned FAQs ready. Free chat needs OPENAI_API_KEY in software/.env (then restart the server).";
      setExplainChatEnabled(false);
    }
  } catch (err) {
    status.textContent = err.message;
    setExplainChatEnabled(false);
  }
}

$("explain-chips").addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-topic]");
  if (!btn) return;
  const topic = btn.dataset.topic;
  appendExplain("user", btn.textContent.trim(), "faq");
  try {
    const data = await api("/api/explain/canned", { method: "POST", body: { topic } });
    appendExplain("assistant", data.reply, "canned");
    state.explainHistory.push({ role: "user", content: btn.textContent.trim() });
    state.explainHistory.push({ role: "assistant", content: data.reply });
  } catch (err) {
    appendExplain("assistant", err.message, "error");
  }
});

$("explain-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!state.explainLlm) return;
  const input = $("explain-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  appendExplain("user", message, "you");
  $("explain-send").disabled = true;
  try {
    const data = await api("/api/explain/chat", {
      method: "POST",
      body: {
        message,
        context: explainContext(),
        history: state.explainHistory.slice(-8),
      },
    });
    appendExplain("assistant", data.reply, data.model || "openai");
    state.explainHistory.push({ role: "user", content: message });
    state.explainHistory.push({ role: "assistant", content: data.reply });
  } catch (err) {
    appendExplain("assistant", err.message, "error");
  } finally {
    setExplainChatEnabled(!!state.explainLlm);
  }
});

async function showApp() {
  $("login").hidden = true;
  $("app").hidden = false;
  $("who").textContent = state.user?.email || "";
  const role = $("who-role");
  if (state.user?.role) {
    role.hidden = false;
    role.textContent = state.user.role;
  } else {
    role.hidden = true;
  }
  setTimeout(() => map.invalidateSize(), 50);
  try {
    await enableGoogleSatellite();
  } catch (err) {
    console.warn("Google satellite unavailable, using Esri imagery", err);
  }
  await loadAoIs();
  await pollJobs();
  await initExplain();
}

(async function boot() {
  if (!state.token) return;
  try {
    state.user = await api("/api/auth/me");
    await showApp();
  } catch (_) {
    logout();
  }
})();
