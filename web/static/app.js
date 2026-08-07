const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

function fmt(x, digits = 4) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(digits);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const t = await res.text();
    let detail = t || res.statusText;
    try {
      const j = JSON.parse(t);
      if (j && j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch (_) {
      /* keep raw */
    }
    throw new Error(`${res.status} ${path}: ${detail}`);
  }
  return res.json();
}

function switchPanel(name) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.panel === name));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${name}`));
  const main = document.querySelector("main");
  if (main) main.classList.toggle("wide", name === "topology");
  if (name === "topology") {
    resizeTopoCanvas();
    pollTopology();
    pollLive().catch(() => {});
    ensureLiveAnim();
  }
  if (name === "demo") pollNetwork();
}

function setupTabs() {
  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", () => switchPanel(btn.dataset.panel));
  });
  $("#goto-topology")?.addEventListener("click", () => switchPanel("topology"));
  $("#goto-demo")?.addEventListener("click", () => switchPanel("demo"));
  $("#topo-to-demo")?.addEventListener("click", () => switchPanel("demo"));
}

/* ---------------- fleet / overview ---------------- */

function renderEdgeFleet(fleet) {
  const grid = $("#edge-fleet-grid");
  if (!grid || !fleet) return;
  const active = fleet.active_id;
  grid.innerHTML = "";
  (fleet.nodes || []).forEach((n) => {
    const art = document.createElement("article");
    art.className = "edge-node-card" + (n.id === active ? " active" : "");
    const net = n.network || {};
    const st = n.stats || {};
    const dist = net.distance_geo_km != null ? `${Number(net.distance_geo_km).toFixed(0)} km` : "—";
    art.innerHTML = `
      <h3>${n.name || n.id}</h3>
      <p class="edge-meta"><code>${n.id}</code> · ${n.category} · ${n.city || net.city || "—"}</p>
      <p class="edge-net">${dist} → cloud
        · prop ${Number(net.prop_rtt_ms || 0).toFixed(1)}ms
        · RTT ${Number(net.rtt_ms || 0).toFixed(0)}ms
        · ${Number(net.bandwidth_mbps || 0).toFixed(1)}Mbps</p>
      <p class="edge-stats">access ${n.access || net.access || "—"}
        · infer ${st.n_infer ?? 0}
        · cloud_ok ${st.n_upload_ok ?? 0}
        · fail ${st.n_upload_fail ?? 0}</p>
    `;
    art.addEventListener("click", () => {
      selectEdgeNode(n.id)
        .then(() => switchPanel("demo"))
        .catch((e) => {
          $("#demo-status").textContent = `切换边缘节点失败：${e.message}`;
        });
    });
    grid.appendChild(art);
  });
}

async function fillEdgeNodeSelect(fleet) {
  const sel = $("#demo-edge-node");
  if (!sel || !fleet) return;
  const prev = sel.value;
  sel.innerHTML = "";
  (fleet.nodes || []).forEach((n) => {
    const opt = document.createElement("option");
    opt.value = n.id;
    const city = n.city || n.network?.city || "?";
    const dist =
      n.network?.distance_geo_km != null ? `${Number(n.network.distance_geo_km).toFixed(0)}km` : "";
    opt.textContent = `${n.name} · ${city} ${dist} · ${n.category}`;
    sel.appendChild(opt);
  });
  const prefer = fleet.active_id || prev || fleet.nodes?.[0]?.id;
  if (prefer) sel.value = prefer;
}

async function loadEdgeFleet() {
  try {
    const data = await api("/api/edge_nodes");
    renderEdgeFleet(data);
    await fillEdgeNodeSelect(data);
    return data;
  } catch (err) {
    const grid = $("#edge-fleet-grid");
    if (grid) {
      grid.innerHTML =
        `<article class="edge-node-card"><h3>边缘舰队 API 不可用</h3>` +
        `<p class="edge-meta">${String(err.message || err)}</p>` +
        `<p class="edge-stats">请重启 Web 服务以加载最新后端（需 /api/edge_nodes）。` +
        `若开着旧进程（如 :7860），请改用已更新端口或杀掉旧进程后重启。</p></article>`;
    }
    throw err;
  }
}

async function selectEdgeNode(nodeId) {
  const fd = new FormData();
  fd.append("edge_node_id", nodeId);
  const res = await fetch("/api/edge_nodes/active", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
  const node = data.edge_node;
  state.selectedEdgeId = node.id;
  const catSel = $("#demo-cat");
  if (catSel && node?.category) {
    const has = [...catSel.options].some((o) => o.value === node.category);
    if (has && catSel.value !== node.category) {
      catSel.value = node.category;
      await loadDemoImages();
    }
  }
  await loadEdgeFleet();
  await pollNetwork();
  if ($("#demo-status")) {
    $("#demo-status").textContent =
      `活动边缘节点 ${node?.name || nodeId} · ${node?.city || ""} · RTT ${Number(data.network?.rtt_ms || 0).toFixed(0)}ms`;
  }
  return data;
}

async function loadOverview() {
  const s = await api("/api/summary");
  if ($("#stack-edge")) $("#stack-edge").textContent = s.stack.edge;
  if ($("#stack-cloud")) $("#stack-cloud").textContent = s.stack.cloud;
  if ($("#stack-collab")) $("#stack-collab").textContent = s.stack.collab;
}

async function fillCategorySelects() {
  const { categories } = await api("/api/categories");
  const demo = $("#demo-cat");
  if (!demo) return;
  demo.innerHTML = "";
  categories.forEach((c) => {
    demo.insertAdjacentHTML("beforeend", `<option value="${c}">${c}</option>`);
  });
  demo.value = categories.includes("bottle") ? "bottle" : categories[0];
  await loadDemoImages();
}

/* ---------------- topology viz ---------------- */

const state = {
  selectedEdgeId: null,
  topo: null, // last env summary
  fleet: null,
  hitRegions: [], // {id, x, y, r}
  localFile: null,
  localObjectUrl: null,
  live: {
    running: false,
    pollTimer: null,
    animTimer: null,
    animT0: performance.now(),
    status: null,
    // nodeId -> {upload: bool, path: string, t: number}
    linkPulse: {},
    // track last rendered image per node to avoid flicker
    cardSig: {},
    // persistent cloud uplink cases: key -> job
    cloudBoard: {},
    cloudBoardOrder: [],
  },
};

function resizeTopoCanvas() {
  const canvas = $("#topo-canvas");
  if (!canvas) return;
  const wrap = canvas.parentElement;
  const w = Math.max(640, Math.floor(wrap.clientWidth - 4));
  const h = Math.max(420, Math.min(620, Math.floor(w * 0.52)));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

function geoBearing(cloud, lat, lon) {
  /** Forward azimuth cloud→site (radians), screen-friendly (0 = east, + = clockwise-ish via atan2). */
  const dLon = ((lon - cloud.lon) * Math.PI) / 180;
  const la1 = (cloud.lat * Math.PI) / 180;
  const la2 = (lat * Math.PI) / 180;
  const y = Math.sin(dLon) * Math.cos(la2);
  const x = Math.cos(la1) * Math.sin(la2) - Math.sin(la1) * Math.cos(la2) * Math.cos(dLon);
  return Math.atan2(y, x);
}

function spreadAngles(rawBearings) {
  /**
   * Keep geographic order around the circle, but enforce equal angular slots
   * so nodes never stay collinear (which stacks labels on one ray).
   */
  const n = rawBearings.length;
  if (n === 0) return [];
  if (n === 1) return [-Math.PI / 2];

  const indexed = rawBearings.map((b, i) => ({ b, i }));
  indexed.sort((a, b) => a.b - b.b);

  // Even fan; start slightly above west so 3-node layouts look balanced
  const gap = (2 * Math.PI) / n;
  const start = -Math.PI / 2 - gap * 0.5;
  const out = new Array(n);
  indexed.forEach((item, k) => {
    out[item.i] = start + k * gap;
  });
  return out;
}

function projectSites(cloud, links, W, H) {
  /** Radial layout: radius ∝ geo distance; angles de-collided by order-preserving fan. */
  const cx = W * 0.5;
  const cy = H * 0.5;
  const maxR = Math.min(W, H) * 0.36;
  const minR = Math.min(W, H) * 0.16;
  const entries = Object.entries(links || {});
  let maxDist = 1;
  for (const [, L] of entries) {
    maxDist = Math.max(maxDist, Number(L.distance_geo_km) || 1);
  }

  const bearings = entries.map(([eid, L], i) => {
    const lat = L.lat;
    const lon = L.lon;
    if (lat != null && lon != null && cloud && cloud.lat != null && cloud.lon != null) {
      return geoBearing(cloud, lat, lon);
    }
    return (i / Math.max(1, entries.length)) * Math.PI * 2;
  });
  const angles = spreadAngles(bearings);

  const nodes = entries.map(([eid, L], i) => {
    const dist = Number(L.distance_geo_km) || 1;
    // sqrt scale: near edges still readable, far edges push out
    const r = minR + Math.sqrt(dist / maxDist) * maxR;
    const angle = angles[i];
    return {
      id: eid,
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
      r: 16 + Math.min(10, dist / 250),
      angle,
      link: L,
      labelSide: i % 2 === 0 ? 1 : -1,
    };
  });
  return { cx, cy, nodes, maxDist, minR, maxR };
}

function linkQualityColor(link) {
  if (link.outage || link.profile === "outage") return "#b42318";
  const rtt = Number(link.rtt_ms) || 0;
  if (rtt < 25) return "#1f7a4c";
  if (rtt < 60) return "#0f5c6e";
  if (rtt < 120) return "#9a6700";
  return "#b42318";
}

function drawTopology() {
  const canvas = $("#topo-canvas");
  if (!canvas || !state.topo) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // soft radial background
  const g = ctx.createRadialGradient(W / 2, H / 2, 20, W / 2, H / 2, Math.max(W, H) * 0.55);
  g.addColorStop(0, "#f4fafb");
  g.addColorStop(1, "#eef1f4");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);

  const cloud = state.topo.cloud;
  const links = state.topo.links || {};
  // attach lat/lon from fleet if available
  const fleetNodes = state.fleet?.nodes || [];
  for (const n of fleetNodes) {
    if (links[n.id] && n.site) {
      links[n.id].lat = n.site.lat;
      links[n.id].lon = n.site.lon;
    }
  }

  const { cx, cy, nodes, maxDist, minR, maxR } = projectSites(cloud, links, W, H);
  state.hitRegions = [];

  // distance rings (match sqrt radius scale roughly at 0.25 / 0.5 / 1.0 of maxDist)
  ctx.strokeStyle = "rgba(15, 92, 110, 0.12)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 6]);
  for (const frac of [0.25, 0.5, 1.0]) {
    const rr = minR + Math.sqrt(frac) * maxR;
    ctx.beginPath();
    ctx.arc(cx, cy, rr, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.fillStyle = "#5b6775";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText(`≈${(maxDist * 0.25).toFixed(0)} km`, cx + 10, cy - (minR + Math.sqrt(0.25) * maxR));
  ctx.fillText(`≈${maxDist.toFixed(0)} km`, cx + 10, cy - (minR + maxR));

  const now = Date.now();
  const anim = ((performance.now() - state.live.animT0) / 1000) % 1;

  // links first (under nodes) — glowing uplink beams when uploading
  for (const n of nodes) {
    const L = n.link;
    const col = linkQualityColor(L);
    const out = L.outage || L.profile === "outage";
    const pulse = state.live.linkPulse[n.id];
    const age = pulse ? (now - pulse.t) / 1000 : 99;
    const activeUp = pulse && age < 5 && pulse.upload;
    const path = pulse?.path || "LOCAL";

    // soft glow under active uplink
    if (activeUp && !out) {
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(n.x, n.y);
      ctx.strokeStyle =
        path === "CLOUD_REVIEW"
          ? "rgba(31,122,76,0.35)"
          : path === "LOCAL_NET_FALLBACK"
            ? "rgba(180,35,24,0.35)"
            : "rgba(15,92,110,0.4)";
      ctx.lineWidth = 14;
      ctx.lineCap = "round";
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(n.x, n.y);
    ctx.strokeStyle = out
      ? "rgba(180,35,24,0.55)"
      : activeUp
        ? path === "CLOUD_REVIEW"
          ? "#1f7a4c"
          : path === "LOCAL_NET_FALLBACK"
            ? "#b42318"
            : "#0f5c6e"
        : col;
    ctx.lineWidth = out ? 2 : activeUp ? 5.5 : 2.5;
    if (out) ctx.setLineDash([6, 5]);
    else if (activeUp) ctx.setLineDash([10, 8]);
    ctx.lineDashOffset = activeUp ? -anim * 36 : 0;
    ctx.globalAlpha = out ? 0.85 : 1;
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
    ctx.lineDashOffset = 0;
    ctx.lineCap = "butt";

    // traffic packets along cloud↔edge link (edge → cloud when upload)
    if (!out && (activeUp || state.live.running)) {
      const packets = activeUp ? 5 : 1;
      for (let k = 0; k < packets; k++) {
        const phase = (anim + k / packets) % 1;
        const t = activeUp ? 1 - phase : phase * 0.28;
        const px = n.x + (cx - n.x) * t;
        const py = n.y + (cy - n.y) * t;
        const r = activeUp ? 6 : 3;
        ctx.beginPath();
        ctx.arc(px, py, r + (activeUp ? 3 : 0), 0, Math.PI * 2);
        ctx.fillStyle = activeUp ? "rgba(255,255,255,0.35)" : "rgba(15,92,110,0.12)";
        ctx.fill();
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = activeUp
          ? path === "CLOUD_REVIEW"
            ? "#1f7a4c"
            : path === "LOCAL_NET_FALLBACK"
              ? "#b42318"
              : "#0f5c6e"
          : "rgba(15,92,110,0.4)";
        ctx.fill();
        if (activeUp) {
          ctx.fillStyle = "#fff";
          ctx.font = "700 8px sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("↑", px, py + 3);
          ctx.textAlign = "left";
        }
      }
    }
  }

  // link labels — along ray but offset perpendicular so they never sit on a nearer node
  for (const n of nodes) {
    const L = n.link;
    const out = L.outage || L.profile === "outage";
    const label = out
      ? "OUTAGE"
      : `${Number(L.rtt_ms).toFixed(0)}ms · ${Number(L.bandwidth_mbps).toFixed(0)}Mb · ${(Number(L.loss_prob) * 100).toFixed(1)}%`;
    ctx.font = "11px ui-monospace, monospace";
    const tw = ctx.measureText(label).width;
    const dx = n.x - cx;
    const dy = n.y - cy;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    // place label at 55% of the segment, then nudge sideways
    const along = 0.55;
    const side = n.labelSide || 1;
    const pad = 18;
    let mx = cx + dx * along + (-uy) * pad * side;
    let my = cy + dy * along + ux * pad * side;
    // keep box inside canvas
    mx = Math.max(tw / 2 + 8, Math.min(W - tw / 2 - 8, mx));
    my = Math.max(14, Math.min(H - 14, my));

    ctx.fillStyle = "rgba(255,255,255,0.96)";
    ctx.strokeStyle = "rgba(180,190,200,0.95)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(mx - tw / 2 - 6, my - 10, tw + 12, 20);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = out ? "#b42318" : "#1b2430";
    ctx.textAlign = "center";
    ctx.fillText(label, mx, my + 4);
    ctx.textAlign = "left";
  }

  // cloud hub + load ring (visualizes cloud concurrent capacity)
  const cloudInfo = state.live.status?.cloud || state.fleet?.cloud || {};
  const inflight = Number(cloudInfo.inflight || 0);
  const maxIn = Math.max(1, Number(cloudInfo.max_inflight || 2));
  const load = Math.min(1, inflight / maxIn);
  ctx.beginPath();
  ctx.arc(cx, cy, 38, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(15,92,110,0.15)";
  ctx.lineWidth = 6;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, 38, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * load);
  ctx.strokeStyle = load > 0.8 ? "#b42318" : "#0f5c6e";
  ctx.lineWidth = 6;
  ctx.lineCap = "round";
  ctx.stroke();
  const pulseR = 30 + (state.live.running ? Math.sin(performance.now() / 220) * 2 : 0);
  ctx.beginPath();
  ctx.arc(cx, cy, pulseR, 0, Math.PI * 2);
  ctx.fillStyle = "#0f5c6e";
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.font = "600 11px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("CLOUD", cx, cy - 2);
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(`${inflight}/${maxIn}`, cx, cy + 12);
  ctx.fillStyle = "#1b2430";
  ctx.font = "600 13px serif";
  ctx.fillText(cloud?.city || cloud?.name || "Cloud", cx, cy + 56);
  ctx.textAlign = "left";

  // edge nodes (on top)
  for (const n of nodes) {
    const selected = n.id === state.selectedEdgeId;
    const L = n.link;
    const out = L.outage || L.profile === "outage";
    const last = state.live.status?.last_by_node?.[n.id];
    const path = last?.path_type || "";
    const pulse = state.live.linkPulse[n.id];
    const age = pulse ? (now - pulse.t) / 1000 : 99;
    const activeUp = pulse && age < 5 && pulse.upload;

    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r + (activeUp ? 4 : 0), 0, Math.PI * 2);
    ctx.fillStyle = selected ? "#e6f2f4" : "#fff";
    ctx.fill();
    ctx.lineWidth = selected || activeUp ? 3.5 : 2;
    ctx.strokeStyle = out
      ? "#b42318"
      : path === "CLOUD_REVIEW"
        ? "#1f7a4c"
        : path === "LOCAL_NET_FALLBACK"
          ? "#b42318"
          : "#0f5c6e";
    ctx.stroke();
    ctx.fillStyle = "#1b2430";
    ctx.font = "600 12px sans-serif";
    ctx.textAlign = "center";
    const lx = n.x + Math.cos(n.angle) * (n.r + 16);
    const ly = n.y + Math.sin(n.angle) * (n.r + 16);
    ctx.fillText(L.city || n.id, lx, ly);
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillStyle = "#5b6775";
    ctx.fillText(`${Number(L.distance_geo_km).toFixed(0)} km`, lx, ly + 12);
    if (path) {
      const short =
        path === "CLOUD_REVIEW" ? "→CLOUD" : path === "LOCAL_NET_FALLBACK" ? "FALLBACK" : "LOCAL";
      ctx.fillStyle =
        path === "CLOUD_REVIEW" ? "#1f7a4c" : path === "LOCAL_NET_FALLBACK" ? "#b42318" : "#5b6775";
      ctx.font = "700 10px sans-serif";
      ctx.fillText(short, lx, ly + 24);
    }
    if (last?.final) {
      ctx.fillStyle = String(last.final).toUpperCase() === "NG" ? "#b42318" : "#1f7a4c";
      ctx.font = "700 12px sans-serif";
      ctx.fillText(String(last.final), n.x, n.y + 4);
    }
    ctx.textAlign = "left";
    state.hitRegions.push({ id: n.id, x: n.x, y: n.y, r: n.r + 8 });
  }
}

function renderTopoSide(fleet, env) {
  const list = $("#topo-link-list");
  if (!list) return;
  list.innerHTML = "";
  const nodes = fleet?.nodes || [];
  nodes
    .slice()
    .sort(
      (a, b) =>
        (a.network?.distance_geo_km || 0) - (b.network?.distance_geo_km || 0)
    )
    .forEach((n) => {
      const net = n.network || {};
      const el = document.createElement("button");
      el.type = "button";
      el.className =
        "topo-link-item" + (n.id === state.selectedEdgeId ? " active" : "");
      const out = net.profile === "outage" || net.outage;
      el.innerHTML = `
        <div class="tli-head">
          <strong>${n.city || net.city || n.id}</strong>
          <span class="tli-chip ${out ? "bad" : "ok"}">${out ? "outage" : "up"}</span>
        </div>
        <div class="tli-meta">${n.id} · ${n.category} · ${n.access || net.access || "—"}</div>
        <div class="tli-stats">
          <span>${Number(net.distance_geo_km || 0).toFixed(0)} km</span>
          <span>prop ${Number(net.prop_rtt_ms || 0).toFixed(1)}ms</span>
          <span>RTT ${Number(net.rtt_ms || 0).toFixed(0)}ms</span>
          <span>${Number(net.bandwidth_mbps || 0).toFixed(1)} Mb</span>
          <span>loss ${(Number(net.loss_prob || 0) * 100).toFixed(1)}%</span>
          <span>cong ${(Number(net.congestion || 0) + Number(net.diurnal || 0) + Number(net.burst || 0)).toFixed(2)}</span>
        </div>
      `;
      el.addEventListener("click", () => {
        state.selectedEdgeId = n.id;
        selectEdgeNode(n.id).catch(() => {});
        updateTopoSelected(n);
        drawTopology();
        renderTopoSide(state.fleet, state.topo);
      });
      list.appendChild(el);
    });

  const cloudLabel = $("#topo-cloud-label");
  if (cloudLabel && env?.cloud) {
    cloudLabel.textContent = `Cloud · ${env.cloud.city || env.cloud.name} (${env.cloud.lat?.toFixed?.(2)}, ${env.cloud.lon?.toFixed?.(2)})`;
  }
}

function updateTopoSelected(node) {
  const pre = $("#topo-selected-json");
  if (!pre) return;
  if (!node) {
    pre.textContent = "点击图中边缘节点";
    return;
  }
  const net = node.network || {};
  pre.textContent = JSON.stringify(
    {
      id: node.id,
      name: node.name,
      city: node.city || net.city,
      category: node.category,
      access: node.access || net.access,
      distance_geo_km: net.distance_geo_km,
      distance_fiber_km: net.distance_fiber_km,
      prop_rtt_ms: net.prop_rtt_ms,
      rtt_ms: net.rtt_ms,
      bandwidth_mbps: net.bandwidth_mbps,
      loss_prob: net.loss_prob,
      congestion: net.congestion,
      diurnal: net.diurnal,
      burst: net.burst,
      outage: net.outage || net.profile === "outage",
      stats: node.stats,
    },
    null,
    2
  );
}

async function pollTopology() {
  try {
    const [fleet, env] = await Promise.all([api("/api/edge_nodes"), api("/api/network/env")]);
    state.fleet = fleet;
    if (env.enabled) {
      // merge live network from fleet nodes into env.links for drawing
      const links = { ...(env.links || {}) };
      for (const n of fleet.nodes || []) {
        links[n.id] = { ...(links[n.id] || {}), ...(n.network || {}), city: n.city || n.network?.city };
      }
      state.topo = { ...env, links };
    } else {
      // synthesize from fleet snapshots
      const links = {};
      for (const n of fleet.nodes || []) links[n.id] = { ...(n.network || {}), city: n.city };
      state.topo = {
        cloud: { city: "Cloud", lat: 31.23, lon: 121.47 },
        links,
      };
    }
    if (!state.selectedEdgeId) state.selectedEdgeId = fleet.active_id;
    const sel = (fleet.nodes || []).find((n) => n.id === state.selectedEdgeId);
    updateTopoSelected(sel);
    renderTopoSide(fleet, state.topo);
    resizeTopoCanvas();
    drawTopology();
    renderEdgeFleet(fleet);
    await fillEdgeNodeSelect(fleet);
    // keep live image cards wired to fleet even before first tick
    if (state.live.status) renderLiveNodeCards(state.live.status);
    else renderLiveNodeCards({ last_by_node: {}, running: false });
  } catch {
    /* ignore */
  }
}

function bindTopoCanvas() {
  const canvas = $("#topo-canvas");
  if (!canvas) return;
  canvas.addEventListener("click", (ev) => {
    const rect = canvas.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((ev.clientY - rect.top) / rect.height) * canvas.height;
    let hit = null;
    for (const h of state.hitRegions) {
      const d = Math.hypot(x - h.x, y - h.y);
      if (d <= h.r) hit = h;
    }
    if (!hit) return;
    state.selectedEdgeId = hit.id;
    selectEdgeNode(hit.id)
      .then(() => {
        const n = (state.fleet?.nodes || []).find((x) => x.id === hit.id);
        updateTopoSelected(n);
        drawTopology();
        renderTopoSide(state.fleet, state.topo);
      })
      .catch(() => {});
  });
  window.addEventListener("resize", () => {
    if ($("#panel-topology")?.classList.contains("active")) {
      resizeTopoCanvas();
      drawTopology();
    }
  });
}

/* ---------------- demo (single node) ---------------- */

async function loadDemoImages() {
  const cat = $("#demo-cat")?.value;
  if (!cat) return;
  const data = await api(`/api/images?category=${encodeURIComponent(cat)}&limit=40`);
  const sel = $("#demo-img");
  sel.innerHTML = "";
  data.images.forEach((img) => {
    const opt = document.createElement("option");
    opt.value = img.path;
    const mark = img.has_llm ? "LLM" : "edge";
    opt.textContent = `[${mark}] ${img.gt} · ${img.defect_type} · ${img.name}`;
    sel.appendChild(opt);
  });
  updatePreview();
}

function clearLocalUpload() {
  const input = $("#demo-file");
  if (input) input.value = "";
  state.localFile = null;
  state.localObjectUrl = revokeLocalPreview(state.localObjectUrl);
  const nameEl = $("#demo-file-name");
  if (nameEl) nameEl.textContent = "未选择 · 上传后优先使用";
  const clr = $("#demo-file-clear");
  if (clr) clr.disabled = true;
  const imgSel = $("#demo-img");
  if (imgSel) imgSel.disabled = false;
  $(".side-panel")?.classList.remove("has-upload");
  updatePreview();
}

function revokeLocalPreview(url) {
  if (url) {
    try {
      URL.revokeObjectURL(url);
    } catch (_) {
      /* ignore */
    }
  }
  return null;
}

function onLocalFileChange() {
  const input = $("#demo-file");
  const file = input && input.files && input.files[0] ? input.files[0] : null;
  state.localObjectUrl = revokeLocalPreview(state.localObjectUrl);
  state.localFile = file;
  const nameEl = $("#demo-file-name");
  const clr = $("#demo-file-clear");
  const imgSel = $("#demo-img");
  if (!file) {
    if (nameEl) nameEl.textContent = "未选择 · 上传后优先使用";
    if (clr) clr.disabled = true;
    if (imgSel) imgSel.disabled = false;
    $(".side-panel")?.classList.remove("has-upload");
    updatePreview();
    return;
  }
  const okType = /^image\/(png|jpeg|jpg|bmp|webp)$/i.test(file.type) || /\.(png|jpe?g|bmp|webp)$/i.test(file.name);
  if (!okType) {
    if (nameEl) nameEl.textContent = "仅支持 png / jpg / bmp / webp";
    clearLocalUpload();
    return;
  }
  state.localObjectUrl = URL.createObjectURL(file);
  if (nameEl) nameEl.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
  if (clr) clr.disabled = false;
  if (imgSel) imgSel.disabled = true;
  $(".side-panel")?.classList.add("has-upload");
  updatePreview();
}

function updatePreview() {
  const img = $("#demo-preview");
  const cap = $("#demo-preview-cap");
  if (!img) return;
  if (state.localFile && state.localObjectUrl) {
    img.src = state.localObjectUrl;
    if (cap) cap.textContent = `本地上传 · ${state.localFile.name}`;
    renderViz(null);
    return;
  }
  const path = $("#demo-img")?.value;
  if (!path) {
    img.removeAttribute("src");
    if (cap) cap.textContent = "Selected sample";
    return;
  }
  img.src = `/api/image?path=${encodeURIComponent(path)}`;
  if (cap) cap.textContent = "Dataset sample";
  renderViz(null);
}

function renderViz(viz, pathType) {
  const edgeFig = $("#viz-edge")?.closest("figure");
  const cloudFig = $("#viz-cloud")?.closest("figure");
  const grid = $("#demo-viz");
  const status = $("#viz-status");
  const wentCloud = pathType === "CLOUD_REVIEW";
  const setFig = (imgId, fig, url) => {
    const img = $(imgId);
    if (!img || !fig) return;
    if (url) {
      fig.classList.remove("is-empty");
      img.src = url;
    } else {
      fig.classList.add("is-empty");
      img.removeAttribute("src");
    }
  };
  if (cloudFig) cloudFig.classList.toggle("is-hidden", !wentCloud);
  if (grid) grid.classList.toggle("edge-only", !wentCloud);

  if (!viz) {
    setFig("#viz-edge", edgeFig, null);
    setFig("#viz-cloud", cloudFig, null);
    if (status) status.textContent = "—";
    return;
  }
  setFig("#viz-edge", edgeFig, viz.edge_strip || null);
  setFig("#viz-cloud", cloudFig, wentCloud ? viz.cloud_strip || null : null);
  if (status) {
    if (!wentCloud) {
      status.textContent = "未上云：仅展示边侧结果，不显示云端热力图 / Cloud LLM。";
      return;
    }
    const parts = [];
    if (viz.edge_strip) parts.push("边侧热力图");
    if (viz.cloud_strip) parts.push("PatchCore 对比图");
    status.textContent = parts.length
      ? `${parts.join(" · ")}。云端 VLM 见下方 Cloud LLM。`
      : "已上云，但暂无 Anomalib 可视化缓存；见下方 Cloud LLM JSON。";
  }
}

function renderLlm(cloud, pathType) {
  const panel = $("#llm-panel");
  const badge = $("#llm-badge");
  const rawEl = $("#out-cloud");
  const wentCloud = pathType === "CLOUD_REVIEW";
  if (panel) panel.classList.toggle("is-hidden", !wentCloud);
  if (!badge || !rawEl) return;
  if (!wentCloud || !cloud || cloud.skipped) {
    badge.textContent = !wentCloud ? "local" : cloud?.skipped ? "skipped" : "no output";
    badge.className = "llm-badge empty";
    $("#llm-decision").textContent = "—";
    $("#llm-conf").textContent = "—";
    $("#llm-type").textContent = "—";
    $("#llm-reason").textContent = !wentCloud
      ? "本次路由为本地，不调用云端。"
      : cloud?.skipped
        ? cloud.reason || "已路由上云，但本次无云端输出（可开 Live cloud）。"
        : "无云端输出。";
    rawEl.textContent = "";
    return;
  }
  const decision = cloud.decision || "—";
  badge.textContent = decision;
  badge.className = `llm-badge ${String(decision).toUpperCase() === "NG" ? "ng" : "ok"}`;
  $("#llm-decision").textContent = decision;
  $("#llm-conf").textContent =
    cloud.confidence == null ? "—" : Number(cloud.confidence).toFixed(2);
  $("#llm-type").textContent = cloud.defect_type || "—";
  $("#llm-reason").textContent = cloud.reason || "(no reason)";
  rawEl.textContent =
    cloud.raw ||
    JSON.stringify(
      {
        decision: cloud.decision,
        confidence: cloud.confidence,
        defect_type: cloud.defect_type,
        reason: cloud.reason,
      },
      null,
      2
    );
}

function renderRouteAgent(ra, pathType, netOut) {
  const badge = $("#route-llm-badge");
  const rawEl = $("#out-route-raw");
  if (!badge) return;
  if (!ra) {
    badge.textContent = "idle";
    badge.className = "llm-badge empty";
    $("#route-llm-reason").textContent = "尚未运行。";
    if (rawEl) rawEl.textContent = "";
    $("#out-route").textContent = "—";
    return;
  }
  const upload = !!ra.upload;
  badge.textContent = upload ? "upload" : "local";
  badge.className = `llm-badge ${upload ? "ng" : "ok"}`;
  // Show final decision only (after rules_snap); hide model draft mismatches.
  let reason = ra.reason || "(no reason)";
  reason = reason.replace(/\s*\|\s*rules_snap:[^|]*/g, "").trim();
  $("#route-llm-reason").textContent = reason;
  const finalJson = {
    upload: ra.upload,
    confidence: ra.confidence,
    reason,
  };
  if (rawEl) {
    rawEl.textContent = JSON.stringify(finalJson, null, 2);
  }
  $("#out-route").textContent = JSON.stringify(
    {
      path: pathType,
      upload_want: upload,
      confidence: ra.confidence,
      source: ra.source,
      backend: ra.backend || null,
      network_profile: ra.network_profile,
      route_latency_ms: ra.latency_ms,
      net_ok: netOut?.ok ?? null,
      net_fail: netOut?.failed_reason ?? null,
      net_rtt_ms: netOut?.rtt_ms ?? null,
      net_tx_ms: netOut?.tx_ms ?? null,
    },
    null,
    2
  );
}

async function runDemo() {
  const status = $("#demo-status");
  const cat = $("#demo-cat").value;
  const path = $("#demo-img").value;
  const live = $("#demo-live").checked;
  const useAgent = $("#demo-route-agent")?.checked ?? true;
  const localFile = state.localFile;
  if (!localFile && !path) {
    status.textContent = "请选择数据集图像，或上传本地图片";
    return;
  }
  status.textContent = localFile ? `运行中（上传 ${localFile.name}）…` : "运行中…";
  $("#demo-run").disabled = true;
  try {
    const edgeNodeId = $("#demo-edge-node")?.value || state.selectedEdgeId || "";
    const fd = new FormData();
    fd.append("category", cat);
    fd.append("live_cloud", live ? "true" : "false");
    fd.append("use_route_agent", useAgent ? "true" : "false");
    if (edgeNodeId) fd.append("edge_node_id", edgeNodeId);
    if (localFile) {
      fd.append("file", localFile, localFile.name);
    } else {
      fd.append("image_path", path);
    }
    const res = await fetch("/api/demo", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));

    const edge = data.edge;
    $("#out-edge").textContent = edge
      ? JSON.stringify(
          {
            pred: edge.edge_pred,
            score: Number(edge.edge_score?.toFixed?.(4) ?? edge.edge_score),
            hard: edge.hard,
            threshold: Number(edge.threshold?.toFixed?.(4) ?? edge.threshold),
          },
          null,
          2
        )
      : localFile
        ? "本地上传图：无预计算边侧分数，RouteAgent 使用默认 CONTEXT；可开 Live cloud 做云端判读"
        : "未找到预计算边侧分数（可换图像或开 Live cloud）";

    renderViz(data.viz || null, data.route);
    renderRouteAgent(data.route_agent, data.route, data.network_outcome);
    // Only show cloud LLM when this run actually routed to cloud.
    renderLlm(data.cloud_live, data.route);

    const final = data.final_decision || data.cached_case?.final || edge?.edge_pred || "—";
    const net = data.network || {};
    $("#out-final").textContent = JSON.stringify(
      {
        edge_node: data.edge_node?.id || edgeNodeId,
        city: data.edge_node?.city || net.city,
        distance_geo_km: net.distance_geo_km,
        prop_rtt_ms: net.prop_rtt_ms,
        rtt_ms: net.rtt_ms,
        bandwidth_mbps: net.bandwidth_mbps,
        path: data.route,
        upload_want: data.upload_want,
        live,
      },
      null,
      2
    );
    $("#out-decision").textContent = String(final);
    const cloud = data.cloud_live;
    status.textContent =
      `完成 · ${data.edge_node?.city || data.edge_node?.name || edgeNodeId} · ${data.route || "—"} · final=${final}` +
      (cloud && !cloud.skipped ? " · cloud LLM" : "");
    await loadEdgeFleet();
    pollNetwork();
    pollTopology();
  } catch (e) {
    status.textContent = `失败：${e.message}`;
    $("#out-final").textContent = e.message;
  } finally {
    $("#demo-run").disabled = false;
  }
}

function renderModelStatus(h) {
  const status = $("#demo-status");
  if (!status || !h) return;
  // Don't overwrite an in-flight demo message
  const cur = status.textContent || "";
  if (/运行中|失败：|完成 ·|切换失败|链路|断网|恢复|节点/.test(cur) && !/模型|就绪|预加载/.test(cur)) {
    return;
  }
  const ra = !!h.route_agent_loaded;
  const cloud = !!h.cloud_loaded;
  const raErr = h.route_agent_error;
  const cloudErr = h.cloud_error;
  if (ra && cloud) {
    const be = h.route_agent?.backend || "gguf";
    status.textContent = `模型就绪 · RouteAgent(${be}) + Cloud LoRA`;
    return;
  }
  if (raErr || cloudErr) {
    const parts = [];
    if (ra) parts.push("RouteAgent ✓");
    else parts.push(raErr ? `RouteAgent ✗` : "RouteAgent…");
    if (cloud) parts.push("Cloud ✓");
    else parts.push(cloudErr ? `Cloud ✗` : "Cloud…");
    status.textContent = parts.join(" · ");
    return;
  }
  if (ra && !cloud) {
    status.textContent = "RouteAgent 就绪 · Cloud 预加载中…";
    return;
  }
  if (!ra && cloud) {
    status.textContent = "Cloud 就绪 · RouteAgent 预加载中…";
    return;
  }
  status.textContent = "模型预加载中…";
}

/* ---------------- network poll (demo strip) ---------------- */

function updateDisconnectButtons(disconnected) {
  const d = $("#net-disconnect");
  const r = $("#net-restore");
  const td = $("#topo-disconnect");
  const tr = $("#topo-restore");
  if (d) d.disabled = !!disconnected;
  if (r) r.disabled = !disconnected;
  if (td) td.disabled = !!disconnected;
  if (tr) tr.disabled = !disconnected;
  $("#net-panel")?.classList.toggle("is-disconnected", !!disconnected);
}

function renderNetStatus(net, last, disconnected) {
  if (!net) return;
  const isOut = disconnected ?? net.profile === "outage";
  const chip = $("#net-chip");
  if (chip) {
    chip.textContent = isOut ? "OUTAGE" : net.profile || "geo";
    chip.className = `net-chip profile-${isOut ? "outage" : "geo"}`;
  }
  const city = net.city || "—";
  const dist = net.distance_geo_km != null ? `${Number(net.distance_geo_km).toFixed(0)} km` : "—";
  if ($("#net-geo")) $("#net-geo").textContent = `${city} · ${dist}`;
  if ($("#net-prop"))
    $("#net-prop").textContent =
      net.prop_rtt_ms != null ? `${Number(net.prop_rtt_ms).toFixed(1)} ms` : "—";
  if ($("#net-rtt"))
    $("#net-rtt").textContent = `${Number(net.rtt_ms).toFixed(0)} ± ${Number(net.rtt_jitter_ms || 0).toFixed(0)} ms`;
  if ($("#net-bw")) $("#net-bw").textContent = `${Number(net.bandwidth_mbps).toFixed(1)} Mbps`;
  if ($("#net-loss")) $("#net-loss").textContent = `${(Number(net.loss_prob) * 100).toFixed(1)}%`;
  if ($("#net-cong")) {
    const c = Number(net.congestion || 0) + Number(net.diurnal || 0) + Number(net.burst || 0);
    $("#net-cong").textContent = Number.isFinite(c) ? c.toFixed(2) : "—";
  }
  if ($("#net-probe") && last) {
    const ok = last.upload_ok ? "ok" : last.failed_reason || "fail";
    $("#net-probe").textContent = isOut ? "断网" : `RTT ${Number(last.rtt_ms || 0).toFixed(0)} · ${ok}`;
  }
  // demo bar chips
  if ($("#demo-link-city")) $("#demo-link-city").textContent = city;
  if ($("#demo-link-rtt")) $("#demo-link-rtt").textContent = `RTT ${Number(net.rtt_ms || 0).toFixed(0)} ms`;
  if ($("#demo-link-dist")) $("#demo-link-dist").textContent = dist;
  if ($("#demo-link-bw"))
    $("#demo-link-bw").textContent = `${Number(net.bandwidth_mbps || 0).toFixed(1)} Mbps`;

  const sel = $("#net-profile");
  if (sel) {
    const want = isOut ? "outage" : "geo";
    if ([...sel.options].some((o) => o.value === want)) sel.value = want;
  }
  updateDisconnectButtons(isOut);
}

function drawWaveform(points) {
  const canvas = $("#net-wave");
  if (!canvas || !points?.length) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#f7f8fa";
  ctx.fillRect(0, 0, W, H);
  const rtts = points.map((p) => Number(p.rtt_ms) || 0);
  const maxR = Math.max(40, ...rtts);
  ctx.strokeStyle = "#0f5c6e";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = (i / Math.max(1, points.length - 1)) * (W - 8) + 4;
    const y = H - 8 - ((Number(p.rtt_ms) || 0) / maxR) * (H - 20);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function activeEdgeNodeId() {
  return $("#demo-edge-node")?.value || state.selectedEdgeId || "";
}

async function pollNetwork() {
  try {
    const nid = activeEdgeNodeId();
    const params = new URLSearchParams({ n: "120" });
    if (nid) params.set("edge_node_id", nid);
    const st = await api(`/api/network${nid ? `?edge_node_id=${encodeURIComponent(nid)}` : ""}`);
    const ts = await api(`/api/network/timeseries?${params}`);
    renderNetStatus(ts.network || st.network, ts.points?.[ts.points.length - 1], st.disconnected);
    drawWaveform(ts.points || []);
  } catch {
    /* ignore */
  }
}

async function setNetworkProfile() {
  const profile = $("#net-profile").value;
  const fd = new FormData();
  fd.append("profile", profile);
  const nid = activeEdgeNodeId();
  if (nid) fd.append("edge_node_id", nid);
  const res = await fetch("/api/network/profile", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
  renderNetStatus(data.network, data.sample, data.disconnected);
  await pollNetwork();
  pollTopology();
}

async function disconnectNetwork(nodeId) {
  const nid = nodeId || activeEdgeNodeId();
  const q = nid ? `?edge_node_id=${encodeURIComponent(nid)}` : "";
  const res = await fetch(`/api/network/disconnect${q}`, { method: "POST" });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
  renderNetStatus(data.network, data.sample, true);
  if ($("#demo-status"))
    $("#demo-status").textContent = `节点 ${data.edge_node_id || nid} 已断网`;
  await pollNetwork();
  pollTopology();
}

async function restoreNetwork(nodeId) {
  const nid = nodeId || activeEdgeNodeId();
  const q = nid ? `?edge_node_id=${encodeURIComponent(nid)}` : "";
  const res = await fetch(`/api/network/restore${q}`, { method: "POST" });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
  renderNetStatus(data.network, data.sample, false);
  if ($("#demo-status"))
    $("#demo-status").textContent = `节点 ${data.edge_node_id || nid} 已恢复物理链路`;
  await pollNetwork();
  pollTopology();
}

async function health() {
  try {
    const h = await api("/api/health");
    const nEdge = h.edge_fleet?.num_nodes ?? "—";
    const mode = h.edge_fleet?.network_mode || "—";
    const models = [
      h.route_agent_loaded ? "RA✓" : "RA…",
      h.cloud_loaded ? "Cloud✓" : "Cloud…",
    ].join(" ");
    $("#health-line").textContent =
      `API ok · ${models} · edges=${nEdge} · ${mode} · active=${h.active_edge_node || "—"} · net=${h.network_profile || "—"}`;
    renderModelStatus(h);
  } catch {
    $("#health-line").textContent = "API unreachable";
  }
}

/* ---------------- fleet live monitor (one-click) ---------------- */

function ensureLiveAnim() {
  if (state.live.animTimer) return;
  const step = () => {
    state.live.animTimer = requestAnimationFrame(step);
    if ($("#panel-topology")?.classList.contains("active")) drawTopology();
  };
  state.live.animT0 = performance.now();
  state.live.animTimer = requestAnimationFrame(step);
}

function pathBadgeClass(path) {
  if (path === "CLOUD_REVIEW") return "cloud";
  if (path === "LOCAL_NET_FALLBACK") return "fallback";
  if (path === "LOCAL") return "local";
  return "idle";
}

function pathShort(path) {
  if (path === "CLOUD_REVIEW") return "→ 云端复核";
  if (path === "LOCAL_NET_FALLBACK") return "链路回退";
  if (path === "LOCAL") return "本地判决";
  return "等待…";
}

function renderLiveNodeCards(st) {
  const grid = $("#live-node-grid");
  if (!grid) return;
  const fleetNodes = state.fleet?.nodes || [];
  const last = st?.last_by_node || {};
  const order = fleetNodes.length
    ? fleetNodes
    : Object.keys(last).map((id) => ({ id, city: last[id]?.city, category: last[id]?.category }));

  if (!order.length) {
    grid.innerHTML = `<div class="hint" style="grid-column:1/-1">加载边缘节点后可显示三路画面</div>`;
    return;
  }

  // rebuild structure once; update images in place to reduce flicker
  const ids = order.map((n) => n.id);
  const existing = [...grid.querySelectorAll(".live-node-card")].map((el) => el.dataset.nodeId);
  const needRebuild =
    existing.length !== ids.length || ids.some((id, i) => existing[i] !== id);

  if (needRebuild) {
    grid.innerHTML = order
      .map((n) => {
        return `<article class="live-node-card path-idle" data-node-id="${n.id}">
          <div class="lnc-head">
            <strong class="lnc-city">${n.city || n.id}</strong>
            <span class="lnc-badge idle">等待…</span>
          </div>
          <div class="lnc-img-wrap">
            <img alt="edge ${n.id}" style="display:none" />
            <div class="lnc-placeholder">启动联调后显示测试图像</div>
            <div class="lnc-overlay"></div>
          </div>
          <div class="lnc-body">
            <div class="lnc-file">—</div>
            <div class="lnc-reason">尚未有检测结果</div>
            <div class="lnc-meta">${n.category || "—"} · ${n.id}</div>
          </div>
        </article>`;
      })
      .join("");
  }

  for (const n of order) {
    const card = grid.querySelector(`.live-node-card[data-node-id="${n.id}"]`);
    if (!card) continue;
    const ev = last[n.id];
    const badge = card.querySelector(".lnc-badge");
    const city = card.querySelector(".lnc-city");
    const img = card.querySelector("img");
    const ph = card.querySelector(".lnc-placeholder");
    const overlay = card.querySelector(".lnc-overlay");
    const fileEl = card.querySelector(".lnc-file");
    const reasonEl = card.querySelector(".lnc-reason");
    const metaEl = card.querySelector(".lnc-meta");

    if (city) city.textContent = (ev?.city || n.city || n.id);
    if (!ev || !ev.ok) {
      card.className = "live-node-card path-idle";
      if (badge) {
        badge.className = "lnc-badge idle";
        badge.textContent = ev && !ev.ok ? "错误" : "等待…";
      }
      if (ev && !ev.ok && reasonEl) reasonEl.textContent = ev.error || "error";
      continue;
    }

    const path = ev.path_type || "LOCAL";
    const bcls = pathBadgeClass(path);
    card.className = `live-node-card path-${bcls === "idle" ? "idle" : bcls}`;
    if (badge) {
      badge.className = `lnc-badge ${bcls}`;
      badge.textContent = pathShort(path);
    }

    const sig = `${ev.image_url || ""}|${ev.t || 0}`;
    if (img && ev.image_url && state.live.cardSig[n.id] !== sig) {
      state.live.cardSig[n.id] = sig;
      img.onload = () => {
        img.style.display = "block";
        if (ph) ph.style.display = "none";
      };
      img.onerror = () => {
        img.style.display = "none";
        if (ph) {
          ph.style.display = "grid";
          ph.textContent = "图像加载失败";
        }
      };
      img.src = ev.image_url;
    } else if (img && ev.image_url && img.src) {
      img.style.display = "block";
      if (ph) ph.style.display = "none";
    }

    if (overlay) {
      const fin = String(ev.final || "—").toUpperCase();
      const gt = ev.gt != null ? String(ev.gt).toUpperCase() : null;
      const score =
        ev.edge_score != null && Number.isFinite(Number(ev.edge_score))
          ? Number(ev.edge_score).toFixed(2)
          : null;
      overlay.innerHTML = `
        <span class="lnc-chip ${fin === "NG" ? "ng" : "ok"}">判 ${fin}</span>
        ${gt ? `<span class="lnc-chip">GT ${gt}</span>` : ""}
        ${score != null ? `<span class="lnc-chip">s=${score}</span>` : ""}
      `;
    }
    if (fileEl) fileEl.textContent = `${ev.category || n.category || "—"} / ${ev.image || "—"}`;
    if (reasonEl) {
      const u = ev.utility != null ? ` U=${Number(ev.utility).toFixed(2)}` : "";
      reasonEl.textContent = `${ev.route_reason || path}${u}`;
    }
    if (metaEl) {
      metaEl.textContent = `RTT ${Number(ev.rtt_ms || 0).toFixed(0)}ms · ${Number(ev.bandwidth_mbps || 0).toFixed(0)}Mb · ${(Number(ev.loss_prob || 0) * 100).toFixed(1)}% · ${Number(ev.distance_geo_km || 0).toFixed(0)}km`;
    }
  }
}

function cloudStatusLabel(status) {
  if (status === "queued") return "排队";
  if (status === "running") return "检测中";
  if (status === "done") return "已完成";
  if (status === "fallback") return "回退";
  return status || "—";
}

function finalCloudStatus(path) {
  if (path === "LOCAL_NET_FALLBACK") return "fallback";
  if (path === "CLOUD_REVIEW") return "done";
  // upload_want but still local → treat as queued intent that settled without cloud hop
  return "done";
}

function clearCloudBoard() {
  state.live.cloudBoard = {};
  state.live.cloudBoardOrder = [];
  const board = $("#cloud-board");
  if (board) {
    board.innerHTML =
      '<div class="cloud-board-empty" id="cloud-board-empty">尚无上云案件 · 拟上云样本会留在这里，不会消失</div>';
  }
}

function findCloudCaseCard(key) {
  const board = $("#cloud-board");
  if (!board) return null;
  return [...board.querySelectorAll(".cloud-case")].find((el) => el.dataset.key === key) || null;
}

function upsertCloudCaseCard(job) {
  const board = $("#cloud-board");
  if (!board) return;
  $("#cloud-board-empty")?.remove();
  let card = findCloudCaseCard(job.key);
  const fin = String(job.final || "—").toUpperCase();
  const stamp = cloudStatusLabel(job.status);
  if (!card) {
    card = document.createElement("article");
    card.className = `cloud-case status-${job.status}`;
    card.dataset.key = job.key;
    card.innerHTML = `
      <div class="cloud-case-img">
        <img alt="${job.image || "cloud case"}" />
        <span class="cloud-case-stamp">${stamp}</span>
        <span class="cloud-case-fin ${fin === "NG" ? "ng" : "ok"}">${fin}</span>
      </div>
      <div class="cloud-case-body">
        <strong></strong>
        <span class="cc-file"></span>
        <span class="cc-path"></span>
      </div>
    `;
    // newest first
    board.prepend(card);
    const img = card.querySelector("img");
    if (img && job.image_url) img.src = job.image_url;
  } else {
    card.className = `cloud-case status-${job.status}`;
    const stampEl = card.querySelector(".cloud-case-stamp");
    if (stampEl) stampEl.textContent = stamp;
    const finEl = card.querySelector(".cloud-case-fin");
    if (finEl) {
      finEl.textContent = fin;
      finEl.className = `cloud-case-fin ${fin === "NG" ? "ng" : "ok"}`;
    }
  }
  const strong = card.querySelector("strong");
  const fileEl = card.querySelector(".cc-file");
  const pathEl = card.querySelector(".cc-path");
  if (strong) strong.textContent = job.city || job.edge_node_id || "—";
  if (fileEl) fileEl.textContent = `${job.category || "—"} / ${job.image || "—"}`;
  if (pathEl) pathEl.textContent = job.path_type || "UPLOAD";
  card.title = `${job.city || job.edge_node_id} · ${stamp} · ${job.path_type || ""}`;
}

function updateCloudBoardMeta(extraAdded = 0) {
  const meta = $("#cloud-lane-meta");
  if (!meta) return;
  const jobs = Object.values(state.live.cloudBoard);
  const nQ = jobs.filter((j) => j.status === "queued").length;
  const nR = jobs.filter((j) => j.status === "running").length;
  const nD = jobs.filter((j) => j.status === "done").length;
  const nF = jobs.filter((j) => j.status === "fallback").length;
  meta.textContent = jobs.length
    ? `共 ${jobs.length} · 排队 ${nQ} · 检测中 ${nR} · 完成 ${nD} · 回退 ${nF}${extraAdded ? ` · +${extraAdded}` : ""}`
    : state.live.running
      ? "监测中 · 等待拟上云案件…"
      : "等待上云流量…";
}

function scheduleCloudCaseLifecycle(job) {
  // Visual pipeline: 排队 → 检测中 → 终态（已完成/回退），卡片本身常驻
  if (job._lifecycleStarted) return;
  job._lifecycleStarted = true;
  const age = job.t ? Date.now() / 1000 - job.t : 0;
  // Old history from poll catch-up: jump straight to final status
  if (age > 6) {
    job.status = finalCloudStatus(job.path_type);
    upsertCloudCaseCard(job);
    updateCloudBoardMeta();
    return;
  }
  job.status = "queued";
  upsertCloudCaseCard(job);
  updateCloudBoardMeta();
  window.setTimeout(() => {
    if (!state.live.cloudBoard[job.key]) return;
    job.status = "running";
    upsertCloudCaseCard(job);
    updateCloudBoardMeta();
  }, 700);
  window.setTimeout(() => {
    if (!state.live.cloudBoard[job.key]) return;
    job.status = finalCloudStatus(job.path_type);
    upsertCloudCaseCard(job);
    updateCloudBoardMeta();
  }, 2200);
}

function renderCloudBoard(st) {
  const events = st?.events || [];
  const uploads = events.filter(
    (e) =>
      e?.ok &&
      e.image_url &&
      (e.upload_want || e.path_type === "CLOUD_REVIEW" || e.path_type === "LOCAL_NET_FALLBACK")
  );
  // events are newest-first; walk older→newer so prepend keeps newest on left/top
  const incoming = uploads.slice().reverse();
  let added = 0;
  for (const e of incoming) {
    const key = `${e.edge_node_id}|${e.t}|${e.image || ""}`;
    if (state.live.cloudBoard[key]) continue;
    const job = {
      key,
      edge_node_id: e.edge_node_id,
      city: e.city,
      category: e.category,
      image: e.image,
      image_url: e.image_url,
      path_type: e.path_type,
      final: e.final,
      t: e.t,
      status: "queued",
    };
    state.live.cloudBoard[key] = job;
    state.live.cloudBoardOrder.unshift(key);
    added += 1;
    scheduleCloudCaseLifecycle(job);
  }
  // hard cap display (keep newest)
  const MAX = 48;
  while (state.live.cloudBoardOrder.length > MAX) {
    const old = state.live.cloudBoardOrder.pop();
    delete state.live.cloudBoard[old];
    findCloudCaseCard(old)?.remove();
  }
  if (!state.live.cloudBoardOrder.length && !$("#cloud-board-empty")) {
    const board = $("#cloud-board");
    if (board) {
      board.innerHTML =
        '<div class="cloud-board-empty" id="cloud-board-empty">尚无上云案件 · 拟上云样本会留在这里，不会消失</div>';
    }
  }
  updateCloudBoardMeta(added);
}

function renderLiveStatus(st) {
  state.live.status = st;
  state.live.running = !!st?.running;
  const btn = $("#live-toggle");
  const dot = $("#live-dot");
  const label = $("#live-status");
  if (btn) {
    btn.textContent = st?.running ? "一键联调 · 停止" : "一键联调 · 启动";
    btn.classList.toggle("danger", !!st?.running);
    btn.classList.toggle("primary", !st?.running);
  }
  $("#live-bar")?.classList.toggle("is-live", !!st?.running);
  if (dot) dot.classList.toggle("on", !!st?.running);
  const stats = st?.stats || {};
  if (label) {
    label.textContent = st?.running
      ? `监测中 · 每 ${Number(st.interval_s || 2).toFixed(1)}s 三节点各跑 1 张 · 轮次 ${stats.ticks || 0}`
      : "未启动 · 启动后三节点自动监测并展示测试图像";
  }
  const hint = $("#live-stage-hint");
  if (hint) {
    hint.textContent = st?.running
      ? "下方三路画面随轮次刷新；Cloud Uplink 案件板常驻显示上云样本（颜色=排队/检测中/完成/回退）"
      : "启动联调后，每个边缘节点的当前测试图、路由路径与判决会显示在这里";
  }
  if ($("#live-kpi-ticks")) $("#live-kpi-ticks").textContent = String(stats.ticks || 0);
  if ($("#live-kpi-local")) $("#live-kpi-local").textContent = String(stats.n_local || 0);
  if ($("#live-kpi-want")) $("#live-kpi-want").textContent = String(stats.n_upload_want || 0);
  if ($("#live-kpi-cloud")) $("#live-kpi-cloud").textContent = String(stats.n_cloud_ok || 0);
  if ($("#live-kpi-fb")) $("#live-kpi-fb").textContent = String(stats.n_fallback || 0);
  const cloud = st?.cloud || {};
  if ($("#live-kpi-inflight")) {
    $("#live-kpi-inflight").textContent = `${cloud.inflight || 0}/${cloud.max_inflight || 2}`;
  }

  // pulse map for link animation
  const last = st?.last_by_node || {};
  for (const [nid, ev] of Object.entries(last)) {
    if (!ev || !ev.ok) continue;
    state.live.linkPulse[nid] = {
      upload: !!ev.upload_want || ev.path_type === "CLOUD_REVIEW" || ev.path_type === "LOCAL_NET_FALLBACK",
      path: ev.path_type || "LOCAL",
      t: (ev.t || 0) * 1000,
    };
  }

  renderLiveNodeCards(st);
  renderCloudBoard(st);
}

async function pollLive() {
  try {
    const st = await api("/api/fleet/live");
    renderLiveStatus(st);
    if (st.running) {
      await pollTopology();
      ensureLiveAnim();
    }
  } catch (e) {
    if ($("#live-status")) $("#live-status").textContent = `联调状态失败：${e.message}`;
  }
}

async function toggleLive() {
  const running = !!state.live.running;
  try {
    if (running) {
      const st = await api("/api/fleet/live/stop", { method: "POST" });
      renderLiveStatus(st);
      if (state.live.pollTimer) {
        clearInterval(state.live.pollTimer);
        state.live.pollTimer = null;
      }
    } else {
      switchPanel("topology");
      clearCloudBoard();
      const fd = new FormData();
      fd.append("interval_s", "2.0");
      fd.append("use_route_agent", "false");
      fd.append("live_cloud", "false");
      const res = await fetch("/api/fleet/live/start", { method: "POST", body: fd });
      const st = await res.json();
      if (!res.ok) throw new Error(st.detail || JSON.stringify(st));
      renderLiveStatus(st);
      ensureLiveAnim();
      if (state.live.pollTimer) clearInterval(state.live.pollTimer);
      state.live.pollTimer = setInterval(() => pollLive().catch(() => {}), 1000);
      await pollLive();
    }
  } catch (e) {
    if ($("#live-status")) $("#live-status").textContent = `操作失败：${e.message}`;
  }
}

function bind() {
  setupTabs();
  bindTopoCanvas();
  $("#demo-cat")?.addEventListener("change", loadDemoImages);
  $("#demo-img")?.addEventListener("change", updatePreview);
  $("#demo-file")?.addEventListener("change", onLocalFileChange);
  $("#demo-file-clear")?.addEventListener("click", clearLocalUpload);
  $("#demo-run")?.addEventListener("click", runDemo);
  $("#demo-edge-node")?.addEventListener("change", () => {
    const id = activeEdgeNodeId();
    if (id) selectEdgeNode(id).catch((e) => {
      $("#demo-status").textContent = `切换失败：${e.message}`;
    });
  });
  $("#net-profile")?.addEventListener("change", () => {
    setNetworkProfile().catch((e) => {
      $("#demo-status").textContent = `链路切换失败：${e.message}`;
    });
  });
  $("#net-disconnect")?.addEventListener("click", () => {
    disconnectNetwork().catch((e) => {
      $("#demo-status").textContent = `断网失败：${e.message}`;
    });
  });
  $("#net-restore")?.addEventListener("click", () => {
    restoreNetwork().catch((e) => {
      $("#demo-status").textContent = `恢复失败：${e.message}`;
    });
  });
  $("#topo-refresh")?.addEventListener("click", () => pollTopology());
  $("#topo-disconnect")?.addEventListener("click", () => {
    disconnectNetwork(state.selectedEdgeId).catch(() => {});
  });
  $("#topo-restore")?.addEventListener("click", () => {
    restoreNetwork(state.selectedEdgeId).catch(() => {});
  });
  $("#live-toggle")?.addEventListener("click", () => toggleLive());
}

function showBootError(err) {
  const el = document.createElement("div");
  el.style.cssText =
    "position:fixed;left:12px;right:12px;bottom:12px;z-index:9999;background:#fdecea;color:#b42318;border:1px solid #f3b0ab;padding:10px 12px;border-radius:8px;font:13px/1.4 sans-serif;white-space:pre-wrap;";
  el.textContent = "前端初始化失败: " + (err && err.message ? err.message : String(err));
  document.body.appendChild(el);
  const line = $("#health-line");
  if (line) line.textContent = "boot error";
}

async function main() {
  try {
    bind();
    switchPanel("overview");
    await health().catch(() => {});
    await loadOverview().catch((e) => console.warn(e));
    let fleet = null;
    try {
      fleet = await loadEdgeFleet();
    } catch (e) {
      console.error(e);
      showBootError(e);
    }
    await fillCategorySelects().catch((e) => console.warn(e));
    if (fleet) {
      state.fleet = fleet;
      state.selectedEdgeId = fleet.active_id;
      const active = (fleet.nodes || []).find((n) => n.id === fleet.active_id);
      const demoCat = $("#demo-cat");
      if (active && active.category && demoCat) {
        const has = Array.prototype.some.call(demoCat.options, (o) => o.value === active.category);
        if (has) {
          demoCat.value = active.category;
          await loadDemoImages().catch(() => {});
        }
      }
    }
    await pollNetwork().catch(() => {});
    await pollTopology().catch(() => {});
    await pollLive().catch(() => {});
    setInterval(() => pollNetwork().catch(() => {}), 1000);
    setInterval(() => pollTopology().catch(() => {}), 1000);
    setInterval(() => health().catch(() => {}), 2000);
    setInterval(() => {
      if (state.live.running || $("#panel-topology")?.classList.contains("active")) {
        pollLive().catch(() => {});
      }
    }, 1500);
  } catch (err) {
    console.error(err);
    showBootError(err);
  }
}

main();
