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
    throw new Error(t || res.statusText);
  }
  return res.json();
}

function setupTabs() {
  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".tab").forEach((b) => b.classList.remove("active"));
      $$(".panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(`#panel-${btn.dataset.panel}`).classList.add("active");
    });
  });
}

async function loadOverview() {
  const s = await api("/api/summary");
  $("#m-b1").textContent = fmt(s.means.B1_f1);
  $("#m-zs").textContent = fmt(s.means.ZS8B_f1);
  $("#m-l8").textContent = fmt(s.means.Lora8B_f1);
  $("#m-l4").textContent = fmt(s.means.Lora4B_f1);
  $("#stack-edge").textContent = s.stack.edge;
  $("#stack-cloud").textContent = s.stack.cloud;
  $("#stack-collab").textContent = s.stack.collab;

  const tbody = $("#metrics-table tbody");
  tbody.innerHTML = "";
  (s.categories || []).forEach((r) => {
    const d = (r.Lora8B_f1 ?? 0) - (r.B1_f1 ?? 0);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.category}</td>
      <td>${r.n ?? "—"}</td>
      <td>${fmt(r.B1_f1)}</td>
      <td>${fmt(r.ZS8B_f1)}</td>
      <td>${fmt(r.Lora4B_f1)}</td>
      <td>${fmt(r.Lora8B_f1)}</td>
      <td class="${d >= 0 ? "delta-pos" : "delta-neg"}">${d >= 0 ? "+" : ""}${fmt(d)}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function fillCategorySelects() {
  const { categories } = await api("/api/categories");
  const demo = $("#demo-cat");
  const cases = $("#case-cat");
  demo.innerHTML = "";
  cases.innerHTML = "";
  categories.forEach((c) => {
    demo.insertAdjacentHTML("beforeend", `<option value="${c}">${c}</option>`);
    cases.insertAdjacentHTML("beforeend", `<option value="${c}">${c}</option>`);
  });
  const prefer = categories.includes("screw") ? "screw" : categories[0];
  demo.value = prefer;
  cases.value = prefer;
  await loadDemoImages();
  await loadCases();
}

async function loadDemoImages() {
  const cat = $("#demo-cat").value;
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

function renderLlm(cloud) {
  const badge = $("#llm-badge");
  const rawEl = $("#out-cloud");
  if (!cloud || cloud.skipped) {
    badge.textContent = cloud?.skipped ? "skipped" : "no output";
    badge.className = "llm-badge empty";
    $("#llm-decision").textContent = "—";
    $("#llm-conf").textContent = "—";
    $("#llm-type").textContent = "—";
    $("#llm-reason").textContent = cloud?.skipped
      ? "边侧置信，未上云（LOCAL）。可勾选 Live cloud 强制调用。"
      : "当前样本无云端 LLM 缓存。请选择带 [LLM] 标记的图像，或勾选 Live cloud LoRA。";
    rawEl.textContent = "";
    return;
  }
  const decision = cloud.decision || "—";
  badge.textContent = decision;
  badge.className = `llm-badge ${String(decision).toUpperCase() === "NG" ? "ng" : "ok"}`;
  $("#llm-decision").textContent = decision;
  $("#llm-conf").textContent =
    cloud.confidence === undefined || cloud.confidence === null
      ? "—"
      : Number(cloud.confidence).toFixed(2);
  $("#llm-type").textContent = cloud.defect_type || "—";
  $("#llm-reason").textContent = cloud.reason || "(no reason)";
  let raw = cloud.raw;
  if (!raw) {
    raw = JSON.stringify(
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
  rawEl.textContent = raw;
}

function updatePreview() {
  const path = $("#demo-img").value;
  const img = $("#demo-preview");
  if (!path) {
    img.removeAttribute("src");
    return;
  }
  img.src = `/api/image?path=${encodeURIComponent(path)}`;
  // clear previous maps until Run
  renderViz(null);
}

function renderViz(viz) {
  const edgeFig = $("#viz-edge")?.closest("figure");
  const cloudFig = $("#viz-cloud")?.closest("figure");
  const status = $("#viz-status");
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
  if (!viz) {
    setFig("#viz-edge", edgeFig, null);
    setFig("#viz-cloud", cloudFig, null);
    if (status) {
      status.textContent =
        "热力图来自 Anomalib，不是 Qwen-VL。云端 VLM 只输出下方 JSON（decision / reason）。";
    }
    return;
  }
  setFig("#viz-edge", edgeFig, viz.edge_strip || null);
  setFig("#viz-cloud", cloudFig, viz.cloud_strip || null);
  if (status) {
    const parts = [];
    if (viz.edge_strip) parts.push("边侧 PaDiM 热力图");
    if (viz.cloud_strip) parts.push("重型 PatchCore 热力图（对比用，非 VLM）");
    if (viz.gt_mask) parts.push("GT mask");
    status.textContent = parts.length
      ? `${parts.join(" · ")}。VLM 结果见下方 Cloud LLM Output。`
      : "该样本暂无 Anomalib 可视化缓存；VLM 仍可能有 JSON 输出。";
  }
}

function renderRouteAgent(ra, pathType, netOut) {
  const badge = $("#route-llm-badge");
  const rawEl = $("#out-route-raw");
  if (!ra) {
    badge.textContent = "idle";
    badge.className = "llm-badge empty";
    $("#route-llm-reason").textContent = "尚未运行路由智能体。";
    rawEl.textContent = "";
    $("#out-route").textContent = "—";
    return;
  }
  const upload = !!ra.upload;
  badge.textContent = upload ? "upload" : "local";
  badge.className = `llm-badge ${upload ? "ng" : "ok"}`;
  $("#route-llm-reason").textContent = ra.reason || "(no reason)";
  rawEl.textContent =
    (ra.raw && String(ra.raw).trim()) ||
    JSON.stringify(
      { upload: ra.upload, confidence: ra.confidence, reason: ra.reason, source: ra.source },
      null,
      2
    );
  $("#out-route").textContent = JSON.stringify(
    {
      path: pathType,
      upload_want: upload,
      confidence: ra.confidence,
      source: ra.source,
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
  status.textContent = useAgent
    ? "运行中（RouteAgent 可能首次加载较慢）…"
    : live
      ? "运行中（可能加载云端模型）…"
      : "读取边侧分数 / 路由…";
  $("#demo-run").disabled = true;
  try {
    const fd = new FormData();
    fd.append("category", cat);
    fd.append("image_path", path);
    fd.append("live_cloud", live ? "true" : "false");
    fd.append("use_route_agent", useAgent ? "true" : "false");
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
      : "未找到预计算边侧分数（可换数据集内图像，或开实时云端）";

    renderViz(data.viz || null);
    renderRouteAgent(data.route_agent, data.route, data.network_outcome);

    const cloud =
      data.cloud_live && !data.cloud_live.skipped
        ? data.cloud_live
        : data.cached_case?.cloud || data.cloud_live;
    renderLlm(cloud);

    const final = data.final_decision || data.cached_case?.final || edge?.edge_pred || "—";
    $("#out-final").textContent = JSON.stringify(
      {
        gt: data.cached_case?.gt,
        path: data.route,
        network_profile: data.network?.profile || data.route_agent?.network_profile,
        live: live,
        use_route_agent: useAgent,
        latency_ms: cloud?.latency_ms,
      },
      null,
      2
    );
    $("#out-decision").textContent = String(final);
    status.textContent =
      `完成 · ${data.route || "—"} · final=${final}` +
      (data.route_agent?.source ? ` · route=${data.route_agent.source}` : "") +
      (cloud && !cloud.skipped ? " · cloud LLM" : "");
    // refresh waveform after demo upload probe
    pollNetwork();
  } catch (e) {
    status.textContent = `失败：${e.message}`;
    $("#out-final").textContent = e.message;
  } finally {
    $("#demo-run").disabled = false;
  }
}

async function loadCases() {
  const cat = $("#case-cat").value;
  const grid = $("#case-grid");
  grid.innerHTML = "<p class='hint'>加载中…</p>";
  try {
    const data = await api(`/api/cases?category=${encodeURIComponent(cat)}&limit=18`);
    grid.innerHTML = "";
    if (!data.cases.length) {
      grid.innerHTML = "<p class='hint'>该类暂无案例</p>";
      return;
    }
    data.cases.forEach((c) => {
      const el = document.createElement("article");
      el.className = "case";
      const cloud = c.cloud;
      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = c.name;
      // Prefer edge heatmap strip when available
      img.src = c.viz?.edge_strip || `/api/image?path=${encodeURIComponent(c.path)}`;
      const body = document.createElement("div");
      body.className = "body";
      const tags = document.createElement("div");
      tags.className = "tags";
      tags.innerHTML = `
        <span class="tag">GT ${c.gt}</span>
        <span class="tag ${c.edge_pred === "NG" ? "ng" : "ok"}">Edge ${c.edge_pred ?? "—"}</span>
        <span class="tag ${c.final === "NG" ? "ng" : "ok"}">Final ${c.final ?? "—"}</span>
        ${c.path_type === "CLOUD_REVIEW" ? '<span class="tag cloud">CLOUD</span>' : '<span class="tag">LOCAL</span>'}
        ${c.viz?.edge_strip ? '<span class="tag cloud">MAP</span>' : ""}
      `;
      const p = document.createElement("p");
      p.textContent = cloud
        ? `${cloud.defect_type || "—"} · ${cloud.reason || ""}`
        : "本地路径（未上云）";
      body.appendChild(tags);
      body.appendChild(p);
      if (cloud) {
        const pre = document.createElement("pre");
        pre.textContent =
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
        body.appendChild(pre);
      }
      el.appendChild(img);
      el.appendChild(body);
      grid.appendChild(el);
    });
  } catch (e) {
    grid.innerHTML = `<p class="hint">加载失败：${e.message}</p>`;
  }
}

async function loadCloud() {
  $("#demo-status").textContent = "正在加载云端 LoRA…";
  try {
    const r = await api("/api/cloud/load", { method: "POST" });
    $("#demo-status").textContent = r.ok ? "云端模型已加载" : `加载失败：${r.error}`;
  } catch (e) {
    $("#demo-status").textContent = `加载失败：${e.message}`;
  }
}

async function loadRouteAgent() {
  $("#demo-status").textContent = "正在加载 RouteAgent（Qwen3.5 全量）…";
  try {
    const r = await api("/api/route_agent/load", { method: "POST" });
    $("#demo-status").textContent = r.ok ? "RouteAgent 已加载" : `加载失败：${r.error}`;
  } catch (e) {
    $("#demo-status").textContent = `RouteAgent 加载失败：${e.message}`;
  }
}

function drawWaveform(points) {
  const canvas = $("#net-wave");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 960;
  const cssH = canvas.clientHeight || 160;
  if (canvas.width !== Math.floor(cssW * dpr) || canvas.height !== Math.floor(cssH * dpr)) {
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cssW;
  const H = cssH;
  ctx.clearRect(0, 0, W, H);
  // background
  ctx.fillStyle = "#f4f7f9";
  ctx.fillRect(0, 0, W, H);
  // grid
  ctx.strokeStyle = "#dde5ec";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (H * i) / 4;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }
  if (!points || points.length < 2) {
    ctx.fillStyle = "#8a95a3";
    ctx.font = "12px IBM Plex Mono, monospace";
    ctx.fillText("waiting for network samples…", 12, H / 2);
    return;
  }

  const rtts = points.map((p) => Number(p.rtt_ms) || 0);
  const bws = points.map((p) => (Number(p.bandwidth_mbps) || 0) * 10);
  const losses = points.map((p) => (Number(p.loss_prob) || 0) * 1000); // 0..1000 scale
  const ymax = Math.max(50, ...rtts, ...bws, ...losses) * 1.15;

  const plot = (arr, color, width) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = (i / (arr.length - 1)) * (W - 8) + 4;
      const y = H - 8 - (Math.min(v, ymax) / ymax) * (H - 16);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  // soft fill under RTT
  ctx.beginPath();
  rtts.forEach((v, i) => {
    const x = (i / (rtts.length - 1)) * (W - 8) + 4;
    const y = H - 8 - (Math.min(v, ymax) / ymax) * (H - 16);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(W - 4, H - 8);
  ctx.lineTo(4, H - 8);
  ctx.closePath();
  ctx.fillStyle = "rgba(15, 92, 110, 0.10)";
  ctx.fill();

  plot(bws, "#2f6f4e", 1.5);
  plot(losses, "#b42318", 1.5);
  plot(rtts, "#0f5c6e", 2.2);

  // outage / fail markers
  points.forEach((p, i) => {
    if (p.upload_ok === false || p.profile === "outage") {
      const x = (i / (points.length - 1)) * (W - 8) + 4;
      ctx.fillStyle = "rgba(180, 35, 24, 0.35)";
      ctx.fillRect(x - 1, 4, 2, H - 12);
    }
  });
}

function updateDisconnectButtons(disconnected) {
  const disc = $("#net-disconnect");
  const rest = $("#net-restore");
  if (disc) disc.disabled = !!disconnected;
  if (rest) rest.disabled = !disconnected;
  const panel = $("#net-panel");
  if (panel) panel.classList.toggle("is-disconnected", !!disconnected);
}

function renderNetStatus(net, last, disconnected) {
  if (!net) return;
  const isOut = disconnected ?? net.profile === "outage";
  const chip = $("#net-chip");
  chip.textContent = isOut ? "OUTAGE" : net.profile || "—";
  chip.className = `net-chip profile-${net.profile || "fair"}`;
  $("#net-rtt").textContent = `${Number(net.rtt_ms).toFixed(0)} ± ${Number(net.rtt_jitter_ms || 0).toFixed(0)} ms`;
  $("#net-bw").textContent = `${Number(net.bandwidth_mbps).toFixed(1)} Mbps`;
  $("#net-loss").textContent = `${(Number(net.loss_prob) * 100).toFixed(1)}%`;
  $("#net-timeout").textContent = `${Number(net.timeout_ms).toFixed(0)} ms`;
  if (last) {
    const ok = last.upload_ok ? "ok" : last.failed_reason || "fail";
    $("#net-probe").textContent = isOut ? "断网 · 不上云" : `RTT ${Number(last.rtt_ms || 0).toFixed(0)} · ${ok}`;
  }
  const sel = $("#net-profile");
  if (sel && net.profile && sel.value !== net.profile) {
    sel.value = net.profile;
  }
  updateDisconnectButtons(isOut);
}

async function pollNetwork() {
  try {
    const st = await api("/api/network");
    const ts = await api("/api/network/timeseries?n=120");
    renderNetStatus(ts.network || st.network, ts.points?.[ts.points.length - 1], st.disconnected);
    drawWaveform(ts.points || []);
  } catch {
    /* ignore transient errors while server starts */
  }
}

async function setNetworkProfile() {
  const profile = $("#net-profile").value;
  const fd = new FormData();
  fd.append("profile", profile);
  const res = await fetch("/api/network/profile", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
  renderNetStatus(data.network, data.sample, data.disconnected);
  await pollNetwork();
}

async function disconnectNetwork() {
  const res = await fetch("/api/network/disconnect", { method: "POST" });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
  renderNetStatus(data.network, data.sample, true);
  $("#demo-status").textContent =
    `已模拟断网（outage）· 恢复将回到 ${data.restore_profile || "fair"}`;
  await pollNetwork();
}

async function restoreNetwork() {
  const res = await fetch("/api/network/restore", { method: "POST" });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
  renderNetStatus(data.network, data.sample, false);
  $("#demo-status").textContent =
    `网络已恢复 · profile=${data.network?.profile || "fair"}`;
  await pollNetwork();
}

async function health() {
  try {
    const h = await api("/api/health");
    $("#health-line").textContent =
      `API ok · net=${h.network_profile || "—"} · route_agent=${h.route_agent_loaded} · cloud=${h.cloud_loaded}`;
  } catch {
    $("#health-line").textContent = "API unreachable";
  }
}

function bind() {
  setupTabs();
  $("#demo-cat").addEventListener("change", loadDemoImages);
  $("#demo-img").addEventListener("change", updatePreview);
  $("#demo-run").addEventListener("click", runDemo);
  $("#demo-load-cloud").addEventListener("click", loadCloud);
  $("#demo-load-route")?.addEventListener("click", loadRouteAgent);
  $("#case-cat").addEventListener("change", loadCases);
  $("#net-profile")?.addEventListener("change", () => {
    setNetworkProfile().catch((e) => {
      $("#demo-status").textContent = `网络剖面切换失败：${e.message}`;
    });
  });
  $("#net-disconnect")?.addEventListener("click", () => {
    disconnectNetwork().catch((e) => {
      $("#demo-status").textContent = `断网失败：${e.message}`;
    });
  });
  $("#net-restore")?.addEventListener("click", () => {
    restoreNetwork().catch((e) => {
      $("#demo-status").textContent = `恢复网络失败：${e.message}`;
    });
  });
  window.addEventListener("resize", () => pollNetwork());
}

async function main() {
  bind();
  await health();
  await loadOverview();
  await fillCategorySelects();
  await pollNetwork();
  setInterval(pollNetwork, 1000);
  setInterval(health, 5000);
}

main();
