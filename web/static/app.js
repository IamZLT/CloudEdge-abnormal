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

async function runDemo() {
  const status = $("#demo-status");
  const cat = $("#demo-cat").value;
  const path = $("#demo-img").value;
  const live = $("#demo-live").checked;
  status.textContent = live ? "运行中（可能加载云端模型）…" : "读取边侧分数 / 缓存案例…";
  $("#demo-run").disabled = true;
  try {
    const fd = new FormData();
    fd.append("category", cat);
    fd.append("image_path", path);
    fd.append("live_cloud", live ? "true" : "false");
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

    $("#out-route").textContent = JSON.stringify(
      {
        route: data.route,
        final: data.final_decision,
        live: live,
      },
      null,
      2
    );

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
        live: live,
        latency_ms: cloud?.latency_ms,
      },
      null,
      2
    );
    $("#out-decision").textContent = String(final);
    status.textContent = `完成 · route=${data.route || "—"} · final=${final}` +
      (cloud && !cloud.skipped ? " · LLM ready" : " · no LLM");
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

async function health() {
  try {
    const h = await api("/api/health");
    $("#health-line").textContent = `API ok · cloud_loaded=${h.cloud_loaded}`;
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
  $("#case-cat").addEventListener("change", loadCases);
}

async function main() {
  bind();
  await health();
  await loadOverview();
  await fillCategorySelects();
}

main();
