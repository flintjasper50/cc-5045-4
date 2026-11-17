// Multi-device render verification panel.
// Every image/stat here is fetched from the backend at click time; this file
// contains no rendered pixels or canned error numbers.
const $ = (id) => document.getElementById(id);
let state = { devices: [], scenes: [], lastJob: null, res: 128 };

function b64Png(b64) {
  return "data:image/png;base64," + b64;
}

async function fetchJson(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.status);
  return res.json();
}

async function init() {
  const [scenes, dev] = await Promise.all([
    fetchJson("/api/scenes"),
    fetchJson("/api/devices"),
  ]);
  state.scenes = scenes.scenes;
  state.devices = dev.devices;

  const sel = $("scene-select");
  state.scenes.forEach((s) => {
    const o = document.createElement("option");
    o.value = s.key;
    o.textContent = s.title;
    sel.appendChild(o);
  });
  sel.onchange = () => {
    const s = state.scenes.find((x) => x.key === sel.value);
    $("scene-desc").textContent = s.description;
  };
  sel.dispatchEvent(new Event("change"));

  const list = $("device-list");
  state.devices.forEach((d) => {
    const lab = document.createElement("label");
    if (!d.available) lab.className = "unavailable";
    lab.innerHTML =
      `<input type="checkbox" value="${d.id}" ${d.available ? "checked" : "disabled"}>` +
      `${d.label} <span class="meta">[${d.precision}]</span>`;
    lab.title = d.executed_note;
    list.appendChild(lab);
  });

  const banner = $("gpu-banner");
  const g = dev.gpu_status;
  banner.classList.remove("hidden");
  banner.classList.toggle("ok", g.gpu_can_execute);
  banner.innerHTML =
    (g.gpu_can_execute ? "✅ " : "⚠️ ") + g.coverage +
    `<br><small>${state.devices.find((d) => d.id === "gpu_cuda").executed_note}</small>`;

  $("res-select").onchange = (e) => { state.res = parseInt(e.target.value, 10); };
  $("render-btn").onclick = runRender;
}

function selectedDevices() {
  return [...document.querySelectorAll(".device-pick input:checked")].map((i) => i.value);
}

async function runRender() {
  const btn = $("render-btn");
  btn.disabled = true;
  $("run-status").textContent = "后端真实计算中…";
  try {
    const job = await fetchJson("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scene: $("scene-select").value,
        devices: selectedDevices(),
        width: state.res,
        height: state.res,
      }),
    });
    state.lastJob = job;
    renderResults(job);
    $("run-status").textContent =
      `完成：执行 ${job.executed.join(", ")}` +
      (job.skipped.length ? `；未执行 ${job.skipped.join(", ")}（无伪造输出）` : "");
  } catch (err) {
    $("run-status").textContent = "渲染失败：" + err.message;
  } finally {
    btn.disabled = false;
  }
}

function renderResults(job) {
  $("result-panel").classList.remove("hidden");
  renderVerdict(job);
  renderDeviceImages(job);
  renderDiagnostics(job);
  renderStats(job);
  renderTolerance(job);
  $("config-box").textContent = JSON.stringify(job.config, null, 2);
  attachImageClickHandlers(job);
}

function renderVerdict(job) {
  const box = $("verdict-box");
  if (!job.comparisons.length) {
    box.className = "verdict";
    box.textContent = "可比较的两条路径未同时执行，无法给出对照结论。";
    return;
  }
  const v = job.comparisons[0].verdict;
  box.className = "verdict " + v.class;
  box.innerHTML = `<span class="tag">${v.title}</span>${v.description}` +
    (v.pass ? ' <span class="pill gpu-on">通过</span>'
            : ' <span class="pill gpu-off">需排查</span>');
}

function card(title, imgB64, meta, dataKey, pxW) {
  const el = document.createElement("div");
  el.className = "card";
  el.dataset.key = dataKey;
  el.dataset.width = pxW;
  el.innerHTML =
    `<h4>${title}</h4>` +
    (imgB64 ? `<img alt="${title}" src="${b64Png(imgB64)}">` : "<p>无图像</p>") +
    (meta ? `<p class="meta">${meta}</p>` : "");
  return el;
}

function renderDeviceImages(job) {
  const grid = $("device-results");
  grid.innerHTML = "";
  Object.values(job.renders).forEach((r) => {
    const wrap = document.createElement("div");
    wrap.style.display = "contents";
    const meta =
      `${r.precision} · 耗时 ${r.elapsed_ms} ms · 覆盖 ${r.covered_count} px`;
    grid.appendChild(card(`${r.label} — sRGB 颜色`, r.images.srgb_png, meta,
      "dev:" + r.device, job.config.width));
    grid.appendChild(card(`${r.label} — 线性 HDR(钳制显示)`, r.images.linear_png,
      "线性光，超 1.0 仅在显示时钳制", "dev:" + r.device, job.config.width));
    grid.appendChild(card(`${r.label} — 深度`, r.images.depth_png,
      "蓝→红表示深度由近到远，黑色为背景", "dev:" + r.device, job.config.width));
    grid.appendChild(card(`${r.label} — 对象标识`, r.images.id_png,
      "不同对象不同颜色，id=0 为背景", "dev:" + r.device, job.config.width));
  });
}

function renderDiagnostics(job) {
  const grid = $("diagnostic-grid");
  grid.innerHTML = "";
  if (!job.comparisons.length) return;
  const c = job.comparisons[0];
  const items = [
    ["颜色绝对误差热图", c.images.color_heatmap_png, "模型区每像素最大通道绝对误差"],
    ["相对误差热图", c.images.relative_heatmap_png, "暗部以下限 eps 归一，封顶 2%"],
    ["深度绝对误差热图", c.images.depth_heatmap_png, "NDC 深度差"],
    ["边缘分布", c.images.edge_png, "紫色为模型边缘带，灰为内部"],
    ["覆盖差异图", c.images.coverage_diff_png, "红=参考独有, 蓝=被测独有, 灰=共有"],
  ];
  items.forEach(([t, img, m]) =>
    grid.appendChild(card(t, img, m, "cmp", job.config.width)));
}

function fmt(x) {
  if (typeof x !== "number") return x;
  return Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(3) : x.toFixed(6).replace(/0+$/, "");
}

function renderStats(job) {
  const box = $("stats-table");
  if (!job.comparisons.length) { box.innerHTML = ""; return; }
  const c = job.comparisons[0];
  const s = c.stats;
  const row = (label, a) =>
    `<tr><td>${label}</td><td>${fmt(a.mean)}</td><td>${fmt(a.max)}</td>` +
    `<td>${fmt(a.p99)}</td><td>${fmt(a.rmse)}</td><td>${a.count}</td></tr>`;
  box.innerHTML = `
  <table>
    <thead><tr><th>指标（仅模型有效区域）</th><th>均值</th><th>最大</th>
      <th>P99</th><th>RMSE</th><th>样本数</th></tr></thead>
    <tbody>
      ${row("sRGB 绝对误差 · 全部模型区", s.all_srgb_abs)}
      ${row("sRGB 绝对误差 · 边缘带", s.edge.srgb_abs)}
      ${row("sRGB 绝对误差 · 内部", s.interior.srgb_abs)}
      ${row("sRGB 相对误差 · 全部", s.all_srgb_rel)}
      ${row("线性 HDR 绝对误差 · 全部", s.all_linear_abs)}
      ${row("深度绝对误差 · 全部", s.all_depth_abs)}
    </tbody>
  </table>
  <p class="meta">总像素 ${s.pixels_total} · 背景(零值,不计入) ${s.background_pixels} ·
     模型并集 ${s.model_union_pixels} · 共有覆盖 ${s.coverage_both} ·
     参考独有 ${s.coverage_ref_only} · 被测独有 ${s.coverage_test_only} ·
     边缘带 ${s.edge_band_pixels} · 内部 ${s.interior_pixels} ·
     非有限值 ${s.nonfinite.count}</p>
  <p class="meta">结构错误合计 ${c.structural.total}（覆盖差异 + 对象标识不一致），
     与数值精度偏差分开归类。</p>`;
}

function renderTolerance(job) {
  const t = job.comparisons.length ? job.comparisons[0].tolerance.provenance : {};
  $("tolerance-box").innerHTML =
    "<ul>" + Object.entries(t).map(([k, v]) =>
      `<li><code>${k}</code>：${v}</li>`).join("") + "</ul>";
}

function attachImageClickHandlers(job) {
  document.querySelectorAll("#device-results img").forEach((img) => {
    img.onclick = (e) => inspectPixel(e, img, job);
  });
}

async function inspectPixel(e, img, job) {
  const rect = img.getBoundingClientRect();
  const x = Math.floor(((e.clientX - rect.left) / rect.width) * job.config.width);
  const y = Math.floor(((e.clientY - rect.top) / rect.height) * job.config.height);
  $("inspector-content").innerHTML = "<p class='hint'>请求该像素真实输入…</p>";
  const traceJob = await fetchJson("/api/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scene: job.scene.key,
      devices: ["cpu_ref", "vector_f32"],
      width: job.config.width,
      height: job.config.height,
      trace_pixel: [x, y],
    }),
  });
  const pt = traceJob.pixel_trace;
  const rows = Object.entries(pt.paths).map(([name, p]) => `
    <h4>${name}</h4>
    <pre>覆盖=${p.covered}  对象id=${p.obj_id}  深度=${p.depth}
sRGB=${JSON.stringify(p.final_srgb)}
线性HDR=${JSON.stringify(p.final_linear)}
逐三角形判定:
${p.trace.map((t) => "  三角形#" + t.triangle + " obj=" + t.obj_id +
  " inside=" + t.inside + " 边权重=[" + t.weights.map((w) => w.toExponential(4)).join(", ") + "]" +
  (t.depth_candidate !== undefined ? " 候选深度=" + t.depth_candidate : "") +
  (t.linear !== undefined ? " 着色线性=" + JSON.stringify(t.linear.map((v) => +v.toFixed(5))) : "")).join("\n")}</pre>`);
  $("inspector-content").innerHTML =
    `<p>像素坐标 (x=${pt.pixel[0]}, y=${pt.pixel[1]})，线性索引 ${pt.index}</p>` +
    rows.join("");
}

init().catch((e) => { $("run-status").textContent = "初始化失败：" + e.message; });
