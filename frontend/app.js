// Multi-device render verification panel.
// Every image/stat here is fetched from the backend at click time; this file
// contains no rendered pixels or canned error numbers.
const $ = (id) => document.getElementById(id);
let state = {
  devices: [], scenes: [], profiles: [], defaults: {}, precisionConfigs: {},
  lastJob: null, res: 128, perDevice: {}, preset: "precision",
};

const STAGE_LABEL = {
  non_finite: "非有限值",
  coverage: "覆盖",
  object_id: "对象标识",
  depth: "深度",
  texture_sampling: "纹理采样",
  shading: "光照/着色",
  blend_write: "写回量化",
  color_mapping: "颜色映射",
};

const PRESETS = {
  precision: { cpu_ref: "reference_f64", vector_f32: "strict_f32" },
  strict: { cpu_ref: "reference_f64", vector_f32: "strict_f32" },
  fast: { cpu_ref: "reference_f64", vector_f32: "fast_math_f32" },
  degraded: {
    cpu_ref: "reference_f64",
    vector_f32: "degraded_rgba8_fixed8_f32",
  },
};

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function b64Png(b64) {
  return "data:image/png;base64," + b64;
}

async function fetchJson(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.status);
  return res.json();
}

async function init() {
  const [scenes, dev, prof] = await Promise.all([
    fetchJson("/api/scenes"),
    fetchJson("/api/devices"),
    fetchJson("/api/profiles"),
  ]);
  state.scenes = scenes.scenes;
  state.devices = dev.devices;
  state.profiles = prof.profiles;
  state.defaults = prof.defaults;
  state.precisionConfigs = prof.precision_configs;
  state.rulesVersion = prof.rules_version;
  state.devices.forEach((d) => {
    state.perDevice[d.id] = state.defaults[d.id] || "reference_f64";
  });

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

  buildDeviceRows();
  applyPreset("precision");
  $("preset-select").onchange = (e) => applyPreset(e.target.value);

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

function buildDeviceRows() {
  const list = $("device-list");
  list.innerHTML = "";
  state.devices.forEach((d) => {
    const row = document.createElement("div");
    row.className = "device-row" + (d.available ? "" : " unavailable");
    const selectable = state.profiles.filter(
      (p) => !p.diagnostic_only && profileSupported(d, p));
    const options = selectable.map(
      (p) => `<option value="${p.id}" ${state.perDevice[d.id] === p.id ? "selected" : ""}>${esc(p.label)}</option>`).join("");
    row.innerHTML =
      `<label class="pick"><input type="checkbox" value="${d.id}" ` +
      `${d.available ? "checked" : "disabled"}>` +
      `<span>${esc(d.label)}</span></label>` +
      `<span class="meta">[${d.precision}]</span>` +
      `<select data-device="${d.id}" class="profile-select" ` +
      `${d.available && selectable.length ? "" : "disabled"}>${options}` +
      `</select>`;
    row.title = d.executed_note;
    list.appendChild(row);
  });
  list.querySelectorAll("select[data-device]").forEach((s) => {
    s.onchange = (e) => { state.perDevice[s.dataset.device] = e.target.value; };
  });
}

function profileSupported(device, profile) {
  const caps = device.capabilities || {};
  return ["precision", "fast_math", "texture_format", "blend_capability",
          "depth_convention", "depth_test"].every(
    (f) => (caps[f] || []).includes(profile[f]));
}

function applyPreset(name) {
  state.preset = name;
  $("preset-select").value = name;
  const map = PRESETS[name];
  if (!map) return; // manual
  document.querySelectorAll(".device-pick input[type=checkbox]").forEach((cb) => {
    if (cb.disabled) return;
    cb.checked = Object.prototype.hasOwnProperty.call(map, cb.value);
  });
  Object.entries(map).forEach(([dev, pid]) => {
    state.perDevice[dev] = pid;
    const sel = document.querySelector(`select[data-device="${dev}"]`);
    if (sel) sel.value = pid;
  });
}

function selectedDevices() {
  return [...document.querySelectorAll(".device-pick input[type=checkbox]:checked")]
    .map((i) => i.value);
}

function selectedProfiles() {
  const out = {};
  selectedDevices().forEach((d) => {
    if (state.perDevice[d]) out[d] = state.perDevice[d];
  });
  return out;
}

async function runRender() {
  const btn = $("render-btn");
  btn.disabled = true;
  $("run-status").textContent = "后端真实计算中…";
  const devices = selectedDevices();
  if (!devices.length) {
    $("run-status").textContent = "请至少选择一个设备。";
    btn.disabled = false;
    return;
  }
  try {
    const job = await fetchJson("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scene: $("scene-select").value,
        devices,
        profiles: selectedProfiles(),
        width: state.res,
        height: state.res,
      }),
    });
    state.lastJob = job;
    renderResults(job);
    $("run-status").textContent =
      `完成：执行 ${job.executed.join(", ") || "无"}` +
      (job.skipped.length ? `；未执行 ${job.skipped.join(", ")}（无伪造输出）` : "") +
      (job.unsupported.length
        ? `；配置不支持 ${job.unsupported.map((u) => u.device + "/" + u.profile).join(", ")}`
        : "");
  } catch (err) {
    $("run-status").textContent = "渲染失败：" + err.message;
  } finally {
    btn.disabled = false;
  }
}

function renderResults(job) {
  $("result-panel").classList.remove("hidden");
  renderRunStatus(job);
  renderVerdict(job);
  renderDeviceImages(job);
  renderDiagnostics(job);
  renderAttribution(job);
  renderAcceptance(job);
  renderStats(job);
  renderTolerance(job);
  renderUnverified(job);
  $("config-box").textContent = JSON.stringify({
    config: job.config,
    rules_version: job.rules_version,
    rules_changelog: job.rules_changelog,
    profiles: Object.fromEntries(
      Object.entries(job.profiles).map(([k, v]) => [k, {
        id: v.id, sha256: v.sha256, precision: v.precision,
        fast_math: v.fast_math, texture_format: v.texture_format,
        blend_capability: v.blend_capability,
        depth_convention: v.depth_convention, depth_test: v.depth_test,
      }])),
  }, null, 2);
  attachImageClickHandlers(job);
}

// ---- run status: execution + per-device image/error health -----------------
function renderRunStatus(job) {
  const s = $("run-summary");
  const chips = job.devices.map((d) => {
    let state_, cls;
    if (job.executed.includes(d.id)) { state_ = "已执行"; cls = "ok"; }
    else if (job.unsupported.some((u) => u.device === d.id)) {
      state_ = "配置不支持"; cls = "bad";
    } else if (job.skipped.includes(d.id)) { state_ = "未执行"; cls = "off"; }
    else { state_ = "未选择"; cls = "off"; }
    const prof = job.profiles[d.id];
    return `<span class="chip ${cls}">${esc(d.id)} · ${state_}` +
      (prof ? ` · ${esc(prof.id)}` : "") + `</span>`;
  });
  s.innerHTML = `<strong>运行状态</strong> ${chips.join(" ")}` +
    ` <span class="meta">规则版本 ${esc(job.rules_version)}</span>`;

  // issues: unsupported configs and any non-finite output on an executed path
  const issues = [];
  job.unsupported.forEach((u) => issues.push({
    cls: "bad",
    text: `设备 ${u.device} 的配置 ${u.profile} 未执行：${u.reason}`,
  }));
  job.comparisons.forEach((c) => {
    if (c.nonfinite.count > 0) {
      issues.push({
        cls: "bad",
        text: `对照 ${c.reference}→${c.test} 检测到 ${c.nonfinite.count} 个非有限值。`,
      });
    }
    c.acceptance.failures.forEach((f) => {
      issues.push({
        cls: "warn",
        text: `对照 ${c.reference}→${c.test} · 操作 ${f.operation} · ` +
          `${f.class} · ${f.count} 像素`,
      });
    });
  });
  // per-device image health: a device marked executed must have produced
  // images and no non-finite channel values.
  job.devices.forEach((d) => {
    if (!job.executed.includes(d.id)) return;
    const r = job.renders[d.id];
    if (!r) {
      issues.push({ cls: "bad", text: `设备 ${d.id} 标记为已执行却没有图像。` });
      return;
    }
    // A rendered image that decodes but is entirely background is also a
    // defect (nothing was drawn), separate from a missing file.
    const missing = ["srgb_png", "linear_png", "depth_png", "id_png"]
      .filter((k) => !r.images[k]);
    if (missing.length) {
      issues.push({ cls: "bad",
        text: `设备 ${d.id} 图像缺失：${missing.join(", ")}` });
    } else if (r.covered_count === 0) {
      issues.push({ cls: "warn", text: `设备 ${d.id} 未覆盖任何模型像素。` });
    }
  });
  const nonfiniteDev = job.comparisons
    .filter((c) => c.nonfinite.count > 0)
    .map((c) => c.test);
  if (nonfiniteDev.length) {
    issues.push({ cls: "bad",
      text: `以下设备输出存在非有限值：${[...new Set(nonfiniteDev)].join(", ")}` });
  }
  const box = $("run-issues");
  if (!issues.length) {
    box.innerHTML = `<span class="chip ok">各设备图像与误差通道无异常</span>`;
  } else {
    box.innerHTML = issues.map(
      (i) => `<div class="issue ${i.cls}">${esc(i.text)}</div>`).join("");
  }
}

function renderVerdict(job) {
  const box = $("verdict-box");
  if (!job.comparisons.length) {
    box.className = "verdict";
    box.textContent = "可比较的两条路径未同时执行，无法给出对照结论。";
    return;
  }
  box.className = "verdict";
  box.innerHTML = job.comparisons.map((c) => {
    const v = c.verdict;
    const cls = v.pass ? "pass" : "fail";
    return `<div class="verdict-line ${cls}">` +
      `<span class="tag">${c.reference} → ${c.test}</span>` +
      `<strong>${v.title}</strong>` +
      (v.primary_class ? ` <span class="pill gpu-off">${v.primary_class}</span>` : "") +
      (v.pass ? ' <span class="pill gpu-on">通过</span>'
              : ' <span class="pill gpu-off">需排查</span>') +
      `<div class="meta">${esc(v.description)}</div></div>`;
  }).join("");
}

function card(title, imgB64, meta, dataKey, pxW) {
  const el = document.createElement("div");
  el.className = "card";
  el.dataset.key = dataKey;
  el.dataset.width = pxW;
  el.innerHTML =
    `<h4>${title}</h4>` +
    (imgB64 ? `<img alt="${esc(title)}" src="${b64Png(imgB64)}">` : "<p>无图像</p>") +
    (meta ? `<p class="meta">${esc(meta)}</p>` : "");
  return el;
}

function renderDeviceImages(job) {
  const grid = $("device-results");
  grid.innerHTML = "";
  Object.values(job.renders).forEach((r) => {
    const meta =
      `${r.precision} · 配置 ${r.profile.id} · 耗时 ${r.elapsed_ms} ms · 覆盖 ${r.covered_count} px`;
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
  job.comparisons.forEach((c, idx) => {
    const items = [
      ["颜色绝对误差热图", c.images.color_heatmap_png, "模型区每像素最大通道绝对误差"],
      ["相对误差热图", c.images.relative_heatmap_png, "暗部以下限 eps 归一，封顶 2%"],
      ["深度绝对误差热图", c.images.depth_heatmap_png, "NDC 深度差"],
      ["边缘分布", c.images.edge_png, "紫色为模型边缘带，灰为内部"],
      ["覆盖差异图", c.images.coverage_diff_png, "红=参考独有, 蓝=被测独有, 灰=共有"],
      ["归因阶段图", c.images.stage_png,
        "按首个差异阶段着色：红=覆盖, 橙=对象, 黄=深度, 绿=纹理采样, " +
        "蓝=光照/着色, 橙黄=写回量化, 紫=颜色映射, 灰=一致"],
    ];
    const head = document.createElement("div");
    head.className = "grid-section";
    head.innerHTML = `<h4>${esc(c.reference)} → ${esc(c.test)}</h4>`;
    grid.appendChild(head);
    items.forEach(([t, img, m]) =>
      grid.appendChild(card(t, img, m, "cmp:" + c.test, job.config.width)));
  });
}

function fmt(x) {
  if (typeof x !== "number") return x;
  return Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(3) : x.toFixed(6).replace(/0+$/, "");
}

function renderAttribution(job) {
  const box = $("attribution-box");
  if (!job.comparisons.length) { box.innerHTML = ""; return; }
  box.innerHTML = job.comparisons.map((c) => {
    const a = c.attribution;
    const stageRows = Object.entries(a.stages)
      .filter(([, v]) => v.count > 0)
      .map(([k, v]) =>
        `<tr><td>${STAGE_LABEL[k] || k}</td><td>${v.count}</td>` +
        `<td>${v.interior_count}</td><td>${v.edge_count}</td>` +
        `<td class="left">${esc(v.provenance)}</td></tr>`).join("");
    const sampleRows = a.samples.slice(0, 6).map((s) =>
      `<tr><td>(${s.x},${s.y})</td><td>${STAGE_LABEL[s.stage] || s.stage}</td>` +
      `<td>${s.ref.covered ? "✓" : "—"}/${s.test.covered ? "✓" : "—"}</td>` +
      `<td>${fmt(s.ref.depth)} / ${fmt(s.test.depth)}</td>` +
      `<td>${JSON.stringify(s.ref.albedo.map(fmt))} / ${JSON.stringify(s.test.albedo.map(fmt))}</td>` +
      `<td>${JSON.stringify(s.ref.linear.map(fmt))} / ${JSON.stringify(s.test.linear.map(fmt))}</td>` +
      `<td>${fmt(s.ref.srgb[0])} / ${fmt(s.test.srgb[0])}</td></tr>`).join("");
    const w = a.repro_window;
    return `<div class="attr-block">
      <h4>${esc(c.reference)} → ${esc(c.test)}</h4>
      <p>首个差异阶段：<strong>${a.first_diverging_stage
        ? (STAGE_LABEL[a.first_diverging_stage] || a.first_diverging_stage)
        : "无（逐像素一致）"}</strong> ·
        最小复现窗口 (x ${w.x0}–${w.x1}, y ${w.y0}–${w.y1}) ·
        贡献像素 ${a.contributing_pixels}</p>
      <p class="meta">复现参数：${esc(a.repro_command.cli)}</p>
      <p class="meta">${esc(a.propagation)}</p>
      ${stageRows ? `<table><thead><tr><th>阶段</th><th>像素数</th><th>内部</th>
        <th>边缘</th><th>含义</th></tr></thead><tbody>${stageRows}</tbody></table>` : ""}
      ${sampleRows ? `<table class="samples"><thead><tr><th>像素</th><th>阶段</th>
        <th>覆盖 ref/test</th><th>深度 ref/test</th><th>反照率 ref/test</th>
        <th>线性 HDR ref/test</th><th>sRGB R ref/test</th></tr></thead>
        <tbody>${sampleRows}</tbody></table>` : ""}
    </div>`;
  }).join("");
}

function renderAcceptance(job) {
  const box = $("acceptance-box");
  if (!job.comparisons.length) { box.innerHTML = ""; return; }
  box.innerHTML = job.comparisons.map((c) => {
    const a = c.acceptance;
    const failSet = new Set(a.failures.map((f) => f.operation));
    const rows = Object.entries(a.rules).map(([op, r]) => {
      const limit = r.allowed_abs_internal !== undefined
        ? `abs≤${fmt(r.allowed_abs_internal)} (内) / ${fmt(r.allowed_abs_edge)} (边)`
        : r.allowed_abs !== undefined ? `abs≤${fmt(r.allowed_abs)}`
        : r.allowed_rel !== undefined ? `rel≤${r.allowed_rel}`
        : r.allowed_disagreement !== undefined ? "0 不允"
        : r.allowed_mismatch !== undefined ? "0 不允"
        : r.allowed_nonfinite !== undefined ? `非有限≤${r.allowed_nonfinite}`
        : "—";
      return `<tr class="${failSet.has(op) ? "breach" : ""}">
        <td>${op}</td><td>${r.failure_class}</td><td>${limit}</td>
        <td class="left">${r.effective_pixels}</td>
        <td class="left">${esc(r.provenance)}</td></tr>`;
    }).join("");
    const swap = a.channel_swap_probe;
    const dprobe = a.depth_convention_probe;
    return `<div class="attr-block">
      <h4>${esc(c.reference)} → ${esc(c.test)} · 规则版本 ${esc(a.rules_version)}</h4>
      <p class="meta">${esc(a.decision_basis)}</p>
      <table><thead><tr><th>操作</th><th>失败类</th><th>允许偏差</th>
        <th>有效像素</th><th>来源</th></tr></thead><tbody>${rows}</tbody></table>
      <p>通道交换探测：${swap.triggered ? "<strong>触发</strong>" : "未触发"} ·
         ${esc(swap.reason)}</p>
      <p>深度约定探测：${dprobe.triggered ? "<strong>触发</strong>" : "未触发"} ·
         ${esc(dprobe.reason)}</p>
      ${a.failures.length ? `<ul class="fail-list">` + a.failures.map(
        (f) => `<li><span class="pill gpu-off">${esc(f.class)}</span> ${esc(f.operation)} ·
           ${f.count} 像素</li>`).join("") + `</ul>`
        : `<p class="chip ok">全部分项通过</p>`}
    </div>`;
  }).join("");
}

function renderStats(job) {
  const box = $("stats-table");
  if (!job.comparisons.length) { box.innerHTML = ""; return; }
  box.innerHTML = job.comparisons.map((c) => {
    const s = c.stats;
    const row = (label, a) =>
      `<tr><td>${label}</td><td>${fmt(a.mean)}</td><td>${fmt(a.max)}</td>` +
      `<td>${fmt(a.p99)}</td><td>${fmt(a.rmse)}</td><td>${a.count}</td></tr>`;
    return `<div class="attr-block"><h4>${esc(c.reference)} → ${esc(c.test)}</h4>
    <table>
      <thead><tr><th>逐像素 v1 指标（保留；不单独决定通过）</th><th>均值</th>
        <th>最大</th><th>P99</th><th>RMSE</th><th>样本数</th></tr></thead>
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
      非有限值 ${s.nonfinite.count}</p></div>`;
  }).join("");
}

function renderTolerance(job) {
  const t = job.comparisons.length ? job.comparisons[0].tolerance.provenance : {};
  const ops = job.comparisons.length ? job.comparisons[0].acceptance.rules : {};
  $("tolerance-box").innerHTML =
    "<ul>" + Object.entries(t).map(([k, v]) =>
      `<li><code>${esc(k)}</code>：${esc(v)}</li>`).join("") +
    Object.entries(ops).map(([k, v]) =>
      `<li><code>${esc(k)}</code>：${esc(v.provenance)}</li>`).join("") + "</ul>";
}

function renderUnverified(job) {
  const box = $("unverified-box");
  const groups = {};
  job.unverified.forEach((u) => {
    const key = u.reason;
    (groups[key] = groups[key] || []).push(u);
  });
  if (!job.unverified.length) {
    box.innerHTML = `<p class="chip ok">本次已覆盖所有设备 × 配置组合。</p>`;
    return;
  }
  box.innerHTML = `<p class="meta">以下组合本次<strong>没有真实执行</strong>，
    因此不构成“已验证”。绿色只代表已执行的组合。</p>` +
    Object.entries(groups).map(([reason, items]) => {
      const runs = items.map((u) =>
        `<li>${u.device} × ${u.profile}` +
        (u.missing_fields.length
          ? ` <span class="pill gpu-off">缺 ${esc(u.missing_fields.join(","))}</span>` : "") +
        ` — ${u.supported_by_device ? "设备支持但未运行" : "设备未声明该能力"}</li>`).join("");
      return `<div class="attr-block"><h4>${esc(reason)}</h4><ul>${runs}</ul></div>`;
    }).join("");
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
      devices: job.executed,
      profiles: Object.fromEntries(
        Object.entries(job.profiles).map(([d, p]) => [d, p.id])),
      width: job.config.width,
      height: job.config.height,
      trace_pixel: [x, y],
    }),
  });
  const pt = traceJob.pixel_trace;
  const rows = Object.entries(pt.paths).map(([name, p]) => `
    <h4>${name} <span class="meta">${p.profile_id || ""}</span></h4>
    <pre>覆盖=${p.covered}  对象id=${p.obj_id}  深度=${p.depth}
sRGB=${JSON.stringify(p.final_srgb)}
线性HDR=${JSON.stringify(p.final_linear)}
反照率=${JSON.stringify(p.final_albedo)}
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
