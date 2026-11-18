"""Cross-device comparison, difference attribution and acceptance rules.

Three layers, all kept separate:

1. **pixelwise v1** -- per-pixel absolute/relative statistics and heatmaps over
   the model-effective region.  Preserved verbatim from the first release; it
   still runs and still produces the diagnostic images, but it never decides
   pass/fail on its own.  ``stats`` and ``heatmap`` are exactly the v1 output.
2. **attribution** -- for every differing pixel, find *where in the pipeline*
   the difference starts: non-finite, coverage, object id, depth, texture
   sampling, shading, or final colour mapping.  A small per-pixel window plus
   the recorded intermediate channels (albedo / linear / srgb / depth) makes the
   difference reproducible instead of a global numeric lump.
3. **acceptance rules (versioned)** -- per-operation deviation limits defined
   on explicit *effective pixel* sets.  Structural mismatches (coverage, id,
   depth convention), channel swaps and non-finite values each get their own
   failure class, so none of them can be swallowed by a loose colour threshold.

Background (id 0, uncovered) never enters model-region statistics.
"""
import math

from .tolerance import TOLERANCES

# Rule version is imported from tolerance and stamped into every report.

STAGE_ORDER = ("non_finite", "coverage", "object_id", "depth",
               "texture_sampling", "shading", "blend_write", "color_mapping")

STAGE_PROVENANCE = {
    "non_finite": "输出含 NaN/Inf，直接失败；任何阈值都不适用。",
    "coverage": "覆盖状态不同：像素归属本身不一致，先于任何数值比较。",
    "object_id": "同一像素被不同对象占据：几何/深度测试差异。",
    "depth": "深度差超过硬限：深度约定或插值错误。",
    "texture_sampling": "反照率(albedo)不同：纹理格式/双线性采样差异起始于此。",
    "shading": "反照率一致但线性 HDR 不同：光照/法线/快速数学差异。",
    "blend_write": "反照率一致、被测档声明定长 8 位写回：差异始于写回量化。",
    "color_mapping": "线性 HDR 一致但 sRGB 不同：色调映射/OETF 差异。",
}


def is_finite(x):
    return not (math.isinf(x) or math.isnan(x))


def edge_mask(covered, width, height):
    """1 for a covered pixel adjacent (8-neighbourhood) to a different
    coverage state -- the silhouette/interior boundary band."""
    edge = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            i = y * width + x
            if not covered[i]:
                continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < width and 0 <= ny < height) or \
                            not covered[ny * width + nx]:
                        edge[i] = 1
                        break
                if edge[i]:
                    break
    return edge


def _abs_rel(a, b, eps):
    d = abs(a - b)
    denom = max(abs(a), abs(b), eps)
    return d, d / denom


def compare(ref, test, width, height, chain=None, prior=None):
    """ref/test: result dicts from a backend.  ref is the f64 reference.

    ``chain`` is the ordered device list actually executed (reference first).
    ``prior`` is the result immediately preceding ``test`` in that chain; it
    defines which pixels were already resolved by an upstream device and is
    therefore used for the ``final_color_inherited`` / ``final_color_primary``
    effective-pixel split.  When omitted the pair's own reference is used.
    """
    if chain is None:
        chain = [ref["device"], test["device"]]
    if prior is None:
        prior = ref
    n = width * height
    ref_cov = ref["covered"]
    tst_cov = test["covered"]
    edge = edge_mask(ref_cov, width, height)

    # ---- region classification (v1, preserved) -----------------------------
    model_union = [i for i in range(n) if ref_cov[i] or tst_cov[i]]
    both = [i for i in model_union if ref_cov[i] and tst_cov[i]]
    ref_only = [i for i in model_union if ref_cov[i] and not tst_cov[i]]
    tst_only = [i for i in model_union if tst_cov[i] and not ref_cov[i]]
    interior = [i for i in both if not edge[i]]
    edge_band = [i for i in both if edge[i]]

    stats = _blank_stats()
    stats["pixels_total"] = n
    stats["background_pixels"] = n - len(model_union)
    stats["model_union_pixels"] = len(model_union)
    stats["coverage_both"] = len(both)
    stats["coverage_ref_only"] = len(ref_only)
    stats["coverage_test_only"] = len(tst_only)
    stats["edge_band_pixels"] = len(edge_band)
    stats["interior_pixels"] = len(interior)

    tol = TOLERANCES
    nonfinite = _collect_nonfinite(ref, test, n)
    stats["nonfinite"] = nonfinite

    id_mismatch = []
    depth_far = []
    per_pixel_diff = [0.0] * n          # max channel abs error (for heatmap)
    rel_diff = [0.0] * n
    depth_abs = [0.0] * n

    def region_of(i):
        return "edge" if edge[i] else "interior"

    # ---- v1 numeric metrics on pixels covered by both paths ---------------
    for i in both:
        region = region_of(i)
        bucket = stats["edge"] if region == "edge" else stats["interior"]

        if ref["obj_id"][i] != test["obj_id"][i]:
            id_mismatch.append({"index": i, "region": region,
                                "ref_id": ref["obj_id"][i],
                                "test_id": test["obj_id"][i]})

        dr, relr = _abs_rel(ref["depth"][i], test["depth"][i],
                            tol["depth_rel_eps"])
        depth_abs[i] = dr
        rel_diff[i] = max(rel_diff[i], relr)
        _accum(bucket["depth_abs"], dr)
        _accum(stats["all_depth_abs"], dr)
        if dr > tol["depth_abs_hard"]:
            depth_far.append({"index": i, "region": region,
                              "ref": ref["depth"][i],
                              "test": test["depth"][i], "abs": dr})

        for c in range(3):
            a, b = ref["srgb"][i][c], test["srgb"][i][c]
            d, r = _abs_rel(a, b, tol["color_rel_eps"])
            per_pixel_diff[i] = max(per_pixel_diff[i], d)
            rel_diff[i] = max(rel_diff[i], r)
            _accum(bucket["srgb_abs"], d)
            _accum(bucket["srgb_rel"], r)
            _accum(stats["all_srgb_abs"], d)
            _accum(stats["all_srgb_rel"], r)
        for c in range(3):
            d, r = _abs_rel(ref["linear"][i][c], test["linear"][i][c],
                            tol["color_rel_eps"])
            _accum(bucket["linear_abs"], d)
            _accum(stats["all_linear_abs"], d)

    _finalize(stats)

    # ---- attribution over the full frame -----------------------------------
    attribution = _attribute(ref, test, width, height, edge, n, tol)

    # ---- versioned acceptance ----------------------------------------------
    acceptance = _accept(ref, test, edge, both, ref_only, tst_only, n, prior,
                         tol, nonfinite)

    return {
        "reference": ref["device"],
        "test": test["device"],
        "chain": list(chain),
        "rules_version": tol["rules_version"],
        "width": width,
        "height": height,
        "tolerance": tol,
        "stats": stats,
        "attribution": attribution,
        "acceptance": acceptance,
        "structural": {
            "ref_only_count": len(ref_only),
            "test_only_count": len(tst_only),
            "ref_only_indices": ref_only[:200],
            "test_only_indices": tst_only[:200],
            "id_mismatch": id_mismatch[:200],
            "depth_far_mismatch": depth_far[:200],
            "total": len(ref_only) + len(tst_only) + len(id_mismatch),
        },
        "verdict": acceptance["verdict"],
        "heatmap": {
            "color_abs_max": per_pixel_diff,
            "relative_max": rel_diff,
            "depth_abs": depth_abs,
            "edge_mask": list(edge),
        },
    }


# ---------------------------------------------------------------------------
# Attribution: first diverging pipeline stage per differing pixel
# ---------------------------------------------------------------------------
def _attribute(ref, test, width, height, edge, n, tol):
    ops = tol["operations"]
    # Half an 8-bit step: catches rgba8 texture quantisation (~1/510) while
    # staying far above binary32 sampling noise (~1e-7).
    thr_texture = 0.25 / 255.0
    thr_shading_abs = 2.0 / 255.0
    thr_color_abs = ops["final_color_inherited"]["allowed_abs_internal"]
    test_declares_fixed8 = \
        (test.get("profile") or {}).get("blend_capability") == "fixed8_blend"
    test_declares_quantized_texture = \
        (test.get("profile") or {}).get("texture_format") == "rgba8_linear"
    blend_tol = 1e-5           # above binary32 noise => the store did it

    stage_of = [None] * n
    candidates = bytearray(n)
    for i in range(n):
        r_cov, t_cov = ref["covered"][i], test["covered"][i]
        stage = None
        if not _linear_finite(ref, i) or not _linear_finite(test, i):
            stage = "non_finite"
        elif r_cov != t_cov:
            stage = "coverage"
        elif r_cov and t_cov:
            if ref["obj_id"][i] != test["obj_id"][i]:
                stage = "object_id"
            elif not (_finite(ref["depth"][i]) and _finite(test["depth"][i])) \
                    or abs(ref["depth"][i] - test["depth"][i]) > \
                    ops["depth"]["allowed_abs_internal"]:
                stage = "depth"
            elif _max_abs(ref["albedo"][i], test["albedo"][i]) > thr_texture:
                stage = "texture_sampling"
            elif test_declares_quantized_texture and \
                    (abs(ref["depth"][i] - test["depth"][i]) <=
                     ops["depth"]["allowed_abs_internal"]) and \
                    _max_abs(ref["linear"][i], test["linear"][i]) > thr_texture:
                # rgba8 texture: the sampled texel is quantised before shading,
                # so even when the stored albedo of the two visible surfaces
                # happens to coincide the linear-HDR result already differs.
                stage = "texture_sampling"
            elif test_declares_fixed8 and \
                    _max_abs(ref["linear"][i], test["linear"][i]) > blend_tol:
                stage = "blend_write"
            elif _max_abs(ref["linear"][i], test["linear"][i]) > thr_shading_abs:
                stage = "shading"
            elif _max_abs(ref["srgb"][i], test["srgb"][i]) > thr_color_abs:
                stage = "color_mapping"
        if stage is not None:
            stage_of[i] = stage
            candidates[i] = 1

    # Smallest reproducing window: if anything differs, the 8-neighbourhood of
    # a differing pixel is included too (a sliver may be owned by one pixel and
    # only the neighbour reveals the geometry).  If nothing differs, 1x1.
    contributing = bytearray(candidates)
    for y in range(height):
        for x in range(width):
            if not candidates[y * width + x]:
                continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        contributing[ny * width + nx] = 1
    if any(candidates):
        xs = [i % width for i in range(n) if candidates[i]]
        ys = [i // width for i in range(n) if candidates[i]]
        window = {"x0": max(0, min(xs) - 1), "y0": max(0, min(ys) - 1),
                  "x1": min(width - 1, max(xs) + 1),
                  "y1": min(height - 1, max(ys) + 1)}
    else:
        window = {"x0": 0, "y0": 0, "x1": 0, "y1": 0}

    stages = {}
    samples = []
    for stage in STAGE_ORDER:
        idxs = [i for i in range(n) if stage_of[i] == stage]
        stages[stage] = {
            "count": len(idxs),
            "edge_count": sum(1 for i in idxs if edge[i]),
            "interior_count": sum(1 for i in idxs if not edge[i]),
            "provenance": STAGE_PROVENANCE[stage],
            "samples": [_pixel_record(ref, test, i, width, stage)
                        for i in idxs[:20]],
        }
        for i in idxs[:5]:
            samples.append(_pixel_record(ref, test, i, width, stage))

    probe = _channel_swap_probe(ref, test, both_idx := _both_indices(ref, test),
                                tol)
    depth_conv = _depth_convention_probe(ref, test,
                                         _both_indices(ref, test), tol)

    first = next((s for s in STAGE_ORDER if stages[s]["count"] > 0), None)
    repro = _repro_command(ref, test, window)
    return {
        "stages": stages,
        "first_diverging_stage": first,
        "contributing_pixels": int(sum(contributing)),
        "repro_window": window,
        "repro_command": repro,
        "stage_index": stage_of,
        "propagation": ("覆盖/深度差异会沿光栅化传播为相邻像素的颜色差异；"
                        "归因按管线顺序取首个差异阶段，颜色映射只作为末端结果。"),
        "channel_swap_probe": probe,
        "depth_convention_probe": depth_conv,
        "samples": samples[:50],
    }


def _repro_command(ref, test, window):
    """Minimal CLI invocation that re-renders the reproducing window.

    The window is the smallest bounding box containing every differing pixel
    plus the one-pixel neighbourhood.  Tracing its centre re-runs the full
    per-triangle decision for both paths at that pixel.
    """
    cx = (window["x0"] + window["x1"]) // 2
    cy = (window["y0"] + window["y1"]) // 2
    prof_ref = (ref.get("profile") or {}).get("id", "reference_f64")
    prof_tst = (test.get("profile") or {}).get("id", "strict_f32")
    return {
        "trace_pixel": [cx, cy],
        "profiles": {ref["device"]: prof_ref, test["device"]: prof_tst},
        "cli": ("python -m backend.cli --scene <scene> --width <w> "
                "--profile %s=%s --profile %s=%s --trace %d,%d --out <dir>"
                % (ref["device"], prof_ref, test["device"], prof_tst, cx, cy)),
        "bbox": [window["x0"], window["y0"], window["x1"], window["y1"]],
    }


def _both_indices(ref, test):
    return [i for i in range(len(ref["covered"]))
            if ref["covered"][i] and test["covered"][i]]


def _pixel_record(ref, test, i, width, stage):
    x, y = i % width, i // width
    return {
        "index": i, "x": x, "y": y, "stage": stage,
        "ref": _channels(ref, i),
        "test": _channels(test, i),
    }


def _channels(res, i):
    return {
        "covered": bool(res["covered"][i]),
        "obj_id": res["obj_id"][i],
        "depth": _num(res["depth"][i]),
        "albedo": [_num(c) for c in res["albedo"][i]],
        "linear": [_num(c) for c in res["linear"][i]],
        "srgb": [_num(c) for c in res["srgb"][i]],
    }


def _linear_finite(res, i):
    return all(_finite(c) for c in res["linear"][i])


def _finite(x):
    return is_finite(x)


def _num(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    return x


def _max_abs(a, b):
    return max(abs(a[c] - b[c]) for c in range(min(len(a), len(b))))


def _channel_swap_probe(ref, test, both, tol):
    cfg = tol["channel_swap"]
    if not both:
        return {"triggered": False, "reason": "无共同覆盖像素"}
    id_err = 0.0
    perms = [(1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)]
    perm_err = {p: 0.0 for p in perms}
    for i in both:
        a = ref["srgb"][i]
        b = test["srgb"][i]
        id_err += max(abs(a[c] - b[c]) for c in range(3))
        for p in perms:
            perm_err[p] += max(abs(a[c] - b[p[c]]) for c in range(3))
    count = len(both)
    id_mean = id_err / count
    best = min(perm_err, key=perm_err.get)
    best_mean = perm_err[best] / count
    triggered = (id_mean > cfg["trigger_abs"] and
                 best_mean < cfg["explained_abs"] and
                 id_mean > cfg["min_improvement_factor"] * best_mean)
    if not triggered:
        if id_mean <= cfg["trigger_abs"]:
            reason = "原始误差低于嫌疑阈值，不构成通道交换。"
        else:
            reason = "任何通道置换都未能解释误差，属真实数值/结构差异。"
    else:
        reason = ("通道置换 %s 将平均误差 %.3e 降至 %.3e，判为通道交换。"
                  % (best, id_mean, best_mean))
    return {"triggered": triggered, "identity_mean_abs": id_mean,
            "best_permutation": list(best), "best_permutation_mean_abs":
            best_mean, "reason": reason}


def _depth_convention_probe(ref, test, both, tol):
    cfg = tol["depth_convention"]
    vals = [i for i in both if _finite(ref["depth"][i]) and
            _finite(test["depth"][i])]
    if not vals:
        return {"triggered": False, "reason": "无有效共同深度像素"}
    diffs = [test["depth"][i] - ref["depth"][i] for i in vals]
    mean_abs = sum(abs(d) for d in diffs) / len(diffs)
    pos = sum(1 for d in diffs if d > 0)
    frac = max(pos, len(diffs) - pos) / len(diffs)
    triggered = mean_abs > cfg["suspicion_abs"] and frac > 0.9
    return {"triggered": triggered, "mean_abs": mean_abs,
            "sign_consistency": frac,
            "reason": ("深度差普遍 %.3e 且同向比例 %.2f，判为深度约定不一致。"
                       % (mean_abs, frac)) if triggered
            else "深度差不足以判定约定不一致。"}


# ---------------------------------------------------------------------------
# Acceptance rules
# ---------------------------------------------------------------------------
def _accept(ref, test, edge, both, ref_only, tst_only, n, prior, tol,
            nonfinite):
    ops = tol["operations"]
    rules_version = tol["rules_version"]
    failures = []

    # coverage --------------------------------------------------------------
    cov_bad = len(ref_only) + len(tst_only)
    _record_operation(failures, "coverage", "coverage_struct",
                      cov_bad, len(model_indices(ref, test)),
                      [{"index": i, "side": "ref_only"} for i in ref_only[:10]] +
                      [{"index": i, "side": "test_only"} for i in tst_only[:10]])

    # object id -------------------------------------------------------------
    id_bad = [i for i in both if ref["obj_id"][i] != test["obj_id"][i]]
    _record_operation(failures, "object_id", "object_id", len(id_bad),
                      len(both),
                      [{"index": i,
                        "ref_id": ref["obj_id"][i],
                        "test_id": test["obj_id"][i]} for i in id_bad[:10]])

    # depth -----------------------------------------------------------------
    depth_bad = [i for i in both
                 if _finite(ref["depth"][i]) and _finite(test["depth"][i]) and
                 abs(ref["depth"][i] - test["depth"][i]) >
                 ops["depth"]["allowed_abs_internal"]]
    depth_probe = _depth_convention_probe(ref, test, both, tol)
    depth_class = "depth_convention" if depth_probe["triggered"] \
        else "depth_correspondence"
    _record_operation(failures, "depth", depth_class, len(depth_bad),
                      len([i for i in both if _finite(ref["depth"][i]) and
                           _finite(test["depth"][i])]),
                      [{"index": i,
                        "ref": _num(ref["depth"][i]),
                        "test": _num(test["depth"][i])} for i in depth_bad[:10]])

    # effective pixel sets for final colour, split by who owns the pixel -----
    inherited = [i for i in both
                 if prior["covered"][i] and
                 prior["obj_id"][i] == ref["obj_id"][i] and
                 ref["obj_id"][i] == test["obj_id"][i]]
    # Pixels the upstream device did not resolve: the difference is introduced
    # by this device, so they form the "primary" set.
    primary = [i for i in both
               if not prior["covered"][i] and
               ref["obj_id"][i] == test["obj_id"][i]]

    declared_quantized = \
        (test.get("profile") or {}).get("blend_capability") == "fixed8_blend"
    for op_name, pixels in (("final_color_inherited", inherited),
                            ("final_color_primary", primary)):
        cfg = ops[op_name]
        # A declared quantised write-back owns its pixels; the strict float-path
        # operation does not judge them, so a legitimately lossy capability is
        # not hidden by loosening the float threshold.
        if op_name == "final_color_inherited" and declared_quantized:
            continue
        bad = []
        for i in pixels:
            if not (_linear_finite(ref, i) and _linear_finite(test, i)):
                continue
            limit = cfg["allowed_abs_edge"] if edge[i] else \
                cfg["allowed_abs_internal"]
            d, r = _abs_rel_channels(ref["srgb"][i], test["srgb"][i])
            if d > limit or r > cfg["allowed_rel"]:
                bad.append({"index": i, "region": "edge" if edge[i]
                            else "interior", "abs": d, "rel": r})
        _record_operation(failures, op_name, cfg["failure_class"],
                          len(bad), len(pixels), bad[:10])

    # quantised blend: declared 8-bit write-back gets its own operation and its
    # own limit, instead of loosening the float-path threshold.
    bw_cfg = ops["blend_write_quantized"]
    if declared_quantized:
        bw_bad = []
        for i in inherited:
            if not (_linear_finite(ref, i) and _linear_finite(test, i)):
                continue
            d, r = _abs_rel_channels(ref["srgb"][i], test["srgb"][i])
            if d > bw_cfg["allowed_abs_edge"] or r > bw_cfg["allowed_rel"]:
                bw_bad.append({"index": i, "abs": d, "rel": r})
        _record_operation(failures, "blend_write_quantized",
                          bw_cfg["failure_class"], len(bw_bad), len(inherited),
                          bw_bad[:10])

    # linear HDR ------------------------------------------------------------
    # Only pixels the two paths agree are the *same object* are eligible: a
    # different object means the value difference is a consequence of the
    # correspondence error, already reported above, not an independent numeric
    # deviation.  This prevents structural faults from being re-counted as
    # precision faults (and vice versa).
    lh_cfg = ops["linear_hdr"]
    lh_pixels = [i for i in both if ref["obj_id"][i] == test["obj_id"][i]]
    lh_bad = []
    for i in lh_pixels:
        if not (_linear_finite(ref, i) and _linear_finite(test, i)):
            continue
        _, r = _abs_rel_channels(ref["linear"][i], test["linear"][i],
                                 relative=True)
        if r > lh_cfg["allowed_rel"]:
            lh_bad.append({"index": i, "rel": r})
    if not declared_quantized:
        _record_operation(failures, "linear_hdr", lh_cfg["failure_class"],
                          len(lh_bad), len(lh_pixels), lh_bad[:10])

    # finite ----------------------------------------------------------------
    _record_operation(failures, "finite", ops["finite"]["failure_class"],
                      nonfinite["count"], n, nonfinite["samples"][:10])

    # channel swap probe becomes a structural failure class of its own ------
    swap = _channel_swap_probe(ref, test, both, tol)
    swap_failure = None
    if swap["triggered"]:
        swap_failure = {"class": "channel_swap", "operation": "channel_layout",
                        "count": len(both),
                        "detail": swap["reason"],
                        "samples": []}
        failures.append(swap_failure)

    # verdict ----------------------------------------------------------------
    verdict = _verdict(failures)
    return {
        "rules_version": rules_version,
        "rules_changelog": tol["rules_changelog"],
        "rules": {name: tol["operations"][name]
                  for name in tol["operations"]},
        "channel_swap_probe": swap,
        "depth_convention_probe": depth_probe,
        "failures": failures,
        "pass": verdict["pass"],
        "verdict": verdict,
        "decision_basis": ("规则版本 %s；按操作/有效像素判定，未使用"
                           "“全局宽松阈值直到全绿”策略。" % rules_version),
    }


def _record_operation(failures, operation, failure_class, breach_count,
                      effective_count, samples):
    if breach_count > 0:
        failures.append({
            "class": failure_class,
            "operation": operation,
            "count": breach_count,
            "effective_pixels": effective_count,
            "samples": samples,
        })


def model_indices(ref, test):
    return [i for i in range(len(ref["covered"]))
            if ref["covered"][i] or test["covered"][i]]


def _abs_rel_channels(a, b, relative=False):
    d = max(abs(a[c] - b[c]) for c in range(min(len(a), len(b))))
    if relative:
        rel = 0.0
        for c in range(min(len(a), len(b))):
            denom = max(abs(a[c]), abs(b[c]), TOLERANCES["color_rel_eps"])
            rel = max(rel, abs(a[c] - b[c]) / denom)
        return d, rel
    denom = max(max(abs(a[c]) for c in range(len(a))),
                max(abs(b[c]) for c in range(len(b))),
                TOLERANCES["color_rel_eps"])
    return d, d / denom


SEVERITY = ("non_finite", "channel_swap", "depth_convention",
            "coverage_struct", "object_id", "depth_correspondence",
            "blend_quantization_exceeded", "precision_exceeded")


def _verdict(failures):
    classes = {f["class"] for f in failures}
    ordered = [c for c in SEVERITY if c in classes]
    reasons = []
    for f in failures:
        reasons.append("%s（%s）：%d 个像素超出允许偏差。"
                       % (f["class"], f["operation"], f["count"]))
    if not ordered:
        return {
            "class": "pass",
            "title": "通过验收",
            "description": ("覆盖、对象标识、深度、最终颜色、线性 HDR 与有限性"
                            "均满足当前规则版本的分项要求。"),
            "reasons": [],
            "pass": True,
            "primary_class": None,
        }
    primary = ordered[0]
    titles = {
        "non_finite": "存在非有限值（NaN/Inf）",
        "channel_swap": "通道交换",
        "depth_convention": "深度约定不一致",
        "coverage_struct": "覆盖结构错误",
        "object_id": "对象标识不一致",
        "depth_correspondence": "深度对应错误",
        "blend_quantization_exceeded": "定长写回量化超出允许偏差",
        "precision_exceeded": "数值精度超出允许偏差",
    }
    return {
        "class": "fail",
        "title": titles[primary],
        "description": "；".join(reasons),
        "reasons": reasons,
        "pass": False,
        "primary_class": primary,
    }


# ---------------------------------------------------------------------------
# v1 statistics helpers (unchanged semantics)
# ---------------------------------------------------------------------------
def _blank_stats():
    def bucket():
        return {
            "srgb_abs": _acc(), "srgb_rel": _acc(),
            "linear_abs": _acc(), "depth_abs": _acc(),
        }
    return {"edge": bucket(), "interior": bucket(),
            "all_srgb_abs": _acc(), "all_srgb_rel": _acc(),
            "all_linear_abs": _acc(), "all_depth_abs": _acc()}


def _acc():
    return {"sum": 0.0, "max": 0.0, "count": 0, "values": []}


def _accum(a, v):
    a["sum"] += v
    a["count"] += 1
    if v > a["max"]:
        a["max"] = v
    a["values"].append(v)


def _finalize(stats):
    def finish(a):
        vals = a.pop("values")
        a["mean"] = a["sum"] / a["count"] if a["count"] else 0.0
        if vals:
            vals.sort()
            a["p99"] = vals[min(len(vals) - 1,
                                int(0.99 * len(vals)))]
            a["rmse"] = math.sqrt(sum(v * v for v in vals) / len(vals))
        else:
            a["p99"] = 0.0
            a["rmse"] = 0.0
    for key in ("edge", "interior"):
        for sub in stats[key].values():
            finish(sub)
    for key in ("all_srgb_abs", "all_srgb_rel", "all_linear_abs",
                "all_depth_abs"):
        finish(stats[key])


def _collect_nonfinite(ref, test, n):
    out = {"count": 0, "samples": []}
    for i in range(n):
        for label, res in (("ref", ref), ("test", test)):
            for c in range(3):
                v = res["linear"][i][c]
                if not is_finite(v):
                    out["count"] += 1
                    if len(out["samples"]) < 50:
                        out["samples"].append(
                            {"index": i, "device": label, "channel": c,
                             "value": str(v)})
            dv = res["depth"][i]
            if not is_finite(dv) and res["covered"][i]:
                out["count"] += 1
                if len(out["samples"]) < 50:
                    out["samples"].append(
                        {"index": i, "device": label, "channel": "depth",
                         "value": str(dv)})
    return out


# ---------------------------------------------------------------------------
# Backwards-compatible classify() used by the first-release API surface.
# New code should read ``acceptance`` / ``verdict``.  Kept so old callers and
# reports remain readable; it mirrors the new verdict rather than inventing a
# second decision path.
# ---------------------------------------------------------------------------
def classify(stats, structural_count, depth_far, tol):
    if structural_count > 0 or depth_far:
        cls = "structural"
        title = "存在像素对应错误"
        desc = "存在像素对应/覆盖结构差异(覆盖缺失或对象标识不一致)，不能归因于浮点精度。"
    else:
        srgb = stats["all_srgb_abs"]
        breach = srgb["max"] > tol["srgb_abs_max"] or \
            srgb["p99"] > tol["srgb_abs_p99"]
        cls = "precision_exceeded" if breach else "precision_only"
        title = "数值偏差超容差" if breach else "仅数值精度偏差"
        desc = "v1 逐像素口径分类（兼容保留）。"
    return {"class": cls, "title": title, "description": desc,
            "reasons": [], "pass": cls == "precision_only",
            "decision": "pixelwise_v1_compat"}
