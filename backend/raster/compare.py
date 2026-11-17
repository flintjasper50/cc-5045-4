"""Cross-device comparison and diagnostics.

Two distinct failure classes are kept separate instead of being lumped into
"hardware difference":

* structural / pixel-correspondence errors -- the two paths disagree about
  *whether* a model pixel exists there (coverage mask, object id, depth far
  beyond precision), e.g. a sliver triangle won or lost on an edge;
* numeric precision deviation -- same pixel covered by the same object, only
  the float value differs within explainable f32/f64 rounding.

Background (id 0, uncovered) never enters the model-region statistics, so a
large black background cannot dilute or hide errors on the actual model.
"""
import math

from .tolerance import TOLERANCES


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


def compare(ref, test, width, height):
    """ref/test: result dicts from a backend.  ref is the f64 reference."""
    n = width * height
    ref_cov = ref["covered"]
    tst_cov = test["covered"]
    edge = edge_mask(ref_cov, width, height)

    # ---- region classification ------------------------------------------------
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

    # ---- numeric metrics on pixels covered by both paths ----------------------
    for i in both:
        region = region_of(i)
        bucket = stats["edge"] if region == "edge" else stats["interior"]

        # object identity must match on shared coverage
        if ref["obj_id"][i] != test["obj_id"][i]:
            id_mismatch.append({"index": i, "region": region,
                                "ref_id": ref["obj_id"][i],
                                "test_id": test["obj_id"][i]})

        # depth
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

        # color: compare final sRGB and linear HDR channels separately
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

    # ---- classification verdict ------------------------------------------------
    structural = len(ref_only) + len(tst_only) + len(id_mismatch)
    verdict = classify(stats, structural, depth_far, tol)

    return {
        "reference": ref["device"],
        "test": test["device"],
        "width": width,
        "height": height,
        "tolerance": tol,
        "stats": stats,
        "structural": {
            "ref_only_count": len(ref_only),
            "test_only_count": len(tst_only),
            "ref_only_indices": ref_only[:200],
            "test_only_indices": tst_only[:200],
            "id_mismatch": id_mismatch[:200],
            "depth_far_mismatch": depth_far[:200],
            "total": structural,
        },
        "verdict": verdict,
        "heatmap": {
            "color_abs_max": per_pixel_diff,
            "relative_max": rel_diff,
            "depth_abs": depth_abs,
            "edge_mask": list(edge),
        },
    }


def classify(stats, structural_count, depth_far, tol):
    """Separate structural correspondence errors from precision deviations."""
    reasons = []
    if structural_count > 0:
        reasons.append("存在像素对应/覆盖结构差异(覆盖缺失或对象标识不一致)，"
                       "不能归因于浮点精度。")
    if depth_far:
        reasons.append("存在远超深度容差的像素，属于深度对应错误而非舍入偏差。")
    srgb = stats["all_srgb_abs"]
    precision_breach = srgb["max"] > tol["srgb_abs_max"] or \
        srgb["p99"] > tol["srgb_abs_p99"]
    if precision_breach:
        reasons.append("共有覆盖区域的颜色误差超出 f32/f64 可解释容差，"
                       "需检查采样或颜色变换是否一致。")

    if not reasons:
        cls = "precision_only"
        title = "仅数值精度偏差"
        desc = ("两路径覆盖与对象标识完全一致，差异处于 f32/f64 舍入可解释范围内，"
                "属正常数值精度偏差。")
    elif structural_count > 0 or depth_far:
        cls = "structural"
        title = "存在像素对应错误"
        desc = "；".join(reasons)
    else:
        cls = "precision_exceeded"
        title = "数值偏差超容差"
        desc = "；".join(reasons)

    return {"class": cls, "title": title, "description": desc,
            "reasons": reasons,
            "pass": cls == "precision_only"}


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
