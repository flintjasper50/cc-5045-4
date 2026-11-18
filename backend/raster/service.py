"""Application service: run real backends, compare, build diagnostic payloads.

Every job resolves a *render profile* per device before execution.  A profile
that declares a capability the device does not support is reported as
``unsupported_capability`` with the offending fields; it is never silently
downgraded.  The executed chain (reference first) is recorded so downstream
reports can tell which pixels an upstream device already resolved.
"""
import base64
import hashlib
import json
import platform
import time

from . import encode
from .backends import BACKENDS, CpuRefPath
from .compare import compare
from .devices import CAPABILITIES, discover_devices, profile_supported
from .geometry import ProceduralTexture
from .profiles import (PROFILE_FIELDS, PROFILES, all_profiles,
                       profile_for_default, resolve_profile)
from .scenes import get_scene, scene_keys
from .tolerance import RULES_CHANGELOG, RULES_VERSION

MAX_RES = 256
DEFAULT_RES = 128

# The two precision configurations delivered for real comparison.  Each maps
# device -> profile and is rendered independently; both paths really run.
PRECISION_CONFIGS = {
    "f64_vs_f32": {"cpu_ref": "reference_f64", "vector_f32": "strict_f32"},
    "f32_vs_f32": {"cpu_ref": "strict_f32", "vector_f32": "strict_f32"},
}


def reproducibility_config(scene_key, width, height):
    """Frozen inputs that make a render reproducible."""
    scene = get_scene(scene_key)
    payload = {
        "scene": scene_key,
        "triangle_count": len(scene["triangles"]),
        "width": width,
        "height": height,
        "camera": scene["camera"],
        "light_dir": scene["light_dir"],
        "exposure": scene["exposure"],
        "texture": "procedural_checker_8x8_linear_wrap",
        "color_pipeline": "linear Blinn-Phong -> Reinhard -> sRGB OETF",
        "coord_system": "right-handed, camera looks -Z, NDC z in [-1,1]",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "fill_convention": "center-sample (x+0.5,y+0.5), top-left >=0",
        "rules_version": RULES_VERSION,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    payload["config_sha256"] = digest
    return payload, scene


def render_one(device_id, scene, width, height, trace_pixel=None,
               profile=None):
    if device_id not in BACKENDS:
        raise ValueError("backend not runnable here: %s" % device_id)
    cls = BACKENDS[device_id]
    tex = ProceduralTexture()
    rast = cls(width, height, tex, trace_pixel=trace_pixel, profile=profile)
    t0 = time.perf_counter()
    result = rast.render(scene)
    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    result["profile_id"] = (profile or {}).get("id")
    return result


def _b64(png_bytes):
    return base64.b64encode(png_bytes).decode("ascii")


def _resolve_requested_profile(device_id, profile_id):
    if profile_id is None:
        profile_id = profile_for_default(device_id)
    profile = resolve_profile(profile_id)
    ok, missing = profile_supported(device_id, profile)
    return profile, ok, missing


def render_job(scene_key, device_ids, width=DEFAULT_RES, height=None,
               trace_pixel=None, profiles=None):
    """Render a scene on the requested devices, each with its resolved profile.

    ``profiles`` maps device_id -> profile_id.  Missing entries fall back to
    the device's declared default.  Returns the job payload consumed by both
    the UI and the CLI.
    """
    width = min(MAX_RES, max(16, int(width)))
    height = height or width
    height = min(MAX_RES, max(16, int(height)))
    config, scene = reproducibility_config(scene_key, width, height)
    devices_meta, gpu_status = discover_devices()
    profiles = profiles or {}

    renders = {}
    raw_results = {}
    executed, skipped = [], []
    unsupported = []
    profile_report = {}
    for dev in device_ids:
        profile_id = profiles.get(dev, profile_for_default(dev))
        try:
            profile = resolve_profile(profile_id)
        except KeyError:
            unsupported.append({"device": dev, "profile": profile_id,
                                "missing_fields": ["profile"],
                                "reason": "未知配置档：%s" % profile_id})
            continue
        profile_report[dev] = profile
        if dev == "gpu_cuda" and not gpu_status["gpu_can_execute"]:
            skipped.append(dev)
            continue
        if dev not in BACKENDS:
            skipped.append(dev)
            continue
        ok, missing = profile_supported(dev, profile)
        if not ok:
            unsupported.append({
                "device": dev, "profile": profile_id,
                "missing_fields": missing,
                "reason": "设备 %s 未声明能力：%s；未执行、未降级替代。"
                          % (dev, ", ".join(missing)),
            })
            continue
        res = render_one(dev, scene, width, height, trace_pixel, profile)
        raw_results[dev] = res
        renders[dev] = _serialize_render(res, width, height)
        executed.append(dev)

    # Executed chain: reference first, then the rest in requested order.  This
    # lets the comparison split inherited vs primary effective pixels.
    chain = _execution_chain(executed)
    comparisons = []
    if len(chain) >= 2:
        ref_id = chain[0]
        for test_id in chain[1:]:
            ref = raw_results[ref_id]
            tst = raw_results[test_id]
            prior = raw_results[chain[chain.index(test_id) - 1]]
            cmp_ = compare(ref, tst, width, height, chain=chain, prior=prior)
            comparisons.append(_serialize_comparison(
                cmp_, ref, tst, width, height, scene_key))

    return {
        "scene": {"key": scene_key, "title": scene["title"],
                  "description": scene["description"]},
        "config": config,
        "rules_version": RULES_VERSION,
        "rules_changelog": RULES_CHANGELOG,
        "gpu_status": gpu_status,
        "devices": devices_meta,
        "profiles": profile_report,
        "executed": executed,
        "skipped": skipped,
        "unsupported": unsupported,
        "chain": chain,
        "renders": renders,
        "comparisons": comparisons,
        "unverified": unverified_combinations(devices_meta, executed, profiles),
        "pixel_trace": _pixel_trace_payload(
            raw_results, width, height, trace_pixel)
        if trace_pixel and raw_results else None,
    }


def _execution_chain(executed):
    """Reference device first, then others in their requested order."""
    if not executed:
        return []
    rest = [d for d in executed if d != CpuRefPath.name]
    if CpuRefPath.name in executed:
        return [CpuRefPath.name] + rest
    return list(executed)


def unverified_combinations(devices_meta, executed, profiles):
    """Profile x device pairs that this environment did *not* actually run.

    The UI must show these explicitly so a green result is never read as
    "every combination is verified".
    """
    out = []
    for dev in devices_meta:
        did = dev["id"]
        caps = dev["capabilities"]
        for pid, raw in PROFILES.items():
            if raw["diagnostic_only"]:
                continue  # never offered by the UI; not a real coverage gap
            supported = True
            missing = []
            for field in PROFILE_FIELDS:
                if raw[field] not in caps.get(field, []):
                    supported = False
                    missing.append(field)
            ran = did in executed and \
                profiles.get(did, profile_for_default(did)) == pid
            if not ran:
                out.append({
                    "device": did,
                    "profile": pid,
                    "supported_by_device": supported,
                    "missing_fields": missing,
                    "reason": ("设备不可用/本次未选择" if not dev["available"]
                               else "本次未执行此组合" if supported
                               else "设备未声明该能力，未验证"),
                })
    return out


def _serialize_render(result, width, height):
    return {
        "device": result["device"],
        "label": result["label"],
        "kind": result["kind"],
        "precision": result["precision"],
        "profile": result["profile"],
        "elapsed_ms": result["elapsed_ms"],
        "covered_count": sum(result["covered"]),
        "images": {
            "srgb_png": _b64(encode.to_bytes_png(
                width, height, result["srgb"])),
            "linear_png": _b64(encode.to_bytes_png(
                width, height, [[min(1.0, c) for c in px]
                                for px in result["linear"]])),
            "depth_png": _b64(encode.depth_png(
                width, height, result["depth"], result["covered"])),
            "id_png": _b64(encode.id_png(
                width, height, result["obj_id"])),
        },
        # raw channels so the pixel inspector shows actual float inputs
        "channels": {
            "srgb": result["srgb"],
            "linear": result["linear"],
            "albedo": result["albedo"],
            "depth": result["depth"],
            "obj_id": result["obj_id"],
            "covered": list(result["covered"]),
        },
    }


def _serialize_comparison(cmp_, ref, tst, width, height, scene_key=None):
    hm = cmp_["heatmap"]
    cov = ref["covered"]
    repro = cmp_["attribution"].get("repro_command")
    if repro and scene_key:
        # The comparison layer does not know the scene/resolution; fill them in
        # here so the emitted command is directly runnable.
        repro["cli"] = repro["cli"].replace("<scene>", scene_key) \
            .replace("<w>", str(width)).replace("<dir>", "reports/repro")
        repro["scene"] = scene_key
        repro["width"] = width
    return {
        "reference": cmp_["reference"],
        "test": cmp_["test"],
        "chain": cmp_["chain"],
        "rules_version": cmp_["rules_version"],
        "verdict": cmp_["verdict"],
        "acceptance": cmp_["acceptance"],
        "attribution": cmp_["attribution"],
        "stats": cmp_["stats"],
        "structural": cmp_["structural"],
        "nonfinite": cmp_["stats"]["nonfinite"],
        "tolerance": cmp_["tolerance"],
        "images": {
            "color_heatmap_png": _b64(encode.heatmap_png(
                width, height, hm["color_abs_max"], cov)),
            "relative_heatmap_png": _b64(encode.heatmap_png(
                width, height, hm["relative_max"], cov, cap=0.02)),
            "depth_heatmap_png": _b64(encode.heatmap_png(
                width, height, hm["depth_abs"], cov)),
            "edge_png": _b64(encode.edge_png(
                width, height, hm["edge_mask"], cov)),
            "coverage_diff_png": _b64(encode.coverage_diff_png(
                width, height, ref["covered"], tst["covered"])),
            "stage_png": _b64(encode.stage_png(
                width, height, cmp_["attribution"]["stage_index"], cov)),
        },
    }


def _pixel_trace_payload(raw_results, width, height, trace_pixel):
    """Per-triangle decisions at one pixel and the exact float inputs that fed
    the eventual shaded color, for every executed path."""
    x, y = trace_pixel
    i = y * width + x
    payload = {"pixel": [x, y], "index": i, "paths": {}}
    for key, r in raw_results.items():
        payload["paths"][key] = {
            "profile_id": r.get("profile_id"),
            "trace": r["trace"],
            "final_srgb": r["srgb"][i],
            "final_linear": r["linear"][i],
            "final_albedo": r["albedo"][i],
            "depth": r["depth"][i],
            "obj_id": r["obj_id"][i],
            "covered": bool(r["covered"][i]),
        }
    return payload


def available_scenes():
    out = []
    for key in scene_keys():
        s = get_scene(key)
        out.append({"key": key, "title": s["title"],
                    "description": s["description"]})
    return out


def profile_catalog():
    """UI-facing catalogue: profiles, device capabilities, defaults, and the
    two delivered precision configs."""
    devices, gpu = discover_devices()
    profiles = all_profiles()
    return {
        "profiles": profiles,
        "devices": devices,
        "gpu_status": gpu,
        "defaults": {d["id"]: profile_for_default(d["id"]) for d in devices},
        "precision_configs": {k: dict(v) for k, v in PRECISION_CONFIGS.items()},
        "rules_version": RULES_VERSION,
        "capabilities": CAPABILITIES,
    }
