"""Application service: run real backends, compare, build diagnostic payloads."""
import base64
import hashlib
import json
import platform
import time

from . import encode
from .backends import BACKENDS, CpuRefPath, VectorPath
from .compare import compare
from .devices import discover_devices
from .geometry import ProceduralTexture
from .scenes import get_scene, scene_keys

MAX_RES = 256
DEFAULT_RES = 128


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
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    payload["config_sha256"] = digest
    return payload, scene


def render_one(device_id, scene, width, height, trace_pixel=None):
    if device_id not in BACKENDS:
        raise ValueError("backend not runnable here: %s" % device_id)
    cls = BACKENDS[device_id]
    tex = ProceduralTexture()
    rast = cls(width, height, tex, trace_pixel=trace_pixel)
    t0 = time.perf_counter()
    result = rast.render(scene)
    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return result


def _b64(png_bytes):
    return base64.b64encode(png_bytes).decode("ascii")


def render_job(scene_key, device_ids, width=DEFAULT_RES, height=None,
               trace_pixel=None):
    width = min(MAX_RES, max(16, int(width)))
    height = height or width
    height = min(MAX_RES, max(16, int(height)))
    config, scene = reproducibility_config(scene_key, width, height)
    devices_meta, gpu_status = discover_devices()

    renders = {}
    raw_results = {}
    executed, skipped = [], []
    for dev in device_ids:
        if dev == "gpu_cuda":
            # Never fake it: an unavailable GPU is reported, not rendered.
            if not gpu_status["gpu_can_execute"]:
                skipped.append(dev)
                continue
        if dev not in BACKENDS:
            skipped.append(dev)
            continue
        res = render_one(dev, scene, width, height, trace_pixel)
        raw_results[dev] = res
        renders[dev] = _serialize_render(res, width, height)
        executed.append(dev)

    comparisons = []
    if CpuRefPath.name in raw_results and VectorPath.name in raw_results:
        ref = raw_results[CpuRefPath.name]
        tst = raw_results[VectorPath.name]
        cmp_ = compare(ref, tst, width, height)
        comparisons.append(_serialize_comparison(cmp_, ref, tst, width, height))

    return {
        "scene": {"key": scene_key, "title": scene["title"],
                  "description": scene["description"]},
        "config": config,
        "gpu_status": gpu_status,
        "devices": devices_meta,
        "executed": executed,
        "skipped": skipped,
        "renders": renders,
        "comparisons": comparisons,
        "pixel_trace": _pixel_trace_payload(
            raw_results, width, height, trace_pixel)
        if trace_pixel and raw_results else None,
    }


def _serialize_render(result, width, height):
    return {
        "device": result["device"],
        "label": result["label"],
        "kind": result["kind"],
        "precision": result["precision"],
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
            "depth": result["depth"],
            "obj_id": result["obj_id"],
            "covered": list(result["covered"]),
        },
    }


def _serialize_comparison(cmp_, ref, tst, width, height):
    hm = cmp_["heatmap"]
    cov = ref["covered"]
    return {
        "reference": cmp_["reference"],
        "test": cmp_["test"],
        "verdict": cmp_["verdict"],
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
        },
    }


def _pixel_trace_payload(raw_results, width, height, trace_pixel):
    """Per-triangle decisions at one pixel and the exact float inputs that fed
    the eventual shaded color, for both real compute paths."""
    x, y = trace_pixel
    i = y * width + x
    payload = {"pixel": [x, y], "index": i, "paths": {}}
    for key in (CpuRefPath.name, VectorPath.name):
        if key not in raw_results:
            continue
        r = raw_results[key]
        payload["paths"][key] = {
            "trace": r["trace"],
            "final_srgb": r["srgb"][i],
            "final_linear": r["linear"][i],
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
