"""Headless runner: render + compare without the UI, emit JSON and PNGs.

Usage:
  python -m backend.cli --scene all --width 128 --out reports/run
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.raster import render_job, available_scenes  # noqa: E402
from backend.raster import encode  # noqa: E402
import base64  # noqa: E402


def _write_png(path, b64):
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(b64))


def run(scene_key, width, out_dir, trace=None):
    os.makedirs(out_dir, exist_ok=True)
    job = render_job(scene_key, ["cpu_ref", "vector_f32", "gpu_cuda"],
                     width=width, height=width, trace_pixel=trace)
    tag = scene_key
    for dev, data in job["renders"].items():
        for name, b64 in data["images"].items():
            _write_png(os.path.join(out_dir, "%s_%s_%s.png" %
                                    (tag, dev, name.replace("_png", ""))), b64)
    for cmp_ in job["comparisons"]:
        for name, b64 in cmp_["images"].items():
            _write_png(os.path.join(out_dir, "%s_cmp_%s.png" %
                                    (tag, name.replace("_png", ""))), b64)
    with open(os.path.join(out_dir, "%s_report.json" % tag), "w") as fh:
        json.dump(_strip_images(job), fh, ensure_ascii=False, indent=2,
                  default=str)
    return job


def _strip_images(job):
    j = json.loads(json.dumps(job, default=str))
    for r in j["renders"].values():
        r.pop("images", None)
        r.pop("channels", None)
    for c in j["comparisons"]:
        c.pop("images", None)
    return j


def main():
    ap = argparse.ArgumentParser(description="多端出图无头验证")
    ap.add_argument("--scene", default="all",
                    help="场景 key 或 all（全部四个场景）")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--out", default="reports/run")
    ap.add_argument("--trace", default=None,
                    help="像素 x,y，输出该像素的完整判定轨迹")
    args = ap.parse_args()
    trace = tuple(int(v) for v in args.trace.split(",")) if args.trace else None
    keys = [s["key"] for s in available_scenes()] \
        if args.scene == "all" else [args.scene]
    summary = []
    for key in keys:
        job = run(key, args.width, args.out, trace)
        verdict = job["comparisons"][0]["verdict"] if job["comparisons"] else None
        summary.append({
            "scene": key,
            "executed": job["executed"],
            "skipped": job["skipped"],
            "gpu_can_execute": job["gpu_status"]["gpu_can_execute"],
            "verdict": verdict,
        })
        print("[%s] executed=%s skipped=%s verdict=%s" %
              (key, job["executed"], job["skipped"],
               verdict["class"] if verdict else "n/a"))
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("报告与 PNG 已写入: %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
