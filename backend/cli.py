"""Headless runner for multi-device render verification.

This is an *auxiliary* entry point: the primary GUI entry point is the render
task panel served by ``backend/server.py`` (see README).  The CLI exists so the
same real renders and comparisons can be reproduced in batch or CI without a
browser.

Usage:
  # Default: two really different compute paths, one scene (or all).
  python -m backend.cli --scene all --width 128 --out reports/run

  # Compare the two delivered precision configurations on every device:
  python -m backend.cli --compare-configs --scene highlight --width 128 \
      --out reports/precision

  # Reproduce a specific capability profile and record it in the report:
  python -m backend.cli --scene depth_near --profile vector_f32=f32_ndc01_depth

  # Inject a diagnostic depth-convention error and a broken channel layout:
  python -m backend.cli --self-check --out reports/selfcheck
"""
import argparse
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.raster import render_job, available_scenes  # noqa: E402
from backend.raster.service import PRECISION_CONFIGS  # noqa: E402
from backend.raster.profiles import PROFILES  # noqa: E402
from backend.raster.tolerance import RULES_VERSION  # noqa: E402
from backend.raster import jsonio  # noqa: E402


def _write_png(path, b64):
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(b64))


def _parse_profiles(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit("--profile 需要 device=profile_id，收到 %r" % item)
        dev, pid = item.split("=", 1)
        if pid not in PROFILES:
            raise SystemExit("未知 profile: %s（可用：%s）"
                             % (pid, ", ".join(sorted(PROFILES))))
        out[dev] = pid
    return out


def _strip_images(job):
    j = json.loads(json.dumps(job, default=str))
    for r in j["renders"].values():
        r.pop("images", None)
        r.pop("channels", None)
    for c in j["comparisons"]:
        c.pop("images", None)
        c["attribution"].pop("stage_index", None)
    return j


def _dump_outputs(job, tag, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for dev, data in job["renders"].items():
        for name, b64 in data["images"].items():
            _write_png(os.path.join(out_dir, "%s_%s_%s.png" %
                                    (tag, dev, name.replace("_png", ""))), b64)
    for cmp_ in job["comparisons"]:
        kind = cmp_["test"]
        for name, b64 in cmp_["images"].items():
            _write_png(os.path.join(out_dir, "%s_cmp_%s_%s.png" %
                                    (tag, kind, name.replace("_png", ""))), b64)
    report_path = os.path.join(out_dir, "%s_report.json" % tag)
    with open(report_path, "w") as fh:
        json.dump(jsonio.sanitize(_strip_images(job)), fh, ensure_ascii=False,
                  indent=2, default=str)
    return report_path


def run(scene_key, width, out_dir, trace=None, profiles=None, devices=None,
        quiet=False):
    devices = devices or ["cpu_ref", "vector_f32", "gpu_cuda"]
    job = render_job(scene_key, devices, width=width, height=width,
                     trace_pixel=trace, profiles=profiles)
    tag = _tag(scene_key, profiles)
    report = _dump_outputs(job, tag, out_dir)
    if not quiet:
        for cmp_ in job["comparisons"]:
            v = cmp_["verdict"]
            a = cmp_["attribution"]
            print("[%s] %s vs %s · 规则 %s · %s%s · 首个差异阶段 %s · 复现窗口 %s"
                  % (scene_key, cmp_["reference"], cmp_["test"],
                     cmp_["rules_version"], v["class"],
                     "/" + v["primary_class"] if v.get("primary_class") else "",
                     a["first_diverging_stage"] or "无（完全一致）",
                     a["repro_window"]))
    return job, report


def _tag(scene_key, profiles):
    if not profiles:
        return scene_key
    return scene_key + "_" + "_".join(
        "%s-%s" % (k, v) for k, v in sorted(profiles.items()))


def _summary_entry(job):
    return {
        "scene": job["scene"]["key"],
        "rules_version": job["rules_version"],
        "executed": job["executed"],
        "skipped": job["skipped"],
        "unsupported": job["unsupported"],
        "profiles": {d: p["id"] for d, p in job["profiles"].items()},
        "comparisons": [{
            "reference": c["reference"],
            "test": c["test"],
            "verdict": c["verdict"],
            "first_diverging_stage":
                c["attribution"]["first_diverging_stage"],
            "repro_window": c["attribution"]["repro_window"],
            "failures": c["acceptance"]["failures"],
        } for c in job["comparisons"]],
        "unverified_combinations": job["unverified"],
        "attribution": [{
            "reference": c["reference"],
            "test": c["test"],
            "first_diverging_stage":
                c["attribution"]["first_diverging_stage"],
            "repro_window": c["attribution"]["repro_window"],
            "repro_command": c["attribution"]["repro_command"],
            "stage_counts": {k: v["count"]
                             for k, v in c["attribution"]["stages"].items()
                             if v["count"]},
        } for c in job["comparisons"]],
    }


def cmd_compare_configs(args):
    """Render both delivered precision configurations on the same scene(s) and
    report them side by side, with unsupported/unverified combinations listed.

    This is the deliverable "real comparison across two precision configs".
    It also runs the cross-device case (same profile, two different compute
    paths) so fill-convention differences are checked, not assumed equal.
    """
    keys = [s["key"] for s in available_scenes()] \
        if args.scene == "all" else [args.scene]
    summary = []
    for key in keys:
        for config_name, profiles in PRECISION_CONFIGS.items():
            job = render_job(key, list(profiles), width=args.width,
                             height=args.width, profiles=profiles,
                             trace_pixel=_trace(args))
            tag = "%s_precision_%s" % (key, config_name)
            report = _dump_outputs(job, tag, args.out)
            entry = _summary_entry(job)
            entry["precision_config"] = config_name
            entry["config_devices"] = dict(profiles)
            entry["report"] = report
            summary.append(entry)
            print("[%s] 精度配置 %s (%s) · 执行 %s · 未支持 %s · 规则 %s"
                  % (key, config_name, profiles, job["executed"],
                     [u["device"] + ":" + ",".join(u["missing_fields"])
                      for u in job["unsupported"]] or "无",
                     job["rules_version"]))
            for c in job["comparisons"]:
                v = c["verdict"]
                print("     %s vs %s → %s%s"
                      % (c["reference"], c["test"], v["class"],
                         "/" + v["primary_class"]
                         if v.get("primary_class") else ""))

    from backend.raster.devices import discover_devices
    _, gpu = discover_devices()
    for case in gpu["cross_device_cases"]:
        width = case.get("width", args.width)
        job = render_job(case["scene"], case["devices"], width=width,
                         height=width, profiles=case["profiles"])
        c = job["comparisons"][0]
        entry = _summary_entry(job)
        entry["cross_device_case"] = case["id"]
        entry["expects"] = case["expects"]
        entry["report"] = _dump_outputs(job, case["id"], args.out)
        summary.append(entry)
        print("[cross-device] %s · %s vs %s @%d → %s/%s"
              % (case["id"], c["reference"], c["test"], width,
                 c["verdict"]["class"], c["verdict"]["primary_class"]))

    path = os.path.join(args.out, "precision_comparison.json")
    os.makedirs(args.out, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(jsonio.sanitize(summary), fh, ensure_ascii=False, indent=2,
                  default=str)
    print("精度对照报告: %s" % os.path.abspath(path))


# Self-check diagnostics are injected inside the comparison, not by faking a
# render: we take a real render pair and mutate one channel set, exactly what a
# device driver bug would do, then confirm the classifier names it.
def _self_check_scene(scene_key, width):
    from backend.raster.compare import compare
    from backend.raster.scenes import get_scene
    from backend.raster.service import render_one
    from backend.raster.profiles import resolve_profile

    scene = get_scene(scene_key)
    ref = render_one("cpu_ref", scene, width, width,
                     profile=resolve_profile("reference_f64"))
    tst = render_one("vector_f32", scene, width, width,
                     profile=resolve_profile("strict_f32"))
    findings = []

    # (1) normal rounding: the untouched pair.
    base = compare(ref, tst, width, width)
    findings.append({"case": "normal_rounding", "verdict": base["verdict"],
                     "first_stage": base["attribution"]["first_diverging_stage"]})

    # (2) deliberate channel swap of the test path's final sRGB.
    import copy
    swapped = copy.deepcopy(tst)
    for i, cov in enumerate(swapped["covered"]):
        if cov:
            r, g, b = swapped["srgb"][i]
            swapped["srgb"][i] = [g, b, r]
    swap_cmp = compare(ref, swapped, width, width)
    findings.append({"case": "channel_swap", "verdict": swap_cmp["verdict"],
                     "probe": swap_cmp["acceptance"]["channel_swap_probe"]})

    # (3) deliberate depth-convention error on the test path.
    convention = copy.deepcopy(tst)
    for i in range(len(convention["depth"])):
        if convention["covered"][i] and convention["depth"][i] == \
                convention["depth"][i]:
            convention["depth"][i] = (convention["depth"][i] + 1.0) * 0.5
    conv_cmp = compare(ref, convention, width, width)
    findings.append({"case": "depth_convention",
                     "verdict": conv_cmp["verdict"],
                     "probe": conv_cmp["acceptance"]["depth_convention_probe"]})

    # (4) non-finite injection.
    broken = copy.deepcopy(tst)
    idx = next(i for i, c in enumerate(broken["covered"]) if c)
    broken["linear"][idx][0] = float("inf")
    nf_cmp = compare(ref, broken, width, width)
    findings.append({"case": "non_finite", "verdict": nf_cmp["verdict"]})
    for f in findings:
        f["scene"] = scene_key
    return findings


def cmd_self_check(args):
    keys = [s["key"] for s in available_scenes()] \
        if args.scene == "all" else [args.scene]
    findings = []
    for key in keys:
        scene_findings = _self_check_scene(key, args.width)
        findings.extend(scene_findings)
        print("[scene %s]" % key)
        for f in scene_findings:
            v = f["verdict"]
            print("  %-16s → %s%s" %
                  (f["case"], v["class"],
                   "/" + v["primary_class"] if v.get("primary_class") else ""))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "self_check.json")
    with open(path, "w") as fh:
        json.dump(jsonio.sanitize({"rules_version": RULES_VERSION,
                                   "scene": args.scene,
                                   "findings": findings}),
                  fh, ensure_ascii=False, indent=2, default=str)
    print("自检报告: %s" % os.path.abspath(path))


def _trace(args):
    return tuple(int(v) for v in args.trace.split(",")) if args.trace else None


def main():
    ap = argparse.ArgumentParser(
        description="多端出图无头验证（图形入口见 README 的出图任务面板）")
    ap.add_argument("--scene", default="all",
                    help="场景 key 或 all（全部四个场景）")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--out", default="reports/run")
    ap.add_argument("--trace", default=None,
                    help="像素 x,y，输出该像素的完整判定轨迹")
    ap.add_argument("--profile", action="append", default=[],
                    metavar="DEVICE=PROFILE_ID",
                    help="为该设备指定能力/精度配置档（可重复）")
    ap.add_argument("--compare-configs", action="store_true",
                    help="交付两种精度配置的真实渲染对照")
    ap.add_argument("--self-check", action="store_true",
                    help="注入通道交换/深度约定/非有限值，验证分类可区分")
    args = ap.parse_args()

    if args.self_check:
        cmd_self_check(args)
        return
    if args.compare_configs:
        cmd_compare_configs(args)
        return

    profiles = _parse_profiles(args.profile)
    trace = _trace(args)
    keys = [s["key"] for s in available_scenes()] \
        if args.scene == "all" else [args.scene]
    summary = []
    for key in keys:
        job, _ = run(key, args.width, args.out, trace, profiles)
        summary.append(_summary_entry(job))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(jsonio.sanitize(summary), fh, ensure_ascii=False, indent=2)
    print("报告与 PNG 已写入: %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
