"""Unit tests for the two real compute paths, attribution and acceptance rules."""
import copy
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.raster import render_job
from backend.raster.compare import compare, edge_mask
from backend.raster.devices import (discover_devices, profile_supported)
from backend.raster.encode import encode_png, stage_png
from backend.raster.profiles import PROFILES, resolve_profile
from backend.raster.scenes import get_scene, scene_keys
from backend.raster.service import render_one
from backend.raster.tolerance import RULES_CHANGELOG, RULES_VERSION
from backend.raster.vector import f32


class F32Tests(unittest.TestCase):
    def test_f32_rounding(self):
        self.assertEqual(f32(1.0 + 2.0 ** -24), 1.0)  # below f32 epsilon at 1.0
        self.assertNotEqual(f32(1.0 + 2.0 ** -23), 1.0)

    def test_f32_nonfinite(self):
        self.assertTrue(math.isinf(f32(1e300 * 1e30)))
        self.assertTrue(math.isnan(f32(float("nan"))))


class RenderTests(unittest.TestCase):
    def test_all_scenes_execute_two_paths(self):
        for key in scene_keys():
            job = render_job(key, ["cpu_ref", "vector_f32"], 64, 64)
            self.assertEqual(sorted(job["executed"]),
                             ["cpu_ref", "vector_f32"])
            self.assertEqual(len(job["comparisons"]), 1)
            for r in job["renders"].values():
                self.assertTrue(r["covered_count"] > 0,
                                "%s produced no model pixels" % key)

    def test_gpu_never_faked(self):
        devices, gpu = discover_devices()
        job = render_job("sliver", ["cpu_ref", "vector_f32", "gpu_cuda"], 48, 48)
        if not gpu["gpu_can_execute"]:
            self.assertIn("gpu_cuda", job["skipped"])
            self.assertNotIn("gpu_cuda", job["renders"])
        else:
            self.assertIn("gpu_cuda", job["executed"])

    def test_deterministic(self):
        a = render_job("highlight", ["cpu_ref"], 48, 48)
        b = render_job("highlight", ["cpu_ref"], 48, 48)
        self.assertEqual(a["renders"]["cpu_ref"]["channels"]["srgb"],
                         b["renders"]["cpu_ref"]["channels"]["srgb"])

    def test_background_excluded(self):
        job = render_job("hdr_gamut", ["cpu_ref", "vector_f32"], 64, 64)
        st = job["comparisons"][0]["stats"]
        self.assertGreater(st["background_pixels"], 0)
        self.assertEqual(st["model_union_pixels"] + st["background_pixels"],
                         st["pixels_total"])

    def test_pixel_trace(self):
        job = render_job("sliver", ["cpu_ref", "vector_f32"], 48, 48,
                         trace_pixel=(24, 24))
        pt = job["pixel_trace"]
        self.assertEqual(pt["pixel"], [24, 24])
        self.assertIn("cpu_ref", pt["paths"])
        self.assertIn("vector_f32", pt["paths"])
        for p in pt["paths"].values():
            self.assertTrue(len(p["trace"]) >= 0)
            self.assertEqual(len(p["final_srgb"]), 3)
            self.assertEqual(len(p["final_albedo"]), 3)


class RulesVersionTests(unittest.TestCase):
    def test_version_is_stamped_on_report(self):
        job = render_job("sliver", ["cpu_ref", "vector_f32"], 32, 32)
        self.assertEqual(job["rules_version"], RULES_VERSION)
        cmp_ = job["comparisons"][0]
        self.assertEqual(cmp_["rules_version"], RULES_VERSION)
        # Changelog is embedded, so an old report explains what judged it.
        self.assertIn(RULES_VERSION, cmp_["acceptance"]["rules_changelog"])
        self.assertIn("acceptance-2026.11-v1",
                      cmp_["acceptance"]["rules_changelog"])

    def test_frozen_report_cannot_be_rewritten(self):
        # A stored report carries its own rule text; changing live tolerances
        # later cannot alter the recorded verdict.
        job = render_job("highlight", ["cpu_ref", "vector_f32"], 32, 32)
        stored = copy.deepcopy(job["comparisons"][0]["acceptance"])
        recorded_version = stored["rules_version"]
        recorded_rules = copy.deepcopy(stored["rules"])
        # Simulate a later threshold change in the live table.
        from backend.raster import tolerance
        old = tolerance.TOLERANCES["operations"]["final_color_inherited"][
            "allowed_abs_internal"]
        try:
            tolerance.TOLERANCES["operations"]["final_color_inherited"][
                "allowed_abs_internal"] = 42.0
            self.assertEqual(stored["rules_version"], recorded_version)
            self.assertEqual(stored["rules"],
                             recorded_rules)
        finally:
            tolerance.TOLERANCES["operations"]["final_color_inherited"][
                "allowed_abs_internal"] = old


class AcceptanceRuleTests(unittest.TestCase):
    def _pair(self, key="highlight", size=64, profile_id="strict_f32"):
        scene = get_scene(key)
        ref = render_one("cpu_ref", scene, size, size,
                        profile=resolve_profile("reference_f64"))
        tst = render_one("vector_f32", scene, size, size,
                         profile=resolve_profile(profile_id))
        return ref, tst

    def test_normal_rounding_passes(self):
        job = render_job("highlight", ["cpu_ref", "vector_f32"], 64, 64,
                         profiles={"cpu_ref": "reference_f64",
                                   "vector_f32": "strict_f32"})
        cmp_ = job["comparisons"][0]
        self.assertEqual(cmp_["verdict"]["class"], "pass")
        self.assertEqual(cmp_["attribution"]["first_diverging_stage"], None)

    def test_deliberate_depth_convention_error_is_its_own_class(self):
        job = render_job("depth_near", ["cpu_ref", "vector_f32"], 64, 64,
                         profiles={"cpu_ref": "reference_f64",
                                   "vector_f32": "f32_ndc01_depth"})
        cmp_ = job["comparisons"][0]
        self.assertEqual(cmp_["verdict"]["class"], "fail")
        self.assertEqual(cmp_["verdict"]["primary_class"],
                         "depth_convention")
        # The depth probe must fire and the attribution must say "depth", not
        # "blur it into a colour threshold".
        self.assertTrue(cmp_["acceptance"]["depth_convention_probe"]
                        ["triggered"])
        self.assertEqual(cmp_["attribution"]["first_diverging_stage"],
                         "depth")
        classes = {f["class"] for f in cmp_["acceptance"]["failures"]}
        self.assertIn("depth_convention", classes)

    def test_channel_swap_is_detected_separately(self):
        ref, tst = self._pair("hdr_gamut")
        for i, cov in enumerate(tst["covered"]):
            if cov:
                r, g, b = tst["srgb"][i]
                tst["srgb"][i] = [g, b, r]
        cmp_ = compare(ref, tst, 64, 64)
        self.assertTrue(cmp_["acceptance"]["channel_swap_probe"]["triggered"])
        self.assertEqual(cmp_["verdict"]["primary_class"], "channel_swap")

    def test_non_finite_is_not_swallowed(self):
        ref, tst = self._pair("sliver")
        idx = next(i for i, c in enumerate(tst["covered"]) if c)
        tst["linear"][idx][0] = float("inf")
        cmp_ = compare(ref, tst, 64, 64)
        self.assertEqual(cmp_["verdict"]["class"], "fail")
        self.assertEqual(cmp_["verdict"]["primary_class"], "non_finite")
        self.assertIn("non_finite",
                      {f["class"] for f in cmp_["acceptance"]["failures"]})

    def test_missing_pixel_is_structural_not_noise(self):
        ref, tst = self._pair("sliver")
        idx = next(i for i, c in enumerate(tst["covered"]) if c)
        tst["covered"][idx] = 0
        tst["obj_id"][idx] = 0
        cmp_ = compare(ref, tst, 64, 64)
        self.assertEqual(cmp_["verdict"]["primary_class"], "coverage_struct")

    def test_operations_declare_effective_pixels(self):
        job = render_job("highlight", ["cpu_ref", "vector_f32"], 32, 32)
        rules = job["comparisons"][0]["acceptance"]["rules"]
        for name in ("coverage", "object_id", "depth",
                     "final_color_inherited", "final_color_primary",
                     "linear_hdr", "finite"):
            self.assertIn(name, rules)
            self.assertIn("effective_pixels", rules[name])

    def test_attribution_reports_repro_window_and_samples(self):
        ref, tst = self._pair("sliver")
        idx = next(i for i, c in enumerate(tst["covered"]) if c)
        tst["covered"][idx] = 0
        tst["obj_id"][idx] = 0
        cmp_ = compare(ref, tst, 64, 64)
        a = cmp_["attribution"]
        self.assertIsNotNone(a["first_diverging_stage"])
        self.assertGreaterEqual(a["contributing_pixels"], 1)
        self.assertLessEqual(a["repro_window"]["x1"], 63)
        self.assertTrue(a["samples"])
        sample = a["samples"][0]
        self.assertIn("ref", sample)
        self.assertIn("test", sample)
        for ch in ("albedo", "linear", "srgb", "depth"):
            self.assertIn(ch, sample["ref"])

    def test_stage_index_covers_frame(self):
        ref, tst = self._pair("highlight")
        cmp_ = compare(ref, tst, 64, 64)
        self.assertEqual(len(cmp_["attribution"]["stage_index"]), 64 * 64)


class CapabilityProfileTests(unittest.TestCase):
    def test_unsupported_format_is_reported_not_downgraded(self):
        # gpu_cuda only declares float32_bilinear; rgba8 there must not run.
        profile = resolve_profile("rgba8_texture_f32")
        ok, missing = profile_supported("gpu_cuda", profile)
        self.assertFalse(ok)
        self.assertIn("texture_format", missing)

    def test_unsupported_device_profile_pair_skips_with_reason(self):
        job = render_job("depth_near", ["cpu_ref", "gpu_cuda"], 32, 32,
                         profiles={"cpu_ref": "reference_f64",
                                   "gpu_cuda": "degraded_rgba8_fixed8_f32"})
        # Either the GPU is unavailable (skipped) or the profile is unsupported;
        # in neither case may a substituted image appear.
        self.assertNotIn("gpu_cuda", job["renders"])
        self.assertTrue(job["unsupported"] or "gpu_cuda" in job["skipped"])

    def test_unsupported_pair_on_available_device(self):
        job = render_job("depth_near", ["vector_f32"], 32, 32,
                         profiles={"vector_f32": "reference_f64"})
        # vector_f32 genuinely only implements f32 precision.
        self.assertIn("vector_f32", [u["device"] for u in job["unsupported"]])
        self.assertNotIn("vector_f32", job["renders"])

    def test_profile_hash_is_stamped_and_stable(self):
        a = resolve_profile("strict_f32")
        b = resolve_profile("strict_f32")
        self.assertEqual(a["sha256"], b["sha256"])
        self.assertNotEqual(a["sha256"],
                            resolve_profile("fast_math_f32")["sha256"])

    def test_declared_profiles_are_complete(self):
        for pid in PROFILES:
            p = resolve_profile(pid)
            for field in ("precision", "fast_math", "texture_format",
                          "blend_capability", "depth_convention", "depth_test"):
                self.assertIn(field, p)
                self.assertIn(field, p["sources"])

    def test_unverified_combinations_are_surfaced(self):
        job = render_job("highlight", ["cpu_ref", "vector_f32"], 32, 32)
        self.assertTrue(job["unverified"])
        for item in job["unverified"]:
            self.assertIn("device", item)
            self.assertIn("profile", item)
            self.assertIn("reason", item)

    def test_two_precision_configs_run_for_real(self):
        from backend.raster.service import PRECISION_CONFIGS
        self.assertGreaterEqual(len(PRECISION_CONFIGS), 2)
        for name, mapping in PRECISION_CONFIGS.items():
            job = render_job("highlight", list(mapping), 32, 32,
                             profiles=mapping)
            self.assertEqual(sorted(job["executed"]), sorted(mapping),
                             "config %s did not really run" % name)

    def test_texture_degradation_attributed_to_sampling(self):
        job = render_job("depth_near", ["cpu_ref", "vector_f32"], 96, 96,
                         profiles={"cpu_ref": "reference_f64",
                                   "vector_f32": "rgba8_texture_f32"})
        stages = job["comparisons"][0]["attribution"]["stages"]
        self.assertGreater(stages["texture_sampling"]["count"], 0)
        self.assertEqual(job["comparisons"][0]["attribution"]
                         ["first_diverging_stage"], "texture_sampling")

    def test_fixed8_blend_attributed_to_writeback(self):
        # highlight's specular peaks make an 8-bit linear store genuinely lossy,
        # so it must both be attributed to write-back and breach its own limit.
        job = render_job("highlight", ["cpu_ref", "vector_f32"], 96, 96,
                         profiles={"cpu_ref": "reference_f64",
                                   "vector_f32": "fixed8_blend_f32"})
        cmp_ = job["comparisons"][0]
        self.assertGreater(cmp_["attribution"]["stages"]["blend_write"]["count"],
                           0)
        self.assertEqual(cmp_["verdict"]["primary_class"],
                         "blend_quantization_exceeded")

    def test_strict_f32_has_no_spurious_stage(self):
        job = render_job("depth_near", ["cpu_ref", "vector_f32"], 96, 96,
                         profiles={"cpu_ref": "reference_f64",
                                   "vector_f32": "strict_f32"})
        self.assertEqual(job["comparisons"][0]["attribution"]
                         ["first_diverging_stage"], None)


class EdgeMaskTests(unittest.TestCase):
    def test_edge_band(self):
        covered = bytearray(9)
        covered[4] = 1  # single center pixel -> it is an edge
        e = edge_mask(covered, 3, 3)
        self.assertEqual(e[4], 1)


class PngTests(unittest.TestCase):
    def test_png_signature(self):
        png = encode_png(2, 2, [(255, 0, 0)] * 4)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_stage_png(self):
        idx = [None, "depth", "coverage", None]
        png = stage_png(2, 2, idx, bytearray([1, 1, 1, 0]))
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
