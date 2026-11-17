"""Unit tests for the two real compute paths and the comparison engine."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.raster import render_job
from backend.raster.compare import compare, edge_mask
from backend.raster.devices import discover_devices
from backend.raster.encode import encode_png
from backend.raster.scenes import scene_keys
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

    def test_structural_vs_precision_classification(self):
        classes = set()
        for key in scene_keys():
            job = render_job(key, ["cpu_ref", "vector_f32"], 128, 128)
            classes.add(job["comparisons"][0]["verdict"]["class"])
        # The fill-convention difference must surface a structural case, and
        # well-conditioned scenes must remain precision-only.
        self.assertIn("precision_only", classes)
        self.assertTrue(classes & {"structural", "precision_exceeded"})

    def test_nonfinite_detection(self):
        # Inject +inf into one path and confirm the engine flags it.
        job = render_job("sliver", ["cpu_ref", "vector_f32"], 32, 32)
        # rebuild raw results via service internals
        from backend.raster.service import render_one
        from backend.raster.scenes import get_scene
        scene = get_scene("sliver")
        ref = render_one("cpu_ref", scene, 32, 32)
        tst = render_one("vector_f32", scene, 32, 32)
        idx = next(i for i, c in enumerate(tst["covered"]) if c)
        tst["linear"][idx][0] = float("inf")
        cmp_ = compare(ref, tst, 32, 32)
        self.assertGreaterEqual(cmp_["stats"]["nonfinite"]["count"], 1)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
