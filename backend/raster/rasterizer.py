"""Shared rasterizer scaffitting: projection, interpolation, per-pixel shading.

The two concrete backends (`cpu_ref` and `vector_path`) reuse these helpers so
that camera, coordinate system, texture sampling and color transform are
identical.  They deliberately differ in *how pixels are visited* and in which
floating-point precision the edge/interpolation math runs.
"""
from .geometry import (mat_vec, shade, tone_map)
from .vector import Framebuffer, f32, f32add, f32mul, f32div, clamp


class Rasterizer:
    name = "base"
    precision = "f64"

    def __init__(self, width, height, texture, trace_pixel=None):
        self.w = width
        self.h = height
        self.tex = texture
        self.fb = Framebuffer(width, height)
        # When set, collect a detailed decision record for this (x, y).
        self.trace_pixel = trace_pixel
        self.trace = []

    # -- transform one triangle to clip/screen space -----------------------
    def project(self, tri, vp):
        """Return screen-space vertices with 1/w for perspective correction."""
        out = []
        for (x, y, z), (u, v) in zip(tri.positions, tri.uvs):
            clip = mat_vec(vp, (x, y, z, 1.0))
            w = clip[3]
            if w <= 1e-8:
                return None
            inv_w = 1.0 / w
            ndc_x = clip[0] * inv_w
            ndc_y = clip[1] * inv_w
            ndc_z = clip[2] * inv_w
            sx = (ndc_x * 0.5 + 0.5) * self.w
            sy = (1.0 - (ndc_y * 0.5 + 0.5)) * self.h
            out.append({"sx": sx, "sy": sy, "ndc_z": ndc_z,
                        "inv_w": inv_w, "world": (x, y, z),
                        "uv": (u, v)})
        return out

    # -- perspective-correct attributes at barycentric weights -------------
    def interpolate(self, verts, w0, w1, w2):
        """verts: projected list; returns world pos, uv, depth, weight sum."""
        if self.precision == "f32":
            add, mul, div = f32add, f32mul, f32div
            w0, w1, w2 = f32(w0), f32(w1), f32(w2)
            inv_sum = div(1.0, add(add(w0, w1), w2))
        else:
            add = lambda a, b: a + b
            mul = lambda a, b: a * b
            div = lambda a, b: a / b
            inv_sum = 1.0 / (w0 + w1 + w2)
        b0, b1, b2 = mul(w0, inv_sum), mul(w1, inv_sum), mul(w2, inv_sum)

        # perspective-correct: divide attributes by w first
        iw0 = mul(b0, verts[0]["inv_w"])
        iw1 = mul(b1, verts[1]["inv_w"])
        iw2 = mul(b2, verts[2]["inv_w"])
        denom = add(add(iw0, iw1), iw2)
        inv_denom = div(1.0, denom)

        world = []
        for c in range(3):
            val = add(add(mul(iw0, verts[0]["world"][c]),
                          mul(iw1, verts[1]["world"][c])),
                      mul(iw2, verts[2]["world"][c]))
            world.append(mul(val, inv_denom))
        uv = []
        for c in range(2):
            val = add(add(mul(iw0, verts[0]["uv"][c]),
                          mul(iw1, verts[1]["uv"][c])),
                      mul(iw2, verts[2]["uv"][c]))
            uv.append(mul(val, inv_denom))
        depth = add(add(mul(b0, verts[0]["ndc_z"]),
                        mul(b1, verts[1]["ndc_z"])),
                    mul(b2, verts[2]["ndc_z"]))
        return tuple(world), tuple(uv), depth

    # -- resolve material base color through the shared texture ------------
    def base_color(self, tri, uv):
        if tri.material.get("use_texture"):
            sampled = self.tex.sample_linear(uv[0], uv[1])
            boost = tri.material.get("boost", 1.0)
            return (sampled[0] * boost, sampled[1] * boost, sampled[2] * boost)
        col = tri.material["color"]
        return (col[0], col[1], col[2])

    def shade_pixel(self, tri, world, uv):
        base = self.base_color(tri, uv)
        linear = shade(world, tri.normal, base, self.scene_light,
                       self.camera_eye, self.precision,
                       tri.material["spec_power"],
                       tri.material["spec_intensity"])
        return linear

    def write_pixel(self, x, y, linear, depth, obj_id):
        i = self.fb.idx(x, y)
        if depth < self.fb.depth[i]:
            self.fb.depth[i] = f32(depth) if self.precision == "f32" else depth
            self.fb.color[i] = list(linear)
            self.fb.obj_id[i] = obj_id
            self.fb.covered[i] = 1

    def finalize(self, exposure):
        """Apply the identical tone-map + sRGB transform to every pixel."""
        out = []
        for i, c in enumerate(self.fb.color):
            if not self.fb.covered[i]:
                out.append([0.0, 0.0, 0.0])
            else:
                out.append(tone_map(c, exposure))
        return out

    def render(self, scene):
        raise NotImplementedError
