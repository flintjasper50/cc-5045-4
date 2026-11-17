"""Two genuinely different rasterization compute paths.

* CpuRefPath  -- triangle-driven, scalar float64, top-left fill convention,
  bounding-box scan with an explicit edge sign computed once per vertex.
* VectorPath  -- pixel-driven 8x8 tiles, every pixel tests every triangle,
  all edge/interpolation math rounded to binary32 (f32), the precision class
  a GPU SIMD lane would use.

The difference in precision and in traversal order is real, not scripted:
sliver coverage and near-coincident depth can legitimately disagree.
"""
from .geometry import view_proj
from .rasterizer import Rasterizer
from .vector import f32, f32sub, f32mul


def _edge64(a, b, px, py):
    return (px - a["sx"]) * (b["sy"] - a["sy"]) - \
           (py - a["sy"]) * (b["sx"] - a["sx"])


def _edge32(a, b, px, py):
    ax, ay, bx, by = f32(a["sx"]), f32(a["sy"]), f32(b["sx"]), f32(b["sy"])
    return f32sub(f32mul(f32sub(px, ax), f32sub(by, ay)),
                  f32mul(f32sub(py, ay), f32sub(bx, ax)))


def _top_left(edge_a, edge_b):
    """GPU top-left fill convention (D3D/Metal/Vulkan): an edge is a
    top-left edge when it is horizontal and above the triangle, or it is
    generally ascending.  Such edges use >=0; other edges use >0."""
    if edge_a["sy"] == edge_b["sy"] and edge_a["sx"] < edge_b["sx"]:
        return True  # horizontal top edge
    return edge_a["sy"] < edge_b["sy"]  # generally ascending edge


def _inside_gpu(weights, edges_tl, area_positive):
    for w, is_tl in zip(weights, edges_tl):
        if is_tl:
            if w < 0.0:
                return False
        elif w <= 0.0:
            return False
    return True


class CpuRefPath(Rasterizer):
    name = "cpu_ref"
    label = "CPU 参考路径 (标量 f64，逐三角形)"
    precision = "f64"
    kind = "cpu"
    fill_convention = "全包含 (所有边 >=0，参考真值约定)"

    def render(self, scene):
        self.scene_light = scene["light_dir"]
        self.camera_eye = scene["camera"]["eye"]
        vp = view_proj(scene["camera"], self.w, self.h)
        for t_index, tri in enumerate(scene["triangles"]):
            verts = self.project(tri, vp)
            if verts is None:
                continue
            xs = [v["sx"] for v in verts]
            ys = [v["sy"] for v in verts]
            x0 = max(0, int(min(xs)) )
            x1 = min(self.w - 1, int(max(xs)))
            y0 = max(0, int(min(ys)))
            y1 = min(self.h - 1, int(max(ys)))
            area = _edge64(verts[0], verts[1], verts[2]["sx"], verts[2]["sy"])
            if area == 0.0:
                continue
            for y in range(y0, y1 + 1):
                py = y + 0.5
                for x in range(x0, x1 + 1):
                    px = x + 0.5
                    w0 = _edge64(verts[1], verts[2], px, py)
                    w1 = _edge64(verts[2], verts[0], px, py)
                    w2 = _edge64(verts[0], verts[1], px, py)
                    inside = (w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0) if area > 0 \
                        else (w0 <= 0.0 and w1 <= 0.0 and w2 <= 0.0)
                    if self.trace_pixel == (x, y):
                        self.trace.append({
                            "triangle": t_index, "obj_id": tri.obj_id,
                            "weights": [w0, w1, w2], "area": area,
                            "inside": bool(inside),
                            "depth_before": self.fb.depth[self.fb.idx(x, y)]})
                    if not inside:
                        continue
                    world, uv, depth = self.interpolate(verts, w0, w1, w2)
                    if self.trace_pixel == (x, y):
                        self.trace[-1].update(
                            {"depth_candidate": depth, "world": list(world),
                             "uv": list(uv)})
                    linear = self.shade_pixel(tri, world, uv)
                    if self.trace_pixel == (x, y):
                        self.trace[-1]["linear"] = list(linear)
                    self.write_pixel(x, y, linear, depth, tri.obj_id)
        return self._result(scene)

    def _result(self, scene):
        return {
            "device": self.name, "label": self.label, "kind": self.kind,
            "precision": self.precision,
            "fill_convention": self.fill_convention,
            "srgb": self.finalize(scene["exposure"]),
            "linear": [list(c) for c in self.fb.color],
            "depth": list(self.fb.depth),
            "obj_id": list(self.fb.obj_id),
            "covered": list(self.fb.covered),
            "trace": self.trace,
        }


class VectorPath(Rasterizer):
    name = "vector_f32"
    label = "向量化计算路径 (分块逐像素，f32，GPU top-left 填充)"
    precision = "f32"
    kind = "cpu-simd"
    fill_convention = "top-left (D3D/Metal/Vulkan 风格，非 top-left 边用严格 >0)"

    TILE = 8

    def render(self, scene):
        self.scene_light = scene["light_dir"]
        self.camera_eye = scene["camera"]["eye"]
        vp = view_proj(scene["camera"], self.w, self.h)
        projected = []
        for t_index, tri in enumerate(scene["triangles"]):
            verts = self.project(tri, vp)
            if verts is None:
                continue
            area = _edge32(verts[0], verts[1], f32(verts[2]["sx"]),
                           f32(verts[2]["sy"]))
            if area == 0.0:
                continue
            bbox = (min(v["sx"] for v in verts), max(v["sx"] for v in verts),
                    min(v["sy"] for v in verts), max(v["sy"] for v in verts))
            # GPU top-left flags, one per edge (w0 uses edge v1->v2, etc.)
            tl = [_top_left(verts[1], verts[2]),
                  _top_left(verts[2], verts[0]),
                  _top_left(verts[0], verts[1])]
            projected.append((t_index, tri, verts, area, bbox, tl))

        # Pixel-driven traversal: each tile visits every triangle.  This is the
        # structure a GPU/SIMD rasterizer uses (one lane per pixel).
        for ty in range(0, self.h, self.TILE):
            for tx in range(0, self.w, self.TILE):
                tile = (tx, ty,
                        min(tx + self.TILE, self.w),
                        min(ty + self.TILE, self.h))
                for y in range(ty, tile[3]):
                    py = f32(y + 0.5)
                    for x in range(tx, tile[2]):
                        px = f32(x + 0.5)
                        for (t_index, tri, verts, area, bbox, tl) in projected:
                            # cheap SIMD-style bounding reject for this tile
                            if bbox[1] < tx or bbox[0] > tile[2] - 1 or \
                               bbox[3] < ty or bbox[2] > tile[3] - 1:
                                continue
                            w0 = _edge32(verts[1], verts[2], px, py)
                            w1 = _edge32(verts[2], verts[0], px, py)
                            w2 = _edge32(verts[0], verts[1], px, py)
                            weights = [w0, w1, w2]
                            if area > 0:
                                inside = _inside_gpu(weights, tl, True)
                            else:
                                # reverse winding: negate weights and swap the
                                # top-left designation to the mirrored edges
                                inside = _inside_gpu(
                                    [-w0, -w1, -w2],
                                    [tl[2], tl[1], tl[0]], False)
                            if self.trace_pixel == (x, y):
                                self.trace.append({
                                    "triangle": t_index, "obj_id": tri.obj_id,
                                    "weights": [w0, w1, w2], "area": area,
                                    "inside": bool(inside),
                                    "depth_before":
                                        self.fb.depth[self.fb.idx(x, y)]})
                            if not inside:
                                continue
                            world, uv, depth = self.interpolate(
                                verts, w0, w1, w2)
                            if self.trace_pixel == (x, y):
                                self.trace[-1].update(
                                    {"depth_candidate": depth,
                                     "world": list(world), "uv": list(uv)})
                            linear = self.shade_pixel(tri, world, uv)
                            if self.trace_pixel == (x, y):
                                self.trace[-1]["linear"] = list(linear)
                            self.write_pixel(x, y, linear, depth, tri.obj_id)
        return self._result(scene)

    def _result(self, scene):
        return {
            "device": self.name, "label": self.label, "kind": self.kind,
            "precision": self.precision,
            "fill_convention": self.fill_convention,
            "srgb": self.finalize(scene["exposure"]),
            "linear": [list(c) for c in self.fb.color],
            "depth": list(self.fb.depth),
            "obj_id": list(self.fb.obj_id),
            "covered": list(self.fb.covered),
            "trace": self.trace,
        }


BACKENDS = {
    CpuRefPath.name: CpuRefPath,
    VectorPath.name: VectorPath,
}
