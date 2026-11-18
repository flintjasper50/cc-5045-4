"""Four deterministic verification scenes.

Each scene targets a failure mode that hides or amplifies cross-device
differences.  All geometry is generated from fixed constants (no RNG state
leakage between scenes) so renders are reproducible bit-for-bit per path.
"""
import math

from .geometry import Triangle, mat_mul, mat_rotation_x, mat_rotation_y, mat_translate, mat_vec


def _xform(pt, model):
    x, y, z = pt
    v = mat_vec(model, (x, y, z, 1.0))
    return (v[0] / v[3], v[1] / v[3], v[2] / v[3])


def _model(rot_y=0.0, rot_x=0.0, trans=(0.0, 0.0, 0.0)):
    m = mat_translate(*trans)
    if rot_x:
        m = mat_mul(m, mat_rotation_x(rot_x))
    if rot_y:
        m = mat_mul(m, mat_rotation_y(rot_y))
    return m


def _quad(p0, p1, p2, p3, uvs, normal, obj_id, material, model=None):
    pts = [p0, p1, p2, p3]
    if model is not None:
        pts = [_xform(p, model) for p in pts]
        # rotate the normal by the model rotation (upper 3x3)
        n = mat_vec(model, (normal[0], normal[1], normal[2], 0.0))
        normal = (n[0], n[1], n[2])
    t1 = Triangle(pts[0:3], uvs[0:3], normal, obj_id, material)
    t2 = Triangle([pts[0], pts[2], pts[3]], [uvs[0], uvs[2], uvs[3]],
                  normal, obj_id, material)
    return [t1, t2]


def _mat(color, use_texture=False, spec_power=32.0, spec=0.6, boost=1.0):
    return {"color": color, "use_texture": use_texture,
            "spec_power": spec_power, "spec_intensity": spec, "boost": boost}


SCENES = {}


def scene_depth_near():
    """Two co-planar-ish planes at large depth: tight depth precision /
    near-coincident surfaces stress the f32 vs f64 depth test."""
    tris = []
    model = _model(rot_y=math.radians(18.0), rot_x=math.radians(10.0))
    mat_a = _mat((0.85, 0.85, 0.8), use_texture=True, spec_power=64.0, spec=0.3)
    mat_b = _mat((0.25, 0.55, 0.3), spec_power=16.0, spec=0.15)
    # Two large planes separated by a tiny depth delta at far distance.  The
    # nearer (green) plane is smaller so the textured plane stays visible in a
    # ring around it -- otherwise texture-format differences would be hidden.
    z0 = -14.0
    s = 1.5
    base = [(-s, -s, 0.0), (s, -s, 0.0), (s, s, 0.0), (-s, s, 0.0)]
    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for k, (dz, scale, mt, oid) in enumerate(
            [(0.0, 1.0, mat_a, 1), (0.0009, 0.7, mat_b, 2)]):
        pts = [(x * scale, y * scale, z0 + dz) for (x, y, _) in base]
        tris += _quad(*pts, uvs=uvs, normal=(0.0, 0.0, 1.0),
                      obj_id=oid, material=mt, model=model)
    # A small foreground cube so the frame has strong depth range.
    cm = _model(rot_y=math.radians(35.0), trans=(1.4, -1.1, -7.0))
    tris += _cube(cm, obj_id=3, material=_mat((0.9, 0.6, 0.2), spec_power=24.0))
    return {
        "key": "depth_near",
        "title": "深度接近 / 近重合表面",
        "description": "远距双平面深度差约 9e-4，检验 f32/f64 深度缓冲精度与覆盖判定。",
        "triangles": tris,
        "light_dir": (0.4, 0.8, 0.9),
        "exposure": 1.0,
        "camera": {"fovy": 45.0, "near": 0.1, "far": 60.0,
                   "eye": (0.0, 0.0, 3.0), "target": (0.0, 0.0, -13.5)},
    }


def scene_highlight():
    """Strong specular highlight: tiny normal/interpolation differences move
    the Blinn-Phong peak and produce real (but bounded) color error."""
    tris = []
    model = _model(rot_y=math.radians(-25.0), rot_x=math.radians(-10.0))
    shiny = _mat((0.95, 0.95, 0.97), spec_power=180.0, spec=10.0)
    s = 1.7
    tris += _quad((-s, -s, 0.0), (s, -s, 0.0), (s, s, 0.0), (-s, s, 0.0),
                  uvs=[(0, 0), (1, 0), (1, 1), (0, 1)],
                  normal=(0.0, 0.0, 1.0), obj_id=1, material=shiny, model=model)
    # Second tilted glossy plane catches a grazing highlight.
    m2 = _model(rot_x=math.radians(50.0), trans=(0.0, -1.2, -0.8))
    tris += _quad((-s, -s, 0.0), (s, -s, 0.0), (s, s, 0.0), (-s, s, 0.0),
                  uvs=[(0, 0), (1, 0), (1, 1), (0, 1)],
                  normal=(0.0, 0.0, 1.0), obj_id=2,
                  material=_mat((0.7, 0.8, 1.0), spec_power=130.0, spec=7.0),
                  model=m2)
    return {
        "key": "highlight",
        "title": "强高光 (高光泽 Blinn-Phong)",
        "description": "高光泽指数与强镜面项，微小法线/插值差异会移动尖峰位置。",
        "triangles": tris,
        "light_dir": (-0.84, 0.19, 0.5),
        "exposure": 1.0,
        "camera": {"fovy": 40.0, "near": 0.1, "far": 50.0,
                   "eye": (1.2, 0.6, 8.5), "target": (0.0, -0.8, -1.0)},
    }


def scene_sliver():
    """Many thin/sliver triangles: edge-function precision and sub-pixel
    coverage decide whether a triangle owns a pixel at all."""
    tris = []
    # Fan of very thin wedges radiating from center, each <1 pixel wide at tip.
    n = 26
    radius = 3.2
    for i in range(n):
        a0 = (2.0 * math.pi) * i / n
        a1 = (2.0 * math.pi) * (i + 0.35) / n  # narrow wedge
        tip = (0.0, 0.0, 0.0)
        p1 = (radius * math.cos(a0), radius * math.sin(a0), 0.0)
        p2 = (radius * math.cos(a1), radius * math.sin(a1), 0.0)
        col = (0.2 + 0.7 * (i % 2), 0.3 + 0.5 * ((i * 3) % 7) / 7.0,
               0.25 + 0.6 * (i % 3) / 3.0)
        tris.append(Triangle([tip, p1, p2],
                             [(0.5, 0.5), (1.0, 0.0), (1.0, 1.0)],
                             (0.0, 0.0, 1.0), obj_id=1 + (i % 4),
                             material=_mat(col, spec_power=8.0, spec=0.1)))
    # A long needle-thin triangle across the frame.
    tris.append(Triangle([(-3.4, -2.1, 0.2), (3.4, 1.9, 0.2),
                          (3.38, 1.86, 0.2)],
                         [(0, 0), (1, 1), (1, 0.9)], (0, 0, 1),
                         obj_id=5, material=_mat((1.0, 1.0, 0.2), spec=0.2)))
    return {
        "key": "sliver",
        "title": "细三角形 / 亚像素覆盖",
        "description": "26 个窄楔形与针状三角形，尖端不足一像素，检验边缘函数与覆盖归属。",
        "triangles": tris,
        "light_dir": (0.3, 0.9, 0.8),
        "exposure": 1.0,
        "camera": {"fovy": 45.0, "near": 0.1, "far": 40.0,
                   "eye": (0.0, 0.0, 5.0), "target": (0.0, 0.0, 0.0)},
    }


def scene_hdr_gamut():
    """HDR colors intentionally exceeding [0,1] before tone mapping; checks the
    out-of-range clamp path and that the two paths agree after the transform."""
    tris = []
    colors = [(2.4, 0.2, 0.15),     # >1 red
              (0.1, 2.6, 0.4),      # >1 green
              (0.2, 0.3, 3.0),      # >1 blue
              (2.2, 2.2, 0.1),      # bright yellow
              (1.8, 0.9, 2.7)]      # bright magenta
    for i, col in enumerate(colors):
        x = -2.4 + i * 1.2
        model = _model(rot_y=math.radians(18.0 * i),
                       trans=(x, 0.0, -2.0 - 0.4 * i))
        s = 0.55
        boosted = tuple(c * 1.0 for c in col)
        tris += _quad((-s, -s, 0.0), (s, -s, 0.0), (s, s, 0.0), (-s, s, 0.0),
                      uvs=[(0, 0), (1, 0), (1, 1), (0, 1)],
                      normal=(0.0, 0.0, 1.0), obj_id=1 + i,
                      material=_mat(boosted, spec_power=48.0, spec=2.4),
                      model=model)
    return {
        "key": "hdr_gamut",
        "title": "超出颜色范围 (HDR/色域)",
        "description": "线性光颜色最高达 3.0，经 Reinhard+sRGB 后才压回 [0,1]，检验越界与钳制。",
        "triangles": tris,
        "light_dir": (0.5, 0.8, 0.7),
        "exposure": 1.2,
        "camera": {"fovy": 45.0, "near": 0.1, "far": 40.0,
                   "eye": (0.0, 0.0, 5.5), "target": (0.0, 0.0, -2.6)},
    }


def _cube(model, obj_id, material):
    """12 triangles forming a unit cube centered at origin, transformed."""
    h = 0.6
    faces = [
        # (normal, 4 corners)
        ((0, 0, 1), [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]),
        ((0, 0, -1), [(h, -h, -h), (-h, -h, -h), (-h, h, -h), (h, h, -h)]),
        ((1, 0, 0), [(h, -h, h), (h, -h, -h), (h, h, -h), (h, h, h)]),
        ((-1, 0, 0), [(-h, -h, -h), (-h, -h, h), (-h, h, h), (-h, h, -h)]),
        ((0, 1, 0), [(-h, h, h), (h, h, h), (h, h, -h), (-h, h, -h)]),
        ((0, -1, 0), [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)]),
    ]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    out = []
    for normal, pts in faces:
        out += _quad(*pts, uvs=uvs, normal=normal, obj_id=obj_id,
                     material=material, model=model)
    return out


SCENES["depth_near"] = scene_depth_near
SCENES["highlight"] = scene_highlight
SCENES["sliver"] = scene_sliver
SCENES["hdr_gamut"] = scene_hdr_gamut


def get_scene(key):
    if key not in SCENES:
        raise KeyError("unknown scene: %s" % key)
    return SCENES[key]()


def scene_keys():
    return ["depth_near", "highlight", "sliver", "hdr_gamut"]
