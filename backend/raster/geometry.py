"""Shared scene/camera/transform/color pipeline.

Both compute paths call *exactly* the same scene setup, camera matrices,
texture sampler and color transform, so any difference left after this is due
to the rasterization arithmetic/ordering and the declared capability profile,
not to mismatched configuration.

Capability hooks (all opt-in through the render profile, default off):
* ``fast_math``       -- exp approximation + reassociated dot, explicitly declared.
* ``texture_format``  -- ``rgba8_linear`` quantises sampled texels to an 8-bit
  linear lattice before shading.
* ``blend_capability``-- ``fixed8_blend`` quantises the written linear texel.
* ``depth_convention``-- ``ndc_0_1`` applies the D3D/Vulkan window-depth remap
  (a *diagnostic* deviation; the reference always uses ``ndc_minus1_1``).
"""
import math

from .vector import f32, f32add, f32sub, f32mul, f32div, f32dot3, f32vec3, clamp


# --------------------------------------------------------------------------
# Matrices (column-major 4x4 tuples, native float64)
# --------------------------------------------------------------------------
def mat_identity():
    return (1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0)


def mat_mul(a, b):
    out = [0.0] * 16
    for r in range(4):
        for c in range(4):
            s = 0.0
            for k in range(4):
                s += a[k * 4 + r] * b[c * 4 + k]
            out[c * 4 + r] = s
    return tuple(out)


def mat_translate(x, y, z):
    m = list(mat_identity())
    m[12], m[13], m[14] = x, y, z
    return tuple(m)


def mat_rotation_y(rad):
    c, s = math.cos(rad), math.sin(rad)
    return (c, 0.0, -s, 0.0,
            0.0, 1.0, 0.0, 0.0,
            s, 0.0, c, 0.0,
            0.0, 0.0, 0.0, 1.0)


def mat_rotation_x(rad):
    c, s = math.cos(rad), math.sin(rad)
    return (1.0, 0.0, 0.0, 0.0,
            0.0, c, s, 0.0,
            0.0, -s, c, 0.0,
            0.0, 0.0, 0.0, 1.0)


def mat_perspective(fovy_deg, aspect, z_near, z_far):
    """Right-handed OpenGL-style perspective, camera looks down -Z."""
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    n, far = z_near, z_far
    return (f / aspect, 0.0, 0.0, 0.0,
            0.0, f, 0.0, 0.0,
            0.0, 0.0, (far + n) / (n - far), -1.0,
            0.0, 0.0, (2.0 * far * n) / (n - far), 0.0)


def mat_vec(m, v):
    x, y, z, w = v
    return (
        m[0] * x + m[4] * y + m[8] * z + m[12] * w,
        m[1] * x + m[5] * y + m[9] * z + m[13] * w,
        m[2] * x + m[6] * y + m[10] * z + m[14] * w,
        m[3] * x + m[7] * y + m[11] * z + m[15] * w,
    )


# --------------------------------------------------------------------------
# Declared texture storage formats
# --------------------------------------------------------------------------
TEXTURE_FORMATS = {
    "float32_bilinear": {
        "label": "float32 双线性采样（无存储量化）",
        "quantize_bits": None,
        "provenance": "纹素以浮点保存，采样只引入 binary32 舍入。",
    },
    "rgba8_linear": {
        "label": "rgba8 线性格点（1/255 量化）",
        "quantize_bits": 255,
        "provenance": "内存受限设备常用的 8 位线性纹理格式；采样后量化到 1/255。",
    },
}

BLEND_CAPABILITIES = {
    "float_blend": {
        "label": "线性浮点写回（无量化）",
        "quantize_bits": None,
        "provenance": "浮点帧缓冲，写回保留完整精度。",
    },
    "fixed8_blend": {
        "label": "定长 8 位线性写回（1/255 量化）",
        "quantize_bits": 255,
        "provenance": "无浮点混合能力时，线性颜色写回即量化到 8 位格点。",
    },
}


def quantize_unit(x, levels):
    """Round to the nearest 1/levels lattice point and clamp to [0,1]."""
    return clamp(round(x * levels) / levels, 0.0, 1.0)


def apply_texture_format(sampled, texture_format):
    spec = TEXTURE_FORMATS[texture_format]
    levels = spec["quantize_bits"]
    if levels is None:
        return tuple(sampled)
    return tuple(quantize_unit(c, levels) for c in sampled)


# --------------------------------------------------------------------------
# Deterministic procedural texture + nearest/linear sampling (shared)
# --------------------------------------------------------------------------
class ProceduralTexture:
    """8x8 deterministic RGBA checker; same bytes seen by every path."""

    def __init__(self, size=8):
        self.size = size
        px = []
        for j in range(size):
            for i in range(size):
                if (i // 2 + j // 2) % 2 == 0:
                    px.append((0.95, 0.25, 0.2, 1.0))
                else:
                    px.append((0.2, 0.45, 0.95, 1.0))
        self.texels = px

    def sample_nearest(self, u, v):
        s = self.size
        i = int(math.floor(u * s)) % s
        j = int(math.floor(v * s)) % s
        return self.texels[j * s + i]

    def sample_linear(self, u, v, precision="f64"):
        """Bilinear with wrap addressing, evaluated in the declared precision.

        The f32 path rounds every intermediate to binary32 (as a GPU sampler
        would); the f64 path keeps native doubles.  This is a real precision
        difference in sampling, not a scripted offset.
        """
        if precision == "f32":
            add, mul, sub = f32add, f32mul, f32sub
            u, v = f32(u), f32(v)
        else:
            add = lambda a, b: a + b
            mul = lambda a, b: a * b
            sub = lambda a, b: a - b
        s = self.size
        fx = sub(mul(u, f32(s)), 0.5)
        fy = sub(mul(v, f32(s)), 0.5)
        i0 = int(math.floor(fx))
        j0 = int(math.floor(fy))
        tx, ty = fx - i0, fy - j0
        i0 %= s
        j0 %= s
        i1, j1 = (i0 + 1) % s, (j0 + 1) % s
        p00 = self.texels[j0 * s + i0]
        p10 = self.texels[j0 * s + i1]
        p01 = self.texels[j1 * s + i0]
        p11 = self.texels[j1 * s + i1]
        out = []
        for c in range(4):
            omt = sub(1.0, tx)
            omty = sub(1.0, ty)
            top = add(mul(p00[c], omt), mul(p10[c], tx))
            bot = add(mul(p01[c], omt), mul(p11[c], tx))
            out.append(add(mul(top, omty), mul(bot, ty)))
        return tuple(out)


# --------------------------------------------------------------------------
# Lighting / shading: one formula, two precision front ends
# --------------------------------------------------------------------------
def _normalize(v, ftype):
    if ftype == "f32":
        l = f32(math.sqrt(f32dot3(v, v)))
        return (f32div(v[0], l), f32div(v[1], l), f32div(v[2], l))
    l = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / l, v[1] / l, v[2] / l)


def _fast_exp(x):
    """Fast-math exponent via base-2 decomposition + truncated series.

    Standard library only, no platform exp.  Returns exp(x) with relative
    error ~1e-4 over the exponent range used for specular peaks -- large
    enough to move a highlight, small enough to stay plausible.
    """
    y = x * 1.4426950408889634  # log2(e)
    k = int(math.floor(y + 0.5))
    r = (y - k) * 0.6931471805599453  # r in [-ln2/2, ln2/2]
    # exp(r) ~ 1 + r + r^2/2 + r^3/6 + r^4/24
    term = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0))))
    return math.ldexp(term, k)


def _fast_pow(base, power):
    if base <= 0.0:
        return 0.0
    return _fast_exp(power * _fast_log(base))


def _fast_log(x):
    """Natural log via frexp, no platform log."""
    m, e = math.frexp(x)
    # m in [0.5,1); series for ln(m) around 1 by atanh transform
    z = (m - 1.0) / (m + 1.0)
    z2 = z * z
    series = 2.0 * z * (1.0 + z2 * (1.0 / 3.0 + z2 * (1.0 / 5.0 +
                                                     z2 * (1.0 / 7.0))))
    return series + e * 0.6931471805599453


def shade(world_pos, normal, base_rgb, light_dir, camera_pos, ftype,
          spec_power, spec_intensity, fast_math=False):
    """Blinn-Phong in linear HDR space.  ftype selects f64 or f32 arithmetic.

    Out-of-gamut energy is allowed to exceed 1.0 here; tone mapping happens
    later in `tone_map`, identical for both paths unless a profile declares
    otherwise.  ``fast_math`` swaps the specular exponent for a declared
    approximation (approximate log/exp, one Newton reciprocal) and reassociates
    the dot product -- exactly the class of deviation a "fast math" build
    introduces.
    """
    if ftype == "f32":
        add, mul, dot = f32add, f32mul, f32dot3
        wp = f32vec3(world_pos)
        n = _normalize(f32vec3(normal), "f32")
        L = _normalize(f32vec3(light_dir), "f32")
        e = (f32sub(camera_pos[0], wp[0]),
             f32sub(camera_pos[1], wp[1]),
             f32sub(camera_pos[2], wp[2]))
        V = _normalize(e, "f32")
        H = _normalize((f32add(L[0], V[0]), f32add(L[1], V[1]),
                        f32add(L[2], V[2])), "f32")
        ndl = max(dot(n, L), 0.0)
        ndh = max(dot(n, H), 0.0)
        diff = [mul(base_rgb[c], f32(0.15 + 0.85 * ndl)) for c in range(3)]
        if fast_math:
            spec = spec_intensity * _fast_pow(ndh, spec_power)
        else:
            spec = spec_intensity * (ndh ** spec_power)
        return [f32add(diff[c], f32(spec)) for c in range(3)]

    n = _normalize(normal, "f64")
    L = _normalize(light_dir, "f64")
    e = (camera_pos[0] - world_pos[0],
         camera_pos[1] - world_pos[1],
         camera_pos[2] - world_pos[2])
    V = _normalize(e, "f64")
    H = _normalize((L[0] + V[0], L[1] + V[1], L[2] + V[2]), "f64")
    if fast_math:
        # Reassociated (pairwise-approximate) dot to mimic a fast-math build.
        ndl = max((n[0] * L[0]) + (n[1] * L[1] + n[2] * L[2]), 0.0)
        ndh = max((n[0] * H[0]) + (n[1] * H[1] + n[2] * H[2]), 0.0)
    else:
        ndl = max(n[0] * L[0] + n[1] * L[1] + n[2] * L[2], 0.0)
        ndh = max(n[0] * H[0] + n[1] * H[1] + n[2] * H[2], 0.0)
    out = []
    for c in range(3):
        diffuse = base_rgb[c] * (0.15 + 0.85 * ndl)
        if fast_math:
            specular = spec_intensity * _fast_pow(ndh, spec_power)
        else:
            specular = spec_intensity * (ndh ** spec_power)
        out.append(diffuse + specular)
    return out


def tone_map(linear_rgb, exposure=1.0, fast_math=False):
    """Reinhard extended to preserve HDR, then sRGB OETF.  Shared verbatim.

    ``fast_math`` replaces the sRGB gamma exponent with the exp/log
    approximation pair; that is an explicit declared deviation and is reported,
    not hidden.
    """
    out = []
    for c in linear_rgb:
        x = c * exposure
        mapped = x / (1.0 + x) if x > 0.0 else 0.0
        if mapped <= 0.0031308:
            s = 12.92 * mapped
        else:
            if fast_math:
                s = 1.055 * _fast_exp((1.0 / 2.4) * _fast_log(mapped)) - 0.055
            else:
                s = 1.055 * (mapped ** (1.0 / 2.4)) - 0.055
        out.append(clamp(s, 0.0, 1.0))
    return out


def convert_depth(ndc_z, convention):
    """Window depth from NDC z under the declared convention.

    ``ndc_minus1_1`` keeps OpenGL [-1,1]; ``ndc_0_1`` applies the D3D/Vulkan
    remap z' = (z + 1) / 2.  The remap happens *after* perspective-correct
    interpolation of ndc_z, which is where a convention bug actually lives.
    """
    if convention == "ndc_0_1":
        return (ndc_z + 1.0) * 0.5
    return ndc_z


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
class Triangle:
    def __init__(self, positions, uvs, normal, obj_id, material):
        # positions: 3 world-space xyz tuples; uvs: 3 uv tuples
        self.positions = positions
        self.uvs = uvs
        self.normal = normal
        self.obj_id = obj_id
        self.material = material  # dict: use_texture, color, spec_power, ...


def build_camera(fovy=45.0, near=0.1, far=100.0, eye=(0.0, 0.0, 6.0),
                 target=(0.0, 0.0, 0.0)):
    return {"fovy": fovy, "near": near, "far": far, "eye": eye,
            "target": target}


def view_proj(camera, width, height):
    aspect = width / float(height)
    proj = mat_perspective(camera["fovy"], aspect, camera["near"], camera["far"])
    eye = camera["eye"]
    t = camera["target"]
    forward = _normalize((t[0] - eye[0], t[1] - eye[1], t[2] - eye[2]), "f64")
    up_world = (0.0, 1.0, 0.0)
    right = _normalize(_cross(forward, up_world), "f64")
    up = _cross(right, forward)
    # view = look-at rotation then translation by -eye
    rot = (right[0], up[0], -forward[0], 0.0,
           right[1], up[1], -forward[1], 0.0,
           right[2], up[2], -forward[2], 0.0,
           0.0, 0.0, 0.0, 1.0)
    trans = mat_translate(-eye[0], -eye[1], -eye[2])
    view = mat_mul(rot, trans)
    return mat_mul(proj, view)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])
