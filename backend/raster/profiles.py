"""Traceable render capability + precision profiles.

A *profile* is the complete set of rendering decisions that can move pixels:
numeric precision, fast-math, texture storage format, blend capability, depth
convention and depth test.  Every field carries provenance text so a report
can explain *why* a configuration is what it is.  Profiles are immutable data;
they are resolved (with a sha256 over the canonical JSON) before rendering and
the resolved copy is embedded in every report.

Profiles are deliberately split from devices.  A device *declares* which
capabilities it can execute; a (device, profile) pair is only runnable when
every requested field is in the device's declared capability set.  Anything
else is reported as ``unsupported_capability`` -- never silently approximated.
"""
import copy
import hashlib
import json

PROFILE_FIELDS = ("precision", "fast_math", "texture_format",
                  "blend_capability", "depth_convention", "depth_test")

FIELD_PROVENANCE = {
    "precision": "运算精度：f64=IEEE 双精度参考；f32=每步按 binary32 舍入（GPU 单精度形态）。",
    "fast_math": "快速数学：启用后用指数近似倒数 + 牛顿迭代、重结合点积，属显式声明的近似通路。",
    "texture_format": ("纹理存储：float32_bilinear=浮点双线性；"
                       "rgba8_linear=采样后量化到 8 位线性格点(1/255)。"),
    "blend_capability": ("混合/写回能力：float_blend=线性浮点写回；"
                         "fixed8_blend=写回时量化到 8 位线性格点(1/255)。"),
    "depth_convention": ("深度约定：ndc_minus1_1=OpenGL 风格 [-1,1(NDC)]；"
                        "ndc_0_1=D3D/Vulkan 风格 [0,1]（诊断用，刻意引入约定差异）。"),
    "depth_test": "深度测试：less=严格小于；less_equal=近重合表面后写覆盖。",
}

# id -> raw profile.  ``diagnostic_only`` profiles are intentionally broken
# (e.g. a depth-convention mismatch) and are used by tests/CLI to prove the
# classifier distinguishes the failure; they are not offered in the UI picker.
PROFILES = {
    "reference_f64": {
        "label": "参考 f64 · 严格 IEEE · 浮点纹理 · 浮点写回",
        "precision": "f64",
        "fast_math": False,
        "texture_format": "float32_bilinear",
        "blend_capability": "float_blend",
        "depth_convention": "ndc_minus1_1",
        "depth_test": "less",
        "diagnostic_only": False,
    },
    "strict_f32": {
        "label": "f32 严格 IEEE · 浮点纹理 · 浮点写回",
        "precision": "f32",
        "fast_math": False,
        "texture_format": "float32_bilinear",
        "blend_capability": "float_blend",
        "depth_convention": "ndc_minus1_1",
        "depth_test": "less",
        "diagnostic_only": False,
    },
    "fast_math_f32": {
        "label": "f32 快速数学 · 浮点纹理 · 浮点写回",
        "precision": "f32",
        "fast_math": True,
        "texture_format": "float32_bilinear",
        "blend_capability": "float_blend",
        "depth_convention": "ndc_minus1_1",
        "depth_test": "less",
        "diagnostic_only": False,
    },
    "rgba8_texture_f32": {
        "label": "f32 · 纹理降级 rgba8 · 浮点写回",
        "precision": "f32",
        "fast_math": False,
        "texture_format": "rgba8_linear",
        "blend_capability": "float_blend",
        "depth_convention": "ndc_minus1_1",
        "depth_test": "less",
        "diagnostic_only": False,
    },
    "fixed8_blend_f32": {
        "label": "f32 · 浮点纹理 · 定长 8 位写回",
        "precision": "f32",
        "fast_math": False,
        "texture_format": "float32_bilinear",
        "blend_capability": "fixed8_blend",
        "depth_convention": "ndc_minus1_1",
        "depth_test": "less",
        "diagnostic_only": False,
    },
    "degraded_rgba8_fixed8_f32": {
        "label": "f32 · 快速数学 · rgba8 纹理 · 定长 8 位写回",
        "precision": "f32",
        "fast_math": True,
        "texture_format": "rgba8_linear",
        "blend_capability": "fixed8_blend",
        "depth_convention": "ndc_minus1_1",
        "depth_test": "less",
        "diagnostic_only": False,
    },
    # Diagnostic probe: identical to strict_f32 except the depth convention.
    # Used to prove the depth-convention classifier is distinct from a loose
    # numeric threshold.  Never offered by the UI.
    "f32_ndc01_depth": {
        "label": "诊断：f32 + 0..1 深度约定（刻意错误）",
        "precision": "f32",
        "fast_math": False,
        "texture_format": "float32_bilinear",
        "blend_capability": "float_blend",
        "depth_convention": "ndc_0_1",
        "depth_test": "less",
        "diagnostic_only": True,
    },
}

DEFAULT_PROFILE_BY_DEVICE = {
    "cpu_ref": "reference_f64",
    "vector_f32": "strict_f32",
    "gpu_cuda": "strict_f32",
}


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str)


def resolve_profile(profile_id):
    """Return a fully described, hashed profile.  Unknown ids raise KeyError."""
    raw = PROFILES[profile_id]
    profile = {"id": profile_id, "label": raw["label"],
               "diagnostic_only": raw["diagnostic_only"],
               "sources": {}}
    for field in PROFILE_FIELDS:
        profile[field] = raw[field]
        profile["sources"][field] = FIELD_PROVENANCE[field]
    profile["sha256"] = hashlib.sha256(
        _canonical(profile).encode("utf-8")).hexdigest()
    return profile


def profile_public(profile_id):
    """UI-facing metadata (safe to serialise)."""
    p = resolve_profile(profile_id)
    return copy.deepcopy(p)


def all_profiles(include_diagnostic=False):
    out = []
    for pid, raw in PROFILES.items():
        if raw["diagnostic_only"] and not include_diagnostic:
            continue
        out.append(profile_public(pid))
    return out


def profile_for_default(device_id):
    return DEFAULT_PROFILE_BY_DEVICE.get(device_id, "reference_f64")
