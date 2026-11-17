"""Tolerances with explicit provenance.

Every threshold states *why* it is that size rather than being a magic
number.  Color tolerances live in sRGB [0,1] units; depth in NDC units.
"""

TOLERANCES = {
    # sRGB 8-bit storage quantization is 1/255 ~= 0.00392.  Two correctly
    # rounded f32/f64 pipelines must agree well inside one code value; we
    # allow half an 8-bit step for ordinary interior pixels.
    "srgb_abs_max": 0.5 / 255.0,
    "srgb_abs_p99": 0.25 / 255.0,

    # Relative comparisons are only meaningful above this floor; below it the
    # absolute threshold governs (avoids dividing by near-zero dark colors).
    "color_rel_eps": 1e-4,
    "color_rel_max": 0.02,

    # Depth is compared in window/NDDC depth.  f32 mantissa is 24 bits, so at
    # a typical depth value ~0.9 the rounding floor is around 1e-7; allow a
    # conservative band.  Anything beyond the "hard" limit is treated as a
    # correspondence error, not precision noise.
    "depth_rel_eps": 1e-7,
    "depth_abs_hard": 1e-4,
    "depth_abs_typical": 1e-6,

    # A coverage / object-id disagreement is structural and has no numeric
    # tolerance: it is always reported as a pixel-correspondence error.
    "coverage_tolerance": 0,

    "provenance": {
        "srgb_abs_max": "0.5/255：8 位 sRGB 半个量化步长，f32 正确舍入不应超过。",
        "srgb_abs_p99": "0.25/255：99 分位收紧到四分之一步长，防止个别高光像素掩盖系统性偏差。",
        "color_rel_eps": "1e-4：暗部相对误差分母下限，低于此值改用绝对误差判定。",
        "depth_abs_hard": "1e-4：超过即判深度对应错误；典型 f32 深度舍入约 1e-6~1e-7。",
        "coverage_tolerance": "覆盖/对象标识无容差，任何不一致都按结构错误上报。",
    },
}
