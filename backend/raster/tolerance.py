"""Acceptance rules with explicit provenance and a frozen version.

Two layers, kept deliberately separate:

* **pixelwise (v1, preserved)** -- the original per-pixel absolute/relative
  statistics and diagnostic heatmaps.  It still runs on every comparison and
  is what the heatmaps are built from.  v1 never decides alone; the versioned
  acceptance rules below own the pass/fail verdict.
* **acceptance rules (versioned)** -- each render *operation* declares which
  pixels are its **effective pixels** and what deviation is allowed on them.
  A failure produces a failure *class* (``non_finite`` / ``channel_swap`` /
  ``depth_convention`` / ``coverage_struct`` / ``object_id`` /
  ``precision_exceeded``) instead of one loose global colour threshold.

Frozen semantics: reports stamp ``rules_version`` and the full rule text.  A
later edit to this file cannot retroactively rewrite an old report -- the
report carries what it was judged against.
"""

RULES_VERSION = "acceptance-2026.11-v2"

RULES_CHANGELOG = {
    "acceptance-2026.11-v1": (
        "初版逐像素比较：仅统计模型有效区域；覆盖/对象标识不一致记为结构差异；"
        "sRGB 绝对误差 0.5/255、深度硬限 1e-4。"),
    RULES_VERSION: (
        "按操作与有效像素制定验收：拆出 coverage/object_id/depth/"
        "final_color_inherited/final_color_primary/linear_hdr/finite 七条规则；"
        "显式通道交换探测与深度约定探测；非有限值独立失败类；"
        "允许偏差按像素类别(内部/边缘)分别给出。"),
}

TOLERANCES = {
    # ---- pixelwise v1 (preserved verbatim; not used for the verdict) --------
    "srgb_abs_max": 0.5 / 255.0,
    "srgb_abs_p99": 0.25 / 255.0,
    "color_rel_eps": 1e-4,
    "color_rel_max": 0.02,
    "depth_rel_eps": 1e-7,
    "depth_abs_hard": 1e-4,
    "depth_abs_typical": 1e-6,
    "coverage_tolerance": 0,

    # ---- acceptance rules (versioned) --------------------------------------
    "rules_version": RULES_VERSION,
    "rules_changelog": RULES_CHANGELOG,

    # Per-operation deviation limits on explicit effective-pixel sets.
    # eps values are "beyond half an 8-bit code" only where an 8-bit store is
    # genuinely part of the operation (rgba8 texture / fixed8 blend); purely
    # float storage must stay at normal binary32 rounding.
    "operations": {
        "coverage": {
            "effective_pixels": "模型并集 (ref_cov 或 test_cov)",
            "allowed_disagreement": 0,
            "failure_class": "coverage_struct",
            "provenance": "覆盖归属无数值容差；缺失/新增一个模型像素即结构差异，"
                          "不得被颜色阈值吸收。",
        },
        "object_id": {
            "effective_pixels": "两路径共同覆盖像素",
            "allowed_mismatch": 0,
            "failure_class": "object_id",
            "provenance": "对象标识不一致代表几何归属错误，与浮点误差无关。",
        },
        "depth": {
            "effective_pixels": "共同覆盖且深度有限",
            "allowed_abs_internal": 1e-4,
            "allowed_abs_edge": 1e-4,
            "failure_class": "depth_convention",
            "provenance": "NDC 深度差超过 1e-4 视为深度约定/对应错误；f32 深度舍入"
                          "典型 1e-6~1e-7，远低于该硬限。",
        },
        "final_color_inherited": {
            "effective_pixels": "共同覆盖、有限、且在上一设备中已被同一对象覆盖",
            "allowed_abs_internal": 0.5 / 255.0,
            "allowed_abs_edge": 1.0 / 255.0,
            "allowed_rel": 0.02,
            "failure_class": "precision_exceeded",
            "provenance": "输入颜色已由上游设备确定，只允许采样/写回引入的偏差；"
                          "内部=半个量化步长，边缘带放宽到一步长。",
        },
        "final_color_primary": {
            "effective_pixels": "上一设备未覆盖但本设备覆盖(差异被本次引入)的共同覆盖像素",
            "allowed_abs_internal": 0.5 / 255.0,
            "allowed_abs_edge": 1.0 / 255.0,
            "allowed_rel": 0.02,
            "failure_class": "precision_exceeded",
            "provenance": "差异由本次设备的采样/颜色路径首次引入，按同口径判定。",
        },
        # fixed8 blend: the linear colour is stored on a 1/255 lattice *before*
        # tone mapping.  Tone mapping has slope <1 but specular values are large
        # in linear space, so an 8-bit linear store is a genuinely lossy
        # capability, not a rounding detail.  Its own operation keeps the limit
        # explicit instead of loosening the float path.
        "blend_write_quantized": {
            "effective_pixels": "共同覆盖、对象一致、被测设备声明 fixed8_blend",
            "allowed_abs_internal": 4.0 / 255.0,
            "allowed_abs_edge": 4.0 / 255.0,
            "allowed_rel": 0.05,
            "failure_class": "blend_quantization_exceeded",
            "provenance": "8 位线性写回在强高光下会被映射放大；允许最多约 4 个 sRGB "
                          "量化步长，超过则说明写回量化已产生可见偏差。",
        },
        "linear_hdr": {
            "effective_pixels": "共同覆盖、对象标识一致且有限",
            "allowed_rel": 0.02,
            "failure_class": "precision_exceeded",
            "provenance": "线性 HDR 相对误差用于定位色调映射之前的颜色路径偏差；"
                          "Reinhard 会压缩差异，故需在映射前检查。",
        },
        "finite": {
            "effective_pixels": "反照率/颜色/深度的所有输出",
            "allowed_nonfinite": 0,
            "failure_class": "non_finite",
            "provenance": "NaN/Inf 独立失败类；任何全局阈值都不得将其吞掉。",
        },
    },

    # Explicit structural probes: a whole-channel permutation that explains the
    # error far better than any numeric tolerance is a channel swap, not noise.
    "channel_swap": {
        "trigger_abs": 2.0 / 255.0,
        "explained_abs": 0.25 / 255.0,
        "min_improvement_factor": 8.0,
        "failure_class": "channel_swap",
        "provenance": "若交换通道后误差降到 0.25/255 以下且至少改善 8 倍，"
                      "而原始误差已超过 2/255，则判为通道交换而非数值噪声。",
    },

    # Depth convention probe: a large, sign-consistent depth offset across the
    # frame is a convention mismatch (e.g. 0..1 vs -1..1), not rounding.
    "depth_convention": {
        "suspicion_abs": 1e-3,
        "failure_class": "depth_convention",
        "provenance": "深度差普遍超过 1e-3 且与几何无关地同向，判为深度约定不一致。",
    },

    "provenance": {
        "srgb_abs_max": "0.5/255：8 位 sRGB 半个量化步长，f32 正确舍入不应超过。",
        "srgb_abs_p99": "0.25/255：99 分位收紧到四分之一步长，防止个别高光像素掩盖系统性偏差。",
        "color_rel_eps": "1e-4：暗部相对误差分母下限，低于此值改用绝对误差判定。",
        "depth_abs_hard": "1e-4：超过即判深度对应错误；典型 f32 深度舍入约 1e-6~1e-7。",
        "coverage_tolerance": "覆盖/对象标识无容差，任何不一致都按结构错误上报。",
    },
}
