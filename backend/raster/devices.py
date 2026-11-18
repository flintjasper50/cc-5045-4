"""Honest device discovery and capability declaration.

A GPU backend is reported as *available* only when a real accelerator API can
be opened on this machine.  When none is present the tool still ships two
runnable compute implementations (the two CPU paths) and the UI clearly marks
the GPU row as "not executed / unavailable".  Nothing here fabricates a GPU
result -- if a GPU cannot be opened, no GPU image is ever produced.

Each device additionally *declares* a capability set.  A render profile is
runnable on a device only when every requested capability is in that set.
Unsupported combinations are reported as ``unsupported_capability`` with the
missing fields spelled out -- the tool never silently downgrades a requested
format and never pretends an untested combination was verified.
"""
import os
import shutil
import subprocess

from .backends import CpuRefPath, VectorPath

# Capability declarations.  These describe what each implementation can
# genuinely execute today; they are not aspirations.
CAPABILITIES = {
    "cpu_ref": {
        "precision": ["f64", "f32"],
        "fast_math": [False, True],
        "texture_format": ["float32_bilinear", "rgba8_linear"],
        "blend_capability": ["float_blend", "fixed8_blend"],
        "depth_convention": ["ndc_minus1_1", "ndc_0_1"],
        "depth_test": ["less", "less_equal"],
    },
    "vector_f32": {
        "precision": ["f32"],
        "fast_math": [False, True],
        "texture_format": ["float32_bilinear", "rgba8_linear"],
        "blend_capability": ["float_blend", "fixed8_blend"],
        "depth_convention": ["ndc_minus1_1", "ndc_0_1"],
        "depth_test": ["less", "less_equal"],
    },
    # A real GPU lane is expected to be f32-only; the exact fast-math /
    # format support of a given vendor is NOT assumed.  Only the conservative
    # subset the CUDA backends would implement is declared.
    "gpu_cuda": {
        "precision": ["f32"],
        "fast_math": [False, True],
        "texture_format": ["float32_bilinear"],
        "blend_capability": ["float_blend"],
        "depth_convention": ["ndc_minus1_1"],
        "depth_test": ["less"],
    },
}


def _has_nvidia():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except (subprocess.SubprocessError, OSError):
        return None
    return None


def _cuda_module_present():
    """A CUDA-capable Python module would back a real GPU rasterizer."""
    for mod in ("cupy", "numba.cuda", "torch"):
        try:
            m = __import__(mod, fromlist=["*"])
        except Exception:
            continue
        try:
            if mod == "torch" and not getattr(m, "cuda", None):
                continue
            if mod == "torch" and not m.cuda.is_available():
                continue
            if mod == "numba.cuda":
                if not m.is_available():
                    continue
            return mod
        except Exception:
            continue
    return None


def profile_supported(device_id, profile):
    """Return (ok, missing_fields).  Every profile field must be declared."""
    caps = CAPABILITIES.get(device_id)
    if caps is None:
        return False, list(profile.keys())
    missing = []
    for field in ("precision", "fast_math", "texture_format",
                  "blend_capability", "depth_convention", "depth_test"):
        if field not in caps or profile.get(field) not in caps[field]:
            missing.append(field)
    return (not missing), missing


def discover_devices():
    """Return metadata for every backend plus an honest GPU status block."""
    gpu_name = _has_nvidia()
    cuda_mod = _cuda_module_present()
    gpu_available = bool(gpu_name and cuda_mod)

    # Cross-device comparison case: same declared profile on two different
    # compute paths.  At identical precision (strict_f32) any remaining
    # silhouette disagreement must come from traversal/fill convention, so the
    # classifier should name it coverage_struct -- not precision_exceeded.
    cross_device_cases = [{
        "id": "fill_convention_same_profile",
        "scene": "depth_near",
        "width": 128,
        "devices": ["cpu_ref", "vector_f32"],
        "profiles": {"cpu_ref": "strict_f32", "vector_f32": "strict_f32"},
        "expects": ("两套实现即使同为 strict_f32，在近重合/剪影像素上仍会不同；"
                    "应判为 coverage_struct/object_id，而非 precision_exceeded。"),
    }]

    devices = [
        {
            "id": CpuRefPath.name,
            "label": CpuRefPath.label,
            "kind": "cpu",
            "precision": "float64",
            "available": True,
            "capabilities": CAPABILITIES[CpuRefPath.name],
            "executed_note": "始终可运行：纯标准库标量光栅化，作为参考真值。",
        },
        {
            "id": VectorPath.name,
            "label": VectorPath.label,
            "kind": "cpu-simd",
            "precision": "float32 (binary32 舍入)",
            "available": True,
            "capabilities": CAPABILITIES[VectorPath.name],
            "executed_note": "始终可运行：分块逐像素、f32 运算，模拟 GPU 单精度数据通路。",
        },
        {
            "id": "gpu_cuda",
            "label": "GPU 渲染路径 (CUDA/单精度)",
            "kind": "gpu",
            "precision": "float32",
            "available": gpu_available,
            "capabilities": CAPABILITIES["gpu_cuda"],
            "executed_note": _gpu_note(gpu_name, cuda_mod),
        },
    ]
    return devices, {
        "gpu_detected": bool(gpu_name),
        "gpu_hardware": gpu_name,
        "cuda_module": cuda_mod,
        "gpu_can_execute": gpu_available,
        "coverage": _coverage_text(gpu_available),
        "cross_device_cases": cross_device_cases,
    }


def _gpu_note(gpu_name, cuda_mod):
    if gpu_name and cuda_mod:
        return "检测到 GPU(%s) 与 %s，可执行真实 GPU 路径。" % (gpu_name, cuda_mod)
    if gpu_name and not cuda_mod:
        return ("检测到硬件 %s，但缺少 CUDA 计算模块(%s)；本次不执行 GPU 渲染，"
                "结果区不会出现 GPU 图像。" % (gpu_name, cuda_mod or "无"))
    return "未检测到可用 GPU 设备/驱动；GPU 路径标记为未执行，不生成任何 GPU 图像。"


def _coverage_text(can_run_gpu):
    if can_run_gpu:
        return "CPU 参考、f32 向量路径与真实 GPU 路径三者均可执行并两两比较。"
    return ("本次覆盖：两条真实 CPU 计算实现(f64 参考 / f32 分块)实际执行并比较。"
            "GPU 路径因无可用设备或计算栈而未执行——未伪造任何 GPU 输出。"
            "GPU 的精度/格式组合尚未在本机验证，界面中会单独标出。")
