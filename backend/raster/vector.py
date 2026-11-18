"""Deterministic float32 emulation and frame buffers (stdlib only).

The whole tool deliberately depends only on the Python standard library so it
runs on a bare interpreter.  `f32` rounds every operation to IEEE-754 binary32
the way a GPU-style single-precision pipeline would, while the reference path
keeps Python native float64 intermediates.  This is where the genuine
numerical differences between the two compute paths come from -- nothing is
injected or faked.
"""
import math
import struct

_F32_INF = struct.unpack("f", struct.pack("f", float("inf")))[0]


def f32(x):
    """Round a Python float to nearest binary32 (ties to even)."""
    if x != x:  # NaN passes through
        return x
    return struct.unpack("f", struct.pack("f", x))[0]


def f32add(a, b):
    return f32(f32(a) + f32(b))


def f32sub(a, b):
    return f32(f32(a) - f32(b))


def f32mul(a, b):
    return f32(f32(a) * f32(b))


def f32div(a, b):
    return f32(f32(a) / f32(b))


def f32dot3(a, b):
    x = f32mul(a[0], b[0])
    y = f32add(x, f32mul(a[1], b[1]))
    z = f32add(y, f32mul(a[2], b[2]))
    return z


def f32vec3(v):
    return (f32(v[0]), f32(v[1]), f32(v[2]))


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


class Framebuffer:
    """Holds linear HDR color (float), device depth (float) and object id.

    Background stays exactly (0,0,0) color with id 0 and depth +inf, so the
    comparison stage can distinguish true background from a rendered model
    pixel that happens to be black.  ``albedo`` records the base color that
    entered shading so the attribution stage can separate texture/sampling
    from the lighting + tone-map tail.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        n = width * height
        self.color = [[0.0, 0.0, 0.0] for _ in range(n)]
        self.albedo = [[0.0, 0.0, 0.0] for _ in range(n)]
        self.depth = [_F32_INF] * n
        self.obj_id = [0] * n
        # True once a primitive covered the pixel (background never sets it).
        self.covered = bytearray(n)

    def idx(self, x, y):
        return y * self.width + x
