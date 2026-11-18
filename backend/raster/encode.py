"""Minial PNG encoder and diagnostic visualizations (stdlib zlib only)."""
import struct
import zlib


def _png_chunk(tag, data):
    chunk = tag + data
    return struct.pack(">I", len(data)) + chunk + \
        struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)


def encode_png(width, height, rgb_rows):
    """rgb_rows: flat list of (r,g,b) tuples in 0..255."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0 (none)
        for x in range(width):
            r, g, b = rgb_rows[y * width + x]
            raw += bytes((r & 255, g & 255, b & 255))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" +
            _png_chunk(b"IHDR", ihdr) +
            _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6)) +
            _png_chunk(b"IEND", b""))


def to_bytes_png(width, height, rgb_float):
    rows = []
    for px in rgb_float:
        rows.append(tuple(int(round(max(0.0, min(1.0, c)) * 255.0))
                          for c in px[:3]))
    return encode_png(width, height, rows)


_ID_PALETTE = [
    (0, 0, 0), (230, 70, 60), (60, 160, 90), (70, 110, 230),
    (220, 180, 50), (180, 80, 200), (60, 200, 200), (240, 130, 60),
]


def id_png(width, height, obj_ids):
    rows = [_ID_PALETTE[oid % len(_ID_PALETTE)] for oid in obj_ids]
    return encode_png(width, height, rows)


def depth_png(width, height, depth, covered):
    finite = [d for d, c in zip(depth, covered) if c and d == d and
              d not in (float("inf"), float("-inf"))]
    if not finite:
        return encode_png(width, height, [(0, 0, 0)] * (width * height))
    lo, hi = min(finite), max(finite)
    span = (hi - lo) or 1.0
    rows = []
    for d, c in zip(depth, covered):
        if not c:
            rows.append((0, 0, 0))
            continue
        t = (d - lo) / span
        t = max(0.0, min(1.0, t))
        # blue->red ramp
        rows.append((int(255 * t), int(60 + 120 * (1 - t)),
                     int(255 * (1 - t))))
    return encode_png(width, height, rows)


def heatmap_png(width, height, values, covered, cap=None):
    vals = [v for v, c in zip(values, covered) if c]
    if cap is None:
        cap = max(vals) if vals else 1.0
    cap = cap if cap > 0 else 1.0
    rows = []
    for v, c in zip(values, covered):
        if not c:
            rows.append((25, 25, 28))  # background distinct from model
            continue
        t = max(0.0, min(1.0, v / cap))
        # green -> yellow -> red
        r = int(255 * min(1.0, t * 2.0))
        g = int(255 * min(1.0, 2.0 - t * 2.0))
        rows.append((r, g, 40))
    return encode_png(width, height, rows)


def edge_png(width, height, edge, covered):
    rows = []
    for e, c in zip(edge, covered):
        if e:
            rows.append((255, 80, 255))
        elif c:
            rows.append((90, 90, 95))
        else:
            rows.append((15, 15, 18))
    return encode_png(width, height, rows)


def coverage_diff_png(width, height, ref_cov, tst_cov):
    rows = []
    for r, t in zip(ref_cov, tst_cov):
        if r and t:
            rows.append((70, 70, 78))
        elif r and not t:
            rows.append((255, 70, 70))     # lost on test path
        elif t and not r:
            rows.append((70, 160, 255))    # added by test path
        else:
            rows.append((15, 15, 18))
    return encode_png(width, height, rows)


_STAGE_COLORS = {
    "non_finite": (255, 0, 255),
    "coverage": (255, 60, 60),
    "object_id": (230, 130, 20),
    "depth": (220, 200, 40),
    "texture_sampling": (60, 200, 120),
    "shading": (60, 140, 240),
    "blend_write": (240, 160, 60),
    "color_mapping": (150, 90, 230),
}


def stage_png(width, height, stage_index, covered):
    """Colour each pixel by the first pipeline stage where it diverged.

    ``stage_index`` is a full-length list of stage names or ``None`` (agreed
    model pixel).  Background is dark; agreed model pixels are neutral grey.
    This is the per-pixel attribution view.
    """
    rows = []
    for i in range(width * height):
        stage = stage_index[i]
        if stage is None:
            rows.append((60, 60, 66) if covered[i] else (18, 18, 20))
        else:
            rows.append(_STAGE_COLORS.get(stage, (255, 255, 255)))
    return encode_png(width, height, rows)
