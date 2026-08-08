"""Per-kernel test vectors: JS kernels vs the NumPy reference on random tensors.

Catching a bug here rather than in the parity test is the difference between
"the dwconv stride-2 path mishandles the bottom row" and "the model is wrong
somewhere". The cases are chosen for where hand-written convolution goes wrong:
stride-2 on odd input sizes (the last output row reads past the edge), SAME-pad
boundaries, and accumulators pushed to int8 saturation.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from reference.fixedpoint import quantize_multiplier
from reference.int8_reference import requantize, sat_u8
from reference.preprocess import crop_resize

OUT = Path(__file__).resolve().parent.parent / "engine" / "test" / "vectors"


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def qparams(rng, n, extreme=False):
    """Per-channel multipliers. `extreme` pushes toward saturation."""
    ms = rng.uniform(1e-4, 0.6, n) if not extreme else rng.uniform(0.3, 0.99, n)
    m0, sh = zip(*(quantize_multiplier(float(v)) for v in ms))
    return np.array(m0, dtype=np.int32), np.array(sh, dtype=np.int32)


def conv3x3_ref(x, w, bias, m0, shift, H, W, inC, outC, stride):
    outH, outW = -(-H // stride), -(-W // stride)
    xp = np.zeros((H + 2, W + 2, inC), dtype=np.int64)
    xp[1:H + 1, 1:W + 1] = x
    cols = np.stack(
        [xp[ky:ky + outH * stride:stride, kx:kx + outW * stride:stride]
         for ky in range(3) for kx in range(3)], axis=2
    ).reshape(outH, outW, 9 * inC)
    acc = cols @ w.reshape(outC, 9 * inC).T.astype(np.int64) + bias
    return sat_u8(requantize(acc, m0, shift))


def dw_ref(x, w, bias, m0, shift, H, W, C, stride):
    outH, outW = -(-H // stride), -(-W // stride)
    xp = np.zeros((H + 2, W + 2, C), dtype=np.int64)
    xp[1:H + 1, 1:W + 1] = x
    acc = np.zeros((outH, outW, C), dtype=np.int64)
    for ky in range(3):
        for kx in range(3):
            acc += xp[ky:ky + outH * stride:stride,
                      kx:kx + outW * stride:stride] * w[ky, kx].astype(np.int64)
    acc = acc + bias
    return sat_u8(requantize(acc, m0, shift))


def pw_ref(x, w, bias, m0, shift, n, inC, outC):
    acc = x.reshape(n, inC).astype(np.int64) @ w.astype(np.int64).T + bias
    return sat_u8(requantize(acc, m0, shift))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(88)
    cases = []

    # ── conv3x3 (stem): dense, 3 input channels ──────────────────────────────
    for (H, W, outC, stride, extreme) in [
        (96, 96, 16, 2, False),
        (32, 32, 8, 1, False),
        (31, 31, 8, 2, False),   # odd input, stride 2: last row/col is a partial window
        (17, 23, 12, 2, False),  # non-square, odd
        (16, 16, 8, 2, True),    # push the requantizer into saturation
        (5, 5, 4, 1, True),
    ]:
        inC = 3
        x = rng.integers(0, 256, (H, W, inC), dtype=np.uint8)
        w = rng.integers(-127, 128, (outC, 3, 3, inC)).astype(np.int8)
        bias = rng.integers(-5000, 5000, outC).astype(np.int32)
        m0, sh = qparams(rng, outC, extreme)
        y = conv3x3_ref(x, w.astype(np.int64), bias.astype(np.int64), m0, sh,
                        H, W, inC, outC, stride)
        cases.append({
            "kernel": "conv3x3", "H": H, "W": W, "inC": inC, "outC": outC,
            "stride": stride, "x": b64(x), "w": b64(w), "bias": b64(bias),
            "m0": b64(m0), "shift": b64(sh), "y": b64(y),
            "outH": y.shape[0], "outW": y.shape[1],
        })

    # ── depthwise 3x3 ────────────────────────────────────────────────────────
    for (H, W, C, stride, extreme) in [
        (48, 48, 16, 1, False),
        (48, 48, 32, 2, False),
        (25, 25, 16, 2, False),   # odd, stride 2
        (13, 7, 24, 2, False),    # non-square, odd
        (6, 6, 256, 1, False),
        (8, 8, 16, 2, True),
        (3, 3, 4, 1, True),       # smaller than the kernel footprint
    ]:
        x = rng.integers(0, 256, (H, W, C), dtype=np.uint8)
        w = rng.integers(-127, 128, (3, 3, C)).astype(np.int8)
        bias = rng.integers(-5000, 5000, C).astype(np.int32)
        m0, sh = qparams(rng, C, extreme)
        y = dw_ref(x, w, bias.astype(np.int64), m0, sh, H, W, C, stride)
        cases.append({
            "kernel": "dwconv3x3", "H": H, "W": W, "C": C, "stride": stride,
            "x": b64(x), "w": b64(w), "bias": b64(bias),
            "m0": b64(m0), "shift": b64(sh), "y": b64(y),
            "outH": y.shape[0], "outW": y.shape[1],
        })

    # ── pointwise 1x1 ────────────────────────────────────────────────────────
    for (n, inC, outC, extreme) in [
        (48 * 48, 16, 32, False),
        (24 * 24, 64, 64, False),
        (6 * 6, 256, 256, False),
        (12 * 12, 128, 128, True),
        (4, 16, 4, True),
    ]:
        x = rng.integers(0, 256, (n, inC), dtype=np.uint8)
        w = rng.integers(-127, 128, (outC, inC)).astype(np.int8)
        bias = rng.integers(-100000, 100000, outC).astype(np.int32)
        m0, sh = qparams(rng, outC, extreme)
        y = pw_ref(x, w, bias.astype(np.int64), m0, sh, n, inC, outC)
        cases.append({
            "kernel": "pwconv1x1", "n": n, "inC": inC, "outC": outC,
            "x": b64(x), "w": b64(w), "bias": b64(bias),
            "m0": b64(m0), "shift": b64(sh), "y": b64(y),
        })

    # ── global average pool ──────────────────────────────────────────────────
    # hw must be > 1: a pool of one element has multiplier exactly 1.0, which
    # needs a left shift. The real network pools 6x6, and export.py asserts
    # every multiplier stays below 1.
    for (hw, C) in [(36, 256), (144, 128), (4, 16), (576, 32)]:
        x = rng.integers(0, 256, (hw, C), dtype=np.uint8)
        m0, sh = quantize_multiplier(1.0 / hw)
        acc = x.astype(np.int64).sum(axis=0)
        y = sat_u8(requantize(acc, np.int64(m0), np.int64(sh)))
        cases.append({
            "kernel": "globalAvgPool", "hw": hw, "C": C, "x": b64(x),
            "m0": int(m0), "shift": int(sh), "y": b64(y),
        })

    # ── crop + resize ────────────────────────────────────────────────────────
    for (H, W) in [(96, 96), (480, 640), (640, 480), (100, 100), (33, 97), (50, 50), (200, 201)]:
        rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        rgba = np.concatenate([rgb, np.full((H, W, 1), 255, np.uint8)], axis=2)
        y = crop_resize(rgb, 96)
        cases.append({
            "kernel": "cropResize", "H": H, "W": W, "size": 96,
            "rgba": b64(rgba), "y": b64(y),
        })

    path = OUT / "kernels.json"
    path.write_text(json.dumps(cases))
    print(f"wrote {path} — {len(cases)} cases, {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
