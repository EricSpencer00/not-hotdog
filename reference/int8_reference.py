"""Integer-only forward pass in NumPy: the ground truth the JS engine must match.

This is not "the model in Python" — it is the same integer program the browser
runs, written a second time against a different set of primitives. Two
independent implementations of the same spec agreeing bit for bit on thousands
of images is evidence the spec was implemented correctly. One implementation
agreeing with itself is not evidence of anything.

Everything is vectorized, because the naive version does roughly 1.2 billion
requantizations over a validation set and Python cannot do that this decade.
Vectorizing does not weaken the guarantee: NumPy's int64 arithmetic, arithmetic
right shift on signed integers, and two's complement bitwise AND all have the
same semantics as the scalar reference in fixedpoint.py, and test_fixedpoint
pins that down over a million random pairs.

Loaded weights come from the same .npz the JS engine's base64 blob is generated
from, so a parity failure can only mean the kernels disagree.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common import REFDIR

INT32_MIN = -(2 ** 31)
INT32_MAX = 2 ** 31 - 1


# ── vectorized fixed point (semantics identical to fixedpoint.py) ────────────

def srdhm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.int64)
    b = np.asarray(b, dtype=np.int64)
    ab = a * b                       # |ab| < 2^62, exact in int64
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30)).astype(np.int64)
    x = ab + nudge
    # Truncate toward zero, matching C integer division (NumPy's // floors).
    q = np.sign(x) * (np.abs(x) >> 31)
    saturate = (a == INT32_MIN) & (b == INT32_MIN)
    return np.where(saturate, INT32_MAX, q).astype(np.int64)


def rdbpot(x: np.ndarray, exponent: np.ndarray) -> np.ndarray:
    x = x.astype(np.int64)
    exponent = np.asarray(exponent, dtype=np.int64)
    mask = (np.int64(1) << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (x < 0).astype(np.int64)
    return (x >> exponent) + (remainder > threshold).astype(np.int64)


def requantize(acc: np.ndarray, m0: np.ndarray, shift: np.ndarray) -> np.ndarray:
    return rdbpot(srdhm(acc, m0), shift)


def sat_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


# ── model ────────────────────────────────────────────────────────────────────

class Int8Model:
    def __init__(self, npz_path: Path | None = None):
        path = npz_path or (REFDIR / "model_int8.npz")
        data = np.load(path)
        self.w = data["weights"].astype(np.int8)
        self.spec = json.loads(bytes(data["spec"]).decode())
        self.size = self.spec["inputSize"]

    def _slice(self, layer: dict) -> np.ndarray:
        o, n = layer["wOffset"], layer["wLen"]
        return self.w[o:o + n]

    @staticmethod
    def _u8(b64: str) -> np.ndarray:
        """Shift amounts, packed as uint8 (they never exceed 31)."""
        return np.frombuffer(base64.b64decode(b64), dtype=np.uint8).astype(np.int64)

    @staticmethod
    def _i32(b64: str) -> np.ndarray:
        """Per-channel params are packed as base64 int32 to keep the browser
        bundle small; both engines decode the same bytes."""
        return np.frombuffer(base64.b64decode(b64), dtype=np.int32).astype(np.int64)

    # -- kernels ------------------------------------------------------------

    def _conv3x3(self, x, layer):
        """x: (H, W, inC) uint8 -> (outH, outW, outC) uint8"""
        H, W, inC = layer["inH"], layer["inW"], layer["inC"]
        outC, stride = layer["outC"], layer["stride"]
        outH, outW = layer["outH"], layer["outW"]
        w = self._slice(layer).reshape(outC, 3, 3, inC).astype(np.int64)

        xp = np.zeros((H + 2, W + 2, inC), dtype=np.int64)
        xp[1:H + 1, 1:W + 1] = x
        # im2col: (outH, outW, 3, 3, inC)
        cols = np.stack(
            [xp[ky:ky + outH * stride:stride, kx:kx + outW * stride:stride]
             for ky in range(3) for kx in range(3)],
            axis=2,
        ).reshape(outH, outW, 9 * inC)
        acc = cols @ w.reshape(outC, 9 * inC).T          # (outH, outW, outC)
        acc = acc + self._i32(layer["bias"])
        return sat_u8(requantize(acc, self._i32(layer["m0"]), self._u8(layer["shift"])))

    def _dwconv3x3(self, x, layer):
        H, W, C = layer["inH"], layer["inW"], layer["inC"]
        stride, outH, outW = layer["stride"], layer["outH"], layer["outW"]
        w = self._slice(layer).reshape(3, 3, C).astype(np.int64)

        xp = np.zeros((H + 2, W + 2, C), dtype=np.int64)
        xp[1:H + 1, 1:W + 1] = x
        acc = np.zeros((outH, outW, C), dtype=np.int64)
        for ky in range(3):
            for kx in range(3):
                acc += xp[ky:ky + outH * stride:stride,
                          kx:kx + outW * stride:stride] * w[ky, kx]
        acc = acc + self._i32(layer["bias"])
        return sat_u8(requantize(acc, self._i32(layer["m0"]), self._u8(layer["shift"])))

    def _pwconv(self, x, layer):
        H, W = layer["inH"], layer["inW"]
        inC, outC = layer["inC"], layer["outC"]
        w = self._slice(layer).reshape(outC, inC).astype(np.int64)
        acc = x.reshape(H * W, inC).astype(np.int64) @ w.T
        acc = acc + self._i32(layer["bias"])
        out = sat_u8(requantize(acc, self._i32(layer["m0"]), self._u8(layer["shift"])))
        return out.reshape(H, W, outC)

    # -- forward ------------------------------------------------------------

    def forward(self, x_u8: np.ndarray, taps: list | None = None) -> int:
        """x_u8: (size, size, 3) uint8, raw RGB. Returns the int32 logit."""
        assert x_u8.shape == (self.size, self.size, 3), x_u8.shape
        assert x_u8.dtype == np.uint8

        x = x_u8
        for layer in self.spec["layers"]:
            if layer["kind"] == "conv":
                x = self._conv3x3(x, layer)
            elif layer["kind"] == "dw":
                x = self._dwconv3x3(x, layer)
            else:
                x = self._pwconv(x, layer)
            if taps is not None and layer["kind"] != "dw":
                taps.append(x.copy())

        pool = self.spec["pool"]
        acc = x.reshape(pool["hw"], pool["C"]).astype(np.int64).sum(axis=0)
        pooled = sat_u8(requantize(acc, np.int64(pool["m0"]), np.int64(pool["shift"])))

        head = self.spec["head"]
        hw = self.w[head["wOffset"]:head["wOffset"] + head["wLen"]].astype(np.int64)
        logit = int(pooled.astype(np.int64) @ hw) + head["bias"]
        return logit

    def logit_float(self, logit_i32: int) -> float:
        return logit_i32 * self.spec["head"]["logitScale"]


if __name__ == "__main__":
    m = Int8Model()
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (m.size, m.size, 3), dtype=np.uint8)
    print("logit:", m.forward(x))
