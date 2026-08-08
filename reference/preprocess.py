"""Centre-crop + box-filter resize, in integers, matching kernels.js cropResize.

Preprocessing is the one place bit-exactness can leak across browsers. The
obvious implementation is canvas `drawImage`, but its resampling filter is not
specified precisely and differs between Safari, Chrome and Firefox, so a model
that matched the reference in one browser would drift in another.

So the resize is done by hand: an area average over integer pixel ranges with
round-half-up, which has exactly one possible answer. This file is the same
arithmetic on the same integers, and test_resize pins the two together.
"""

from __future__ import annotations

import numpy as np


def crop_resize(rgb: np.ndarray, size: int) -> np.ndarray:
    """rgb: (H, W, 3) uint8 -> (size, size, 3) uint8."""
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    h, w = rgb.shape[:2]
    side = min(w, h)
    off_x = (w - side) >> 1
    off_y = (h - side) >> 1

    out = np.zeros((size, size, 3), dtype=np.uint8)
    for oy in range(size):
        sy0 = (oy * side) // size
        sy1 = ((oy + 1) * side) // size
        if sy1 <= sy0:
            sy1 = sy0 + 1
        for ox in range(size):
            sx0 = (ox * side) // size
            sx1 = ((ox + 1) * side) // size
            if sx1 <= sx0:
                sx1 = sx0 + 1
            patch = rgb[off_y + sy0:off_y + sy1, off_x + sx0:off_x + sx1]
            count = patch.shape[0] * patch.shape[1]
            total = patch.reshape(-1, 3).astype(np.int64).sum(axis=0)
            out[oy, ox] = (total + (count >> 1)) // count
    return out


def from_rgba(rgba: np.ndarray, w: int, h: int, size: int) -> np.ndarray:
    """Convenience for canvas-shaped input: (h*w*4,) uint8 -> (size, size, 3)."""
    return crop_resize(rgba.reshape(h, w, 4)[:, :, :3].copy(), size)
