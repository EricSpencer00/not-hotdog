"""Emit test vectors for the JS fixed-point implementation.

Writes a flat little-endian int32 binary of (a, b, srdhm, shift, requantize)
tuples. Vectors are generated rather than committed because a million of them
is 20 MB; the generator is deterministic, so CI reproduces the exact same set.

Coverage is deliberately not just uniform random: the failure modes of a
hand-rolled 64-bit multiply cluster at sign boundaries, at powers of two, and
at the exact values where the high/low split carries. Those get enumerated
exhaustively as a cross product before the random bulk.
"""

from __future__ import annotations

import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reference.fixedpoint import (  # noqa: E402
    INT32_MAX,
    INT32_MIN,
    rounding_divide_by_pot,
    saturating_rounding_doubling_high_mul,
)

N_RANDOM = 1_000_000
OUT = Path(__file__).resolve().parent.parent / "engine" / "test" / "vectors"

BOUNDARY = [
    0, 1, -1, 2, -2,
    255, 256, -256,
    32767, 32768, -32768, -32769,
    65535, 65536, -65536,
    (1 << 30), -(1 << 30), (1 << 30) - 1, (1 << 30) + 1,
    INT32_MAX, INT32_MIN, INT32_MAX - 1, INT32_MIN + 1,
    0x7fff0000, -0x7fff0000, 0x0000ffff, 0x00010000,
    1431655765, -1431655765,  # 0x55555555 pattern
    -1431655766,
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260808)

    pairs: list[tuple[int, int, int]] = []

    # Exhaustive boundary cross product, across several shift amounts.
    for a in BOUNDARY:
        for b in BOUNDARY:
            for shift in (0, 1, 7, 15, 23, 31):
                pairs.append((a, b, shift))

    # Random bulk. `b` is biased toward the [2^30, 2^31) range that real
    # quantized multipliers actually occupy, so most vectors exercise the case
    # the model will hit rather than only the arithmetic corners.
    for _ in range(N_RANDOM):
        a = rng.randint(INT32_MIN, INT32_MAX)
        if rng.random() < 0.6:
            b = rng.randint(1 << 30, (1 << 31) - 1)
        else:
            b = rng.randint(INT32_MIN, INT32_MAX)
        pairs.append((a, b, rng.randint(0, 31)))

    buf = bytearray()
    for a, b, shift in pairs:
        h = saturating_rounding_doubling_high_mul(a, b)
        r = rounding_divide_by_pot(h, shift)
        buf += struct.pack("<5i", a, b, h, shift, r)

    path = OUT / "fixedpoint.bin"
    path.write_bytes(buf)
    print(f"wrote {path} — {len(pairs):,} vectors, {len(buf) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
