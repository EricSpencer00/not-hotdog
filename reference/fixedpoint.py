"""gemmlowp fixed-point primitives, in Python, as the ground truth.

These are the operations that turn an int32 accumulator back into a uint8
activation. They are the only place in the whole pipeline where getting the
arithmetic subtly wrong produces output that still looks plausible — a bad
convolution gives you garbage, a bad requantization gives you a model that is
three points less accurate for no visible reason.

Python integers are arbitrary precision, so this file is trivially correct and
serves as the reference that engine/fixedpoint.js must match bit for bit.
"""

from __future__ import annotations

INT32_MIN = -(2 ** 31)
INT32_MAX = 2 ** 31 - 1


def _trunc_div(n: int, d: int) -> int:
    """Integer division truncating toward zero (C semantics), not floor."""
    q = abs(n) // abs(d)
    return -q if (n < 0) != (d < 0) else q


def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:
    """int32 x int32 -> int32, returning round(a*b / 2^31).

    The 'doubling' is because both operands are Q0.31 fixed-point, so the
    product is Q0.62 and taking the high word needs a shift of 31, not 32.
    """
    assert INT32_MIN <= a <= INT32_MAX, a
    assert INT32_MIN <= b <= INT32_MAX, b
    if a == INT32_MIN and b == INT32_MIN:
        # The one product that will not fit: 2^62 rounds to 2^31, off by one.
        return INT32_MAX
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    return _trunc_div(ab + nudge, 1 << 31)


def rounding_divide_by_pot(x: int, exponent: int) -> int:
    """Divide by 2^exponent, rounding half away from zero.

    Uses an arithmetic (flooring) shift plus an explicit remainder test, which
    is what gemmlowp does and what makes the result independent of the host's
    division semantics.
    """
    assert exponent >= 0
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> exponent) + (1 if remainder > threshold else 0)


def multiply_by_quantized_multiplier(acc: int, m0: int, shift: int) -> int:
    """Apply the real multiplier M = m0 * 2^-(31+shift) to an int32 accumulator."""
    return rounding_divide_by_pot(saturating_rounding_doubling_high_mul(acc, m0), shift)


def quantize_multiplier(m: float) -> tuple[int, int]:
    """Decompose a real multiplier in (0, 1) into (m0, shift).

    m == m0 * 2^-(31+shift), with m0 in [2^30, 2^31) so the fixed-point
    representation always uses its full precision.
    """
    if m <= 0:
        raise ValueError(f"multiplier must be positive, got {m}")
    shift = 0
    while m < 0.5:
        m *= 2.0
        shift += 1
    while m >= 1.0:
        m /= 2.0
        shift -= 1
    q = int(round(m * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        shift -= 1
    if shift < 0:
        raise ValueError(
            f"multiplier >= 1 needs a left shift (shift={shift}); "
            "the export path assumes right shifts only"
        )
    assert (1 << 30) <= q < (1 << 31), q
    return q, shift


def saturate_uint8(x: int) -> int:
    return 0 if x < 0 else (255 if x > 255 else x)
