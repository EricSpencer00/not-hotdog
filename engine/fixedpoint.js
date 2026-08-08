// gemmlowp fixed-point primitives in plain JavaScript.
//
// The problem this file exists to solve: requantizing an int32 accumulator
// requires the high 32 bits of a signed 64-bit product, and JavaScript has no
// 64-bit integers. A Number is an IEEE-754 double with a 53-bit mantissa, so
// `a * b` for two int32 operands silently loses the low bits and is unusable.
// BigInt is exact but far too slow to run once per output element.
//
// The way out is to never form the 64-bit product at all. Split each operand
// into a signed high half and an unsigned low half:
//
//     aHi = a >> 16        (signed, |aHi| <= 2^15)
//     aLo = a & 0xffff     (unsigned, < 2^16)
//     a   = aHi * 2^16 + aLo          <- exact for two's complement int32
//
// then
//
//     a*b = A*2^32 + B*2^16 + C,  A = aHi*bHi, B = aHi*bLo + aLo*bHi, C = aLo*bLo
//
// Every one of A, B and C fits in a double exactly (|A| <= 2^30, |B| < 2^32,
// C < 2^32), and so does T = B*2^16 + C + nudge, because |T| < 2^49. Since
// a*b + nudge = A*2^32 + T = (2A)*2^31 + T, the quotient by 2^31 falls out of
// a floor-divide of T alone. No 64-bit value is ever materialised.
//
// Verified against reference/fixedpoint.py over 1,000,000 random pairs plus
// every boundary combination. See engine/test/fixedpoint.test.js.

export const INT32_MIN = -2147483648;
export const INT32_MAX = 2147483647;

const TWO_31 = 2147483648; // 2^31
const TWO_16 = 65536;

/**
 * round(a * b / 2^31) for int32 a, b, saturating the single case that overflows.
 * @param {number} a int32
 * @param {number} b int32
 * @returns {number} int32
 */
export function srdhm(a, b) {
  if (a === INT32_MIN && b === INT32_MIN) return INT32_MAX;

  const aHi = a >> 16;
  const aLo = a & 0xffff;
  const bHi = b >> 16;
  const bLo = b & 0xffff;

  const A = aHi * bHi;                 // |A| <= 2^30
  const B = aHi * bLo + aLo * bHi;     // |B| < 2^32
  const C = aLo * bLo;                 // C  < 2^32

  // Sign of the full product, without computing it.
  const negative = (a < 0) !== (b < 0) && a !== 0 && b !== 0;
  const nudge = negative ? 1 - (1 << 30) : 1 << 30;

  const T = B * TWO_16 + C + nudge;    // |T| < 2^49, exact in a double

  // a*b + nudge == (2A)*2^31 + T. Split T so the remainder is non-negative,
  // which makes the floor quotient exact.
  const tq = Math.floor(T / TWO_31);
  const tr = T - tq * TWO_31;          // 0 <= tr < 2^31

  let q = 2 * A + tq;                  // floor((a*b + nudge) / 2^31)
  if (q < 0 && tr > 0) q += 1;         // convert floor to truncate-toward-zero
  return q | 0;
}

/**
 * Divide by 2^exponent, rounding half away from zero.
 * @param {number} x int32
 * @param {number} exponent >= 0
 * @returns {number} int32
 */
export function rdbpot(x, exponent) {
  if (exponent === 0) return x;
  const mask = (1 << exponent) - 1;
  const remainder = x & mask;
  const threshold = (mask >> 1) + (x < 0 ? 1 : 0);
  return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

/**
 * Apply the real multiplier M = m0 * 2^-(31+shift) to an accumulator.
 * @param {number} acc int32
 * @param {number} m0 int32 in [2^30, 2^31)
 * @param {number} shift >= 0
 * @returns {number} int32
 */
export function requantize(acc, m0, shift) {
  return rdbpot(srdhm(acc, m0), shift);
}

/** Clamp to uint8. */
export function satU8(x) {
  return x < 0 ? 0 : x > 255 ? 255 : x;
}
