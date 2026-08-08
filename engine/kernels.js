// The four kernels. This is the entire numerical surface area of the model.
//
// Layout is NHWC everywhere. That makes the pointwise convolution a contiguous
// dot product over the channel axis (the inner loop walks both the activation
// and the weight linearly), and it makes the activation-visualization taps a
// simple strided read instead of a transpose.
//
// Quantization contract, relied on by every kernel here:
//   * activations are uint8 with zero-point 0
//   * weights are int8, per-output-channel symmetric, zero-point 0
//   * accumulators are int32
//
// Because the activation zero-point is 0, two things fall out for free. There
// is no zero-point correction term in any accumulator, and SAME padding is
// genuinely zero rather than "the quantized value that represents zero" — so
// skipping out-of-bounds taps is exactly right rather than approximately.
//
// All products here are int8 x uint8, bounded by 127*255 = 32,385, and the
// longest accumulation is 3*3*256 = 2,304 terms, so |acc| < 7.5e7. That is far
// inside the 2^53 range where a double is an exact integer, which is why the
// inner loops use plain `*` and `+` and only coerce to int32 at the end.

import { requantize, satU8 } from "./fixedpoint.js";

/**
 * Dense 3x3 convolution, SAME padding. Used only for the stem (3 input
 * channels); every other spatial convolution in the network is depthwise.
 *
 * Weights: [outC][kh][kw][inC]
 */
export function conv3x3(x, w, bias, m0, shift, inH, inW, inC, outC, stride, out) {
  const outH = Math.ceil(inH / stride);
  const outW = Math.ceil(inW / stride);
  let o = 0;
  for (let oy = 0; oy < outH; oy++) {
    const iy0 = oy * stride - 1;
    for (let ox = 0; ox < outW; ox++) {
      const ix0 = ox * stride - 1;
      for (let oc = 0; oc < outC; oc++) {
        let acc = 0;
        const wBase = oc * 9 * inC;
        for (let ky = 0; ky < 3; ky++) {
          const iy = iy0 + ky;
          if (iy < 0 || iy >= inH) continue;
          const rowBase = (iy * inW) << 0;
          for (let kx = 0; kx < 3; kx++) {
            const ix = ix0 + kx;
            if (ix < 0 || ix >= inW) continue;
            const xBase = (rowBase + ix) * inC;
            const wB = wBase + (ky * 3 + kx) * inC;
            for (let c = 0; c < inC; c++) {
              acc += x[xBase + c] * w[wB + c];
            }
          }
        }
        acc += bias[oc];
        out[o++] = satU8(requantize(acc | 0, m0[oc], shift[oc]));
      }
    }
  }
  return out;
}

/**
 * Depthwise 3x3 convolution, SAME padding. One filter per channel, no mixing.
 *
 * Weights: [kh][kw][C] — channel-minor so the innermost loop is contiguous in
 * both the activation and the weight.
 */
export function dwconv3x3(x, w, bias, m0, shift, inH, inW, C, stride, out) {
  const outH = Math.ceil(inH / stride);
  const outW = Math.ceil(inW / stride);
  let o = 0;
  for (let oy = 0; oy < outH; oy++) {
    const iy0 = oy * stride - 1;
    for (let ox = 0; ox < outW; ox++) {
      const ix0 = ox * stride - 1;
      const accBase = o;
      for (let c = 0; c < C; c++) {
        let acc = 0;
        for (let ky = 0; ky < 3; ky++) {
          const iy = iy0 + ky;
          if (iy < 0 || iy >= inH) continue;
          for (let kx = 0; kx < 3; kx++) {
            const ix = ix0 + kx;
            if (ix < 0 || ix >= inW) continue;
            acc += x[((iy * inW) + ix) * C + c] * w[(ky * 3 + kx) * C + c];
          }
        }
        acc += bias[c];
        out[accBase + c] = satU8(requantize(acc | 0, m0[c], shift[c]));
      }
      o += C;
    }
  }
  return out;
}

/**
 * Pointwise 1x1 convolution — a GEMM over the channel axis, and about 80% of
 * the network's multiply-accumulates.
 *
 * Weights: [outC][inC]
 */
export function pwconv1x1(x, w, bias, m0, shift, n, inC, outC, out) {
  let o = 0;
  for (let p = 0; p < n; p++) {
    const xBase = p * inC;
    for (let oc = 0; oc < outC; oc++) {
      const wBase = oc * inC;
      let acc = 0;
      // Unrolled by 4: the trip count is always a multiple of 16 here (channel
      // counts are 16/32/64/128/256), so no remainder loop is needed.
      for (let c = 0; c < inC; c += 4) {
        acc += x[xBase + c] * w[wBase + c] +
               x[xBase + c + 1] * w[wBase + c + 1] +
               x[xBase + c + 2] * w[wBase + c + 2] +
               x[xBase + c + 3] * w[wBase + c + 3];
      }
      acc += bias[oc];
      out[o++] = satU8(requantize(acc | 0, m0[oc], shift[oc]));
    }
  }
  return out;
}

/**
 * Global average pool over H*W, NHWC in, C out.
 *
 * Input and output share a scale (an average of same-scale values is on the
 * same scale), so the only multiplier is 1/(H*W), applied through the same
 * fixed-point path as everything else rather than as a float divide.
 */
export function globalAvgPool(x, hw, C, m0, shift, out) {
  for (let c = 0; c < C; c++) {
    let acc = 0;
    for (let p = 0; p < hw; p++) acc += x[p * C + c];
    out[c] = satU8(requantize(acc | 0, m0, shift));
  }
  return out;
}

/**
 * Fully connected layer to a single logit, left in int32.
 *
 * The sign of this value is the decision, and sign is preserved by the
 * (strictly positive) output scale, so the classification itself never touches
 * a float. The scale is applied later, for display only.
 */
export function denseToLogit(x, w, bias, inC) {
  let acc = 0;
  for (let c = 0; c < inC; c++) acc += x[c] * w[c];
  return (acc + bias) | 0;
}

/**
 * Centre-crop to the largest square, then box-filter down to size x size.
 *
 * Deliberately not `drawImage`: browser resampling differs between engines and
 * is not specified precisely enough to reproduce, which would break bit-exact
 * parity with the Python reference across browsers. This is a plain area
 * average with round-half-up, and reference/preprocess.py is the same
 * arithmetic on the same integers.
 *
 * @param {Uint8ClampedArray} rgba source pixels, length w*h*4
 * @param {Uint8Array} out destination, length size*size*3, NHWC
 */
export function cropResize(rgba, w, h, size, out) {
  const side = Math.min(w, h);
  const offX = (w - side) >> 1;
  const offY = (h - side) >> 1;

  for (let oy = 0; oy < size; oy++) {
    let sy0 = Math.floor((oy * side) / size);
    let sy1 = Math.floor(((oy + 1) * side) / size);
    if (sy1 <= sy0) sy1 = sy0 + 1;
    for (let ox = 0; ox < size; ox++) {
      let sx0 = Math.floor((ox * side) / size);
      let sx1 = Math.floor(((ox + 1) * side) / size);
      if (sx1 <= sx0) sx1 = sx0 + 1;

      let r = 0, g = 0, b = 0;
      const count = (sy1 - sy0) * (sx1 - sx0);
      for (let sy = sy0; sy < sy1; sy++) {
        const row = ((sy + offY) * w + offX) << 2;
        for (let sx = sx0; sx < sx1; sx++) {
          const i = row + (sx << 2);
          r += rgba[i];
          g += rgba[i + 1];
          b += rgba[i + 2];
        }
      }
      const half = count >> 1;
      const o = (oy * size + ox) * 3;
      out[o] = ((r + half) / count) | 0;
      out[o + 1] = ((g + half) / count) | 0;
      out[o + 2] = ((b + half) / count) | 0;
    }
  }
  return out;
}
