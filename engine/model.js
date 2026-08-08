// The layer graph driver.
//
// Buffers are allocated once at construction and reused for every inference.
// In camera mode this runs 15-30 times a second, and allocating ~400 KB of
// typed arrays per frame would hand the garbage collector a steady job and put
// a visible stutter in the verdict.
//
// Two scratch buffers alternate as input and output between layers, sized to
// the largest activation in the network so either can hold any intermediate.

import { conv3x3, dwconv3x3, pwconv1x1, globalAvgPool, denseToLogit, cropResize } from "./kernels.js";

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function b64ToInt8(b64) {
  const b = b64ToBytes(b64);
  return new Int8Array(b.buffer, b.byteOffset, b.byteLength);
}

function b64ToInt32(b64) {
  const b = b64ToBytes(b64);
  return new Int32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}

export class NotHotdog {
  /**
   * @param {object} spec layer graph from train/export.py
   * @param {string} weightsB64 int8 weight blob
   */
  constructor(spec, weightsB64) {
    this.spec = spec;
    this.weights = b64ToInt8(weightsB64);
    this.size = spec.inputSize;

    // Per-layer views into the weight blob, plus typed copies of the
    // per-channel requantization parameters (plain arrays would box).
    this.layers = spec.layers.map((L) => ({
      ...L,
      w: this.weights.subarray(L.wOffset, L.wOffset + L.wLen),
      bias: b64ToInt32(L.bias),
      m0: b64ToInt32(L.m0),
      shift: b64ToBytes(L.shift),
    }));

    const head = spec.head;
    this.headW = this.weights.subarray(head.wOffset, head.wOffset + head.wLen);

    let maxAct = this.size * this.size * 3;
    for (const L of spec.layers) maxAct = Math.max(maxAct, L.outH * L.outW * L.outC);
    this.bufA = new Uint8Array(maxAct);
    this.bufB = new Uint8Array(maxAct);
    this.input = new Uint8Array(this.size * this.size * 3);
    this.pooled = new Uint8Array(spec.pool.C);

    // Activation taps need their own storage. A view into bufA/bufB would be
    // overwritten two layers later, so every tap but the last two would be
    // garbage by the time the caller read it. ~240 KB, allocated once.
    this.taps = spec.layers
      .map((L, i) => ({ L, i }))
      .filter(({ L }) => L.kind !== "dw")
      .map(({ L, i }) => ({
        name: i === 0 ? "stem" : `b${(i + 1) >> 1}`,
        h: L.outH, w: L.outW, c: L.outC,
        data: new Uint8Array(L.outH * L.outW * L.outC),
      }));

    this.lastMs = 0;
  }

  /** Total shipped parameters (int8 bytes). */
  get paramCount() {
    return this.weights.length;
  }

  /**
   * Centre-crop and box-resize RGBA canvas pixels into the input tensor.
   * The result is fed to the first convolution unchanged: the input scale is
   * exactly 1/255 with zero-point 0, so a raw pixel byte *is* its own
   * quantized value.
   */
  preprocess(rgba, w, h) {
    return cropResize(rgba, w, h, this.size, this.input);
  }

  /**
   * @param {Uint8Array} input size*size*3 uint8, NHWC
   * @param {boolean} withTaps capture per-block activations for visualization
   * @returns {{logit: number, isHotdog: boolean, prob: number, ms: number}}
   */
  run(input, withTaps = false) {
    const t0 = performance.now();
    // `input` belongs to the caller and is never written to; the first layer
    // reads from it and writes into bufA, after which the two scratch buffers
    // simply alternate.
    let src = input;
    let dst = this.bufA;
    let tapIdx = 0;

    for (let i = 0; i < this.layers.length; i++) {
      const L = this.layers[i];
      if (L.kind === "conv") {
        conv3x3(src, L.w, L.bias, L.m0, L.shift, L.inH, L.inW, L.inC, L.outC, L.stride, dst);
      } else if (L.kind === "dw") {
        dwconv3x3(src, L.w, L.bias, L.m0, L.shift, L.inH, L.inW, L.inC, L.stride, dst);
      } else {
        pwconv1x1(src, L.w, L.bias, L.m0, L.shift, L.inH * L.inW, L.inC, L.outC, dst);
      }

      if (L.kind !== "dw") {
        const tap = this.taps[tapIdx++];
        if (withTaps) tap.data.set(dst.subarray(0, tap.data.length));
      }

      if (i === 0) {
        src = this.bufA;
        dst = this.bufB;
      } else {
        const t = src;
        src = dst;
        dst = t;
      }
    }

    const pool = this.spec.pool;
    globalAvgPool(src, pool.hw, pool.C, pool.m0, pool.shift, this.pooled);
    const logit = denseToLogit(this.pooled, this.headW, this.spec.head.bias, this.spec.head.inC);

    this.lastMs = performance.now() - t0;
    // Float only after the decision: sign(logit) already is the answer.
    const real = logit * this.spec.head.logitScale;
    return {
      logit,
      isHotdog: logit > 0,
      prob: 1 / (1 + Math.exp(-real)),
      ms: this.lastMs,
    };
  }
}
