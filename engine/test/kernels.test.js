// Each kernel against the NumPy reference on random tensors.
//
// The parity test proves the whole network matches; this proves *which* kernel
// is wrong when it does not. Without it, a stride-2 off-by-one in the depthwise
// path presents as "the logits differ" 200 layers downstream.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  conv3x3,
  dwconv3x3,
  pwconv1x1,
  globalAvgPool,
  cropResize,
} from "../kernels.js";

const here = dirname(fileURLToPath(import.meta.url));
const CASES = join(here, "vectors", "kernels.json");

function dec(b64, Type) {
  const bin = Buffer.from(b64, "base64");
  return new Type(bin.buffer, bin.byteOffset, bin.byteLength / Type.BYTES_PER_ELEMENT);
}

function firstDiff(got, want) {
  for (let i = 0; i < want.length; i++) {
    if (got[i] !== want[i]) return `index ${i}: got ${got[i]}, want ${want[i]}`;
  }
  return null;
}

let cases;
try {
  cases = JSON.parse(readFileSync(CASES, "utf8"));
} catch {
  cases = null;
}

test("kernel vectors are present", () => {
  assert.ok(cases, `missing ${CASES}\nrun: python reference/gen_kernel_vectors.py`);
  assert.ok(cases.length > 15, `expected >15 cases, got ${cases?.length}`);
});

test("conv3x3 matches the reference", () => {
  let n = 0;
  for (const c of cases.filter((c) => c.kernel === "conv3x3")) {
    const want = dec(c.y, Uint8Array);
    const out = new Uint8Array(want.length);
    conv3x3(
      dec(c.x, Uint8Array), dec(c.w, Int8Array), dec(c.bias, Int32Array),
      dec(c.m0, Int32Array), dec(c.shift, Int32Array),
      c.H, c.W, c.inC, c.outC, c.stride, out
    );
    assert.equal(firstDiff(out, want), null,
      `conv3x3 ${c.H}x${c.W}x${c.inC}->${c.outC} s${c.stride}: ${firstDiff(out, want)}`);
    n++;
  }
  assert.ok(n >= 5, `only ${n} conv3x3 cases`);
});

test("dwconv3x3 matches the reference", () => {
  let n = 0;
  for (const c of cases.filter((c) => c.kernel === "dwconv3x3")) {
    const want = dec(c.y, Uint8Array);
    const out = new Uint8Array(want.length);
    dwconv3x3(
      dec(c.x, Uint8Array), dec(c.w, Int8Array), dec(c.bias, Int32Array),
      dec(c.m0, Int32Array), dec(c.shift, Int32Array),
      c.H, c.W, c.C, c.stride, out
    );
    assert.equal(firstDiff(out, want), null,
      `dwconv3x3 ${c.H}x${c.W}x${c.C} s${c.stride}: ${firstDiff(out, want)}`);
    n++;
  }
  assert.ok(n >= 6, `only ${n} dwconv3x3 cases`);
});

test("pwconv1x1 matches the reference", () => {
  let n = 0;
  for (const c of cases.filter((c) => c.kernel === "pwconv1x1")) {
    const want = dec(c.y, Uint8Array);
    const out = new Uint8Array(want.length);
    pwconv1x1(
      dec(c.x, Uint8Array), dec(c.w, Int8Array), dec(c.bias, Int32Array),
      dec(c.m0, Int32Array), dec(c.shift, Int32Array),
      c.n, c.inC, c.outC, out
    );
    assert.equal(firstDiff(out, want), null,
      `pwconv1x1 n=${c.n} ${c.inC}->${c.outC}: ${firstDiff(out, want)}`);
    n++;
  }
  assert.ok(n >= 4, `only ${n} pwconv1x1 cases`);
});

test("globalAvgPool matches the reference", () => {
  for (const c of cases.filter((c) => c.kernel === "globalAvgPool")) {
    const want = dec(c.y, Uint8Array);
    const out = new Uint8Array(c.C);
    globalAvgPool(dec(c.x, Uint8Array), c.hw, c.C, c.m0, c.shift, out);
    assert.equal(firstDiff(out, want), null,
      `globalAvgPool hw=${c.hw} C=${c.C}: ${firstDiff(out, want)}`);
  }
});

test("cropResize matches the reference across aspect ratios", () => {
  let n = 0;
  for (const c of cases.filter((c) => c.kernel === "cropResize")) {
    const want = dec(c.y, Uint8Array);
    const out = new Uint8Array(c.size * c.size * 3);
    cropResize(dec(c.rgba, Uint8Array), c.W, c.H, c.size, out);
    assert.equal(firstDiff(out, want), null,
      `cropResize ${c.H}x${c.W}: ${firstDiff(out, want)}`);
    n++;
  }
  assert.ok(n >= 6, `only ${n} cropResize cases`);
});
