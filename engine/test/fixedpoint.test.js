// The gate. If a single vector here disagrees with the Python reference, the
// whole port is unsound and nothing downstream is worth building — a broken
// requantization does not crash, it just quietly makes the model worse.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { srdhm, rdbpot, requantize, satU8, INT32_MIN, INT32_MAX } from "../fixedpoint.js";

const here = dirname(fileURLToPath(import.meta.url));
const VECTORS = join(here, "vectors", "fixedpoint.bin");

test("srdhm and rdbpot match the Python reference exactly", () => {
  let raw;
  try {
    raw = readFileSync(VECTORS);
  } catch {
    assert.fail(
      `missing ${VECTORS}\nrun: python reference/gen_fixedpoint_vectors.py`
    );
  }

  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const n = raw.byteLength / 20;
  assert.ok(n > 1_000_000, `expected >1M vectors, got ${n}`);

  let mismatches = 0;
  const examples = [];

  for (let i = 0; i < n; i++) {
    const o = i * 20;
    const a = view.getInt32(o, true);
    const b = view.getInt32(o + 4, true);
    const wantHigh = view.getInt32(o + 8, true);
    const shift = view.getInt32(o + 12, true);
    const wantReq = view.getInt32(o + 16, true);

    const gotHigh = srdhm(a, b);
    const gotReq = requantize(a, b, shift);

    if (gotHigh !== wantHigh || gotReq !== wantReq) {
      mismatches++;
      if (examples.length < 5) {
        examples.push(
          `a=${a} b=${b} shift=${shift} ` +
            `srdhm got=${gotHigh} want=${wantHigh} ` +
            `req got=${gotReq} want=${wantReq}`
        );
      }
    }
  }

  assert.equal(
    mismatches,
    0,
    `${mismatches} of ${n} vectors mismatched:\n  ${examples.join("\n  ")}`
  );
  console.log(`    ${n.toLocaleString()} vectors, 0 mismatches`);
});

test("srdhm saturates the one product that cannot fit", () => {
  assert.equal(srdhm(INT32_MIN, INT32_MIN), INT32_MAX);
});

test("srdhm returns int32", () => {
  for (const [a, b] of [
    [INT32_MAX, INT32_MAX],
    [INT32_MIN, INT32_MAX],
    [0, INT32_MIN],
    [1, 1],
  ]) {
    const r = srdhm(a, b);
    assert.ok(Number.isInteger(r), `${r} not an integer`);
    assert.ok(r >= INT32_MIN && r <= INT32_MAX, `${r} out of int32 range`);
  }
});

test("rdbpot rounds half away from zero", () => {
  assert.equal(rdbpot(3, 1), 2); // 1.5 -> 2
  assert.equal(rdbpot(-3, 1), -2); // -1.5 -> -2
  assert.equal(rdbpot(5, 2), 1); // 1.25 -> 1
  assert.equal(rdbpot(6, 2), 2); // 1.5 -> 2
  assert.equal(rdbpot(-6, 2), -2);
  assert.equal(rdbpot(7, 0), 7);
});

test("satU8 clamps", () => {
  assert.equal(satU8(-1), 0);
  assert.equal(satU8(0), 0);
  assert.equal(satU8(255), 255);
  assert.equal(satU8(9999), 255);
});
