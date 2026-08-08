// The claim: the JavaScript engine and the NumPy reference produce the *same*
// int32 logit on every image. Not within a tolerance — the same integer.
//
// A tolerance would be meaningless here. There is no floating point in the
// forward pass, so there is no rounding to be forgiving about; any difference
// at all means one of the two implementations has a bug, and a model that is
// "almost right" is just a model that is wrong in a way nobody noticed.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { NotHotdog } from "../model.js";
import { SPEC, WEIGHTS_B64 } from "../../dist/model.js";

const here = dirname(fileURLToPath(import.meta.url));
const VECTORS = join(here, "vectors", "parity.bin");

test("JS engine matches the Python integer reference exactly", () => {
  let raw;
  try {
    raw = readFileSync(VECTORS);
  } catch {
    assert.fail(
      `missing ${VECTORS}\nrun: python reference/gen_parity_vectors.py --source val --n 0`
    );
  }

  const model = new NotHotdog(SPEC, WEIGHTS_B64);
  const px = model.size * model.size * 3;
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const n = view.getInt32(0, true);
  assert.ok(n > 0, "no records");
  assert.equal(raw.byteLength, 4 + n * (px + 4), "vector file is truncated");

  const input = new Uint8Array(px);
  let mismatches = 0;
  const examples = [];
  let hot = 0;

  for (let i = 0; i < n; i++) {
    const o = 4 + i * (px + 4);
    input.set(new Uint8Array(raw.buffer, raw.byteOffset + o, px));
    const want = view.getInt32(o + px, true);
    const got = model.run(input, false);
    if (got.logit !== want) {
      mismatches++;
      if (examples.length < 5) examples.push(`record ${i}: got ${got.logit}, want ${want}`);
    }
    if (want > 0) hot++;
  }

  assert.equal(
    mismatches,
    0,
    `${mismatches} of ${n} logits differ:\n  ${examples.join("\n  ")}`
  );
  console.log(`    ${n} images, 0 mismatches (${hot} classified hot dog)`);
});

test("taps do not change the result", () => {
  // The internals panel must be observational. If capturing activations
  // perturbed the output, the visualization would be of a different network
  // than the one giving the verdict.
  const model = new NotHotdog(SPEC, WEIGHTS_B64);
  const px = model.size * model.size * 3;
  const input = new Uint8Array(px);
  for (let i = 0; i < px; i++) input[i] = (i * 37 + 11) & 255;

  const a = model.run(input, false);
  const b = model.run(input, true);
  const c = model.run(input, false);
  assert.equal(a.logit, b.logit);
  assert.equal(b.logit, c.logit);
});

test("repeated inference is deterministic and does not leak state", () => {
  // Buffers are reused across frames; a kernel that failed to overwrite every
  // output would carry the previous frame's values forward, which in camera
  // mode looks like plausible lag rather than a bug.
  const model = new NotHotdog(SPEC, WEIGHTS_B64);
  const px = model.size * model.size * 3;
  const x = new Uint8Array(px).fill(200);
  const y = new Uint8Array(px).fill(20);

  const x1 = model.run(x, false).logit;
  model.run(y, false);
  const x2 = model.run(x, false).logit;
  assert.equal(x1, x2, "output depended on the previous frame");
});
