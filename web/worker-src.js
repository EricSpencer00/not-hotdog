// Worker body. build.py prepends the engine modules and the weight blob, so by
// the time this runs, NotHotdog / SPEC / WEIGHTS_B64 are already in scope.
//
// Inference happens off the main thread for one reason: at 96x96 the forward
// pass is ~14M multiply-accumulates, and doing that inside requestAnimationFrame
// would block the compositor and make the verdict transition stutter exactly
// when the user is looking at it.

let model = null;

const TILE_COLS = 4;
const TILE_ROWS = 4;
const MAX_CELL = 24;

/**
 * Render one block's activations as a small grid of per-channel tiles.
 *
 * Done here rather than on the main thread because the raw activations are
 * ~240 KB per frame and the rendered grid is ~2-9 KB, so this is the difference
 * between a transfer that keeps up at 30fps and one that does not.
 *
 * Each channel is normalised against its own maximum. Absolute magnitudes vary
 * enormously between channels, and without per-channel normalisation the grid
 * is one bright tile and fifteen black ones.
 */
function makeTile(tap) {
  const { h, w, c, data } = tap;
  const step = Math.max(1, Math.ceil(h / MAX_CELL));
  const cw = Math.ceil(w / step);
  const ch = Math.ceil(h / step);
  const outW = cw * TILE_COLS;
  const outH = ch * TILE_ROWS;
  const out = new Uint8Array(outW * outH);
  const nCh = Math.min(c, TILE_COLS * TILE_ROWS);

  for (let k = 0; k < nCh; k++) {
    // Spread the sampled channels across the block rather than taking the
    // first 16, which in a 256-channel layer would show one corner of it.
    const src = Math.floor((k * c) / nCh);
    let max = 1;
    for (let p = 0; p < h * w; p++) {
      const v = data[p * c + src];
      if (v > max) max = v;
    }
    const gx = (k % TILE_COLS) * cw;
    const gy = Math.floor(k / TILE_COLS) * ch;
    for (let y = 0, sy = 0; y < ch; y++, sy += step) {
      for (let x = 0, sx = 0; x < cw; x++, sx += step) {
        const v = data[(sy * w + sx) * c + src];
        out[(gy + y) * outW + gx + x] = ((v * 255) / max) | 0;
      }
    }
  }
  return { name: tap.name, w: outW, h: outH, cols: TILE_COLS, shape: `${h}x${w}x${c}`, data: out };
}

self.onmessage = (e) => {
  const d = e.data;

  if (d.type === "init") {
    model = new NotHotdog(SPEC, WEIGHTS_B64);
    // Warm the JIT so the first user-visible timing is not an outlier.
    const warm = new Uint8Array(model.size * model.size * 3);
    for (let i = 0; i < 3; i++) model.run(warm, false);
    self.postMessage({
      type: "ready",
      params: model.paramCount,
      bytes: model.weights.length,
      layers: model.taps.map((t) => ({ name: t.name, shape: `${t.h}x${t.w}x${t.c}` })),
    });
    return;
  }

  if (d.type === "infer") {
    const input = model.preprocess(new Uint8ClampedArray(d.rgba), d.w, d.h);
    const r = model.run(input, d.withTaps);
    const tiles = d.withTaps ? model.taps.map(makeTile) : null;
    self.postMessage(
      { type: "result", seq: d.seq, ...r, tiles },
      tiles ? tiles.map((t) => t.data.buffer) : []
    );
  }
};
