# Not Hotdog — Specification

A binary image classifier that answers exactly one question — *is this a hot dog?* —
trained with knowledge distillation, quantized to int8 with QAT, and executed in the
browser by hand-written JavaScript kernels with **zero dependencies**.

Live at `ericspencer.us/hotdog`. Source at `github.com/EricSpencer00/not-hotdog` (public, MIT).

Inspired by the SeeFood gag in HBO's *Silicon Valley*. Not affiliated with HBO.

---

## 0. Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Inference runtime | Pure JS, zero deps. No TF.js, no ONNX Runtime, no WASM, no npm at runtime. |
| 2 | Positives | Food-101 `hot_dog` + Open Images V7 `Hot dog` crops |
| 3 | Training | Fine-tuned EfficientNet-B0 teacher → KD into tiny student |
| 4 | Student | 96×96 input, ~136K params, 8 depthwise-separable blocks |
| 5 | Quantization | QAT int8, **integer-only** inference, bit-exact JS↔Python parity |
| 6 | Input paths | Live camera + drag/drop/picker/paste + bundled samples |
| 7 | Naming | Repo `not-hotdog`, app "Not Hotdog". Descriptive, not appropriated. |
| 8 | Design | Editorial chrome + full-bleed verdict + live activation visualization |
| 9 | Structure | Separate repo; `make deploy` rsyncs `dist/` → site repo `hotdog/` on `dev` |
| 10 | Verification | Kernel unit tests + bit-exact parity + adversarial eval set + CI |
| 11 | Artifacts | README/model card, HF publish, homepage listing. Blog post deferred. |

**Explicitly out of scope:** URL-paste input (CORS breaks it), synthetic training data,
the "SeeFood" name, a blog post (write it once real numbers exist), any server-side
component, any analytics, any telemetry.

---

## 1. Constraints

These are hard. Every design choice below descends from them.

- **C1 — Zero runtime dependencies.** The shipped page loads `index.html`, one CSS file,
  and JS files authored in this repo. Nothing else. No CDN, no import map, no WASM binary.
- **C2 — Integer-only forward pass.** After preprocessing, no floating-point arithmetic
  in the inference path. int8 weights, uint8 activations, int32 accumulators,
  fixed-point requantization.
- **C3 — Bit-exact portability.** The JS engine and the Python reference must produce
  *identical* int32 logits on every image in the validation set. Not "close" — identical.
- **C4 — No network after load.** No image, no frame, no result ever leaves the device.
  Verifiable by reading the source; the page makes zero fetches post-load except
  lazily-loaded sample thumbnails.
- **C5 — Page weight ≤ 250 KB** for the critical path (HTML + CSS + JS + weights).
- **C6 — Honest numbers.** Every figure on the page and in the README is measured,
  not estimated. Estimates in this spec are marked as such and get replaced.

---

## 2. Data

### 2.1 Sources

| Source | Role | Count (approx) | License |
|---|---|---|---|
| `ethz/food101` class `hot_dog` | positives | 1,000 | Unknown / research use (documented) |
| Open Images V7, class `Hot dog` | positives (bbox crops) | 2,000–3,000 | CC BY 2.0 (per-image attribution retained) |
| `ethz/food101`, other 100 classes | hard negatives (food) | 12,000 sampled | Unknown / research use |
| `frgfm/imagenette` | easy negatives (non-food) | 8,000 | Apache-2.0 |

**No images are committed to the repo.** `data/` holds manifests (source ID, URL, license,
split assignment, SHA-256) and the scripts that materialize a local cache. `make data`
reconstructs the full set from scratch. The manifests are the reproducible artifact.

### 2.2 Negative sampling policy

The class balance and the *composition* of the negative class determine what the model
actually learns. Naive sampling produces a model that has learned "food-coloured blob."

- Food negatives are **oversampled from confusable classes**: `hamburger`,
  `lobster_roll_sandwich`, `club_sandwich`, `pulled_pork_sandwich`,
  `grilled_cheese_sandwich`, `french_fries`, `tacos`, `spring_rolls`, `garlic_bread`,
  `breakfast_burrito`. These get 3× the per-class sampling rate of the other 90 classes.
- Non-food negatives ensure the model doesn't just learn "is this a photo of food."
- Final ratio target ≈ **1 : 5** positive : negative. Class imbalance is handled with a
  weighted loss, not by discarding negatives — the negatives are where the difficulty is.

### 2.3 Splits

- `train` / `val` / `test` = 70 / 15 / 15, stratified by class **and by source**, so val
  isn't accidentally all-Food-101 while train is all-OpenImages.
- Split assignment is deterministic: `sha256(source_id) % 100` bucketed. Rerunning ingest
  never reshuffles an image into a different split.
- Open Images crops from the same source photo are pinned to the same split (no leakage
  via near-duplicate crops).

### 2.4 Adversarial eval set (`data/hard/`)

~200 images assembled by hand specifically to break the model. Committed as a manifest
with URLs + hashes. Composition:

- **Should be HOT DOG:** extreme angles, partially eaten, heavily topped (chili, slaw),
  in-hand, low light, cartoon/illustrated hot dogs, hot dogs at distance in a crowd.
- **Should be NOT HOT DOG:** corn dog, bratwurst on a plate, sausage roll, kolache,
  submarine sandwich, baguette sandwich, an empty bun, a dachshund, a "hot dog" costume,
  a plain sausage with no bun, a taco.

The corn-dog and empty-bun cases are the interesting ones and are expected to be where
the model is weakest. **Both accuracy numbers get published.** The standard-val number
will look good and mean little; the hard-set number is the one worth reading.

---

## 3. Models

### 3.1 Teacher

- `timm` `efficientnet_b0`, ImageNet-pretrained, binary head.
- 224×224, RandAugment + random-resized-crop + horizontal flip, mixup off (hurts KD targets).
- AdamW, cosine schedule, ~15 epochs, MPS on the M1 Max.
- Target: **≥96% val accuracy.** If it doesn't clear this, the teacher is not good enough
  to distill from and the data pipeline gets revisited before proceeding.
- Teacher exists only to produce soft targets. It is never shipped to the browser.

### 3.2 Student — architecture

Input `96×96×3`. MobileNet-style depthwise-separable stack. BatchNorm after every conv
during training, **folded into the preceding conv at export**. ReLU only — no hard-swish,
no SE blocks, no residual connections. Every op below maps to one of four JS kernels.

| Layer | Op | Out shape | Params |
|---|---|---|---:|
| stem | conv3×3 s2, 3→16 | 48×48×16 | 432 |
| b1 | dw3×3 s1, 16 | 48×48×16 | 144 |
| | pw1×1, 16→32 | 48×48×32 | 512 |
| b2 | dw3×3 s2, 32 | 24×24×32 | 288 |
| | pw1×1, 32→64 | 24×24×64 | 2,048 |
| b3 | dw3×3 s1, 64 | 24×24×64 | 576 |
| | pw1×1, 64→64 | 24×24×64 | 4,096 |
| b4 | dw3×3 s2, 64 | 12×12×64 | 576 |
| | pw1×1, 64→128 | 12×12×128 | 8,192 |
| b5 | dw3×3 s1, 128 | 12×12×128 | 1,152 |
| | pw1×1, 128→128 | 12×12×128 | 16,384 |
| b6 | dw3×3 s2, 128 | 6×6×128 | 1,152 |
| | pw1×1, 128→256 | 6×6×256 | 32,768 |
| b7 | dw3×3 s1, 256 | 6×6×256 | 2,304 |
| | pw1×1, 256→256 | 6×6×256 | 65,536 |
| head | global avg pool | 256 | 0 |
| | fc 256→1 | 1 | 257 |
| | | **total** | **136,417** |

**14.0M MACs** per forward pass. int8 weights = 136 KB raw, **182 KB base64**.

### 3.3 Student — training

Two-phase, one script:

1. **KD phase (float).** Loss = `α · KL(student_soft ‖ teacher_soft, T=4) · T² + (1−α) · BCE(student, label)`,
   with `α = 0.7`. Teacher logits precomputed once over the augmented pool and cached to
   disk, so the teacher isn't re-run every epoch. ~40 epochs.
2. **QAT phase.** Fake-quant observers inserted (per-channel symmetric int8 weights,
   per-tensor asymmetric uint8 activations), BN folded, LR dropped 10×, KD loss retained,
   ~10 further epochs. Straight-through estimator for the quantizer gradient.

Expected: **~93% standard val** (estimate — replaced by the measured figure).
Hard-set accuracy is not predicted; it gets measured.

---

## 4. Quantization and the integer pipeline

### 4.1 Scheme

- **Weights:** int8, **per-output-channel** symmetric. Scale `s_w[c]`, zero-point 0.
- **Activations:** uint8, **per-tensor** asymmetric. Scale `s_a`, zero-point `z_a ∈ [0,255]`.
- **Bias:** int32, quantized at `s_w[c] · s_a_in` (the standard bias scale identity),
  folded with the BatchNorm shift and the input zero-point correction term.
- **Accumulate:** int32.
- **Requantize:** `out_u8 = saturate_u8( z_out + RoundingDoublingHighMul(acc, M0[c]) >> n[c] )`
  where the real multiplier `M = (s_w[c] · s_a_in) / s_a_out` is decomposed into
  `M0 ∈ [2^30, 2^31)` int32 and a right shift `n`. gemmlowp semantics exactly.

### 4.2 The one hard part

`RoundingDoublingHighMul(a, b)` requires the high 32 bits of a signed 64-bit product.
JavaScript numbers are float64 with a 53-bit mantissa, so `a * b` is **not exact** for
32×32 operands and cannot be used. Implementation splits both operands into 16-bit halves
and reassembles via `Math.imul`, producing the exact 64-bit product, then applies
gemmlowp's rounding and the `INT32_MIN × INT32_MIN` saturation special case.

This function is the single highest-risk piece of the port. It gets its own unit test
against a Python reference over ~10^6 random pairs including all boundary values before
anything else is built on it.

### 4.3 Preprocessing (the only float in the pipeline)

Canvas → `getImageData` → resize to 96×96 → uint8 RGB → quantize to the input tensor's
`(s_a, z_a)`. Resizing uses a fixed box-filter implemented in JS, **not** the browser's
`drawImage` scaler, because `drawImage` resampling differs across engines and would break
bit-exactness across browsers. The Python reference uses the identical box filter.

### 4.4 Export

`train/export.py` emits:

- `dist/weights.b64` semantics: int8 weight blob, base64, inlined into `hotdog.weights.js`
- `dist/model.json` → inlined: layer specs, `M0`/`n` per channel, zero-points, shapes
- `reference/model_int8.npz` → the Python integer reference reads this, same bytes

Both engines consume the **same** exported artifact. Divergence is then always a code bug,
never a data-loading bug.

---

## 5. The JS engine (`engine/`)

Four kernels. That is the entire numerical surface area.

| Kernel | Signature | Notes |
|---|---|---|
| `conv3x3` | dense 3×3, stride 1\|2, SAME pad | stem only (3 input channels) |
| `dwconv3x3` | depthwise 3×3, stride 1\|2, SAME pad | 7 call sites |
| `pwconv1x1` | pointwise = blocked int8 GEMM | 7 call sites, ~80% of MACs |
| `requantize` | int32 → uint8, fixed-point | fused into the three above |

Plus `globalAvgPool` (integer mean with rounding) and `dequantLogit` (int32 → float, for
display only, after the decision has already been made).

- All buffers are preallocated `Int8Array` / `Uint8Array` / `Int32Array`, reused across
  frames. Zero allocation in the steady-state camera loop.
- Layout is **NHWC** throughout — makes pointwise conv a contiguous GEMM and makes the
  activation-visualization taps trivial to slice.
- Runs in a **Web Worker**. The main thread only posts a `Uint8ClampedArray` frame and
  receives `{logit, activations[]}`. UI never blocks.
- **Activation taps:** after each block, the uint8 activation buffer is optionally
  min-max normalized into a small `Uint8ClampedArray` tile grid and transferred back for
  rendering. Off by default; enabled when the internals panel is open. Costs one extra
  pass over ≤ 73 KB of data.

---

## 6. Frontend (`web/`)

### 6.1 Layout

```
┌─ not hotdog ───────────────────────── ericspencer.us ─┐
│                                                        │
│   ┌──────────────────┐   confidence  ▓▓▓▓▓▓▓▓░░  0.94  │
│   │                  │   inference    71 ms            │
│   │   camera / img   │   params       136,417          │
│   │                  │   weights      182 KB int8      │
│   └──────────────────┘   deps         0                │
│                                                        │
├────────────────────────────────────────────────────────┤
│                                                        │
│                     HOT DOG                            │  ← full-bleed
│                                                        │
├────────────────────────────────────────────────────────┤
│  inside the network                              [ ─ ] │
│  stem  b1   b2   b3   b4   b5   b6   b7                │
│  ▦▦▦   ▦▦▦  ▦▦▦  ▦▦▦  ▦▦▦  ▦▦▦  ▦▦▦  ▦▦▦               │  ← live tiles
├────────────────────────────────────────────────────────┤
│  [🌭] [🌽] [🥖] [🍔] [🐕] [💻]   ← tap to classify      │
└────────────────────────────────────────────────────────┘
```

### 6.2 Design system

Inherits `ericspencer.us` tokens exactly:

```css
--paper:  oklch(0.978 0.006 80)
--ink:    oklch(0.11  0.015 60)
--dim:    oklch(0.52  0.01  70)
--rule:   oklch(0.87  0.005 80)
--accent: oklch(0.30  0.13  22)
```

Plus two verdict colors, chosen to sit in the same warm space rather than screaming
sRGB green/red:

```css
--yes: oklch(0.62 0.16 145)
--no:  oklch(0.55 0.19 25)
```

Type: IBM Plex Mono for the verdict, all data readouts, and layer labels;
Plus Jakarta Sans for prose. Verdict set at `clamp(3rem, 14vw, 9rem)`, tight tracking.

### 6.3 Motion

Verdict flip: spring transition on background color and a small scale/opacity on the
label. `prefers-reduced-motion: reduce` → instant swap, no scale. Verdict is
**debounced/hysteretic** in camera mode — it requires 3 consecutive agreeing frames
before flipping, otherwise a borderline frame makes the whole page strobe.

### 6.4 Accessibility

- Verdict announced via `aria-live="polite"`; the text says "hot dog" / "not hot dog",
  so color is never the sole carrier of meaning.
- Full keyboard path: file picker, sample selection, internals toggle all focusable.
- Camera is opt-in behind an explicit button. Never auto-requested on load.
- Camera-denied and no-camera states are designed, not left as a broken black box.

---

## 7. Verification

| Test | What it proves | Where |
|---|---|---|
| `test_fixedpoint` | `RoundingDoublingHighMul` exact over 10^6 random + all boundary pairs | `engine/test/` + `reference/` |
| `test_kernels` | Each JS kernel matches NumPy on random tensors; stride-2 odd inputs, SAME-pad boundaries, int8 saturation | `engine/test/` |
| `test_resize` | JS box filter ≡ Python box filter, pixel-exact, across aspect ratios | both |
| **`test_parity`** | **JS engine logit ≡ Python integer reference logit, exactly, on every val image** | `engine/test/` |
| `eval_standard` | Accuracy / precision / recall / F1 on held-out val | `train/eval.py` |
| `eval_hard` | Same metrics on the 200-image adversarial set, with a per-category breakdown | `train/eval.py` |

CI (GitHub Actions) runs fixedpoint, kernels, resize, and parity on every push. The
exported 136 KB model is committed so CI needs no dataset. Accuracy evals are run locally
and their outputs committed as JSON, which the README and the page both read from —
so a number can never drift between the three places it appears.

**Definition of done:** parity test passes with zero mismatches, both eval numbers are
measured and published, page weight is measured and under 250 KB, and the page works on
desktop Safari, Chrome, and iOS Safari.

---

## 8. Repo layout

```
not-hotdog/
├── SPEC.md                    this file
├── README.md                  + model card
├── LICENSE                    MIT
├── Makefile                   data | teacher | student | export | test | eval | deploy
├── pyproject.toml             uv-managed, torch + timm + datasets
├── data/
│   ├── ingest_food101.py
│   ├── ingest_openimages.py
│   ├── ingest_imagenette.py
│   ├── build_manifest.py      splits, dedupe, license tracking
│   ├── manifests/*.jsonl      committed
│   └── hard/manifest.jsonl    committed
├── train/
│   ├── teacher.py             efficientnet_b0 fine-tune
│   ├── distill.py             KD + QAT, one script two phases
│   ├── export.py              BN fold, per-channel int8, emit artifacts
│   ├── eval.py                standard + hard, writes results/*.json
│   └── results/*.json         committed
├── reference/
│   ├── int8_reference.py      the integer-only ground truth
│   └── fixedpoint.py
├── engine/
│   ├── fixedpoint.js
│   ├── kernels.js
│   ├── model.js               layer graph driver + activation taps
│   ├── worker.js
│   └── test/*.test.js         node:test, no deps
├── web/
│   ├── index.html
│   ├── app.js                 camera, drop, paste, samples, UI
│   ├── style.css
│   └── samples/*.webp
├── dist/                      built output, rsync source
└── .github/workflows/ci.yml
```

`make deploy` → `rsync -a --delete dist/ ../EricSpencer00.github.io/hotdog/` on the
site repo's `dev` branch. **Before merging `dev` → `main`, diff the file trees** —
`dev` has silently dropped prod pages before.

---

## 9. Build order

| # | Phase | Est. | Gate |
|---|---|---|---|
| 1 | Repo scaffold, Makefile, uv env, torch/MPS smoke test | 20 min | `import torch; torch.backends.mps.is_available()` |
| 2 | Data ingest + manifests + splits + hard set | 90 min | counts + license table printed, no split leakage |
| 3 | Teacher fine-tune | 30 min | **≥96% val or stop and revisit data** |
| 4 | Student KD + QAT | 60 min | fake-quant val within 1pt of float val |
| 5 | Export + Python integer reference | 60 min | reference matches QAT model within tolerance |
| 6 | `fixedpoint.js` + its test | 45 min | **10^6-pair exact match or nothing proceeds** |
| 7 | Kernels + model driver + kernel tests | 90 min | all kernel tests green |
| 8 | Parity test | 45 min | **zero mismatches on full val set** |
| 9 | Frontend: UI, camera, worker, samples | 2.5 h | works in Safari + Chrome |
| 10 | Activation visualization | 60 min | tiles update live without dropping frames |
| 11 | Eval runs, README/model card, CI, HF push | 60 min | numbers measured, CI green |
| 12 | Deploy to site `dev`, projects.html + selected.txt | 30 min | dev.ericspencer.us/hotdog renders |

**Total: ~10 hours** of build time, plus dataset download wall-time.

Phases 3, 6, and 8 are hard gates. If the teacher misses 96%, or the fixed-point test
finds a single mismatch, or parity finds a single divergent logit, the correct action is
to stop and fix rather than proceed and paper over it with "close enough."

---

## 10. Known risks

1. **Open Images ingest is the flakiest step.** The V7 CSV annotations are large and the
   image URLs are Flickr-hosted with meaningful link rot. Mitigation: manifest records
   which URLs failed; the pipeline tolerates a fetch failure rate up to 30% and reports
   the realized positive count rather than assuming it.
2. **Bit-exactness across browsers** could break on the preprocessing path if anything
   engine-specific leaks in. Mitigation: box-filter resize implemented in JS, and the
   parity test runs on pre-resized 96×96 tensors so engine differences are isolated to
   the one testable function.
3. **Pure-JS latency on mobile** could be 3-5× worse than desktop. Mitigation: measure on
   an actual iPhone; if the camera loop is too slow, drop camera-mode inference to every
   Nth frame rather than degrading the model.
4. **The hard-set number may be genuinely mediocre** (corn dogs are legitimately hard).
   That is an acceptable outcome and gets published as-is. The interesting artifact is the
   pipeline and the parity proof, not a leaderboard score.
5. **Food-101's license is "unknown"** on HF. Images are not redistributed — only
   manifests — which keeps this to a documented caveat rather than a problem.
