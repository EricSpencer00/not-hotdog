# Not Hotdog

**[ericspencer.us/hotdog](https://ericspencer.us/hotdog)**

A convolutional network that answers one question — *is this a hot dog?* — with
136,417 int8 parameters and hand-written JavaScript kernels. No TensorFlow.js,
no ONNX Runtime, no WebAssembly, no npm. The page is an HTML file and one
script, and after it loads it never talks to the network again.

Inspired by the SeeFood gag in HBO's *Silicon Valley*. Not affiliated with HBO.

<!--NUMBERS-->
## Numbers

Measured on the **int8 model** — the one that ships — with the browser's exact
preprocessing. Not the float checkpoint.

| | value |
|---|---|
| parameters | 136,417 (int8) |
| weights | 133 KB |
| page weight | 239 KB raw, ~155 KB gzipped |
| dependencies | 0 |
| inference | ~26 ms (Node, M1 Max) |
| MACs | 14.0M |

### Accuracy

| split | n | positives | accuracy | majority baseline | F1 | precision | recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 3,369 | 263 | 95.3% | 92.2% | 0.672 | 0.745 | 0.612 |
| test | 3,415 | 253 | 94.8% | 92.6% | 0.634 | 0.661 | 0.609 |
| adversarial | 150 | 14 | 81.3% | 90.7% | 0.176 | 0.150 | 0.214 |

**Read F1, not accuracy.** The evaluation sets are built to be hard: roughly
92.6% of the test split is negative, so a model that
says "not a hot dog" to literally everything already scores
92.6%. The accuracy column is there because people expect
it, not because it means much.

The precision figure is also measured against that same deliberately hostile
mix — 12,000 photographs of other food and 8,000 photographs of unrelated
objects against roughly 1,800 hot dogs. Someone pointing a camera at their lunch
is not sampling from that distribution.

### Adversarial set, by class

Six classes held out entirely from training.

| class | n | correct | accuracy |
|---|---:|---:|---:|
| `bratwurst` | 40 | 34/40 | 85.0% |
| `chili_dog` | 12 | 3/12 | 25.0% |
| `corn_dog` | 10 | 8/10 | 80.0% |
| `dachshund` | 40 | 40/40 | 100.0% |
| `hot_dog_bun` | 40 | 31/40 | 77.5% |
| `hot_dog_wild` | 2 | 0/2 | 0.0% |
| `sausage_roll` | 6 | 6/6 | 100.0% |

Small-n classes are reported as-is rather than dropped: Open Images contains 46
corn dogs in total, and only the subset on the CVDF mirror is fetchable.

Note that adversarial accuracy (81.3%) is *below* the
majority baseline (90.7%). That is not a rounding
artefact, it is the result: on this set, the model is worse than a rock with
"not a hot dog" written on it.

The per-class split says why, and it is more interesting than the aggregate. The
model is good at *rejecting* things that are nearly hot dogs — it gets
119/136 of the held-out negatives, never once mistakes a dachshund for
food, and correctly refuses 8 of 10 corn dogs. What it fails is
hard *positives*: 3/14, driven by chili dogs, which are hot dogs
buried under chili and look nothing like the canonical article.

So the model has learned a narrow, prototypical hot dog and is conservative
about anything outside it. Given 1,834 positives and 136K parameters that is the
failure mode you would predict, but it is worth stating plainly rather than
reporting the flattering test number and moving on.

### The pipeline, end to end

| stage | metric |
|---|---|
| teacher (EfficientNet-B0, 5.3M params) | F1 0.837, accuracy 97.6% |
| student after distillation (float) | F1 0.584 |
| student after QAT (int8) | F1 0.586 |
| cost of quantization | -0.003 F1 |
| decision threshold (int32, folded into bias) | 55,983 |

The teacher-to-student gap is the honest headline: a 136K-parameter network is
not going to match a 5.3M-parameter one, and distillation narrows that gap
rather than closing it. What quantization costs on top of that is small, which
is the part QAT was for.

### Verification

| check | result |
|---|---|
| fixed-point vs Python reference | 1,005,400 vectors, 0 mismatches |
| per-kernel vs NumPy | 29 cases, 0 mismatches |
| JS engine vs Python integer reference | 3,369 validation images, **0 mismatches** |
<!--/NUMBERS-->

## Why this is interesting

Running a neural network in a browser is not hard; there are three good
libraries for it. What is mildly interesting is running one with *nothing* —
writing the convolutions yourself, in integers, and then proving the result is
correct rather than asserting it.

Three things fall out of that constraint.

**The multiply.** Requantizing an int32 accumulator needs the high 32 bits of a
signed 64-bit product. JavaScript's only number is a double with a 53-bit
mantissa, so `a * b` silently loses the low bits and is unusable. BigInt is
exact but far too slow to call once per output element. See
[the hard part](#the-hard-part).

**The zero-point.** Every activation in the network is the output of a ReLU and
therefore non-negative, so activations are quantized as unsigned symmetric with
zero-point 0 rather than asymmetric. This costs nothing — asymmetric
quantization only buys range when values go negative — and it deletes the
zero-point correction term from every kernel and makes SAME padding *genuinely*
zero instead of "the quantized value that represents zero".

**The proof.** The JavaScript engine and a NumPy implementation of the same
integer program must produce identical int32 logits on every validation image.
Not close. Identical. A convolution bug does not crash; it makes the model
slightly worse in a way nobody would ever notice.

## The hard part

`SaturatingRoundingDoublingHighMul(a, b)` — gemmlowp's core primitive — is the
one function the whole port rests on. The trick is to never form the 64-bit
product at all. Split each int32 operand into a signed high half and an
unsigned low half:

```js
const aHi = a >> 16;        // signed,   |aHi| <= 2^15
const aLo = a & 0xffff;     // unsigned, < 2^16
// a === aHi * 2^16 + aLo, exactly, for any two's complement int32
```

Then with `A = aHi*bHi`, `B = aHi*bLo + aLo*bHi`, `C = aLo*bLo`:

```
a*b = A*2^32 + B*2^16 + C
```

`|A| <= 2^30`, `|B| < 2^32` and `C < 2^32`, so each is an exact double. The
useful part is that `T = B*2^16 + C + nudge` stays below `2^49` and is therefore
*also* exact, and since `a*b + nudge = (2A)*2^31 + T`, the quotient by `2^31`
falls out of a floor-divide of `T` alone.

Verified against the Python reference over 1,000,000 random pairs plus every
boundary combination — `INT32_MIN`, `INT32_MAX`, powers of two, the values where
the high/low split carries. It runs a million multiplies in ~50 ms, which is why
it is a plain function call in the inner loop and not a BigInt.

See [`engine/fixedpoint.js`](engine/fixedpoint.js) and
[`reference/fixedpoint.py`](reference/fixedpoint.py).

## Architecture

96×96×3 input, MobileNet-style depthwise-separable stack, ReLU only. No
hard-swish, no squeeze-excite, no residual connections — every operation maps to
one of four kernels, because every operation had to be written by hand.

| Stage | Op | Output | Params |
|---|---|---|---:|
| stem | conv 3×3 s2, 3→16 | 48×48×16 | 432 |
| b1 | dw 3×3 + pw 16→32 | 48×48×32 | 656 |
| b2 | dw 3×3 s2 + pw 32→64 | 24×24×64 | 2,336 |
| b3 | dw 3×3 + pw 64→64 | 24×24×64 | 4,672 |
| b4 | dw 3×3 s2 + pw 64→128 | 12×12×128 | 8,768 |
| b5 | dw 3×3 + pw 128→128 | 12×12×128 | 17,536 |
| b6 | dw 3×3 s2 + pw 128→256 | 6×6×256 | 33,920 |
| b7 | dw 3×3 + pw 256→256 | 6×6×256 | 67,840 |
| head | global avg pool → fc | 1 | 257 |
| | | **total** | **136,417** |

14.0M multiply-accumulates per forward pass.

### Quantization

| | scheme |
|---|---|
| weights | int8, per-output-channel symmetric, zero-point 0 |
| activations | uint8, per-tensor symmetric, zero-point 0 |
| bias | int32 at scale `s_w[c] · s_in` |
| accumulate | int32 |
| requantize | `RoundingDivideByPOT(SRDHM(acc, m0[c]), shift[c])` |

BatchNorm is folded into the convolution **during** QAT, not at export. Folding
only at export would mean the weights QAT trained on are not the weights that
ship: the fold rescales each output channel by `γ/√(σ²+ε)`, which changes every
per-channel maximum and therefore every weight scale. Training against unfolded
weights and shipping folded ones is a silent accuracy leak.

The input is fed as `pixel/255`, so the input quantizer sits at exactly scale
1/255, zero-point 0 — the quantized input tensor is bit-identical to the raw RGB
bytes off the canvas, and preprocessing is a byte copy.

## Training

1. **Teacher.** EfficientNet-B0, ImageNet-pretrained, fine-tuned at 224px.
2. **Distillation.** 30 epochs against the teacher's soft targets at `T=4`,
   `α=0.7`. For a student 200× smaller this is most of the point: the hard label
   says "hot dog", the soft target says "0.83 hot dog", and the gap encodes what
   the teacher knows about *how* hot-dog-like a particular image is.
3. **Calibration.** Activation ranges observed with the weights frozen. Doing
   this as a separate pass matters — if the weights were still moving, the EMA
   would track a distribution that no longer exists by the time training ends.
4. **QAT.** 8 further epochs with BN folded and fake-quant in the graph, LR/10.

## Data

No images are redistributed. `data/manifests/*.jsonl` records the source, URL,
licence and split of every image, and `make data` reconstructs the set.

| Source | Role | Licence |
|---|---|---|
| [Food-101](https://huggingface.co/datasets/ethz/food101) `hot_dog` | positives | unknown / research |
| [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) `Hot dog` boxes | positives, cropped | CC BY 2.0 |
| Open Images V7 `Hot dog` image labels | positives | CC BY 2.0 |
| Food-101, other 100 classes | food negatives | unknown / research |
| Open Images: hamburger, sub, bagel, burrito, pretzel | confusable negatives | CC BY 2.0 |
| [Imagenette](https://github.com/fastai/imagenette) | non-food negatives | ImageNet terms |

Food negatives are oversampled 3× from the classes most confusable with a hot
dog. Non-food negatives exist so the model learns "is this a hot dog" rather
than "is this a photo of food" — without them every negative is a plated dish.

Splits are deterministic (`sha256(group_key) % 100`) and grouped by source
photo, so multiple crops of one image can never straddle a split boundary.

### The adversarial set

Six classes held out **entirely** — not one image of any of them appears in
training:

`corn_dog` · `bratwurst` · `sausage_roll` · `hot_dog_bun` · `dachshund` ·
`chili_dog`

This makes it a test of whether the model learned what a hot dog *is*, rather
than whether it memorised the negatives it was shown. A corn dog is the case the
whole exercise turns on: sausage in a cylindrical carbohydrate, and the answer
is no. An empty hot dog bun is the mirror image. A dachshund is there because
somebody was always going to try it.

Per-class counts are reported with the accuracy because some of these classes
are genuinely rare — Open Images contains 46 corn dogs in total, and only the
subset hosted on the CVDF mirror is fetchable.

## Verification

```
make test        # kernels, fixed point, and parity over every val image
make test-fast   # what CI runs: no dataset, parity on 300 synthetic images
```

| Test | What it establishes |
|---|---|
| `fixedpoint.test.js` | 1,000,000 random pairs + all boundary combinations, exact |
| `kernels.test.js` | each kernel vs NumPy: stride-2 on odd inputs, SAME-pad edges, int8 saturation |
| `parity.test.js` | JS logit ≡ Python logit, exactly, on every validation image |
| | taps are observational — capturing activations cannot change the output |
| | repeated inference is stateless — reused buffers cannot leak the previous frame |

CI runs all of it on every push with numpy and node and no dataset, since the
136 KB exported model is committed.

## Layout

```
data/        ingest + manifests (no images committed)
train/       teacher, distillation, QAT, export, eval
reference/   the integer program in NumPy — the ground truth
engine/      the integer program in JavaScript + its tests
web/         index.html, app, worker
dist/        bundled output
```

`build.py` concatenates the ES modules, strips the import/export syntax and
embeds the worker as a string, so the deployed page is two files and makes no
requests. The modules exist so `node:test` can import individual kernels.

## Reproducing

```
make setup
make data       # ~6 GB of downloads
make teacher
make student
make build
make test
make eval
```

Trained on an M1 Max via MPS. The teacher is ~20 minutes, the student ~70.

## Limitations

- The model is small and it is not subtle. It is good at "obvious hot dog" and
  "obviously not food"; it is much less good at the boundary, which is exactly
  where the adversarial numbers above come from.
- Positives are the binding constraint. Clean, licence-clear hot dog images are
  scarce; the whole of Food-101 contains 1,000 and Open Images adds fewer than
  most people would guess.
- Food-101's licence is listed as unknown. Nothing from it is redistributed —
  only manifests — but it is a caveat rather than a non-issue.
- Bit-exact parity is established between this JavaScript and this NumPy. Any
  browser running the same JavaScript gets the same answer, because the forward
  pass has no floating point in it and the resize is a hand-written integer box
  filter rather than `drawImage`, whose resampling differs between engines.

## Licence

MIT. See [`LICENSE`](LICENSE). Sample thumbnails are Open Images, CC BY 2.0; see
[`web/samples/ATTRIBUTION.md`](web/samples/ATTRIBUTION.md).
