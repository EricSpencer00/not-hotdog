"""Push the int8 model and its card to huggingface.co/EricSpencer00/not-hotdog.

Uploads the exported artifact and the bundled engine, not a PyTorch checkpoint,
because the artifact is the thing worth having: 136 KB of int8 weights plus the
two implementations that read them. The teacher stays local — it exists only to
produce soft targets and has no independent value.

Numbers in the card are read from train/results/eval.json rather than typed, so
they cannot drift from what was actually measured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi

from common import DIST, REFDIR, RESULTS, ROOT

REPO = "EricSpencer00/not-hotdog"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def card(ev: dict, student: dict) -> str:
    t, h = ev.get("test", {}), ev.get("hard", {})
    rows = "\n".join(
        f"| `{k}` | {v['correct']}/{v['n']} | {pct(v['accuracy'])} |"
        for k, v in h.get("per_class", {}).items()
    )
    return f"""---
license: mit
tags:
  - image-classification
  - quantized
  - int8
  - tinyml
  - javascript
pipeline_tag: image-classification
---

# Not Hotdog

A {ev['params']:,}-parameter int8 CNN that decides whether an image is a hot dog,
built to run in a browser with hand-written JavaScript kernels and no
dependencies of any kind.

**Live demo:** https://ericspencer.us/hotdog
**Source:** https://github.com/EricSpencer00/not-hotdog

## Results

Measured on the int8 model — the one that actually ships — not on the float
checkpoint.

| Split | n | accuracy | majority baseline | F1 | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| test | {t.get('n', 0)} | {pct(t.get('accuracy', 0))} | {pct(t.get('majority_baseline', 0))} | {t.get('f1', 0):.3f} | {t.get('precision', 0):.3f} | {t.get('recall', 0):.3f} |
| adversarial | {h.get('n', 0)} | {pct(h.get('accuracy', 0))} | {pct(h.get('majority_baseline', 0))} | {h.get('f1', 0):.3f} | {h.get('precision', 0):.3f} | {h.get('recall', 0):.3f} |

The adversarial split is six classes held out entirely from training:

| class | correct | accuracy |
|---|---:|---:|
{rows}

Read F1, not accuracy. The evaluation sets are heavily negative, so predicting
"not a hot dog" unconditionally already scores the majority baseline above.

## Architecture

96x96x3 input, MobileNet-style depthwise-separable stack, ReLU only, global
average pool, one logit. 14.0M MACs.

- weights: int8, per-output-channel symmetric, zero-point 0
- activations: uint8, per-tensor symmetric, zero-point 0
- accumulators: int32, requantized with a fixed-point multiply and shift

Trained by distilling a fine-tuned EfficientNet-B0 (T=4, alpha=0.7), then
quantization-aware training with BatchNorm folded into the convolutions.

## Files

| file | what it is |
|---|---|
| `model_int8.npz` | weights + layer graph, read by the NumPy reference |
| `model.js` | the same weights, base64, as an ES module |
| `hotdog.js` | the complete bundled engine — drop it in a page |

## Limitations

Small model, and not subtle. Good at obvious hot dogs and obvious non-food,
much weaker at the boundary, which is what the adversarial numbers show. Clean
licence-clear hot dog images are scarce; positives were the binding constraint
throughout.

## Licence

MIT.
"""


def main() -> None:
    ev_path = RESULTS / "eval.json"
    if not ev_path.exists():
        sys.exit("missing train/results/eval.json — run train/eval.py first")
    ev = json.loads(ev_path.read_text())
    student = json.loads((RESULTS / "student.json").read_text())

    api = HfApi()
    who = api.whoami()["name"]
    print(f"authenticated as {who}")
    api.create_repo(REPO, repo_type="model", exist_ok=True)

    (ROOT / "MODEL_CARD.md").write_text(card(ev, student))

    uploads = [
        (ROOT / "MODEL_CARD.md", "README.md"),
        (REFDIR / "model_int8.npz", "model_int8.npz"),
        (DIST / "model.js", "model.js"),
        (DIST / "hotdog.js", "hotdog.js"),
        (RESULTS / "eval.json", "eval.json"),
        (RESULTS / "student.json", "training.json"),
    ]
    for src, dst in uploads:
        if not src.exists():
            print(f"  ! skipping missing {src}")
            continue
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                        repo_id=REPO, repo_type="model")
        print(f"  uploaded {dst} ({src.stat().st_size / 1024:.1f} KB)")

    print(f"\nhttps://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
