"""Pick the decision threshold, in int32, on the validation split.

Training uses a class-balanced sampler: positives are outnumbered roughly 11:1,
and letting the network see that ratio makes it learn "say no" rather than
"find hot dogs". Balancing fixes the learning problem and creates a calibration
one — the model's implied prior is now 50/50, but the world it is deployed into
is not, so thresholding the logit at zero fires far too eagerly.

The fix is to move the threshold rather than retrain: sweep every distinct int32
logit on the validation set, take the one that maximises F1, and fold it into
the final bias at export. After that the shipped model's decision really is
`accumulator > 0` and the browser needs to know nothing about any of this.

The threshold is chosen on validation and reported on test, which is the only
ordering that keeps the test number honest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from tqdm import tqdm

from common import IMAGES, RESULTS, all_rows
from reference.int8_reference import Int8Model
from reference.preprocess import crop_resize


def f1_at(logits: np.ndarray, labels: np.ndarray, t: float) -> tuple[float, float, float]:
    pred = logits > t
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return 2 * prec * rec / max(1e-9, prec + rec), prec, rec


def main() -> None:
    model = Int8Model()
    rows = [r for r in all_rows() if r["split"] == "val"]
    logits, labels = [], []
    for r in tqdm(rows, desc="val logits"):
        img = np.asarray(Image.open(IMAGES / r["path"]).convert("RGB"), dtype=np.uint8)
        logits.append(model.forward(crop_resize(img, model.size)))
        labels.append(r["label"])

    logits = np.array(logits, dtype=np.int64)
    labels = np.array(labels, dtype=np.int64)

    base_f1, base_p, base_r = f1_at(logits, labels, 0)

    # Candidate thresholds sit between adjacent distinct logits, so the sweep is
    # exhaustive over every distinct classification the model can produce.
    uniq = np.unique(logits)
    cands = np.concatenate([[uniq[0] - 1], (uniq[:-1] + uniq[1:]) // 2, [uniq[-1] + 1]])

    best = (-1.0, 0, 0.0, 0.0)
    for t in cands:
        f1, p, r = f1_at(logits, labels, t)
        if f1 > best[0]:
            best = (f1, int(t), p, r)

    f1, t, p, r = best
    out = {
        "threshold_int32": t,
        "f1_at_threshold": f1, "precision_at_threshold": p, "recall_at_threshold": r,
        "f1_at_zero": base_f1, "precision_at_zero": base_p, "recall_at_zero": base_r,
        "n_val": len(rows),
    }
    (RESULTS / "calibration.json").write_text(json.dumps(out, indent=2))

    print(f"\nthreshold 0      F1={base_f1:.4f}  P={base_p:.3f}  R={base_r:.3f}")
    print(f"threshold {t:<7d} F1={f1:.4f}  P={p:.3f}  R={r:.3f}   <- folded into the bias")
    print(f"\nwrote {RESULTS / 'calibration.json'}")
    print("now re-run: python train/export.py")


if __name__ == "__main__":
    main()
