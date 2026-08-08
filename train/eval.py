"""Measure the model that actually ships.

Evaluation runs the int8 integer reference, not the float checkpoint, and
preprocesses with the same crop-and-box-resize the browser uses. Reporting a
float model's accuracy for a page that serves a quantized one would be quietly
dishonest, and the gap is exactly the thing worth knowing.

Every accuracy figure is printed next to the majority-class baseline. On a set
that is 92% negative, "97% accurate" sounds like a result and is worth five
points over predicting "not a hot dog" every single time. F1 is the number to
read; accuracy is reported because people expect it.

Outputs train/results/eval.json, which the README and the page both read, so a
number cannot drift between the places it appears.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from tqdm import tqdm

from common import IMAGES, RESULTS, all_rows, read_manifest
from reference.int8_reference import Int8Model
from reference.preprocess import crop_resize


def metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n = len(y_true)
    pos = tp + fn
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    return {
        "n": n,
        "positives": pos,
        "accuracy": (tp + tn) / max(1, n),
        "majority_baseline": max(pos, n - pos) / max(1, n),
        "balanced_accuracy": (rec + spec) / 2,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": 2 * prec * rec / max(1e-9, prec + rec),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def run(model: Int8Model, rows: list[dict], desc: str):
    y_true, y_pred, per_class = [], [], defaultdict(lambda: [0, 0])
    for r in tqdm(rows, desc=desc, leave=False):
        img = np.asarray(Image.open(IMAGES / r["path"]).convert("RGB"), dtype=np.uint8)
        logit = model.forward(crop_resize(img, model.size))
        pred = 1 if logit > 0 else 0
        y_true.append(r["label"])
        y_pred.append(pred)
        cls = per_class[r.get("class_name", "?")]
        cls[1] += 1
        if pred == r["label"]:
            cls[0] += 1
    return y_true, y_pred, per_class


def fmt(name: str, m: dict) -> str:
    return (
        f"{name:12s} n={m['n']:5d}  pos={m['positives']:4d}  "
        f"acc={m['accuracy']:.4f} (baseline {m['majority_baseline']:.4f})  "
        f"bal={m['balanced_accuracy']:.4f}  P={m['precision']:.3f}  "
        f"R={m['recall']:.3f}  F1={m['f1']:.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap rows per split (0 = all)")
    args = ap.parse_args()

    model = Int8Model()
    out: dict = {"model": "int8", "params": int(model.w.size)}

    print("Evaluating the int8 model (the one that ships).\n")

    for split in ("val", "test"):
        rows = [r for r in all_rows() if r["split"] == split]
        if args.limit:
            rows = rows[: args.limit]
        yt, yp, _ = run(model, rows, split)
        m = metrics(yt, yp)
        out[split] = m
        print(fmt(split, m))

    hard = read_manifest("hard")
    if hard:
        yt, yp, per_class = run(model, hard, "hard")
        m = metrics(yt, yp)
        m["per_class"] = {
            k: {"correct": v[0], "n": v[1], "accuracy": v[0] / max(1, v[1])}
            for k, v in sorted(per_class.items())
        }
        out["hard"] = m
        print("\n" + fmt("hard", m))
        print("\n  held-out classes (never seen in training):")
        for k, v in m["per_class"].items():
            warn = "   <- small n" if v["n"] < 20 else ""
            print(f"    {k:20s} {v['correct']:3d}/{v['n']:3d}  {v['accuracy']:.3f}{warn}")

    (RESULTS / "eval.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'eval.json'}")


if __name__ == "__main__":
    main()
