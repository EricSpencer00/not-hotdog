"""Stage 1: fine-tune an ImageNet-pretrained EfficientNet-B0 as the KD teacher.

The teacher never ships. Its only job is to produce soft targets good enough
that a 136K-parameter student can learn from them, which means it has to be
comfortably better than the student could ever be on its own.

Hard gate (SPEC.md §9): >= 96% validation accuracy. Below that the teacher is
not worth distilling from and the data pipeline is the thing to fix, not the
training recipe.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from common import CKPT, RESULTS
from train.data import HotDogDataset, describe, load_split

GATE = 0.96
MODEL = "efficientnet_b0"


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_loader(rows, train: bool, bs: int, workers: int) -> DataLoader:
    ds = HotDogDataset(rows, train=train)
    if train:
        # Positives are heavily outnumbered here. Sample them up rather than
        # throwing negatives away — the negatives are where the difficulty is.
        n_pos = sum(r["label"] for r in rows)
        n_neg = len(rows) - n_pos
        w = [(1.0 / n_pos) if r["label"] else (1.0 / n_neg) for r in rows]
        sampler = WeightedRandomSampler(w, num_samples=len(rows), replacement=True)
        return DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=workers,
                          drop_last=True, persistent_workers=workers > 0)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                      persistent_workers=workers > 0)


@torch.no_grad()
def evaluate(model, loader, dev) -> dict:
    model.eval()
    tp = fp = tn = fn = 0
    for teacher_x, _, y in loader:
        logits = model(teacher_x.to(dev)).squeeze(1).float().cpu()
        pred = (logits > 0).float()
        tp += int(((pred == 1) & (y == 1)).sum())
        fp += int(((pred == 1) & (y == 0)).sum())
        tn += int(((pred == 0) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum())
    n = tp + fp + tn + fn
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "accuracy": (tp + tn) / max(1, n),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(1e-9, prec + rec),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    dev = device()
    tr, va = load_split("train"), load_split("val")
    print(describe(tr, "train"))
    print(describe(va, "val"))

    model = timm.create_model(MODEL, pretrained=True, num_classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, args.epochs * (len(tr) // args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss()

    tl = make_loader(tr, True, args.bs, args.workers)
    vl = make_loader(va, False, args.bs, args.workers)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for i, (tx, _, y) in enumerate(tl):
            tx, y = tx.to(dev), y.to(dev)
            loss = lossf(model(tx).squeeze(1), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()
            tot += float(loss) * len(y)
            seen += len(y)
            if i % 50 == 0:
                print(f"  ep{ep} step{i}/{len(tl)} loss={tot / max(1, seen):.4f}", flush=True)
        m = evaluate(model, vl, dev)
        print(f"epoch {ep}: loss={tot / max(1, seen):.4f} val_acc={m['accuracy']:.4f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if m["accuracy"] > best:
            best = m["accuracy"]
            torch.save(model.state_dict(), CKPT / "teacher.pt")
            (RESULTS / "teacher.json").write_text(json.dumps(
                {"model": MODEL, "epoch": ep, **m}, indent=2))

    print(f"\nbest val accuracy: {best:.4f}  (gate {GATE})")
    if best < GATE:
        print("GATE FAILED — teacher is not good enough to distil from.")
        sys.exit(1)
    print("gate passed")


if __name__ == "__main__":
    main()
