"""Stage 2: distil the teacher into the 136K-parameter student, then QAT it.

Three phases in one run:

  1. KD (float)   student learns from the teacher's soft targets
  2. observe      activation ranges calibrated with the network frozen
  3. QAT          BN folded, weights and activations fake-quantized, LR/10

Phase 2 exists because you cannot fake-quantize an activation before you know
its range, and you cannot know its range without running data through the
trained network. Doing it as a separate pass with no weight updates keeps the
observed ranges honest — if the weights were still moving, the EMA would be
tracking a distribution that no longer exists by the time training ends.

The distillation loss is binary cross-entropy against the teacher's *soft*
probability at temperature T, not against the hard label. For a 200x smaller
student that difference is most of the point: the hard label says "hot dog",
the soft target says "0.83 hot dog", and the gap encodes what the teacher
knows about how hot-dog-like this particular image is.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import CKPT, RESULTS
from train.data import describe, load_split
from train.model import Student, param_count
from train.teacher import MODEL as TEACHER_MODEL
from train.teacher import device, make_loader


def kd_loss(z_s: torch.Tensor, z_t: torch.Tensor, T: float) -> torch.Tensor:
    """Binary KD. Cross-entropy of the student against the teacher's soft
    probability, scaled by T^2 so the gradient magnitude does not collapse as T
    rises (the standard Hinton correction)."""
    soft = torch.sigmoid(z_t / T)
    return F.binary_cross_entropy_with_logits(z_s / T, soft) * (T * T)


@torch.no_grad()
def evaluate(model, loader, dev) -> dict:
    model.eval()
    tp = fp = tn = fn = 0
    for _, sx, y in loader:
        logits = model(sx.to(dev)).float().cpu()
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


def run_epochs(model, teacher, tl, vl, dev, epochs, lr, args, tag, best, hist):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=4e-5)
    steps = max(1, epochs * len(tl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    for ep in range(epochs):
        model.train()
        if tag == "qat":
            # Freeze BN statistics: the fold uses running_mean/var, so those must
            # stop moving or the folded weights drift under the quantizer.
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        t0, tot, seen = time.time(), 0.0, 0
        for i, (tx, sx, y) in enumerate(tl):
            tx, sx, y = tx.to(dev), sx.to(dev), y.to(dev)
            with torch.no_grad():
                z_t = teacher(tx).squeeze(1)
            z_s = model(sx)
            loss = args.alpha * kd_loss(z_s, z_t, args.temp) + \
                (1 - args.alpha) * F.binary_cross_entropy_with_logits(z_s, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()
            tot += float(loss) * len(y)
            seen += len(y)
            if i % 50 == 0:
                print(f"  [{tag}] ep{ep} {i}/{len(tl)} loss={tot / max(1, seen):.4f}", flush=True)
        m = evaluate(model, vl, dev)
        hist.append({"phase": tag, "epoch": ep, "loss": tot / max(1, seen), **m})
        print(f"[{tag}] epoch {ep}: loss={tot / max(1, seen):.4f} "
              f"val_acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        # Select on F1, not accuracy. The validation split is ~92% negative, so
        # accuracy rewards a model that quietly stops predicting the positive
        # class — and no amount of threshold calibration afterwards can recover
        # a checkpoint that has already collapsed to "no".
        if m["f1"] >= best[tag]:
            best[tag] = m["f1"]
            best[tag + "_acc"] = m["accuracy"]
            torch.save(model.state_dict(), CKPT / f"student_{tag}.pt")
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kd-epochs", type=int, default=30)
    ap.add_argument("--qat-epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--temp", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    dev = device()
    tr, va = load_split("train"), load_split("val")
    print(describe(tr, "train"))
    print(describe(va, "val"))

    teacher = timm.create_model(TEACHER_MODEL, pretrained=False, num_classes=1)
    teacher.load_state_dict(torch.load(CKPT / "teacher.pt", map_location="cpu"))
    teacher = teacher.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    model = Student().to(dev)
    print(f"student shipped params: {param_count(model)}")

    tl = make_loader(tr, True, args.bs, args.workers)
    vl = make_loader(va, False, args.bs, args.workers)

    hist: list[dict] = []
    best = {"kd": 0.0, "qat": 0.0, "kd_acc": 0.0, "qat_acc": 0.0}

    # ── phase 1: float KD ────────────────────────────────────────────────────
    model.set_quant("off", weights=False)
    best = run_epochs(model, teacher, tl, vl, dev, args.kd_epochs, args.lr,
                      args, "kd", best, hist)
    model.load_state_dict(torch.load(CKPT / "student_kd.pt", map_location=dev))
    float_f1, float_acc = best["kd"], best["kd_acc"]

    # ── phase 2: calibrate activation ranges, no weight updates ──────────────
    print("\ncalibrating activation ranges...")
    model.set_quant("observe", weights=False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    model.train()  # observers only update their EMA in train mode
    with torch.no_grad():
        for i, (_, sx, _) in enumerate(tl):
            model(sx.to(dev))
            if i >= 60:
                break

    # ── phase 3: QAT with BN folded ──────────────────────────────────────────
    for m in model.modules():
        from train.model import ConvBNReLU
        if isinstance(m, ConvBNReLU):
            m.fold_bn = True
    model.set_quant("fake", weights=True)
    best["qat"] = 0.0
    best = run_epochs(model, teacher, tl, vl, dev, args.qat_epochs, args.lr / 10,
                      args, "qat", best, hist)

    out = {
        "shipped_params": param_count(model),
        "float_val_f1": float_f1,
        "float_val_accuracy": float_acc,
        "qat_val_f1": best["qat"],
        "qat_val_accuracy": best["qat_acc"],
        "quant_drop_f1": float_f1 - best["qat"],
        "temperature": args.temp,
        "alpha": args.alpha,
        "selection_metric": "f1",
        "history": hist,
    }
    (RESULTS / "student.json").write_text(json.dumps(out, indent=2))
    print(f"\nfloat KD  F1={float_f1:.4f} acc={float_acc:.4f}")
    print(f"QAT       F1={best['qat']:.4f} acc={best['qat_acc']:.4f} "
          f"(F1 cost of quantization: {float_f1 - best['qat']:+.4f})")


if __name__ == "__main__":
    main()
