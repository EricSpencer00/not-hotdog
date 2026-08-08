"""Pull hot dogs and food negatives out of ethz/food101.

Positives: the whole `hot_dog` class (1,000 images).
Negatives: N_FOOD_NEG images sampled from the other 100 classes, with the
classes listed in common.CONFUSABLE drawn at CONFUSABLE_BOOST x the base rate.

Images land in cache/images/ (gitignored). A manifest row per image is written
to data/manifests/food101.jsonl (committed) so the set is reproducible without
redistributing anything.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common import (  # noqa: E402
    CONFUSABLE,
    CONFUSABLE_BOOST,
    IMAGES,
    N_FOOD_NEG,
    split_for,
    write_manifest,
)

MAX_SIDE = 256
OUT = IMAGES / "food101"
LICENSE = "food101-unknown-research"


def save(img: Image.Image, path: Path) -> None:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=92)


def main() -> None:
    rng = random.Random(1337)
    ds = load_dataset("ethz/food101")
    names = ds["train"].features["label"].names
    hot_dog_idx = names.index("hot_dog")
    confusable_idx = {names.index(c) for c in CONFUSABLE}

    # Two-pass: first collect (split, row_index, label) so we can sample the
    # negative pool with the right per-class weights before decoding any JPEG.
    index: list[tuple[str, int, int]] = []
    for split in ("train", "validation"):
        labels = ds[split]["label"]
        index.extend((split, i, lab) for i, lab in enumerate(labels))

    positives = [r for r in index if r[2] == hot_dog_idx]
    negatives = [r for r in index if r[2] != hot_dog_idx]

    weights = [CONFUSABLE_BOOST if r[2] in confusable_idx else 1 for r in negatives]
    total_w = sum(weights)
    keep_p = min(1.0, N_FOOD_NEG / total_w)
    chosen_neg = [r for r, w in zip(negatives, weights) if rng.random() < keep_p * w]
    rng.shuffle(chosen_neg)
    chosen_neg = chosen_neg[:N_FOOD_NEG]

    print(f"food101: {len(positives)} positives, {len(chosen_neg)} negatives selected")

    rows: list[dict] = []
    for split, i, lab in tqdm(positives + chosen_neg, desc="food101"):
        label = 1 if lab == hot_dog_idx else 0
        sid = f"food101:{split}:{i}"
        rel = Path("food101") / ("pos" if label else "neg") / f"{split}_{i}.jpg"
        out = IMAGES / rel
        if not out.exists():
            save(ds[split][i]["image"], out)
        rows.append(
            {
                "source_id": sid,
                "path": str(rel),
                "label": label,
                "split": split_for(sid),
                "source": "food101",
                "class_name": names[lab],
                "license": LICENSE,
                "url": "https://huggingface.co/datasets/ethz/food101",
            }
        )

    p = write_manifest("food101", rows)
    n_pos = sum(r["label"] for r in rows)
    print(f"wrote {p} — {len(rows)} rows ({n_pos} pos / {len(rows) - n_pos} neg)")


if __name__ == "__main__":
    main()
