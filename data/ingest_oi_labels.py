"""Open Images image-level labels: extra positives, and the adversarial set.

Two outputs, and the split between them is deliberate.

`oi_labels` (training)
    Hot dog positives that have a human-verified image-level label but no
    bounding box, plus confusable-food negatives: hamburger, submarine
    sandwich, bagel, burrito, pretzel. These are in-the-wild photos rather
    than Food-101's centred restaurant framing.

`hard` (evaluation only)
    Corn dog, bratwurst, sausage roll, hot dog bun, dachshund, chili dog.
    These classes are held out *entirely* — no image of any of them appears in
    training. That makes the hard set a test of whether the model learned what
    a hot dog is, rather than a test of whether it memorised the negatives it
    was shown. A corn dog is the case the whole exercise turns on: sausage in a
    cylindrical carbohydrate, and the answer is no.

Fine-grained classes have no human-verified labels in Open Images, only
machine-generated ones, so the hard set is drawn from those at a high
confidence threshold and then spot-checked.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common import IMAGES, read_manifest, split_for, write_manifest  # noqa: E402

MIRROR = "https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg"
LICENSE = "CC BY 2.0 (Open Images V7; per-image attribution at https://storage.googleapis.com/openimages/web/download_v7.html)"
MAX_SIDE = 256
WORKERS = 6
TIMEOUT = 30

HUMAN_CSV = Path("/tmp/oi-imagelabels-filtered.csv")
MACHINE_CSV = Path("/tmp/oi-machine-filtered.csv")

# label -> (class name, is_hot_dog)
TRAIN_CLASSES = {
    "/m/01b9xk": ("hot_dog", 1),
    "/m/0cdn1": ("hamburger", 0),
    "/m/06pcq": ("submarine_sandwich", 0),
    "/m/01fb_0": ("bagel", 0),
    "/m/01j3zr": ("burrito", 0),
    "/m/01f91_": ("pretzel", 0),
}
HARD_CLASSES = {
    "/m/02_ty7": ("corn_dog", 0),
    "/m/01gxnj": ("bratwurst", 0),
    "/m/018_zj": ("sausage_roll", 0),
    "/m/07vh6y": ("hot_dog_bun", 0),
    "/m/02cj3": ("dachshund", 0),
    "/m/0ch5ncy": ("chili_dog", 1),
    # Hot dogs in the wild carrying neither a human-verified label nor a
    # bounding box, so none of them can appear in training. Without hard
    # positives the adversarial set would only measure precision; these are
    # what measure recall.
    "/m/01b9xk": ("hot_dog_wild", 1),
}

CAP_TRAIN_NEG = 400   # per class, so no single negative class dominates
CAP_HARD = 40         # per class, keeps the hard set ~250 and hand-checkable
# Machine labels carry a confidence. Corn dog has only 46 instances in all of
# Open Images, so thresholding would gut the single most important adversarial
# class; rank by confidence and take the top CAP_HARD instead.
MACHINE_MIN_CONF = 0.0


def fetch(image_id: str, rel: Path, attempts: int = 4) -> bool:
    """Download one image. Idempotent — an existing file is a hit, so reruns
    only retry what failed.

    The mirror throttles under concurrency, which shows up as a uniform partial
    failure rate rather than as missing images (the same IDs return 200 when
    probed one at a time). Retry with backoff instead of treating a throttled
    request as a dead image.
    """
    dst = IMAGES / rel
    if dst.exists():
        return True
    img = None
    for attempt in range(attempts):
        try:
            r = requests.get(MIRROR.format(image_id=image_id), timeout=TIMEOUT)
            if r.status_code == 404:
                return False
            if r.status_code != 200:
                raise RuntimeError(f"http {r.status_code}")
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            break
        except Exception:
            if attempt == attempts - 1:
                return False
            time.sleep(1.5 * (attempt + 1))
    if img is None:
        return False
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=92)
    return True


def training_image_ids() -> set[str]:
    """Every Open Images ID already used for training, from the committed
    manifests. The adversarial set must not contain any of them."""
    seen: set[str] = set()
    for name in ("openimages", "oi_labels"):
        for r in read_manifest(name):
            sid = r["source_id"]
            parts = sid.split(":")
            if parts[0] == "openimages" and len(parts) >= 4:
                seen.add(parts[2])       # openimages:<split>:<id>:<box>
            elif parts[0] == "oi_labels" and len(parts) >= 2:
                seen.add(parts[1])       # oi_labels:<id>
    return seen


def read_csv(path: Path, classes: dict, min_conf: float | None,
             exclude: set[str] | None = None) -> dict[str, list[str]]:
    """class label -> [image_id], best confidence first.

    Images carrying more than one of our target labels are dropped: a photo of
    a hot dog next to a hamburger is a clean example of neither.
    """
    per_image: dict[str, dict[str, float]] = defaultdict(dict)
    if not path.exists():
        print(f"  ! missing {path}")
        return {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            image_id, _, label, conf = row[0], row[1], row[2], row[3]
            if label not in classes:
                continue
            if exclude and image_id in exclude:
                continue
            try:
                c = float(conf)
            except ValueError:
                c = 1.0
            if min_conf is not None and c < min_conf:
                continue
            per_image[image_id][label] = c

    ranked: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for image_id, labels in per_image.items():
        if len(labels) != 1:
            continue  # ambiguous, skip
        label, c = next(iter(labels.items()))
        ranked[label].append((c, image_id))

    return {lab: [i for _, i in sorted(v, reverse=True)] for lab, v in ranked.items()}


def build(csv_path: Path, classes: dict, cap_pos: int, cap_neg: int,
          min_conf: float | None, subdir: str, manifest: str,
          fixed_split: str | None, exclude: set[str] | None = None) -> None:
    groups = read_csv(csv_path, classes, min_conf, exclude)
    if not groups:
        print(f"nothing to do for {manifest}")
        return

    rows: list[dict] = []
    stats: list[tuple[str, int, int, int]] = []

    for label, ids in sorted(groups.items()):
        name, is_pos = classes[label]
        cap = cap_pos if is_pos else cap_neg

        # Only ~1.9M of Open Images' 9M photos live on the CVDF mirror (the
        # boxable + segmentation subset), but machine labels cover all 9M, so
        # roughly three in four candidate IDs are simply not fetchable. Walk
        # down the confidence-ranked list until the cap is met rather than
        # truncating to `cap` first and losing 75% of it.
        kept = 0
        tried = 0
        batch = 0
        while kept < cap and batch * WORKERS * 8 < len(ids):
            chunk = ids[batch * WORKERS * 8: (batch + 1) * WORKERS * 8]
            batch += 1
            if not chunk:
                break
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futs = {}
                for image_id in chunk:
                    rel = Path(subdir) / name / f"{image_id}.jpg"
                    futs[pool.submit(fetch, image_id, rel)] = (image_id, rel)
                for fut in as_completed(futs):
                    image_id, rel = futs[fut]
                    tried += 1
                    if not fut.result() or kept >= cap:
                        continue
                    kept += 1
                    sid = f"{manifest}:{image_id}"
                    rows.append({
                        "source_id": sid,
                        "path": str(rel),
                        "label": is_pos,
                        "split": fixed_split or split_for(sid),
                        "source": manifest,
                        "class_name": name,
                        "license": LICENSE,
                        "url": MIRROR.format(image_id=image_id),
                    })
        stats.append((name, kept, tried, len(ids)))
        flag = "" if kept >= cap else "  <- exhausted the class"
        print(f"  {name:20s} kept={kept:3d}  tried={tried:4d}  "
              f"candidates={len(ids):4d}{flag}")

    p = write_manifest(manifest, rows)
    n_pos = sum(r["label"] for r in rows)
    print(f"wrote {p} — {len(rows)} rows ({n_pos} pos / {len(rows) - n_pos} neg)")
    thin = [n for n, k, _, _ in stats if k < 20]
    if thin:
        print(f"!! thin classes (n<20), report n alongside accuracy: {', '.join(thin)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "hard", "both"], default="both")
    args = ap.parse_args()

    if args.mode in ("train", "both"):
        print("== training labels (human-verified) ==")
        build(HUMAN_CSV, TRAIN_CLASSES, cap_pos=10**9, cap_neg=CAP_TRAIN_NEG,
              min_conf=None, subdir="oi_labels", manifest="oi_labels",
              fixed_split=None)

    if args.mode in ("hard", "both"):
        exclude = training_image_ids()
        print(f"\n== adversarial set (machine labels, held out by class) ==")
        print(f"  excluding {len(exclude)} image IDs already used in training")
        build(MACHINE_CSV, HARD_CLASSES, cap_pos=CAP_HARD, cap_neg=CAP_HARD,
              min_conf=MACHINE_MIN_CONF, subdir="hard", manifest="hard",
              fixed_split="hard", exclude=exclude)


if __name__ == "__main__":
    main()
