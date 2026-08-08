"""Shared paths, constants and split logic. Imported by data/, train/ and reference/."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"
IMAGES = CACHE / "images"
MANIFESTS = ROOT / "data" / "manifests"
HARD = ROOT / "data" / "hard"
RESULTS = ROOT / "train" / "results"
CKPT = ROOT / "train" / "checkpoints"
DIST = ROOT / "dist"
REFDIR = ROOT / "reference"

for _d in (CACHE, IMAGES, MANIFESTS, HARD, RESULTS, CKPT, DIST):
    _d.mkdir(parents=True, exist_ok=True)

# ── model geometry ───────────────────────────────────────────────────────────
INPUT_SIZE = 96
TEACHER_SIZE = 224

# ── negative sampling policy ─────────────────────────────────────────────────
# Food-101 classes most confusable with a hot dog. Sampled at CONFUSABLE_BOOST x
# the rate of the other 90 classes, because that is where the decision boundary
# actually lives.
CONFUSABLE = [
    "hamburger",
    "lobster_roll_sandwich",
    "club_sandwich",
    "pulled_pork_sandwich",
    "grilled_cheese_sandwich",
    "french_fries",
    "tacos",
    "spring_rolls",
    "garlic_bread",
    "breakfast_burrito",
]
CONFUSABLE_BOOST = 3

N_FOOD_NEG = 12000
N_NONFOOD_NEG = 8000

# ── deterministic splits ─────────────────────────────────────────────────────
# Bucketed on a hash of the *group* key, so near-duplicate crops from one source
# photo always land in the same split. Rerunning ingest never reshuffles.
SPLIT_BOUNDS = (70, 85)  # train < 70 <= val < 85 <= test


def split_for(group_key: str) -> str:
    h = hashlib.sha256(group_key.encode()).hexdigest()
    bucket = int(h[:8], 16) % 100
    if bucket < SPLIT_BOUNDS[0]:
        return "train"
    if bucket < SPLIT_BOUNDS[1]:
        return "val"
    return "test"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(name: str, rows: list[dict]) -> Path:
    path = MANIFESTS / f"{name}.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return path


def read_manifest(name: str) -> list[dict]:
    path = MANIFESTS / f"{name}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# Manifests that make up train/val/test. `hard` is deliberately absent: its
# classes are held out entirely so the adversarial eval measures generalization
# rather than recall of negatives the model was trained on.
TRAIN_MANIFESTS = ("food101", "openimages", "imagenette", "oi_labels")


def all_rows() -> list[dict]:
    rows: list[dict] = []
    for name in TRAIN_MANIFESTS:
        rows.extend(read_manifest(name))
    return rows
