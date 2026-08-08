"""Non-food negatives from Imagenette (10-class ImageNet subset).

Without these the model learns "is this a photo of food" rather than "is this a
hot dog" — every negative would be a plated dish. Imagenette contributes
churches, gas pumps, parachutes and (usefully) English springers, so the model
sees dogs of the four-legged kind too.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common import IMAGES, N_NONFOOD_NEG, split_for, write_manifest  # noqa: E402

MAX_SIDE = 256
LICENSE = "imagenette-research (ImageNet subset, fastai)"
URL = "https://huggingface.co/datasets/johnowhitaker/imagenette2-320"

WNID = {
    "n01440764": "tench",
    "n02102040": "english_springer",
    "n02979186": "cassette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "french_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute",
}


def save(img: Image.Image, path: Path) -> None:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=92)


def main() -> None:
    rng = random.Random(4242)
    ds = load_dataset("johnowhitaker/imagenette2-320", split="train")
    names = ds.features["label"].names

    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[:N_NONFOOD_NEG]
    print(f"imagenette: {len(idx)} negatives selected of {len(ds)}")

    rows: list[dict] = []
    for i in tqdm(idx, desc="imagenette"):
        sid = f"imagenette:{i}"
        rel = Path("imagenette") / f"{i}.jpg"
        out = IMAGES / rel
        if not out.exists():
            save(ds[i]["image"], out)
        wnid = names[ds[i]["label"]]
        rows.append(
            {
                "source_id": sid,
                "path": str(rel),
                "label": 0,
                "split": split_for(sid),
                "source": "imagenette",
                "class_name": WNID.get(wnid, wnid),
                "license": LICENSE,
                "url": URL,
            }
        )

    p = write_manifest("imagenette", rows)
    print(f"wrote {p} — {len(rows)} rows (0 pos / {len(rows)} neg)")


if __name__ == "__main__":
    main()
