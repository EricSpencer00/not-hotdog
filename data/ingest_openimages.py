"""Hot dog positives from Open Images V7, cropped from bounding boxes.

Food-101 gives us 1,000 hot dogs and that is the hard ceiling from clean food
datasets. Open Images class /m/01b9xk ("Hot dog") adds a few thousand more, and
crucially they are *in the wild* — held in hands, at ballparks, half-eaten —
rather than the centred restaurant-plate framing Food-101 is full of.

Images come from the CVDF S3 mirror (open-images-dataset.s3.amazonaws.com),
not the original Flickr URLs, so there is no link rot to tolerate.

Bounding boxes are padded by CONTEXT so the bun and the plate stay in frame; a
pixel-tight crop teaches the model to classify sausage texture alone.
"""

from __future__ import annotations

import csv
import io
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common import IMAGES, split_for, write_manifest  # noqa: E402

MID = "/m/01b9xk"
MIRROR = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
LICENSE = "CC BY 2.0 (Open Images V7; per-image attribution at https://storage.googleapis.com/openimages/web/download_v7.html)"
MAX_SIDE = 256
CONTEXT = 0.12  # fraction of box size added on each side
MIN_AREA = 0.02  # skip boxes under 2% of the image
WORKERS = 16
TIMEOUT = 30

CSVS = {
    "train": Path("/tmp/oi-train-hotdog.csv"),
    "validation": Path("/tmp/oi-valid-hotdog.csv"),
    "test": Path("/tmp/oi-test-hotdog.csv"),
}
COLS = [
    "ImageID", "Source", "LabelName", "Confidence",
    "XMin", "XMax", "YMin", "YMax",
    "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside",
]


def load_boxes() -> dict[tuple[str, str], list[dict]]:
    """(oi_split, image_id) -> list of box dicts."""
    boxes: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for oi_split, path in CSVS.items():
        if not path.exists():
            print(f"  ! missing {path}, skipping {oi_split}")
            continue
        with open(path) as f:
            for row in csv.DictReader(f, fieldnames=COLS):
                if row["LabelName"] != MID:
                    continue
                if row["IsGroupOf"] == "1":
                    continue  # crowd boxes are a pile of hot dogs, not a hot dog
                x0, x1 = float(row["XMin"]), float(row["XMax"])
                y0, y1 = float(row["YMin"]), float(row["YMax"])
                if (x1 - x0) * (y1 - y0) < MIN_AREA:
                    continue
                boxes[(oi_split, row["ImageID"])].append(
                    {"x0": x0, "x1": x1, "y0": y0, "y1": y1,
                     "depiction": row["IsDepiction"] == "1"}
                )
    return boxes


def fetch_and_crop(oi_split: str, image_id: str, bxs: list[dict]) -> list[dict]:
    url = MIRROR.format(split=oi_split, image_id=image_id)
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return []
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return []

    W, H = img.size
    out_rows: list[dict] = []
    for k, b in enumerate(bxs):
        bw, bh = (b["x1"] - b["x0"]) * W, (b["y1"] - b["y0"]) * H
        left = max(0, int(b["x0"] * W - bw * CONTEXT))
        right = min(W, int(b["x1"] * W + bw * CONTEXT))
        top = max(0, int(b["y0"] * H - bh * CONTEXT))
        bottom = min(H, int(b["y1"] * H + bh * CONTEXT))
        if right - left < 32 or bottom - top < 32:
            continue
        crop = img.crop((left, top, right, bottom))
        w, h = crop.size
        if max(w, h) > MAX_SIDE:
            s = MAX_SIDE / max(w, h)
            crop = crop.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)

        rel = Path("openimages") / oi_split / f"{image_id}_{k}.jpg"
        dst = IMAGES / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        crop.save(dst, "JPEG", quality=92)
        out_rows.append(
            {
                "source_id": f"openimages:{oi_split}:{image_id}:{k}",
                "path": str(rel),
                "label": 1,
                # group on ImageID: every crop of one photo stays in one split
                "split": split_for(f"openimages:{image_id}"),
                "source": "openimages",
                "class_name": "hot_dog",
                "license": LICENSE,
                "url": url,
                "depiction": b["depiction"],
            }
        )
    return out_rows


def main() -> None:
    boxes = load_boxes()
    print(f"openimages: {len(boxes)} source photos, {sum(len(v) for v in boxes.values())} boxes")

    rows: list[dict] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {
            pool.submit(fetch_and_crop, s, i, b): (s, i)
            for (s, i), b in boxes.items()
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="openimages"):
            got = fut.result()
            if not got:
                failed += 1
            rows.extend(got)

    rate = failed / max(1, len(boxes))
    p = write_manifest("openimages", rows)
    print(f"wrote {p} — {len(rows)} crops from {len(boxes) - failed} photos")
    print(f"fetch failure rate: {rate:.1%} ({failed}/{len(boxes)})")
    if rate > 0.30:
        print("!! failure rate above the 30% tolerance in SPEC.md §10.1")


if __name__ == "__main__":
    main()
