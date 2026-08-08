"""Pick the six tap-to-classify thumbnails and write them as WebP.

All six come from Open Images, which is CC BY 2.0, so they can actually be
redistributed in the repo — Food-101's licence is listed as unknown and
ImageNet-derived images are murkier, so neither is used for anything that ships.

The set is chosen to demonstrate the model's real decision boundary rather than
to flatter it: a hot dog, then four things that are nearly a hot dog, then a
dog that is not a food.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from common import IMAGES, ROOT, read_manifest

OUT = ROOT / "web" / "samples"
SIZE = 200

# filename -> (manifest, class_name)
WANT = [
    ("hotdog", "openimages", "hot_dog"),
    ("corndog", "hard", "corn_dog"),
    ("bratwurst", "hard", "bratwurst"),
    ("sub", "oi_labels", "submarine_sandwich"),
    ("burger", "oi_labels", "hamburger"),
    ("dachshund", "hard", "dachshund"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    picked = []

    for fname, manifest, cls in WANT:
        rows = [r for r in read_manifest(manifest) if r.get("class_name") == cls]
        if not rows:
            print(f"  ! no rows for {cls} in {manifest}")
            continue
        # Deterministic pick so re-running does not silently swap the images.
        row = sorted(rows, key=lambda r: r["source_id"])[0]
        src = IMAGES / row["path"]
        img = Image.open(src).convert("RGB")
        side = min(img.size)
        img = img.crop((
            (img.width - side) // 2, (img.height - side) // 2,
            (img.width + side) // 2, (img.height + side) // 2,
        )).resize((SIZE, SIZE), Image.LANCZOS)
        dst = OUT / f"{fname}.webp"
        img.save(dst, "WEBP", quality=80, method=6)
        picked.append({
            "file": dst.name,
            "class": cls,
            "source_id": row["source_id"],
            "url": row["url"],
            "license": row["license"],
            "bytes": dst.stat().st_size,
        })
        print(f"  {fname:12s} {cls:20s} {dst.stat().st_size / 1024:5.1f} KB")

    (OUT / "sources.json").write_text(json.dumps(picked, indent=2))
    total = sum(p["bytes"] for p in picked)
    print(f"\n{len(picked)} samples, {total / 1024:.1f} KB total (lazy-loaded)")


if __name__ == "__main__":
    main()
