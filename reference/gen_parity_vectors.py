"""Emit (input, expected int32 logit) pairs for the JS<->Python parity test.

Two sources, and both matter.

    --source val      every held-out validation image, preprocessed exactly as
                      the browser preprocesses it. This is the real claim: the
                      two engines agree on the data the model was measured on.

    --source random   uniformly random pixels. Runs in CI without the dataset,
                      and is in one way a harder test — natural images leave
                      most of each activation's range unvisited, while random
                      input pushes accumulators toward saturation and exercises
                      requantization paths real photographs never reach.

Written as a flat binary: an int32 count, then per record 96*96*3 uint8 pixels
followed by the int32 logit.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from reference.int8_reference import Int8Model

# Pillow and tqdm are only needed to read real images. CI runs --source random
# and installs nothing but numpy, so these are imported at point of use.

OUT = Path(__file__).resolve().parent.parent / "engine" / "test" / "vectors"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["val", "test", "random"], default="random")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    model = Int8Model()
    size = model.size
    OUT.mkdir(parents=True, exist_ok=True)

    inputs: list[np.ndarray] = []
    if args.source == "random":
        rng = np.random.default_rng(4242)
        for _ in range(args.n):
            inputs.append(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
        progress = list
    else:
        from PIL import Image
        from tqdm import tqdm

        from common import IMAGES, all_rows
        from reference.preprocess import crop_resize

        progress = tqdm
        rows = [r for r in all_rows() if r["split"] == args.source]
        if args.n > 0:
            rows = rows[: args.n]
        for r in tqdm(rows, desc="preprocess"):
            img = np.asarray(Image.open(IMAGES / r["path"]).convert("RGB"), dtype=np.uint8)
            inputs.append(crop_resize(img, size))

    buf = bytearray(struct.pack("<i", len(inputs)))
    for x in progress(inputs):
        logit = model.forward(x)
        buf += x.tobytes()
        buf += struct.pack("<i", logit)

    path = OUT / "parity.bin"
    path.write_bytes(buf)
    print(f"wrote {path} — {len(inputs)} records, {len(buf) / 1e6:.1f} MB "
          f"(source={args.source})")


if __name__ == "__main__":
    main()
