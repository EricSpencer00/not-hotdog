"""Datasets and transforms.

One augmented crop feeds both networks. The teacher sees it at 224 with
ImageNet normalization (that is what its pretrained weights expect); the student
sees the same crop at 96 scaled to [0,1] with no mean/std shift, because the
student's input quantizer has to land on scale 1/255, zero-point 0 so that the
browser can hand raw canvas bytes straight to the first convolution.

Returning both views from one crop is what makes the distillation targets
meaningful — if the teacher and student saw different augmentations, the soft
labels would describe an image the student never sees.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from common import IMAGES, INPUT_SIZE, TEACHER_SIZE, all_rows, read_manifest

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def train_transform() -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(TEACHER_SIZE, scale=(0.4, 1.0), ratio=(0.75, 1.333)),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=7),
            T.ToTensor(),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    )


def eval_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize(int(TEACHER_SIZE * 1.14)),
            T.CenterCrop(TEACHER_SIZE),
            T.ToTensor(),
        ]
    )


class HotDogDataset(Dataset):
    """Yields (teacher_view, student_view, label)."""

    def __init__(self, rows: list[dict], train: bool):
        self.rows = rows
        self.tf = train_transform() if train else eval_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = Image.open(IMAGES / r["path"]).convert("RGB")
        crop = self.tf(img)  # float [0,1], 3 x 224 x 224

        teacher = (crop - IMAGENET_MEAN) / IMAGENET_STD
        student = F.interpolate(
            crop.unsqueeze(0), size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear", align_corners=False, antialias=True,
        ).squeeze(0)
        return teacher, student, torch.tensor(float(r["label"]))


def load_split(split: str) -> list[dict]:
    return [r for r in all_rows() if r["split"] == split]


def load_hard() -> list[dict]:
    return read_manifest("hard")


def describe(rows: list[dict], name: str) -> str:
    pos = sum(r["label"] for r in rows)
    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    srcs = ", ".join(f"{k}={v}" for k, v in sorted(by_source.items()))
    return f"{name:6s} n={len(rows):6d}  pos={pos:5d}  neg={len(rows) - pos:6d}  [{srcs}]"


if __name__ == "__main__":
    for s in ("train", "val", "test"):
        print(describe(load_split(s), s))
    hard = load_hard()
    if hard:
        print(describe(hard, "hard"))
