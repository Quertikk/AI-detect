"""
Downloads the "140k Real and Fake Faces" Kaggle dataset and builds a
subsampled train/valid/test split under data/faces/{split}/{real,fake}.

Requires Kaggle API credentials (~/.kaggle/kaggle.json or the
KAGGLE_USERNAME / KAGGLE_KEY environment variables) — see
https://github.com/Kaggle/kaggle-api#api-credentials

Usage:
    python training/dataset_prep.py --per-class 5000
"""
import argparse
import random
import shutil
from pathlib import Path

import kagglehub

DATASET_HANDLE = "xhlulu/140k-real-and-fake-faces"
SPLITS = ("train", "valid", "test")
CLASSES = ("real", "fake")


def find_split_dir(root: Path, split: str) -> Path:
    """The Kaggle archive nests folders inconsistently across mirrors,
    so search for a directory named `split` that contains real/fake subdirs."""
    candidates = [p for p in root.rglob(split) if p.is_dir()]
    for c in candidates:
        if (c / "real").is_dir() and (c / "fake").is_dir():
            return c
    raise FileNotFoundError(f"Could not locate a '{split}' folder with real/fake subdirs under {root}")


def build_subset(src_root: Path, dst_root: Path, per_class: int, seed: int) -> None:
    rng = random.Random(seed)
    for split in SPLITS:
        split_dir = find_split_dir(src_root, split)
        for cls in CLASSES:
            src_dir = split_dir / cls
            images = sorted(p for p in src_dir.iterdir() if p.is_file())
            rng.shuffle(images)
            take = per_class if split == "train" else max(1, per_class // 5)
            chosen = images[:take]

            dst_dir = dst_root / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for img in chosen:
                shutil.copy2(img, dst_dir / img.name)
            print(f"{split}/{cls}: copied {len(chosen)} of {len(images)} available images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=5000,
                         help="Number of training images per class (valid/test use 1/5 of this)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/faces")
    args = parser.parse_args()

    print(f"Downloading {DATASET_HANDLE} via kagglehub (cached after first run)...")
    dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    print(f"Dataset cached at: {dataset_path}")

    out_root = Path(args.out)
    build_subset(dataset_path, out_root, args.per_class, args.seed)
    print(f"Done. Subset written to {out_root.resolve()}")


if __name__ == "__main__":
    main()
