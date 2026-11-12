"""Clean, check, and split the rock/paper/scissors hand crops made by eda.py.

Run from any directory after eda.py has created DATA/rps_data:

    python SCRIPTS/preprocess.py

Steps
1. Combine rock_meta.csv, paper_meta.csv and scissors_meta.csv into one table
   and replace each person's absolute file path with a path relative to the
   project folder, so the data works on every group member's computer.
2. Check every image: it exists, opens, is 224 x 224 RGB, and is not an exact
   duplicate of another image. Problem images are removed and listed.
3. Record two quality measures per image (brightness and sharpness). These
   are kept as columns for analysis only; no images are removed for them.
4. Split the data into train / validation / test by user_id, so that every
   person appears in exactly one split. This is what makes test accuracy a
   measure of performance on new players.

Outputs
    DATA/rps_manifest.csv      one row per image, with a "split" column
    OUTPUT/split_summary.csv   image and person counts per split and gesture

Pixel scaling and data augmentation are not done here. They happen inside the
model's data loader, so the saved images stay unchanged.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "DATA" / "rps_data"
MANIFEST_PATH = PROJECT_ROOT / "DATA" / "rps_manifest.csv"
SUMMARY_PATH = PROJECT_ROOT / "OUTPUT" / "split_summary.csv"
LABELS = ["rock", "paper", "scissors"]
HAGRID_GESTURE = {"rock": "fist", "paper": "palm", "scissors": "peace"}
TEST_SHARE, VAL_SHARE = 0.15, 0.15
SEED = 42


def load_metadata():
    """Combine the three metadata files and rebuild portable relative paths."""
    frames = []
    for label in LABELS:
        meta = pd.read_csv(DATA_DIR / f"{label}_meta.csv", dtype={"user_id": str})
        meta["label"] = label
        frames.append(meta)
    data = pd.concat(frames, ignore_index=True)
    filename = data["file"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
    data["path"] = "DATA/rps_data/" + data["label"] + "/" + filename
    stem = filename.str.removesuffix(".jpg")
    data["hagrid_image_id"] = stem.str.rsplit("_", n=1).str[0]
    data["hand_index"] = stem.str.rsplit("_", n=1).str[1].astype(int)
    data["hagrid_gesture"] = data["label"].map(HAGRID_GESTURE)
    return data.drop(columns="file")


def check_images(data):
    """Drop missing, unreadable, wrongly sized, or duplicated images."""
    problems, keep, hashes = [], [], {}
    brightness, sharpness = [], []
    for row in data.itertuples():
        path = PROJECT_ROOT / row.path
        reason = None
        if not path.exists():
            reason = "missing file"
        else:
            raw = path.read_bytes()
            digest = hashlib.md5(raw).hexdigest()
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.size != (224, 224) or image.mode != "RGB":
                        reason = f"unexpected size/mode {image.size} {image.mode}"
                    gray = np.asarray(image.convert("L"), dtype=float)
            except OSError:
                reason = "unreadable image"
            if reason is None and digest in hashes:
                reason = f"exact duplicate of {hashes[digest]}"
        if reason:
            problems.append((row.path, reason))
            keep.append(False)
            brightness.append(np.nan)
            sharpness.append(np.nan)
            continue
        hashes[digest] = row.path
        keep.append(True)
        brightness.append(gray.mean())
        # Variance of the Laplacian: low values mean a blurry image.
        laplacian = (4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1]
                     - gray[1:-1, :-2] - gray[1:-1, 2:])
        sharpness.append(laplacian.var())
    data = data.assign(mean_brightness=brightness, sharpness=sharpness)
    return data[keep].reset_index(drop=True), problems


def split_by_person(data):
    """Assign whole people to train, validation, or test."""
    groups = data["user_id"]
    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SHARE, random_state=SEED)
    rest_idx, test_idx = next(outer.split(data, groups=groups))
    rest = data.iloc[rest_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=VAL_SHARE / (1 - TEST_SHARE),
                              random_state=SEED)
    train_idx, val_idx = next(inner.split(rest, groups=rest["user_id"]))
    split = pd.Series("train", index=data.index)
    split.iloc[test_idx] = "test"
    split.loc[rest.index[val_idx]] = "val"
    data = data.assign(split=split)

    people = data.groupby("split")["user_id"].apply(set)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = people[a] & people[b]
        if overlap:
            raise RuntimeError(f"{len(overlap)} people appear in both {a} and {b}")
    return data


def summarize(data):
    images = data.pivot_table(index="split", columns="label", values="path",
                              aggfunc="count").reindex(["train", "val", "test"])[LABELS]
    images["total_images"] = images.sum(axis=1)
    images["share_of_images"] = (images["total_images"] / len(data)).round(3)
    images["people"] = data.groupby("split")["user_id"].nunique()
    return images


def main():
    data = load_metadata()
    print(f"Loaded {len(data):,} images from {data['user_id'].nunique():,} people")
    print("People appearing in more than one gesture class:",
          (data.groupby("user_id")["label"].nunique() > 1).sum())

    data, problems = check_images(data)
    print(f"Removed {len(problems)} problem images")
    for path, reason in problems:
        print(f"  {path}: {reason}")

    data = split_by_person(data)
    summary = summarize(data)
    print("\nImages per split (no person appears in more than one split):")
    print(summary.to_string())

    columns = ["path", "label", "split", "user_id", "hagrid_gesture",
               "hagrid_image_id", "hand_index", "mean_brightness", "sharpness"]
    data[columns].to_csv(MANIFEST_PATH, index=False, float_format="%.3f")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH)
    print(f"\nSaved {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
