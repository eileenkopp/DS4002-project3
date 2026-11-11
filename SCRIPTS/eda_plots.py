"""Compare brightness and original hand area; run from any directory.

Requires Pillow, NumPy, pandas, and Matplotlib. Reads the class folders written
by eda.py without downloading, preprocessing, or modifying any images.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.ticker import PercentFormatter

from eda import find_ann


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "DATA" / "rps_data"
OUTPUT_PATH = PROJECT_ROOT / "OUTPUT" / "brightness_distributions.png"
GESTURE_MAP = {"fist": "rock", "palm": "paper", "peace": "scissors"}
AREA_OUTPUT_PATH = OUTPUT_PATH.with_name("relative_hand_area_distributions.png")


def load_brightness():
    """Measure every processed JPG, retaining filenames and gesture labels."""
    rows = []
    for gesture in GESTURE_MAP.values():
        paths = sorted((DATA_DIR / gesture).glob("*.jpg"))
        if not paths:
            raise FileNotFoundError(
                f"No processed JPG images found in {DATA_DIR / gesture}. "
                "The processed hand crops produced by SCRIPTS/eda.py are required."
            )
        for path in paths:
            with Image.open(path) as image:
                if image.size != (224, 224):
                    raise ValueError(f"Expected a 224 x 224 processed crop: {path}")
                brightness = np.asarray(image.convert("L"), dtype=np.float64).mean()
            rows.append({"filename": path.name, "gesture": gesture,
                         "mean_brightness": float(brightness)})
    return pd.DataFrame(rows)


def plot_brightness(data):
    """Use common bins and per-class densities to accommodate unequal counts."""
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    bins = np.linspace(0, 255, 31)
    colors = ("#0072B2", "#D55E00", "#009E73")
    for gesture, color in zip(GESTURE_MAP.values(), colors):
        values = data.loc[data["gesture"] == gesture, "mean_brightness"]
        ax.hist(values, bins=bins, density=True, alpha=0.40,
                color=color, label=f"{gesture.title()} (n={len(values):,})",
                edgecolor="white", linewidth=0.4)
    ax.set(title="Brightness distributions of processed hand images",
           xlabel="Mean grayscale pixel brightness (0 = black, 255 = white)",
           ylabel="Probability density", xlim=(0, 255))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Gesture", frameon=False)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, facecolor="white")
    plt.close(fig)


def load_relative_area(images):
    """Match <image-id>_<box-index>.jpg to the original, unpadded XYWH box.

    eda.py interprets normalized boxes as x, y, width, height, multiplying
    by the original photograph dimensions. Thus width * height is the
    fraction of the photograph covered by the annotated box.
    """
    rows = []
    unmatched = []
    for source_gesture, gesture in GESTURE_MAP.items():
        try:
            annotation_path = find_ann(source_gesture)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"{error}. Restore the HaGRID annotations under DATA/ann "
                "(the existing DATA/annotations.zip may contain them). "
                "This analysis does not download or change dataset files."
            ) from error
        print(f"\n{gesture.title()} annotations: {annotation_path}")
        with open(annotation_path, encoding="utf-8") as source:
            annotations = json.load(source)
        filenames = images.loc[images["gesture"] == gesture, "filename"]
        matched = 0
        for filename in filenames:
            try:
                image_id, index = Path(filename).stem.rsplit("_", 1)
                index = int(index)
                if index < 0:
                    raise ValueError("negative bounding-box index")
                annotation = annotations[image_id]
                if annotation["labels"][index] != source_gesture:
                    raise ValueError("indexed gesture label does not match class")
                x, y, width, height = annotation["bboxes"][index]
                box = np.array([x, y, width, height], dtype=float)
                if (not np.isfinite(box).all() or (box < 0).any()
                        or (box > 1).any() or width <= 0 or height <= 0
                        or x + width > 1.000001 or y + height > 1.000001):
                    raise ValueError("invalid normalized XYWH bounding box")
                rows.append({"filename": filename, "gesture": gesture,
                             "relative_area": width * height})
                matched += 1
            except (KeyError, IndexError, TypeError, ValueError) as error:
                unmatched.append((gesture, filename, str(error)))
        print(f"Matched {matched}/{len(filenames)} processed images; "
              f"unmatched: {len(filenames) - matched}")
    for gesture, filename, reason in unmatched:
        print(f"Unmatched {gesture}/{filename}: {reason}")
    data = pd.DataFrame(rows, columns=["filename", "gesture", "relative_area"])
    missing = set(GESTURE_MAP.values()) - set(data["gesture"])
    if missing:
        raise ValueError(f"No valid bounding-box matches for: {sorted(missing)}")
    return data


def plot_relative_area(data):
    """Show original box-area fractions, including outliers, by class."""
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    values = [data.loc[data["gesture"] == gesture, "relative_area"]
              for gesture in GESTURE_MAP.values()]
    boxes = ax.boxplot(values, patch_artist=True, widths=0.5,
                       medianprops={"color": "black", "linewidth": 1.5},
                       flierprops={"marker": ".", "markersize": 3, "alpha": 0.35})
    for patch, color in zip(boxes["boxes"], ("#0072B2", "#D55E00", "#009E73")):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks([1, 2, 3],
                  [f"{gesture.title()}\n(n={len(group):,})"
                   for gesture, group in zip(GESTURE_MAP.values(), values)])
    ax.set(title="Original hand bounding-box area by gesture",
           xlabel="Gesture", ylabel="Bounding-box area / original photograph area")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_ylim(bottom=0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    AREA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(AREA_OUTPUT_PATH, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    data = load_brightness()
    summary = (data.groupby("gesture")["mean_brightness"]
               .agg(["count", "mean", "median", "std", "min", "max"])
               .reindex(list(GESTURE_MAP.values())))
    print("Mean grayscale brightness by class (0–255; sample standard deviation):")
    print(summary.to_string(float_format=lambda value: f"{value:.2f}"))
    plot_brightness(data)
    print(f"\nSaved figure: {OUTPUT_PATH}")
    area = load_relative_area(data)
    summary = (area.groupby("gesture")["relative_area"]
               .agg(["count", "mean", "median", "std", "min", "max"])
               .reindex(list(GESTURE_MAP.values())))
    print("\nRelative bounding-box area (fraction; sample standard deviation):")
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))
    plot_relative_area(area)
    print(f"\nSaved figure: {AREA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
