"""Download HaGRID annotations and save cropped rock/paper/scissors images locally."""

import argparse
from urllib.request import urlretrieve
from zipfile import ZipFile

from remotezip import RemoteZip
import io, os, json, random, glob
from pathlib import Path
from PIL import Image
import pandas as pd

GESTURE_MAP = {"fist": "rock", "palm": "paper", "peace": "scissors"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "DATA"
ANN_DIR = DATA_DIR / "ann"
OUT = DATA_DIR / "rps_data"
ANNOTATIONS_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
BASE = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset"

def prepare_annotations():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if all(glob.glob(str(ANN_DIR / "**" / f"{gesture}.json"), recursive=True)
           for gesture in GESTURE_MAP):
        return
    archive = DATA_DIR / "annotations.zip"
    if not archive.exists():
        print("Downloading annotations...", flush=True)
        temporary = archive.with_suffix(".zip.part")
        urlretrieve(ANNOTATIONS_URL, temporary)
        temporary.replace(archive)
    print("Extracting annotations...", flush=True)
    with ZipFile(archive) as zipped:
        zipped.extractall(ANN_DIR)

def find_ann(gesture):
    paths = sorted(glob.glob(str(ANN_DIR / "**" / f"{gesture}.json"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No {gesture}.json found under {ANN_DIR}")
    train = [p for p in paths if "train" in p]
    return (train or paths)[0]

def process(gesture, n_images=1500, pad=0.15):
    with open(find_ann(gesture), encoding="utf-8") as source:
        ann = json.load(source)
    label = GESTURE_MAP[gesture]
    (OUT / label).mkdir(parents=True, exist_ok=True)
    rows = []
    with RemoteZip(f"{BASE}/{gesture}.zip") as z:
        names = {Path(n).stem: n for n in z.namelist() if n.endswith(".jpg")}
        keys = [k for k in ann if k in names]
        random.seed(0); random.shuffle(keys)
        for k in keys[:n_images]:
            a = ann[k]
            img = Image.open(io.BytesIO(z.read(names[k]))).convert("RGB")
            W, H = img.size
            for i, (box, lab) in enumerate(zip(a["bboxes"], a["labels"])):
                if lab != gesture:
                    continue
                x, y, w, h = box
                x0, y0 = max(0, (x - pad*w)*W), max(0, (y - pad*h)*H)
                x1, y1 = min(W, (x + w + pad*w)*W), min(H, (y + h + pad*h)*H)
                dest = f"{OUT}/{label}/{k}_{i}.jpg"
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                img.crop((x0, y0, x1, y1)).resize((224, 224)).save(dest, quality=90)
                rows.append({"file": dest, "label": label, "user_id": a.get("user_id")})
    pd.DataFrame(rows, columns=["file", "label", "user_id"]).to_csv(f"{OUT}/{label}_meta.csv", index=False)
    print(gesture, "→", label, ":", len(rows), "hand images saved")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-images", type=int, default=1500,
                        help="Source images per gesture (default: 1500).")
    args = parser.parse_args()
    if args.n_images < 1:
        parser.error("--n-images must be at least 1")
    prepare_annotations()
    for gesture in GESTURE_MAP:
        process(gesture, n_images=args.n_images)


if __name__ == "__main__":
    main()
