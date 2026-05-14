"""
Assignment 1 — Image Search: Colour Histograms vs. VGG16 Embeddings

Visual Analytics, Spring 2026 — Cultural Data Science

This script:
  1. Loads a target flower image
  2. Part A: extracts colour histograms and ranks images by Chi-Squared distance
  3. Part B: extracts VGG16 embeddings and ranks images by cosine distance
  4. Saves both result CSVs to out/
  5. Saves a side-by-side comparison plot to out/comparison.png

Run from the assignment root:
    python src/main.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import argparse


SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

OUT_DIR    = os.path.join(PROJECT_ROOT, "out")
TOP_N      = 5

os.makedirs(OUT_DIR, exist_ok=True)

#arguments

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image search using colour histograms (Part A) and VGG16 embeddings (Part B)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "in", "flowers"),
        help="Path to the folder containing the flower images. "
             "Defaults to 'in/flowers/' in the project root.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="image_0484.jpg",
        help="Filename of the target image (must live inside --data-dir). "
             "Default: image_0484.jpg",
    )
    return parser.parse_args()

# ----------------------- Part A: histograms -----------------------

def get_histogram(image_path: str) -> np.ndarray | None:
    """Load an image and return its normalised 3-channel BGR colour histogram."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARNING] Could not read: {image_path}")
        return None
    hist = cv2.calcHist(
        [img],
        [0, 1, 2],          # BGR
        None,               # no mask
        [8, 8, 8],          # 8 bins per channel
        [0, 256, 0, 256, 0, 256],
    )
    return cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)


def run_histogram_search(target_path: str, all_paths: list[str]) -> pd.DataFrame:
    target_hist = get_histogram(target_path)
    if target_hist is None:
        raise FileNotFoundError(f"Could not load target image: {target_path}")

    results = []
    for path in all_paths:
        hist = get_histogram(path)
        if hist is None:
            continue
        d = round(cv2.compareHist(target_hist, hist, cv2.HISTCMP_CHISQR), 4)
        results.append({"filename": os.path.basename(path), "distance": d})

    return (pd.DataFrame(results)
              .sort_values("distance")
              .head(TOP_N)
              .reset_index(drop=True))


# --------------------- Part B: VGG16 embeddings -------------------

def build_vgg_model():
    """Load VGG16 with ImageNet weights, no classifier head, global avg pooling."""
    return VGG16(
        weights="imagenet",
        include_top=False,
        pooling="avg",          
        input_shape=(224, 224, 3),
    )


def get_embedding(image_path: str, model) -> np.ndarray | None:
    """Load an image and return its L2-normalised VGG16 embedding (512-d)."""
    try:
        img = load_img(image_path, target_size=(224, 224))
    except Exception as e:
        print(f"  [WARNING] Could not read: {image_path} ({e})")
        return None

    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)           

    feat = model.predict(arr, verbose=0).flatten()
    return feat / np.linalg.norm(feat)


def run_vgg_search(target_path: str, all_paths: list[str], model) -> pd.DataFrame:
    target_emb = get_embedding(target_path, model)
    if target_emb is None:
        raise FileNotFoundError(f"Could not load target image: {target_path}")

    results = []
    for i, path in enumerate(all_paths, start=1):
        emb = get_embedding(path, model)
        if emb is None:
            continue
        sim = cosine_similarity(target_emb.reshape(1, -1), emb.reshape(1, -1))[0][0]
        # Cosine similarity is in [-1, 1]; turn into a distance in [0, 2]
        d = round(float(1 - sim), 4)
        results.append({"filename": os.path.basename(path), "distance": d})
        if i % 100 == 0:
            print(f"  processed {i}/{len(all_paths)}")

    return (pd.DataFrame(results)
              .sort_values("distance")
              .head(TOP_N)
              .reset_index(drop=True))


# extra visualisation

def _style_image_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _read_rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def plot_comparison(target_path: str,
                    hist_df: pd.DataFrame,
                    vgg_df:  pd.DataFrame,
                    save_path: str,
                    data_dir: str) -> None:

    fig, axes = plt.subplots(3, TOP_N, figsize=(15, 9))

    # row 0 — target image, centred
    target_img = _read_rgb(target_path)
    for j in range(TOP_N):
        _style_image_axis(axes[0, j])
        if j == TOP_N // 2:
            axes[0, j].imshow(target_img)
            axes[0, j].set_title(f"TARGET\n{os.path.basename(target_path)}",
                                  fontsize=11, fontweight="bold")

    # rows 1 & 2 — top-N for each method
    method_rows = [
        (1, hist_df, "Part A:\nHistogram\n+ χ²"),
        (2, vgg_df,  "Part B:\nVGG16\n+ Cosine"),
    ]
    for row_idx, df, label in method_rows:
        for j, row in df.iterrows():
            img = _read_rgb(os.path.join(data_dir, row["filename"]))
            axes[row_idx, j].imshow(img)
            axes[row_idx, j].set_title(
                f"#{j + 1}   d = {row['distance']}\n{row['filename']}",
                fontsize=9,
            )
            _style_image_axis(axes[row_idx, j])
        # method label as a y-label on the leftmost subplot of the row
        axes[row_idx, 0].set_ylabel(
            label, fontsize=11, fontweight="bold",
            rotation=0, ha="right", va="center", labelpad=30,
        )

    fig.suptitle("Image Search Comparison: Colour Histograms vs. VGG16 Embeddings",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0.04, 0, 1, 0.97))
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved to: {save_path}")


# main function

def main() -> None:
    args = parse_args()
    data_dir    = args.data_dir
    target_img  = args.target
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_path = os.path.join(data_dir, target_img)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Pass --data-dir to point at your flower images, "
            "or place them at 'in/flowers/' in the project root."
        )
    if not os.path.isfile(target_path):
        raise FileNotFoundError(
            f"Target image not found: {target_path}\n"
            "Pass --target with a filename that exists inside --data-dir."
        )

    all_paths = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and f != target_img
    )
    print(f"Target image     : {target_path}")
    print(f"Comparing against: {len(all_paths)} other images\n")

    # Part A 
    print("Part A — Colour histograms + Chi-Squared distance")
    hist_df = run_histogram_search(target_path, all_paths)
    hist_df_out = hist_df.copy()
    hist_df_out.index = hist_df_out.index + 1
    print(hist_df_out.to_string())
    hist_csv = os.path.join(OUT_DIR, f"similar_images_histogram_{timestamp}.csv")
    hist_df_out.to_csv(hist_csv, index_label="rank")
    print(f"Saved: {hist_csv}\n")

    # B

    print("Part B — VGG16 embeddings + Cosine distance")
    print("Loading VGG16 (ImageNet weights) …")
    model = build_vgg_model()
    vgg_df = run_vgg_search(target_path, all_paths, model)
    vgg_df_out = vgg_df.copy()
    vgg_df_out.index = vgg_df_out.index + 1
    print(vgg_df_out.to_string())
    vgg_csv = os.path.join(OUT_DIR, f"similar_images_vgg16_{timestamp}.csv")
    vgg_df_out.to_csv(vgg_csv, index_label="rank")
    print(f"Saved: {vgg_csv}\n")

    # visualisation 
    print("Generating comparison plot …")
    plot_path = os.path.join(OUT_DIR, f"comparison_{timestamp}.png")
    plot_comparison(target_path, hist_df, vgg_df, plot_path, data_dir)

if __name__ == "__main__":
    main()