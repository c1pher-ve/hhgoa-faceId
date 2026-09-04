"""
pipeline.py — Face Reverse Image Search Pipeline
==================================================
Loads a photo → detects & crops the face (SCRFD) → uploads the cropped
face to Yandex, Bing, and TinEye → filters results to social-media links.

Architecture:
  Input photo
      ↓
  SCRFD — detect & crop face
      ↓
  Yandex + Bing + TinEye  (searcher.py)
      ↓
  Collect all result URLs
      ↓
  Keep social-media URLs   (downloader.py)
      ↓
  Return / print results

Usage:
  python pipeline.py --input path/to/photo.jpg [--output results.json]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from downloader import filter_social_media_urls
from models import SCRFD
from searcher import collect_all_results
from utils.logging import setup_logging

setup_logging(log_to_file=True)
logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reverse-image-search pipeline — finds social media links for a face photo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input photo (jpg/png). Can be a full photo — face will be auto-detected.",
    )
    parser.add_argument(
        "--det-weight",
        type=str,
        default="./weights/det_10g.onnx",
        help="Path to SCRFD detection ONNX model.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.5,
        help="SCRFD face detection confidence threshold.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.2,
        help="Fractional padding added around the cropped face (e.g. 0.2 = 20%%).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional path to write the JSON results file.",
    )
    return parser.parse_args()


# ── Face detection & crop ─────────────────────────────────────────────────────

def detect_and_crop_face(
    image: np.ndarray,
    detector: SCRFD,
    padding: float = 0.2,
) -> np.ndarray:
    """
    Detect the largest/most-confident face in the image and return a
    padded crop of it.

    Args:
        image:    BGR image array (the full input photo).
        detector: Loaded SCRFD model instance.
        padding:  Fractional padding around the bounding box (0.2 = 20%).

    Returns:
        Cropped BGR face image.

    Raises:
        ValueError: If no face is detected in the image.
    """
    bboxes, _ = detector.detect(image, max_num=1)

    if len(bboxes) == 0:
        raise ValueError(
            "No face detected in the input image. "
            "Please provide a photo with a clear, visible face."
        )

    # bboxes shape: (N, 5) → [x1, y1, x2, y2, score]
    x1, y1, x2, y2, score = bboxes[0]
    h, w = image.shape[:2]

    # Add padding around the detected face box
    face_w = x2 - x1
    face_h = y2 - y1
    pad_x = int(face_w * padding)
    pad_y = int(face_h * padding)

    x1 = max(0, int(x1) - pad_x)
    y1 = max(0, int(y1) - pad_y)
    x2 = min(w, int(x2) + pad_x)
    y2 = min(h, int(y2) + pad_y)

    cropped = image[y1:y2, x1:x2]
    logger.info(
        f"Face detected (confidence={score:.2f}) → "
        f"crop [{x1},{y1}] to [{x2},{y2}]  ({x2-x1}×{y2-y1}px)"
    )
    return cropped


# ── Output helper ─────────────────────────────────────────────────────────────

def _print_results(results: list[dict]) -> None:
    """Pretty-print grouped social-media results to stdout."""
    if not results:
        print("\n  No social-media links found.\n")
        return

    # Group by platform
    grouped: dict[str, list[str]] = {}
    for r in results:
        grouped.setdefault(r["platform"], []).append(r["url"])

    print("\n" + "=" * 70)
    print(f"  RESULTS  —  {len(results)} social-media link(s) found")
    print("=" * 70)

    for platform, urls in grouped.items():
        print(f"\n📌 {platform} ({len(urls)}):")
        for url in urls:
            print(f"   {url}")

    print("\n" + "=" * 70 + "\n")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> list[dict]:
    """
    Execute the pipeline:
      load image → detect & crop face → search → filter → return social links.

    Args:
        args: Parsed CLI arguments.

    Returns:
        List of dicts: [{"url": ..., "platform": ...}, ...]
    """
    # ── Step 0: Load image ────────────────────────────────────────────────────
    if not os.path.exists(args.input):
        logger.error(f"Input image not found: {args.input}")
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        logger.error(f"Could not decode image: {args.input}")
        sys.exit(1)

    logger.info(f"Loaded image: {args.input}  ({image.shape[1]}×{image.shape[0]}px)")

    # ── Step 1: Detect & crop the face ───────────────────────────────────────
    logger.info("Loading SCRFD face detector…")
    try:
        detector = SCRFD(args.det_weight, input_size=(640, 640), conf_thres=args.conf_thresh)
    except Exception as e:
        logger.error(f"Failed to load SCRFD model: {e}")
        sys.exit(1)

    try:
        face_crop = detect_and_crop_face(image, detector, padding=args.padding)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Face crop ready: {face_crop.shape[1]}×{face_crop.shape[0]}px")

    # ── Step 2: Reverse image search using the face crop ─────────────────────
    all_urls = collect_all_results(face_crop)

    if not all_urls:
        logger.warning("No URLs returned by any search engine.")
        _print_results([])
        return []

    # ── Step 3: Filter to social-media URLs ──────────────────────────────────
    social_links = filter_social_media_urls(all_urls)

    # ── Step 4: Output ────────────────────────────────────────────────────────
    _print_results(social_links)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(social_links, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {out_path}")

    return social_links


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline(parse_args())
