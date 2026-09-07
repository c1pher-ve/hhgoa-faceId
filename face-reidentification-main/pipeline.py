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
import io
import json
import logging
import os
import sys

# Force UTF-8 output on Windows to avoid cp1252 emoji errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path

import cv2
import numpy as np

from downloader import filter_social_media_urls
from models import SCRFD
from searcher import collect_all_results
from utils.logging import setup_logging
from blockchain import SearchRecord, write_record, hash_image, NETWORKS, DEFAULT_NETWORK
from face_verifier import build_query_embedding, filter_by_face_presence
from models import SCRFD, ArcFace

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
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show the browser window while searching (useful for debugging).",
    )
    parser.add_argument(
        "--blockchain",
        action="store_true",
        help="Write the match result to the blockchain as a tamper-evident record.",
    )
    parser.add_argument(
        "--network",
        type=str,
        default=DEFAULT_NETWORK,
        choices=list(NETWORKS.keys()),
        help="Blockchain network to write the record to.",
    )
    parser.add_argument(
        "--rec-weight",
        type=str,
        default="./weights/w600k_r50.onnx",
        help="Path to ArcFace recognition ONNX model (used for identity comparison).",
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
    print(f"  RESULTS  --  {len(results)} social-media link(s) found")
    print("=" * 70)

    for platform, urls in grouped.items():
        print(f"\n[{platform}] ({len(urls)} link(s)):")
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

    # ── Step 1b: Load ArcFace recognizer + compute query embedding ────────────
    recognizer = None
    query_embedding = None
    rec_weight = getattr(args, "rec_weight", "./weights/w600k_r50.onnx")
    if os.path.exists(rec_weight):
        try:
            recognizer = ArcFace(rec_weight)
            logger.info("ArcFace recognizer loaded — identity comparison enabled")
            query_embedding = build_query_embedding(face_crop, detector, recognizer)
        except Exception as e:
            logger.warning(f"ArcFace load failed ({e}) — running without identity comparison")
    else:
        logger.warning(
            f"ArcFace weights not found at {rec_weight} — running face-detection filter only.\n"
            "  To enable identity comparison, download weights:\n"
            "  python -c \"import requests; open('weights/w600k_r50.onnx','wb').write("
            "requests.get('https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx').content)\""
        )

    # ── Step 2: Reverse image search using the face crop ─────────────────────
    headless = not args.show_browser
    if not headless:
        logger.info("Running in VISIBLE browser mode (--show-browser)")
    all_urls = collect_all_results(face_crop, headless=headless)

    if not all_urls:
        logger.warning("No URLs returned by any search engine.")
        _print_results([])
        return []

    # ── Step 3: Filter to social-media URLs ──────────────────────────────────
    social_links = filter_social_media_urls(all_urls)

    # ── Step 3b: Discard results with no human face / wrong identity ────────
    if social_links:
        logger.info("Running face-presence + identity verification on result pages…")
        social_links = filter_by_face_presence(
            social_links,
            detector,
            recognizer=recognizer,
            query_embedding=query_embedding,
        )

    # ── Step 4: Output ──────────────────────────────────────────────────
    _print_results(social_links)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(social_links, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {out_path}")

    # ── Step 5: Write to blockchain (optional) ──────────────────────────────
    if args.blockchain:
        logger.info("Preparing blockchain record…")

        # Hash the face crop image for the record
        _, img_encoded = cv2.imencode(".jpg", face_crop)
        img_hash = hash_image(img_encoded.tobytes())

        record = SearchRecord(
            image_hash=img_hash,
            timestamp_utc=int(__import__("time").time()),
            urls_found=[r["url"] for r in social_links],
            engines_used=["Yandex", "Bing", "TinEye"],
            total_raw_urls=len(all_urls),
            network=args.network,
        )

        logger.info(f"Record hash: {record.record_hash}")
        receipt = write_record(record, network=args.network)

        if receipt.success:
            print("\n" + "=" * 70)
            print("  BLOCKCHAIN RECORD WRITTEN")
            print("=" * 70)
            print(f"  TX Hash    : {receipt.tx_hash}")
            print(f"  Block      : #{receipt.block_number}")
            print(f"  Network    : {args.network}")
            print(f"  Explorer   : {receipt.explorer_url}")
            print(f"  Record Hash: {record.record_hash}")
            print("=" * 70 + "\n")

            # Append blockchain receipt to output file if requested
            if args.output:
                receipt_path = Path(args.output).with_suffix(".receipt.json")
                with open(receipt_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "tx_hash": receipt.tx_hash,
                        "block_number": receipt.block_number,
                        "block_timestamp": receipt.block_timestamp,
                        "network": args.network,
                        "explorer_url": receipt.explorer_url,
                        "gas_used": receipt.gas_used,
                        "record_hash": record.record_hash,
                        "image_hash": img_hash,
                    }, f, indent=2)
                logger.info(f"Blockchain receipt saved to {receipt_path}")
        else:
            logger.error(f"Blockchain write failed: {receipt.error}")

    return social_links


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline(parse_args())
