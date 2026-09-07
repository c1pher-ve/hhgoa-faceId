"""
face_verifier.py — Post-filter: Face presence + identity match
===============================================================
After the reverse-image-search returns social-media URLs, this module:

  Stage 1 — Fetch the page's representative image (og:image / twitter:image /
             first <img>) and run SCRFD to confirm a human face is present.
             Results with no face are dropped immediately.

  Stage 2 — Run ArcFace on the detected face to get a 512-d embedding, then
             compare it against the query face's embedding via cosine similarity.
             Results below the similarity threshold are dropped (different person).

If the ArcFace model is unavailable, Stage 2 is skipped and only Stage 1 runs.

Usage (called from pipeline.py):
    from face_verifier import build_query_embedding, filter_by_face_presence
    query_emb = build_query_embedding(face_crop, detector, recognizer)
    verified  = filter_by_face_presence(social_links, detector, recognizer, query_emb)
"""

import logging
import os
from typing import Optional

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Tuneable thresholds ───────────────────────────────────────────────────────

# SCRFD minimum confidence to consider a detected region a real face
_DETECT_MIN_CONF: float = 0.45

# ArcFace cosine similarity threshold — tune to trade recall vs. precision:
#   0.30 = loose  (more results, risk of false positives)
#   0.40 = balanced  (recommended default)
#   0.50 = strict (fewer results, high precision)
SIMILARITY_THRESHOLD: float = 0.35

# HTTP settings
_HTTP_TIMEOUT: int = 10
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Image fetching helpers ────────────────────────────────────────────────────

def _fetch_image_from_url(img_url: str) -> Optional[np.ndarray]:
    """Download an image URL and decode it as a BGR NumPy array. Returns None on failure."""
    try:
        resp = requests.get(img_url, timeout=_HTTP_TIMEOUT, headers=_HEADERS, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and "octet-stream" not in content_type:
            return None
        arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _get_representative_image_url(page_url: str) -> Optional[str]:
    """
    Fetch a page and extract the best candidate image URL.

    Priority:
      1. og:image  (most reliable — platforms set this to the post's main image)
      2. twitter:image
      3. First <img> that isn't an icon/logo
    """
    try:
        resp = requests.get(page_url, timeout=_HTTP_TIMEOUT, headers=_HEADERS)
        resp.raise_for_status()
    except Exception as e:
        logger.debug(f"Could not fetch page {page_url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    og = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
    if og and og.get("content"):
        return og["content"]

    tw = soup.find("meta", attrs={"name": "twitter:image"})
    if tw and tw.get("content"):
        return tw["content"]

    for img_tag in soup.find_all("img", src=True):
        src = img_tag["src"]
        if src.startswith("//"):
            src = "https:" + src
        if not src.startswith("http"):
            continue
        low = src.lower()
        if any(kw in low for kw in ("icon", "logo", "avatar", "sprite", "pixel", "badge")):
            continue
        return src

    return None


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _get_embedding(image: np.ndarray, detector, recognizer) -> Optional[np.ndarray]:
    """
    Detect the best face in `image`, align it, and return its ArcFace embedding.

    Returns None if no face is detected or recognizer is unavailable.
    """
    if recognizer is None:
        return None
    try:
        bboxes, kpss = detector.detect(image, max_num=1)
        if len(bboxes) == 0 or float(bboxes[0][4]) < _DETECT_MIN_CONF:
            return None
        if kpss is None or len(kpss) == 0:
            return None
        landmarks = kpss[0]  # shape (5, 2)
        embedding = recognizer.get_embedding(image, landmarks, normalized=True)
        return embedding
    except Exception as e:
        logger.debug(f"Embedding extraction failed: {e}")
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a = a.ravel().astype(np.float32)
    b = b.ravel().astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── Public API — query embedding ──────────────────────────────────────────────

def build_query_embedding(
    face_crop: np.ndarray,
    detector,
    recognizer,
) -> Optional[np.ndarray]:
    """
    Compute the ArcFace embedding for the query (input) face crop.

    Args:
        face_crop:  BGR crop of the query face (from pipeline).
        detector:   Loaded SCRFD instance.
        recognizer: Loaded ArcFace instance (or None if unavailable).

    Returns:
        Normalised 512-d embedding, or None if extraction fails.
    """
    if recognizer is None:
        logger.info("[verifier] ArcFace not available — identity comparison disabled")
        return None

    emb = _get_embedding(face_crop, detector, recognizer)
    if emb is None:
        logger.warning("[verifier] Could not extract embedding from query face — identity comparison disabled")
    else:
        logger.info(f"[verifier] Query face embedding computed (dim={emb.shape[0]})")
    return emb


# ── Public API — URL filter ───────────────────────────────────────────────────

def filter_by_face_presence(
    social_links: list[dict],
    detector,
    recognizer=None,
    query_embedding: Optional[np.ndarray] = None,
) -> list[dict]:
    """
    Two-stage filter on social-media result URLs:

      Stage 1: Does the page's image contain a human face?  (SCRFD)
      Stage 2: Does that face match the query face?          (ArcFace cosine similarity)

    Stage 2 only runs when both `recognizer` and `query_embedding` are provided.

    Args:
        social_links:     [{"url": ..., "platform": ...}, ...]
        detector:         Loaded SCRFD instance.
        recognizer:       Loaded ArcFace instance, or None to skip Stage 2.
        query_embedding:  Pre-computed query embedding from build_query_embedding().

    Returns:
        Filtered list in the same format, with an added "similarity" key when
        Stage 2 ran (float, cosine similarity to query face).
    """
    if not social_links:
        return social_links

    use_identity = (recognizer is not None and query_embedding is not None)
    mode = "face detection + identity match" if use_identity else "face detection only"
    logger.info(f"[verifier] Checking {len(social_links)} result(s) — mode: {mode}")

    verified: list[dict] = []
    dropped_no_face: list[str] = []
    dropped_diff_person: list[str] = []

    for item in social_links:
        url = item["url"]
        platform = item.get("platform", "?")

        try:
            # ── Fetch representative image ────────────────────────────────────
            img_url = _get_representative_image_url(url)
            if img_url is None:
                logger.info(f"[verifier] KEPT (no image to check) [{platform}] {url}")
                verified.append(item)
                continue

            img = _fetch_image_from_url(img_url)
            if img is None:
                logger.info(f"[verifier] KEPT (image undecodable) [{platform}] {url}")
                verified.append(item)
                continue

            # ── Stage 1: Face presence check ──────────────────────────────────
            bboxes, kpss = detector.detect(img, max_num=5)
            face_found = len(bboxes) > 0 and float(bboxes[0][4]) >= _DETECT_MIN_CONF

            if not face_found:
                dropped_no_face.append(url)
                logger.info(f"[verifier] DROPPED — no face   [{platform}] {url}")
                continue

            # ── Stage 2: Identity match (ArcFace) ────────────────────────────
            if use_identity and kpss is not None and len(kpss) > 0:
                result_embedding = recognizer.get_embedding(img, kpss[0], normalized=True)
                similarity = _cosine_similarity(query_embedding, result_embedding)

                if similarity >= SIMILARITY_THRESHOLD:
                    logger.info(
                        f"[verifier] KEPT   sim={similarity:.3f} [{platform}] {url}"
                    )
                    verified.append({**item, "similarity": round(similarity, 4)})
                else:
                    dropped_diff_person.append(url)
                    logger.info(
                        f"[verifier] DROPPED — different person sim={similarity:.3f} "
                        f"(threshold={SIMILARITY_THRESHOLD}) [{platform}] {url}"
                    )
            else:
                # Stage 2 unavailable — keep anything that has a face
                logger.info(f"[verifier] KEPT (face detected, no ID check) [{platform}] {url}")
                verified.append(item)

        except Exception as e:
            logger.warning(f"[verifier] Error checking {url}: {e} — keeping")
            verified.append(item)

    total_dropped = len(dropped_no_face) + len(dropped_diff_person)
    logger.info(
        f"[verifier] Done: kept {len(verified)}/{len(social_links)}, "
        f"dropped {len(dropped_no_face)} (no face) + "
        f"{len(dropped_diff_person)} (different person)"
    )
    return verified

