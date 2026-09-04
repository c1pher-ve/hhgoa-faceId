"""
searcher.py — Reverse Image Search Module
=========================================
Sends a face photo to Yandex, Bing, and TinEye via reverse-image search
and collects all result page URLs.

Strategy:
  - Yandex : Multipart POST upload → parse JSON response for page URLs.
  - Bing   : Multipart POST upload → parse HTML for result links.
  - TinEye : Multipart POST upload → parse JSON API response.

All three functions return a flat list of raw result URLs (strings).
"""

import io
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── HTTP session shared across helpers ──────────────────────────────────────
_SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _image_to_bytes(image: np.ndarray, ext: str = ".jpg") -> bytes:
    """Encode an OpenCV BGR image to JPEG bytes."""
    success, buf = cv2.imencode(ext, image)
    if not success:
        raise ValueError("Failed to encode image to bytes.")
    return buf.tobytes()


def _build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_SESSION_HEADERS)
    return s


# ── Yandex ───────────────────────────────────────────────────────────────────

def search_yandex(image: np.ndarray, timeout: int = 15) -> list[str]:
    """
    Reverse-image search on Yandex Images.

    Uploads the image via the Yandex upload endpoint, follows the redirect
    to the CBIR (content-based image retrieval) result page, and scrapes
    all external 'source' URLs from the JSON block embedded in the page.

    Args:
        image: BGR image as a NumPy array.
        timeout: Request timeout in seconds.

    Returns:
        List of result page URLs from Yandex.
    """
    urls: list[str] = []
    session = _build_session()
    img_bytes = _image_to_bytes(image)

    try:
        upload_url = "https://yandex.com/images/search"
        files = {"upfile": ("face.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        params = {"rpt": "imageview", "format": "json", "request": '{"blocks":[{"block":"cbir-uploader__get-image"}]}'}

        resp = session.post(upload_url, params=params, files=files, timeout=timeout)
        resp.raise_for_status()

        # Yandex returns a JSON with an image upload URL we follow.
        data = resp.json()
        blocks = data.get("blocks", [])
        cbir_url = None
        for block in blocks:
            params_data = block.get("params", {})
            if "url" in params_data:
                cbir_url = params_data["url"]
                break

        if not cbir_url:
            logger.warning("[Yandex] Could not extract CBIR redirect URL from upload response.")
            return urls

        # Follow the CBIR result page
        cbir_url = f"https://yandex.com{cbir_url}" if cbir_url.startswith("/") else cbir_url
        page_resp = session.get(cbir_url, timeout=timeout)
        page_resp.raise_for_status()

        soup = BeautifulSoup(page_resp.text, "html.parser")

        # Extract links from image result cards
        for tag in soup.select("a.serp-item__link, a.CbirSites-ItemTitle, a[href]"):
            href = tag.get("href", "")
            if href.startswith("http") and "yandex" not in href:
                urls.append(href)

        logger.info(f"[Yandex] Found {len(urls)} result URLs.")
    except Exception as e:
        logger.error(f"[Yandex] Search failed: {e}")

    return urls


# ── Bing ─────────────────────────────────────────────────────────────────────

def search_bing(image: np.ndarray, timeout: int = 15) -> list[str]:
    """
    Reverse-image search on Bing Visual Search.

    Uploads the image to Bing's visual search endpoint and parses the
    HTML result page for linked URLs.

    Args:
        image: BGR image as a NumPy array.
        timeout: Request timeout in seconds.

    Returns:
        List of result page URLs from Bing.
    """
    urls: list[str] = []
    session = _build_session()
    img_bytes = _image_to_bytes(image)

    try:
        upload_url = "https://www.bing.com/images/search"
        files = {"imgurl": ("face.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        params = {"view": "detailv2", "iss": "sbiupload", "FORM": "SBIVSP"}

        resp = session.post(upload_url, params=params, files=files, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Bing result page: look for page-source links in result cards
        for tag in soup.select("a.iusc, a[m]"):
            href = tag.get("href", "")
            if href.startswith("http") and "bing.com" not in href and "microsoft.com" not in href:
                urls.append(href)

        # Also try structured JSON-like data Bing embeds in anchor 'm' attrs
        import json
        for tag in soup.select("a[m]"):
            try:
                m_data = json.loads(tag["m"])
                page_url = m_data.get("purl") or m_data.get("murl", "")
                if page_url.startswith("http"):
                    urls.append(page_url)
            except (json.JSONDecodeError, KeyError):
                continue

        urls = list(dict.fromkeys(urls))  # dedupe preserving order
        logger.info(f"[Bing] Found {len(urls)} result URLs.")
    except Exception as e:
        logger.error(f"[Bing] Search failed: {e}")

    return urls


# ── TinEye ───────────────────────────────────────────────────────────────────

def search_tineye(image: np.ndarray, timeout: int = 20) -> list[str]:
    """
    Reverse-image search on TinEye.

    Submits the image via TinEye's public upload API and parses the
    JSON response for image page URLs.

    Args:
        image: BGR image as a NumPy array.
        timeout: Request timeout in seconds.

    Returns:
        List of result page URLs from TinEye.
    """
    urls: list[str] = []
    session = _build_session()
    img_bytes = _image_to_bytes(image)

    try:
        upload_url = "https://tineye.com/search"
        files = {"image": ("face.jpg", io.BytesIO(img_bytes), "image/jpeg")}

        resp = session.post(upload_url, files=files, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # TinEye result page: match links contain the source image page
        for tag in soup.select("a.match-link, .result-image-link a, a[class*='item']"):
            href = tag.get("href", "")
            if href.startswith("http") and "tineye.com" not in href:
                urls.append(href)

        # Fallback: grab any external links in results container
        results_div = soup.find("div", {"id": "results"}) or soup.find("div", class_="matches")
        if results_div:
            for a in results_div.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http") and "tineye.com" not in href:
                    urls.append(href)

        urls = list(dict.fromkeys(urls))
        logger.info(f"[TinEye] Found {len(urls)} result URLs.")
    except Exception as e:
        logger.error(f"[TinEye] Search failed: {e}")

    return urls


# ── Aggregator ───────────────────────────────────────────────────────────────

def collect_all_results(image: np.ndarray) -> list[str]:
    """
    Run all three reverse image search engines and merge their results.

    Args:
        image: BGR face/photo image as a NumPy array.

    Returns:
        Deduplicated list of all result page URLs.
    """
    logger.info("Starting reverse image search across Yandex, Bing, and TinEye…")
    all_urls: list[str] = []

    all_urls.extend(search_yandex(image))
    all_urls.extend(search_bing(image))
    all_urls.extend(search_tineye(image))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_urls: list[str] = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)

    logger.info(f"Total unique URLs collected: {len(unique_urls)}")
    return unique_urls
