"""
downloader.py — Social Media URL Filter
========================================
Filters a raw list of URLs, keeping only those that belong to known
social-media platforms, and returns them grouped by platform.
"""

import logging
from urllib.parse import urlparse
from typing import Optional

logger = logging.getLogger(__name__)

# ── Social-media domain registry (extend as needed) ───────────────────────────
SOCIAL_MEDIA_DOMAINS: dict[str, str] = {
    "instagram.com": "Instagram",
    "twitter.com": "Twitter",
    "x.com": "Twitter/X",
    "facebook.com": "Facebook",
    "fb.com": "Facebook",
    "linkedin.com": "LinkedIn",
    "reddit.com": "Reddit",
    "pinterest.com": "Pinterest",
    "tumblr.com": "Tumblr",
    "vk.com": "VKontakte",
    "ok.ru": "Odnoklassniki",
    "snapchat.com": "Snapchat",
    "tiktok.com": "TikTok",
    "flickr.com": "Flickr",
    "500px.com": "500px",
    "youtube.com": "YouTube",
    "threads.net": "Threads",
}


def detect_platform(url: str) -> Optional[str]:
    """
    Return the platform name if the URL belongs to a known social-media domain.

    Args:
        url: Page URL to check.

    Returns:
        Platform name (e.g. "Instagram"), or None if not social media.
    """
    try:
        host = urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return None

    for domain, name in SOCIAL_MEDIA_DOMAINS.items():
        if host == domain or host.endswith(f".{domain}"):
            return name
    return None


def filter_social_media_urls(urls: list[str]) -> list[dict]:
    """
    Keep only social-media URLs from a raw list.

    Args:
        urls: Raw list of page URLs from search engines.

    Returns:
        List of dicts: [{"url": ..., "platform": ...}, ...]
        Deduplicated and ordered as found.
    """
    results: list[dict] = []
    seen: set[str] = set()

    for url in urls:
        if url in seen:
            continue
        platform = detect_platform(url)
        if platform:
            results.append({"url": url, "platform": platform})
            seen.add(url)

    logger.info(f"Kept {len(results)}/{len(urls)} social-media URLs.")
    return results
