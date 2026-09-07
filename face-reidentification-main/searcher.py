"""
searcher.py — Reverse Image Search via Playwright
===================================================
Uses a real Chromium browser (Playwright) to upload a face image to
Yandex Images, Bing Visual Search, and TinEye, then scrapes result URLs.

Supports --show-browser flag for visual debugging.
On any failure, screenshots are saved to ./debug_screenshots/.
"""

import logging
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright, Page, BrowserContext, TimeoutError as PWTimeout

logger = logging.getLogger(__name__)

NAV_TIMEOUT  = 30_000
ELEM_TIMEOUT = 12_000
RESULT_WAIT  = 4_000

DEBUG_DIR = Path("./debug_screenshots")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_temp_image(image: np.ndarray) -> str:
    """Write BGR image to a temp JPEG file; return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, image)
    return tmp.name


def _screenshot(page: Page, name: str) -> None:
    """Save a debug screenshot (always — success or failure)."""
    try:
        DEBUG_DIR.mkdir(exist_ok=True)
        path = DEBUG_DIR / f"{name}.png"
        page.screenshot(path=str(path), full_page=True)
        logger.info(f"Screenshot saved → {path}")
    except Exception as e:
        logger.warning(f"Could not save screenshot '{name}': {e}")


def _extract_external_links(page: Page, exclude: list[str]) -> list[str]:
    """Pull all <a href> links, drop internal/noise ones."""
    try:
        raw: list[str] = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
    except Exception:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for url in raw:
        if not url.startswith("http"):
            continue
        if any(kw in url.lower() for kw in exclude):
            continue
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


# ── Yandex ───────────────────────────────────────────────────────────────────

def search_yandex(context: BrowserContext, image_path: str) -> list[str]:
    """Upload face to Yandex Images and return result page URLs."""
    urls: list[str] = []
    page = context.new_page()

    try:
        logger.info("[Yandex] Opening Yandex Images…")
        page.goto("https://yandex.com/images/", timeout=NAV_TIMEOUT)
        page.wait_for_load_state("domcontentloaded")
        _screenshot(page, "yandex_01_home")

        # Click the camera / search-by-image button
        camera_selectors = [
            'button.input__search-by-image',
            '[class*="SearchByImage"]',
            '[class*="search-by-image"]',
            '[data-type="cbir"]',
            'button[aria-label*="image"]',
            'button[title*="image"]',
        ]
        clicked = False
        for sel in camera_selectors:
            try:
                page.click(sel, timeout=3_000)
                clicked = True
                logger.debug(f"[Yandex] Camera clicked via: {sel}")
                break
            except PWTimeout:
                continue

        if not clicked:
            logger.warning("[Yandex] Could not find camera button — trying direct URL upload")
            # Try navigating directly to the upload URL
            page.goto("https://yandex.com/images/search?rpt=imageview", timeout=NAV_TIMEOUT)

        _screenshot(page, "yandex_02_after_camera_click")

        # Handle file upload via file chooser
        try:
            with page.expect_file_chooser(timeout=ELEM_TIMEOUT) as fc_info:
                # Try clicking various upload triggers
                for upload_sel in [
                    '[class*="upload"]',
                    'label[class*="upload"]',
                    'input[type="file"]',
                    '[class*="cbir"]',
                    'text=Upload',
                    'text=Select file',
                ]:
                    try:
                        page.click(upload_sel, timeout=2_000)
                        break
                    except PWTimeout:
                        continue
            fc_info.value.set_files(image_path)
            logger.info("[Yandex] File uploaded via file chooser")
        except Exception as e:
            logger.warning(f"[Yandex] File chooser method failed ({e}), trying set_input_files directly")
            try:
                page.set_input_files('input[type="file"]', image_path, timeout=ELEM_TIMEOUT)
                logger.info("[Yandex] File set directly on input")
            except Exception as e2:
                logger.error(f"[Yandex] Upload failed entirely: {e2}")
                _screenshot(page, "yandex_ERROR_upload_failed")
                return urls

        # Wait for results
        page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT)
        page.wait_for_timeout(RESULT_WAIT)
        _screenshot(page, "yandex_03_results")

        logger.info(f"[Yandex] Results page URL: {page.url}")

        # Scrape result links (try specific selectors first, fallback to all links)
        site_links: list[str] = page.eval_on_selector_all(
            '.CbirSites-ItemTitle a, '
            '.CbirItem a, '
            '.serp-item__link, '
            'a.Link[target="_blank"]',
            "els => els.map(e => e.href)"
        )

        if not site_links:
            logger.debug("[Yandex] Specific selectors returned nothing, using fallback")
            site_links = _extract_external_links(
                page, exclude=["yandex.", "ya.ru", "javascript:", "mailto:", "google."]
            )

        urls = list(dict.fromkeys(u for u in site_links if u.startswith("http")))
        logger.info(f"[Yandex] Found {len(urls)} URLs")

    except PWTimeout as e:
        logger.error(f"[Yandex] Timeout: {e}")
        _screenshot(page, "yandex_ERROR_timeout")
    except Exception as e:
        logger.error(f"[Yandex] Error: {e}")
        _screenshot(page, "yandex_ERROR_general")
    finally:
        page.close()

    return urls


# ── Bing ─────────────────────────────────────────────────────────────────────

def search_bing(context: BrowserContext, image_path: str) -> list[str]:
    """Upload face to Bing Visual Search and return result page URLs."""
    urls: list[str] = []
    page = context.new_page()

    try:
        logger.info("[Bing] Opening Bing Images…")
        page.goto("https://www.bing.com/images", timeout=NAV_TIMEOUT)
        page.wait_for_load_state("domcontentloaded")
        _screenshot(page, "bing_01_home")

        # Click the camera icon
        camera_selectors = [
            '.sbi_b',
            '#sbi_b',
            '[aria-label*="Search using an image"]',
            '[title*="Search using an image"]',
            'label[for="sbi_fld"]',
            'a[href*="visualsearch"]',
        ]
        clicked = False
        for sel in camera_selectors:
            try:
                page.click(sel, timeout=3_000)
                clicked = True
                logger.debug(f"[Bing] Camera clicked via: {sel}")
                break
            except PWTimeout:
                continue

        if not clicked:
            logger.warning("[Bing] Camera button not found, navigating directly to visual search")
            page.goto("https://www.bing.com/visualsearch", timeout=NAV_TIMEOUT)

        _screenshot(page, "bing_02_after_camera_click")

        # Upload file
        try:
            with page.expect_file_chooser(timeout=ELEM_TIMEOUT) as fc_info:
                for upload_sel in [
                    '#sbi_fld',
                    'input[type="file"]',
                    '[class*="upload"]',
                    'text=Upload from device',
                    'text=Upload an image',
                    '.sbi_dl_btn',
                ]:
                    try:
                        page.click(upload_sel, timeout=2_000)
                        break
                    except PWTimeout:
                        continue
            fc_info.value.set_files(image_path)
            logger.info("[Bing] File uploaded via file chooser")
        except Exception as e:
            logger.warning(f"[Bing] File chooser failed ({e}), trying direct input")
            try:
                page.set_input_files('input[type="file"]', image_path, timeout=ELEM_TIMEOUT)
            except Exception as e2:
                logger.error(f"[Bing] Upload failed: {e2}")
                _screenshot(page, "bing_ERROR_upload_failed")
                return urls

        page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT)
        page.wait_for_timeout(RESULT_WAIT)
        _screenshot(page, "bing_03_results")

        logger.info(f"[Bing] Results page URL: {page.url}")

        page_links: list[str] = page.eval_on_selector_all(
            '.iusc a, .richcard a, a.inflnk, .img_info a, .b_algo a',
            "els => els.map(e => e.href)"
        )

        if not page_links:
            page_links = _extract_external_links(
                page, exclude=["bing.com", "microsoft.com", "msn.com", "javascript:", "mailto:"]
            )

        urls = list(dict.fromkeys(u for u in page_links if u.startswith("http")))
        logger.info(f"[Bing] Found {len(urls)} URLs")

    except PWTimeout as e:
        logger.error(f"[Bing] Timeout: {e}")
        _screenshot(page, "bing_ERROR_timeout")
    except Exception as e:
        logger.error(f"[Bing] Error: {e}")
        _screenshot(page, "bing_ERROR_general")
    finally:
        page.close()

    return urls


# ── TinEye ───────────────────────────────────────────────────────────────────

def search_tineye(context: BrowserContext, image_path: str) -> list[str]:
    """Upload face to TinEye and return result page URLs."""
    urls: list[str] = []
    page = context.new_page()

    try:
        logger.info("[TinEye] Opening TinEye…")
        page.goto("https://tineye.com/", timeout=NAV_TIMEOUT)
        page.wait_for_load_state("domcontentloaded")
        _screenshot(page, "tineye_01_home")

        # TinEye uses a hidden <input type="file"> — set files directly,
        # then click the submit button. No native file chooser dialog opens.
        try:
            # Make the hidden input visible so Playwright can interact with it
            page.eval_on_selector(
                'input[type="file"]',
                "el => el.style.display = 'block'"
            )
            page.set_input_files('input[type="file"]', image_path, timeout=ELEM_TIMEOUT)
            logger.info("[TinEye] File set on input element")

            # Submit the form
            for submit_sel in [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:has-text("Search")',
                'form button',
            ]:
                try:
                    page.click(submit_sel, timeout=3_000)
                    logger.debug(f"[TinEye] Submitted via: {submit_sel}")
                    break
                except PWTimeout:
                    continue
        except Exception as e:
            logger.error(f"[TinEye] Upload failed: {e}")
            _screenshot(page, "tineye_ERROR_upload_failed")
            return urls

        # TinEye renders results fast — networkidle waits forever on background requests
        try:
            page.wait_for_load_state("domcontentloaded", timeout=NAV_TIMEOUT)
        except PWTimeout:
            pass  # Results may already be rendered
        page.wait_for_timeout(RESULT_WAIT)
        _screenshot(page, "tineye_02_results")

        logger.info(f"[TinEye] Results page URL: {page.url}")

        match_links: list[str] = page.eval_on_selector_all(
            '.match-url a, .result-image-link a, a.item-link, .link a',
            "els => els.map(e => e.href)"
        )

        if not match_links:
            match_links = _extract_external_links(
                page, exclude=["tineye.com", "javascript:", "mailto:"]
            )

        urls = list(dict.fromkeys(u for u in match_links if u.startswith("http")))
        logger.info(f"[TinEye] Found {len(urls)} URLs")

    except PWTimeout as e:
        logger.error(f"[TinEye] Timeout: {e}")
        _screenshot(page, "tineye_ERROR_timeout")
    except Exception as e:
        logger.error(f"[TinEye] Error: {e}")
        _screenshot(page, "tineye_ERROR_general")
    finally:
        page.close()

    return urls


# ── Aggregator ────────────────────────────────────────────────────────────────

def collect_all_results(image: np.ndarray, headless: bool = True) -> list[str]:
    """
    Run all three search engines; merge and deduplicate results.

    Args:
        image:    BGR face crop as NumPy array.
        headless: False = show the browser window (useful for debugging).

    Returns:
        Deduplicated list of all result page URLs.
    """
    image_path = _save_temp_image(image)
    all_urls: list[str] = []

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
                locale="en-US",
            )

            all_urls.extend(search_yandex(context, image_path))
            all_urls.extend(search_bing(context, image_path))
            all_urls.extend(search_tineye(context, image_path))

            browser.close()
    finally:
        try:
            os.unlink(image_path)
        except Exception:
            pass

    seen: set[str] = set()
    unique: list[str] = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    logger.info(f"Total unique URLs: {len(unique)}")
    return unique
