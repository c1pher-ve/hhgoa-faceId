"""
searcher.py — Reverse Image Search via Playwright
===================================================
Uses a real Chromium browser (Playwright) to upload a face image to
Yandex Images, Bing Visual Search, and TinEye, then scrapes the result
page URLs.

Why Playwright and not requests?
  All three engines render results with JavaScript and have bot-detection.
  A real headless browser bypasses this reliably.

Flow per engine:
  1. Open the search engine's image-search page
  2. Upload the face crop via the file input
  3. Wait for results to fully render
  4. Scrape all external result page URLs
  5. Return them

Install:
  pip install playwright
  playwright install chromium
"""

import logging
import os
import tempfile
from typing import Optional

import cv2
import numpy as np
from playwright.sync_api import sync_playwright, Page, BrowserContext, TimeoutError as PWTimeout

logger = logging.getLogger(__name__)

# How long (ms) to wait for pages / elements
NAV_TIMEOUT   = 30_000   # page navigation
ELEM_TIMEOUT  = 15_000   # waiting for an element to appear
RESULT_WAIT   = 5_000    # extra settle time after results load


# ── Temp file helper ──────────────────────────────────────────────────────────

def _save_temp_image(image: np.ndarray) -> str:
    """
    Write the BGR NumPy image to a temporary JPEG file on disk.

    Playwright's file-upload API requires a real file path, not bytes.

    Returns:
        Absolute path to the temp file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, image)
    logger.debug(f"Saved temp face image to {tmp.name}")
    return tmp.name


# ── URL helpers ───────────────────────────────────────────────────────────────

def _extract_external_links(page: Page, exclude_keywords: list[str]) -> list[str]:
    """
    Pull all <a href> links from the current page, drop internal/irrelevant ones.

    Args:
        page:             Playwright Page object.
        exclude_keywords: Strings — any URL containing one of these is dropped.

    Returns:
        Deduplicated list of external URLs.
    """
    try:
        raw: list[str] = page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => e.href)"
        )
    except Exception:
        raw = []

    seen: set[str] = set()
    result: list[str] = []
    for url in raw:
        if not url.startswith("http"):
            continue
        if any(kw in url.lower() for kw in exclude_keywords):
            continue
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


# ── Yandex ───────────────────────────────────────────────────────────────────

def search_yandex(context: BrowserContext, image_path: str) -> list[str]:
    """
    Reverse-image search on Yandex Images.

    Steps:
      1. Go to yandex.com/images
      2. Click the camera icon → switch to "Upload file" tab
      3. Set the file on the hidden <input type=file>
      4. Wait for the CBIR results page to load
      5. Scrape result source links ("Sites with this image")

    Args:
        context:    Playwright BrowserContext (shared across engines).
        image_path: Absolute path to the face crop JPEG.

    Returns:
        List of result page URLs.
    """
    urls: list[str] = []
    page = context.new_page()

    try:
        logger.info("[Yandex] Navigating to Yandex Images…")
        page.goto("https://yandex.com/images/", timeout=NAV_TIMEOUT)

        # Click the search-by-image (camera) button
        page.click(
            'button.input__search-by-image, '
            '[class*="SearchByImageButton"], '
            '[data-type="cbir"], '
            '.cbir-panel__search-button',
            timeout=ELEM_TIMEOUT,
        )
        logger.debug("[Yandex] Clicked camera icon")

        # In the dialog that pops up, click "Upload file" tab if present
        try:
            page.click(
                'text=Upload file, '
                '[class*="upload"], '
                '.cbir-panel__tab:nth-child(2)',
                timeout=5_000,
            )
            logger.debug("[Yandex] Switched to Upload tab")
        except PWTimeout:
            pass  # Some layouts go straight to file input

        # Upload the image via the file input
        with page.expect_file_chooser(timeout=ELEM_TIMEOUT) as fc_info:
            # Try clicking the upload area if the input isn't directly visible
            try:
                page.click('[class*="uploader"], [class*="upload-area"]', timeout=3_000)
            except PWTimeout:
                pass
        file_chooser = fc_info.value
        file_chooser.set_files(image_path)
        logger.debug("[Yandex] File uploaded via file chooser")

        # Wait for results to render
        page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT)
        page.wait_for_timeout(RESULT_WAIT)

        # Scrape: "Sites with this image" section links
        # These sit in .CbirSites or .serp-item wrappers
        site_links: list[str] = page.eval_on_selector_all(
            '.CbirSites-ItemTitle a, .serp-item__link, a.Link[href^="http"]',
            "els => els.map(e => e.href)"
        )

        # Fallback — grab all external links on the page
        if not site_links:
            site_links = _extract_external_links(
                page,
                exclude_keywords=["yandex.", "ya.ru", "javascript:", "mailto:"]
            )

        urls = list(dict.fromkeys(u for u in site_links if u.startswith("http")))
        logger.info(f"[Yandex] Collected {len(urls)} URLs")

    except PWTimeout as e:
        logger.error(f"[Yandex] Timed out: {e}")
    except Exception as e:
        logger.error(f"[Yandex] Error: {e}")
    finally:
        page.close()

    return urls


# ── Bing ─────────────────────────────────────────────────────────────────────

def search_bing(context: BrowserContext, image_path: str) -> list[str]:
    """
    Reverse-image search on Bing Visual Search.

    Steps:
      1. Go to bing.com/images
      2. Click the camera icon → upload tab
      3. Set the file on the file input
      4. Wait for visual-search results
      5. Scrape "Pages with this image" links

    Args:
        context:    Playwright BrowserContext.
        image_path: Absolute path to the face crop JPEG.

    Returns:
        List of result page URLs.
    """
    urls: list[str] = []
    page = context.new_page()

    try:
        logger.info("[Bing] Navigating to Bing Images…")
        page.goto("https://www.bing.com/images", timeout=NAV_TIMEOUT)

        # Click the camera / visual search icon
        page.click(
            '.sbi_b, '
            '[title="Search using an image"], '
            '#sbi_b, '
            'label[for="sbi_fld"]',
            timeout=ELEM_TIMEOUT,
        )
        logger.debug("[Bing] Clicked camera icon")

        # Click "Upload from device" or the upload tab
        try:
            page.click(
                'text=Upload from device, '
                'text=Upload an image, '
                '.sbi_dl_btn',
                timeout=5_000,
            )
            logger.debug("[Bing] Clicked upload tab")
        except PWTimeout:
            pass

        # Set the file
        with page.expect_file_chooser(timeout=ELEM_TIMEOUT) as fc_info:
            try:
                page.click('#sbi_fld, input[type=file], .upload-btn', timeout=3_000)
            except PWTimeout:
                pass
        file_chooser = fc_info.value
        file_chooser.set_files(image_path)
        logger.debug("[Bing] File uploaded")

        # Wait for results page to fully render
        page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT)
        page.wait_for_timeout(RESULT_WAIT)

        # Scrape result links — "Pages with matching images"
        page_links: list[str] = page.eval_on_selector_all(
            '.iusc a, .richcard a, a.inflnk, .img_info a',
            "els => els.map(e => e.href)"
        )

        if not page_links:
            page_links = _extract_external_links(
                page,
                exclude_keywords=["bing.com", "microsoft.com", "msn.com", "javascript:", "mailto:"]
            )

        urls = list(dict.fromkeys(u for u in page_links if u.startswith("http")))
        logger.info(f"[Bing] Collected {len(urls)} URLs")

    except PWTimeout as e:
        logger.error(f"[Bing] Timed out: {e}")
    except Exception as e:
        logger.error(f"[Bing] Error: {e}")
    finally:
        page.close()

    return urls


# ── TinEye ───────────────────────────────────────────────────────────────────

def search_tineye(context: BrowserContext, image_path: str) -> list[str]:
    """
    Reverse-image search on TinEye.

    Steps:
      1. Go to tineye.com
      2. Upload the image via the file input
      3. Wait for match results
      4. Scrape source page links for each match

    Args:
        context:    Playwright BrowserContext.
        image_path: Absolute path to the face crop JPEG.

    Returns:
        List of result page URLs.
    """
    urls: list[str] = []
    page = context.new_page()

    try:
        logger.info("[TinEye] Navigating to TinEye…")
        page.goto("https://tineye.com/", timeout=NAV_TIMEOUT)

        # TinEye has a visible file input on the homepage
        with page.expect_file_chooser(timeout=ELEM_TIMEOUT) as fc_info:
            page.click(
                '#upload_button, '
                'label[for="upload_url"], '
                'button[type="button"]:has-text("Upload")',
                timeout=ELEM_TIMEOUT,
            )
        file_chooser = fc_info.value
        file_chooser.set_files(image_path)
        logger.debug("[TinEye] File uploaded")

        # Wait for results to load (TinEye can be slow)
        page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT)
        page.wait_for_timeout(RESULT_WAIT)

        # Scrape match source links
        match_links: list[str] = page.eval_on_selector_all(
            '.match-url a, .result-image-link, .link a, a.item-link',
            "els => els.map(e => e.href)"
        )

        if not match_links:
            match_links = _extract_external_links(
                page,
                exclude_keywords=["tineye.com", "javascript:", "mailto:"]
            )

        urls = list(dict.fromkeys(u for u in match_links if u.startswith("http")))
        logger.info(f"[TinEye] Collected {len(urls)} URLs")

    except PWTimeout as e:
        logger.error(f"[TinEye] Timed out: {e}")
    except Exception as e:
        logger.error(f"[TinEye] Error: {e}")
    finally:
        page.close()

    return urls


# ── Aggregator ────────────────────────────────────────────────────────────────

def collect_all_results(image: np.ndarray, headless: bool = True) -> list[str]:
    """
    Run all three reverse image search engines using a shared Playwright
    browser, merge and deduplicate the results.

    Args:
        image:    BGR face crop as a NumPy array.
        headless: Run browser in headless mode (no visible window).
                  Set to False for debugging.

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
                    "--disable-blink-features=AutomationControlled",  # hide bot signals
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

            logger.info("Browser launched. Starting searches…")

            all_urls.extend(search_yandex(context, image_path))
            all_urls.extend(search_bing(context, image_path))
            all_urls.extend(search_tineye(context, image_path))

            browser.close()

    finally:
        # Clean up temp file
        try:
            os.unlink(image_path)
        except Exception:
            pass

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    logger.info(f"Total unique URLs from all engines: {len(unique)}")
    return unique
