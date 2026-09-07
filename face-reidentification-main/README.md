# Mantis — Face Re-Identification Pipeline

> **Reverse-image-search a face, verify it is the same person using AI, and optionally notarise the findings immutably on the blockchain.**

---

## What This Is

Mantis is a privacy-intelligence pipeline that takes a single photo as input and automatically discovers where that person appears across public social-media platforms. Unlike a naive reverse-image search (which returns anything visually similar — dog posts, landscapes, memes), Mantis applies **two layers of AI-powered verification** to ensure every result actually contains the same person.

The system runs fully locally, operates in headless mode by default, and produces machine-readable JSON output. An optional blockchain mode writes a tamper-evident, timestamped proof of the findings to the Polygon network.

---

## Pipeline Architecture

```
Input Photo
      |
      v
+------------------------------------------------------+
|  Step 1 . Face Detection & Crop  (SCRFD det_10g)     |
|  Detects the largest face, pads the bounding box,    |
|  and extracts a clean face crop.                     |
+-------------------------+----------------------------+
                          |
                          v
+------------------------------------------------------+
|  Step 1b . Query Embedding  (ArcFace w600k_r50)      |
|  Aligns the face crop and encodes it as a            |
|  512-dimensional identity vector for later           |
|  comparison against search results.                  |
+-------------------------+----------------------------+
                          |
                          v
+------------------------------------------------------+
|  Step 2 . Reverse Image Search  (Playwright)         |
|  Uploads the face crop to Yandex Images,             |
|  Bing Visual Search, and TinEye in sequence.         |
|  Collects and deduplicates all result URLs.          |
|  (~70 raw URLs typically)                            |
+-------------------------+----------------------------+
                          |
                          v
+------------------------------------------------------+
|  Step 3 . Social-Media Domain Filter                 |
|  Keeps only URLs on known platforms:                 |
|  Instagram . Facebook . YouTube . Twitter/X          |
|  Reddit . TikTok . LinkedIn . Pinterest . Tumblr     |
|  Flickr . VK . Snapchat . Threads . 500px           |
|  (~10-15 URLs typically)                             |
+-------------------------+----------------------------+
                          |
                          v
+------------------------------------------------------+
|  Step 3b . Two-Stage Face Verification               |
|                                                      |
|  For each result URL:                                |
|  +---------------------------------------------------+|
|  |  Stage 1 . Face Presence Check  (SCRFD)          ||
|  |  Fetches og:image / twitter:image from page.     ||
|  |  Runs SCRFD on it.                               ||
|  |  No face detected -> DROPPED                     ||
|  |  (eliminates dog posts, landscapes, memes)       ||
|  +------------------------+-------------------------+|
|                           | face found               |
|  +------------------------v-------------------------+|
|  |  Stage 2 . Identity Match  (ArcFace)            ||
|  |  Encodes the face on the result page.            ||
|  |  Computes cosine similarity vs. query embed.     ||
|  |  Below threshold (0.35) -> DROPPED              ||
|  |  (eliminates different people)                   ||
|  +---------------------------------------------------+|
+-------------------------+----------------------------+
                          | verified matches only
                          v
+------------------------------------------------------+
|  Step 4 . Output                                     |
|  Prints grouped results to stdout.                   |
|  Saves results.json with URL + platform + similarity.|
+-------------------------+----------------------------+
                          | (optional --blockchain flag)
                          v
+------------------------------------------------------+
|  Step 5 . Blockchain Notarisation  (Polygon Amoy)    |
|  Sends a zero-value Polygon transaction containing:  |
|    . SHA-256 hash of the face crop image             |
|    . SHA-256 hash of the full result record          |
|    . All found URLs, engines used, timestamp         |
|  Returns a PolygonScan URL as immutable proof.       |
+------------------------------------------------------+
```

---

## Models Used

| Model | Weight File | Size | Purpose |
|---|---|---|---|
| **SCRFD** (`det_10g`) | `weights/det_10g.onnx` | ~16 MB | Face detection — finds & crops faces in any image |
| **ArcFace** (`w600k_r50`) | `weights/w600k_r50.onnx` | ~174 MB | Face recognition — produces 512-d identity embeddings |

Both models run on CPU via ONNX Runtime. GPU acceleration (CUDA) is used automatically if available.

---

## File Structure

```
face-reidentification-main/
|
+-- pipeline.py          <- Main entry point — orchestrates all steps
+-- face_verifier.py     <- Two-stage result filter (face presence + identity match)
+-- searcher.py          <- Playwright-based Yandex / Bing / TinEye scraper
+-- downloader.py        <- Social-media domain registry and URL filter
+-- blockchain.py        <- Polygon blockchain record writer and verifier
|
+-- models/
|   +-- scrfd.py         <- SCRFD face detector wrapper (ONNX Runtime)
|   +-- arcface.py       <- ArcFace face recognizer wrapper (ONNX Runtime)
|
+-- utils/
|   +-- helpers.py       <- Face alignment, cosine similarity, bbox drawing
|   +-- logging.py       <- Structured file + console logging setup
|
+-- weights/
    +-- det_10g.onnx     <- SCRFD detection model
    +-- w600k_r50.onnx   <- ArcFace recognition model
```

---

## Installation

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Download model weights

**Windows (PowerShell):**
```powershell
python -c "
import requests
print('Downloading SCRFD...')
open('weights/det_10g.onnx','wb').write(requests.get('https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx').content)
print('Downloading ArcFace...')
open('weights/w600k_r50.onnx','wb').write(requests.get('https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx').content)
print('Done.')
"
```

**Linux / macOS:**
```bash
bash download.sh
```

---

## Usage

### Find social-media links for a face (basic)

```bash
python pipeline.py --input photo.jpg
```

### Save verified results to JSON

```bash
python pipeline.py --input photo.jpg --output results.json
```

**Example `results.json` output:**
```json
[
  {
    "url": "https://www.instagram.com/p/ABC123/",
    "platform": "Instagram",
    "similarity": 0.512
  },
  {
    "url": "https://www.youtube.com/watch?v=XYZ789",
    "platform": "YouTube",
    "similarity": 0.441
  }
]
```

The `similarity` field (0-1) shows how closely the face in that post matches the query photo.

### Show browser window (debug)

```bash
python pipeline.py --input photo.jpg --show-browser
```

### Write results to the blockchain

```bash
# One-time wallet setup
python blockchain.py --generate-wallet
# Fund the printed address at https://faucet.polygon.technology/ (Amoy testnet)

# Run with blockchain recording (PowerShell)
$env:WALLET_PRIVATE_KEY = "0xyour_private_key_here"
python pipeline.py --input photo.jpg --output results.json --blockchain
```

**Output on success:**
```
======================================================================
  BLOCKCHAIN RECORD WRITTEN
======================================================================
  TX Hash    : 0xabc123...def456
  Block      : #12345678
  Network    : polygon-amoy
  Explorer   : https://amoy.polygonscan.com/tx/0xabc123...
  Record Hash: d615e4ed...
======================================================================
```

A `.receipt.json` file is also saved alongside `results.json`.

### Verify a past blockchain record

```bash
python blockchain.py --verify 0xyour_tx_hash_here
```

---

## CLI Reference

```
python pipeline.py [OPTIONS]

Required:
  --input  / -i   PATH    Path to input photo (jpg / png)

Optional:
  --output / -o   PATH    Save results as JSON
  --det-weight    PATH    SCRFD model  (default: ./weights/det_10g.onnx)
  --rec-weight    PATH    ArcFace model (default: ./weights/w600k_r50.onnx)
  --conf-thresh   FLOAT   Face detection confidence threshold (default: 0.5)
  --padding       FLOAT   Padding fraction around face crop (default: 0.2)
  --show-browser          Show Chromium window during search
  --blockchain            Write result record to Polygon blockchain
  --network       NAME    Blockchain network:
                            polygon-amoy     (default — free test MATIC)
                            polygon-mainnet  (real MATIC)
                            ethereum-sepolia (ETH testnet)
```

---

## How the Identity Verification Works

The `face_verifier.py` module runs after the social-media domain filter and eliminates false positives that reverse-image search cannot catch on its own.

### Stage 1 — Face Presence (SCRFD)

For each result URL:
1. Fetches the page HTML
2. Extracts the best image URL: `og:image` -> `twitter:image` -> first `<img>` tag
3. Runs SCRFD on that image
4. If no human face detected at >= 0.45 confidence -> **DROPPED**

This eliminates dog posts, cat photos, landscapes, products, and memes.

### Stage 2 — Identity Match (ArcFace cosine similarity)

For results that pass Stage 1:
1. ArcFace aligns the detected face to a canonical 112x112 crop
2. Encodes it as a normalised 512-dimensional embedding
3. Computes cosine similarity against the query face embedding
4. If similarity < `SIMILARITY_THRESHOLD` -> **DROPPED** (different person)
5. If similarity >= threshold -> **KEPT** with score attached

**Fail-safe:** If a page is inaccessible (login wall, network error), the result is kept by default. Genuine matches are never silently discarded due to access issues.

### Tuning sensitivity

Edit `SIMILARITY_THRESHOLD` in `face_verifier.py`:

```python
SIMILARITY_THRESHOLD: float = 0.35   # 0.30 loose . 0.40 balanced . 0.50 strict
```

---

## How Blockchain Notarisation Works

When `--blockchain` is passed, the pipeline constructs a `SearchRecord` containing:

| Field | Value |
|---|---|
| `image_hash` | SHA-256 of the face crop JPEG bytes |
| `timestamp_utc` | Unix timestamp (seconds) of the search |
| `urls_found` | All verified social-media URLs |
| `engines_used` | `["Yandex", "Bing", "TinEye"]` |
| `total_raw_urls` | Total URLs scraped before filtering |
| `record_hash` | SHA-256 of all the above (tamper-proof fingerprint) |

This record is JSON-encoded and written to the `data` field of a zero-value Polygon transaction with the prefix `MANTIS_FACEID:`.

Because the **image hash** proves which face was searched, the **record hash** cannot be altered after block inclusion, and the **block timestamp** is set by the Polygon network — the record is **cryptographically tamper-evident** and permanently verifiable by anyone with the transaction hash.

---

## Blockchain Networks

| Key | Network | Chain ID | Notes |
|---|---|---|---|
| `polygon-amoy` | Polygon Amoy Testnet | 80002 | **Default.** Free test MATIC from faucet |
| `polygon-mainnet` | Polygon Mainnet | 137 | Production. Real MATIC required |
| `ethereum-sepolia` | Ethereum Sepolia | 11155111 | Ethereum testnet alternative |

Free test MATIC faucet: **https://faucet.polygon.technology/**

---

## Requirements

```
python >= 3.10
onnxruntime          # use onnxruntime-gpu for CUDA acceleration
numpy
opencv-python
playwright           # run: playwright install chromium
requests
beautifulsoup4
lxml
web3                 # only needed for --blockchain
```

---

## Known Limitations

| Limitation | Detail |
|---|---|
| Login-walled pages | Instagram/Facebook pages requiring login cannot be fetched — results are kept by default (fail-safe) |
| Search engine UI drift | Yandex/Bing/TinEye update their selectors periodically; check `searcher.py` if timeouts increase |
| Threshold tuning | `0.35` works well for frontal photos; lower-quality or profile-angle photos may need a lower threshold |
| Blockchain gas | Use the free Amoy testnet faucet for testing; mainnet requires real MATIC for gas fees |

---

## Credits

- Face detection & recognition: [InsightFace / SCRFD + ArcFace](https://github.com/deepinsight/insightface)
- Model weights hosted by: [yakhyo/face-reidentification](https://github.com/yakhyo/face-reidentification)
- Browser automation: [Playwright](https://playwright.dev/)
- Blockchain: [Polygon](https://polygon.technology/) and [web3.py](https://web3py.readthedocs.io/)
