"""
blockchain.py — Tamper-Evident Record on Polygon Blockchain
=============================================================
Writes face-search match results as an immutable, timestamped record
to the Polygon Amoy testnet (free — uses test MATIC).

How it works:
  1. Build a structured record from the search result
     (image SHA-256 hash + timestamp + social-media URLs found).
  2. Compute the SHA-256 hash of that record.
  3. Send a zero-value transaction to the Polygon network with the
     record hash encoded in the `data` field.
  4. Return the transaction hash as proof — anyone can look it up
     on https://amoy.polygonscan.com/ and verify the record.

Why this is tamper-evident:
  - The image hash proves which face was searched.
  - The record hash in the transaction cannot be altered retroactively.
  - The block timestamp is set by the network, not by us.
  - The transaction hash uniquely identifies the record forever.

Setup (one-time):
  1. Generate a wallet:
       python blockchain.py --generate-wallet
  2. Fund it with free test MATIC:
       https://faucet.polygon.technology/  (select Amoy testnet)
  3. Set environment variable:
       set WALLET_PRIVATE_KEY=0xyour_private_key_here

Usage:
  Automatically called from pipeline.py when --blockchain flag is used.
  Or standalone:
       python blockchain.py --test
"""

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Network config ────────────────────────────────────────────────────────────

NETWORKS = {
    "polygon-amoy": {
        "name": "Polygon Amoy Testnet",
        "rpc": "https://rpc-amoy.polygon.technology/",
        "chain_id": 80002,
        "explorer": "https://amoy.polygonscan.com/tx/",
        "currency": "MATIC (test)",
        "faucet": "https://faucet.polygon.technology/",
    },
    "polygon-mainnet": {
        "name": "Polygon Mainnet",
        "rpc": "https://polygon-rpc.com/",
        "chain_id": 137,
        "explorer": "https://polygonscan.com/tx/",
        "currency": "MATIC (real)",
        "faucet": None,
    },
    "ethereum-sepolia": {
        "name": "Ethereum Sepolia Testnet",
        "rpc": "https://rpc.sepolia.org/",
        "chain_id": 11155111,
        "explorer": "https://sepolia.etherscan.io/tx/",
        "currency": "ETH (test)",
        "faucet": "https://sepoliafaucet.com/",
    },
}

DEFAULT_NETWORK = "polygon-amoy"


# ── Record data model ─────────────────────────────────────────────────────────

@dataclass
class SearchRecord:
    """
    Represents one face-search event written to the blockchain.

    Fields:
        image_hash      : SHA-256 of the face crop image bytes (hex string).
        timestamp_utc   : Unix timestamp (integer seconds) of the search.
        urls_found      : Social-media URLs discovered.
        engines_used    : Which search engines returned results.
        total_raw_urls  : Total URLs scraped before filtering.
        network         : Blockchain network name.
        record_hash     : SHA-256 of all above fields (computed, not set by caller).
    """
    image_hash: str
    timestamp_utc: int
    urls_found: list[str]
    engines_used: list[str]
    total_raw_urls: int
    network: str = DEFAULT_NETWORK
    record_hash: str = field(default="", init=False)

    def __post_init__(self):
        self.record_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """SHA-256 of the record fields (excluding record_hash itself)."""
        payload = {
            "image_hash": self.image_hash,
            "timestamp_utc": self.timestamp_utc,
            "urls_found": sorted(self.urls_found),   # sorted for determinism
            "engines_used": sorted(self.engines_used),
            "total_raw_urls": self.total_raw_urls,
            "network": self.network,
        }
        canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)


# ── Image hashing ─────────────────────────────────────────────────────────────

def hash_image(image_bytes: bytes) -> str:
    """
    Compute the SHA-256 hash of raw image bytes.

    Args:
        image_bytes: Raw bytes of the face crop image.

    Returns:
        Hex string of the SHA-256 hash.
    """
    return hashlib.sha256(image_bytes).hexdigest()


def hash_image_file(path: str) -> str:
    """SHA-256 hash of an image file on disk."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ── Blockchain writer ─────────────────────────────────────────────────────────

@dataclass
class BlockchainReceipt:
    """Result of writing a record to the blockchain."""
    success: bool
    tx_hash: Optional[str] = None
    block_number: Optional[int] = None
    block_timestamp: Optional[int] = None
    network: str = DEFAULT_NETWORK
    explorer_url: Optional[str] = None
    gas_used: Optional[int] = None
    error: Optional[str] = None
    record: Optional[SearchRecord] = None


def write_record(
    record: SearchRecord,
    private_key: Optional[str] = None,
    network: str = DEFAULT_NETWORK,
) -> BlockchainReceipt:
    """
    Write a SearchRecord to the blockchain as a zero-value transaction.

    The record_hash is encoded in the transaction's `data` field.
    The full record JSON is also prepended so it can be decoded later.

    Args:
        record:      The SearchRecord to persist.
        private_key: Wallet private key (hex string with 0x prefix).
                     Falls back to WALLET_PRIVATE_KEY env var.
        network:     Network key from NETWORKS dict.

    Returns:
        BlockchainReceipt with tx_hash and explorer URL on success.
    """
    try:
        from web3 import Web3
        from web3.middleware import ExtraDataToPOAMiddleware
    except ImportError:
        msg = (
            "web3 package not installed. Run: pip install web3\n"
            "Then get a free wallet and test MATIC from the faucet."
        )
        logger.error(msg)
        return BlockchainReceipt(success=False, error=msg, network=network)

    # Resolve private key
    pk = private_key or os.environ.get("WALLET_PRIVATE_KEY")
    if not pk:
        msg = (
            "No wallet private key found.\n"
            "Set the WALLET_PRIVATE_KEY environment variable:\n"
            "  set WALLET_PRIVATE_KEY=0xyour_key_here\n"
            "To generate a new wallet, run: python blockchain.py --generate-wallet"
        )
        logger.error(msg)
        return BlockchainReceipt(success=False, error=msg, network=network)

    net = NETWORKS.get(network)
    if not net:
        msg = f"Unknown network '{network}'. Options: {list(NETWORKS.keys())}"
        return BlockchainReceipt(success=False, error=msg, network=network)

    # Connect
    w3 = Web3(Web3.HTTPProvider(net["rpc"]))
    # PoA middleware for Polygon
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

    if not w3.is_connected():
        msg = f"Could not connect to {net['name']} RPC: {net['rpc']}"
        logger.error(msg)
        return BlockchainReceipt(success=False, error=msg, network=network)

    logger.info(f"Connected to {net['name']} (chain ID: {net['chain_id']})")

    account = w3.eth.account.from_key(pk)
    address = account.address
    logger.info(f"Wallet address: {address}")

    # Check balance
    balance_wei = w3.eth.get_balance(address)
    balance = w3.from_wei(balance_wei, "ether")
    logger.info(f"Wallet balance: {balance:.6f} {net['currency']}")

    if balance_wei == 0:
        msg = (
            f"Wallet has 0 balance on {net['name']}.\n"
            f"Get free test MATIC from: {net['faucet']}\n"
            f"Your address: {address}"
        )
        logger.error(msg)
        return BlockchainReceipt(success=False, error=msg, network=network)

    # Encode data: prefix + record_hash in hex
    # Format: "MANTIS_FACEID:" + record_hash (hex) + "|" + full JSON
    record_json = json.dumps(record.to_dict(), separators=(",", ":"))
    data_str = f"MANTIS_FACEID:{record.record_hash}|{record_json}"
    data_bytes = data_str.encode("utf-8")

    # Build transaction
    nonce = w3.eth.get_transaction_count(address)
    gas_price = w3.eth.gas_price

    tx = {
        "nonce": nonce,
        "to": address,              # send to self — zero-value record transaction
        "value": 0,
        "gas": 100_000,             # sufficient for data payload
        "gasPrice": gas_price,
        "chainId": net["chain_id"],
        "data": data_bytes,
    }

    # Estimate gas more accurately
    try:
        estimated_gas = w3.eth.estimate_gas(tx)
        tx["gas"] = int(estimated_gas * 1.2)   # 20% buffer
        logger.debug(f"Estimated gas: {estimated_gas}, using: {tx['gas']}")
    except Exception as e:
        logger.warning(f"Gas estimation failed ({e}), using default 100k")

    # Sign and send
    signed = w3.eth.account.sign_transaction(tx, pk)
    logger.info("Sending transaction to blockchain…")

    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    tx_hash_hex = tx_hash.hex()
    logger.info(f"Transaction sent: {tx_hash_hex}")

    # Wait for confirmation
    logger.info("Waiting for block confirmation…")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    block = w3.eth.get_block(receipt["blockNumber"])
    explorer_url = net["explorer"] + tx_hash_hex

    logger.info(f"Confirmed in block #{receipt['blockNumber']}")
    logger.info(f"Explorer URL: {explorer_url}")

    return BlockchainReceipt(
        success=True,
        tx_hash=tx_hash_hex,
        block_number=receipt["blockNumber"],
        block_timestamp=block["timestamp"],
        network=network,
        explorer_url=explorer_url,
        gas_used=receipt["gasUsed"],
        record=record,
    )


# ── Wallet generator ──────────────────────────────────────────────────────────

def generate_wallet() -> dict:
    """
    Generate a new Ethereum-compatible wallet (address + private key).

    Returns:
        Dict with 'address' and 'private_key'.

    WARNING: Store the private key securely. Never commit it to git.
    """
    try:
        from eth_account import Account
        Account.enable_unaudited_hdwallet_features()
        acct, mnemonic = Account.create_with_mnemonic()
        return {
            "address": acct.address,
            "private_key": acct.key.hex(),
            "mnemonic": mnemonic,
        }
    except ImportError:
        try:
            from web3 import Web3
            w3 = Web3()
            acct = w3.eth.account.create()
            return {
                "address": acct.address,
                "private_key": acct.key.hex(),
                "mnemonic": None,
            }
        except ImportError:
            return {"error": "web3 not installed. Run: pip install web3"}


# ── Verifier ──────────────────────────────────────────────────────────────────

def verify_record(tx_hash: str, network: str = DEFAULT_NETWORK) -> dict:
    """
    Retrieve and verify a record from the blockchain by transaction hash.

    Args:
        tx_hash: The transaction hash returned when the record was written.
        network: Network key from NETWORKS dict.

    Returns:
        Dict with the decoded record and verification status.
    """
    try:
        from web3 import Web3
        from web3.middleware import ExtraDataToPOAMiddleware
    except ImportError:
        return {"error": "web3 not installed"}

    net = NETWORKS.get(network)
    w3 = Web3(Web3.HTTPProvider(net["rpc"]))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

    tx = w3.eth.get_transaction(tx_hash)
    data_bytes = bytes(tx["input"])
    data_str = data_bytes.decode("utf-8", errors="replace")

    if not data_str.startswith("MANTIS_FACEID:"):
        return {"error": "Transaction data is not a MANTIS_FACEID record."}

    # Parse record hash and JSON
    content = data_str[len("MANTIS_FACEID:"):]
    record_hash, _, record_json = content.partition("|")

    try:
        record_data = json.loads(record_json)
    except json.JSONDecodeError:
        return {"error": "Could not parse record JSON from transaction data."}

    # Recompute hash to verify integrity
    record_obj = SearchRecord(
        image_hash=record_data["image_hash"],
        timestamp_utc=record_data["timestamp_utc"],
        urls_found=record_data["urls_found"],
        engines_used=record_data["engines_used"],
        total_raw_urls=record_data["total_raw_urls"],
        network=record_data.get("network", network),
    )
    recomputed_hash = record_obj.record_hash
    integrity_ok = recomputed_hash == record_hash

    block = w3.eth.get_block(tx["blockNumber"])

    return {
        "tx_hash": tx_hash,
        "block_number": tx["blockNumber"],
        "block_timestamp": block["timestamp"],
        "explorer_url": net["explorer"] + tx_hash,
        "integrity_ok": integrity_ok,
        "stored_hash": record_hash,
        "recomputed_hash": recomputed_hash,
        "record": record_data,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Blockchain record utility")
    parser.add_argument("--generate-wallet", action="store_true",
                        help="Generate a new wallet address and private key")
    parser.add_argument("--verify", type=str, metavar="TX_HASH",
                        help="Verify a record by transaction hash")
    parser.add_argument("--network", type=str, default=DEFAULT_NETWORK,
                        choices=list(NETWORKS.keys()),
                        help="Blockchain network to use")
    parser.add_argument("--test", action="store_true",
                        help="Write a test record to the blockchain")
    args = parser.parse_args()

    if args.generate_wallet:
        wallet = generate_wallet()
        if "error" in wallet:
            print(f"Error: {wallet['error']}")
        else:
            print("\n=== NEW WALLET GENERATED ===")
            print(f"Address     : {wallet['address']}")
            print(f"Private Key : {wallet['private_key']}")
            if wallet.get("mnemonic"):
                print(f"Mnemonic    : {wallet['mnemonic']}")
            print(f"\nFund this address with test MATIC:")
            print(f"  {NETWORKS[args.network]['faucet']}")
            print(f"\nThen set the env var:")
            print(f"  set WALLET_PRIVATE_KEY={wallet['private_key']}")
            print("\nWARNING: Keep your private key secret. Never share it.\n")

    elif args.verify:
        result = verify_record(args.verify, network=args.network)
        print(json.dumps(result, indent=2))

    elif args.test:
        # Write a dummy test record
        record = SearchRecord(
            image_hash="a" * 64,
            timestamp_utc=int(time.time()),
            urls_found=["https://instagram.com/test"],
            engines_used=["Yandex", "Bing"],
            total_raw_urls=42,
            network=args.network,
        )
        receipt = write_record(record, network=args.network)
        if receipt.success:
            print(f"\nRecord written successfully!")
            print(f"TX Hash     : {receipt.tx_hash}")
            print(f"Block       : #{receipt.block_number}")
            print(f"Explorer    : {receipt.explorer_url}")
        else:
            print(f"\nFailed: {receipt.error}")
