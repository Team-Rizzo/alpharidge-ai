"""Signed API request headers; the API checks them with the same message format."""
from __future__ import annotations

import hashlib
import secrets
import time
from typing import Dict, Optional
from urllib.parse import unquote, urlsplit

AUTH_PREFIX = "alpharidge-ai-auth:v2"


def body_sha256(body: Optional[bytes]) -> str:
    return hashlib.sha256(body or b"").hexdigest()


def request_target(url: str) -> str:
    """Decoded path plus raw query string, as the server sees them."""
    parts = urlsplit(str(url))
    target = unquote(parts.path) or "/"
    return f"{target}?{parts.query}" if parts.query else target


def auth_message(method: str, target: str, body_hash: str, timestamp: int, nonce: str) -> str:
    return "|".join((AUTH_PREFIX, method.upper(), target, body_hash, str(int(timestamp)), nonce))


def signed_headers(keypair, method: str, url: str, body: Optional[bytes] = b"",
                   timestamp: Optional[float] = None, nonce: Optional[str] = None) -> Dict[str, str]:
    ts = int(time.time() if timestamp is None else timestamp)
    nonce = nonce or secrets.token_hex(16)
    body_hash = body_sha256(body)
    message = auth_message(method, request_target(url), body_hash, ts, nonce)
    return {
        "X-Auth-SS58Address": keypair.ss58_address,
        "X-Auth-Signature": keypair.sign(message.encode("utf-8")).hex(),
        "X-Auth-Message": message,
        "X-Auth-Timestamp": str(ts),
        "X-Auth-Nonce": nonce,
        "X-Auth-Body-SHA256": body_hash,
    }
