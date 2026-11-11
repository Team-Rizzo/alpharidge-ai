"""Keyed audit selection.

Selection is decided by a per-validator secret and keys on the item id alone:
never on order, batch position, timestamp, or who holds the item.
"""
from __future__ import annotations

import hashlib
import hmac

_DENOM = float(1 << 64)


def keyed_fraction(secret: bytes, item_id, domain: str) -> float:
    """A uniform value in [0,1) from the key and the item id."""
    msg = f"{domain}:{int(item_id)}".encode("utf-8")
    digest = hmac.new(secret, msg, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / _DENOM


class Selector:
    """Holds this validator's audit key. The key stays in memory and is never logged."""

    def __init__(self, secret: bytes):
        if not secret:
            raise ValueError("audit key must not be empty")
        self._secret = secret if isinstance(secret, bytes) else str(secret).encode("utf-8")

    def __repr__(self) -> str:
        return "<Selector key=redacted>"

    def fraction(self, item_id, domain: str = "select") -> float:
        return keyed_fraction(self._secret, item_id, domain)
