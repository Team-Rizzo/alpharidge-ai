"""Canonical encoding and hashing. Anything signed is signed over these bytes."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

# Lists whose order carries no meaning; sorted before hashing.
_UNORDERED = ("actors", "facts")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite number")
        return 0.0 if value == 0.0 else value
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def canonical_payload(model) -> dict:
    """A model as plain JSON values: nulls dropped, unordered lists sorted."""
    data = _clean(model.model_dump(mode="json"))
    for key in _UNORDERED:
        if isinstance(data.get(key), list):
            data[key] = sorted(data[key], key=canonical_json)
    return data


def model_hash(model) -> str:
    return sha256_hex(canonical_json(canonical_payload(model)))


def canonical_body(model) -> bytes:
    """The exact bytes a model is sent and signed as."""
    return canonical_json(canonical_payload(model)).encode("utf-8")
