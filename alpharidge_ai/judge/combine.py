"""One verdict from the listed validators' posted answers: the mean of their probabilities per
question, read with the usual thresholds, and the fixed check passed on every posting
validator's fetch. The API and every validator compute it the same way."""
from __future__ import annotations

import json
from typing import Iterable, List, Optional

from alpharidge_ai.judge.prompt import verdict_of
from alpharidge_ai.ledger import request_auth

VERDICTS_PATH = "/judge/verdicts"


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1


def mean_answers(raws: Iterable[dict]) -> dict:
    """Per question, the mean of each probability every answer gives; a question or an option
    some answer leaves out or gives badly is left out. The pick is the shared one, else the
    option with the highest mean."""
    raws = [r for r in raws if isinstance(r, dict) and r]
    if not raws:
        return {}
    out = {}
    for name in sorted(set.intersection(*(set(r) for r in raws))):
        got = [r[name] for r in raws]
        if not all(isinstance(a, dict) for a in got):
            continue
        kinds = {a.get("type") for a in got}
        if kinds == {"noul"}:
            if all(_num(a.get("noul")) for a in got):
                out[name] = {"type": "noul", "noul": sum(a["noul"] for a in got) / len(got)}
            continue
        if kinds != {"choice"} or not all(isinstance(a.get("probabilities"), dict) for a in got):
            continue
        options = sorted(set.intersection(*(set(a["probabilities"]) for a in got)))
        probs = {o: sum(a["probabilities"][o] for a in got) / len(got) for o in options
                 if all(_num(a["probabilities"][o]) for a in got)}
        picks = {a.get("choice") for a in got}
        pick = picks.pop() if len(picks) == 1 else (
            max(sorted(probs), key=lambda o: probs[o]) if probs else None)
        out[name] = {"type": "choice", "choice": pick, "probabilities": probs}
    return out


def fixed_passed(bodies: Iterable[dict]) -> bool:
    return all((b.get("fixed_check") or {}).get("passed") is True for b in bodies)


def combined(bodies: List[dict], claim, filing: Optional[dict] = None) -> dict:
    """The verdict fields from the posted bodies, in the order the record keeps them. A failed
    fixed check on any posting validator's fetch rejects."""
    judged = [b.get("answers") for b in bodies if (b.get("fixed_check") or {}).get("passed")]
    out = verdict_of(mean_answers(judged), claim, filing)
    if not fixed_passed(bodies):
        out.update(outcome="reject", reason="fixed_check")
    return out


def post_signed(post: dict) -> bool:
    """The validator's signature over its posted body, as the API checked it on arrival."""
    try:
        from bittensor_wallet import Keypair
        message = request_auth.auth_message(
            "POST", VERDICTS_PATH, request_auth.body_sha256(str(post["body"]).encode("utf-8")),
            int(post["signed_at"]), str(post["nonce"]))
        return Keypair(ss58_address=str(post["validator"])).verify(
            message.encode("utf-8"), bytes.fromhex(str(post["signature"])))
    except Exception:
        return False


def bodies_of(record: dict) -> Optional[List[dict]]:
    """The posted bodies a record keeps, or None when it keeps none or one is not JSON."""
    posts = record.get("posts")
    if not isinstance(posts, list) or not posts:
        return None
    try:
        out = [json.loads(p["body"]) for p in posts]
    except (KeyError, TypeError, ValueError):
        return None
    return out if all(isinstance(b, dict) for b in out) else None
