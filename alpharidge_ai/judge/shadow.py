"""A judge wording tried in shadow: for its types, a validator also asks the judge with the
wording in `data/types_shadow.json` and logs both outcomes. Only the live answers are posted."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from alpharidge_ai.judge import prompt
from alpharidge_ai.schema.types import TYPES_PATH, load_types

SHADOW_PATH = TYPES_PATH.with_name("types_shadow.json")
SHADOW_VERSION = "v18"


@lru_cache(maxsize=1)
def table() -> dict:
    return load_types(overlay=SHADOW_PATH)


@lru_cache(maxsize=1)
def types() -> frozenset:
    with open(SHADOW_PATH, encoding="utf-8") as f:
        return frozenset(json.load(f))


def wanted(claim) -> bool:
    return claim is not None and str(claim.event_type.value) in types()


def outcome(answers: dict, claim, shadow: bool = False) -> Optional[str]:
    if not answers:
        return None
    if shadow:
        with prompt.wording(table()):
            return prompt.verdict_of(answers, claim).get("outcome")
    return prompt.verdict_of(answers, claim).get("outcome")


def ask(judge, claim, text: str, published: Optional[str], kinds=None) -> dict:
    """The judge's answers to the shadow wording (run in a worker thread)."""
    with prompt.wording(table()):
        return judge.judge_page(claim, text, published, kinds)


def log(path: Path, row: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        pass
