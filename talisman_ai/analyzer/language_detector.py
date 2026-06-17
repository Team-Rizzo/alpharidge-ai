"""Deterministic language detection.

Replaces the hardcoded ``language="en"`` (§3.4) that mislabeled every Italian
and Russian article as English and fed them to English-only NER models.

Uses ``langdetect`` with a fixed RNG seed so the result is reproducible across
runs and machines (the validator hard-gates ``detected_language`` as an exact
match — non-determinism here would fail consensus). Short or undetectable text
falls back to ``en`` with confidence 0.0, preserving the previous behavior
without raising.
"""

from __future__ import annotations

from typing import NamedTuple

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

# Fixed seed => deterministic tie-breaking. Must be set before any detection.
DetectorFactory.seed = 0

_DEFAULT_CODE = "en"


class LanguageResult(NamedTuple):
    code: str          # ISO 639-1 code, e.g. "en", "it", "ru"
    confidence: float  # 0.0–1.0


def detect_language(text: str) -> LanguageResult:
    """Detect the dominant language of ``text``.

    Args:
        text: Article title and/or body.

    Returns:
        LanguageResult(code, confidence). Falls back to ("en", 0.0) for empty
        or undetectable input — never raises.
    """
    if not text or not text.strip():
        return LanguageResult(_DEFAULT_CODE, 0.0)

    try:
        langs = detect_langs(text)
    except LangDetectException:
        return LanguageResult(_DEFAULT_CODE, 0.0)

    if not langs:
        return LanguageResult(_DEFAULT_CODE, 0.0)

    best = langs[0]
    return LanguageResult(best.lang, float(best.prob))
