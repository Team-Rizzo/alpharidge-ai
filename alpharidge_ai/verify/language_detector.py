"""Deterministic language detection for the page check.

Uses ``langdetect`` with a fixed RNG seed so the API and validators get the same result.
Short or undetectable text falls back to ``en`` with confidence 0.0.
"""

from __future__ import annotations

import threading
from typing import NamedTuple

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

# Fixed seed => deterministic tie-breaking. Must be set before any detection.
DetectorFactory.seed = 0

_DEFAULT_CODE = "en"
# langdetect loads its language profiles on first use, and a detection racing that load in
# another thread can come out wrong; one detection at a time.
_LOCK = threading.Lock()


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
        with _LOCK:
            langs = detect_langs(text)
    except LangDetectException:
        return LanguageResult(_DEFAULT_CODE, 0.0)

    if not langs:
        return LanguageResult(_DEFAULT_CODE, 0.0)

    best = langs[0]
    return LanguageResult(best.lang, float(best.prob))
