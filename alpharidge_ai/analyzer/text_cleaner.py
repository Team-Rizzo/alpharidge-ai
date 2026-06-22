"""Deterministic boilerplate stripping, run before NER / asset extraction.

The miner-data-quality review (§3.1, §3.3) found that social-follow widgets
("Follow us on Google / Facebook"), photo credits ("Getty Images", "Photo by
…"), and share/subscribe chrome leak into the extractors as fake tickers and
entities. Those artifacts are not article content, so we remove them up front.

Every transform is a pure regex — same input always yields the same output, so
the miner and validator stay byte-identical (the validator hard-gates
determinism). A conservative guard reverts to the HTML-stripped baseline if a
rule ever removes nearly everything, so cleaning can never annihilate a real
article.
"""

from __future__ import annotations

import re

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")

# Boilerplate phrases removed up to the next sentence terminator / newline.
_BOILERPLATE = [
    # "Follow/Like/Connect with us on Google and Facebook for updates"
    re.compile(r"(?im)\b(?:follow|like|connect with|find)\s+(?:us|me)\s+on\b[^.!?\n]*[.!?]?"),
    # "Share (this) on Twitter, Facebook, LinkedIn"
    re.compile(r"(?im)\bshare\s+(?:this\s+)?on\b[^.!?\n]*[.!?]?"),
    # Parenthetical / colon photo & image credits: "(Photo: Getty Images)",
    # "Image: AFP", "Credit: Reuters"
    re.compile(r"(?im)\(?\s*(?:photo|image|picture|credit|getty)\s*[:/][^)\n.!?]*\)?[.!?]?"),
    # "Photo by Jane Doe / Reuters"
    re.compile(r"(?im)\bphoto by\b[^.!?\n]*[.!?]?"),
    # Bare wire-service credit "Getty Images" / "REUTERS/John Smith"
    re.compile(r"(?im)\bgetty images\b"),
    re.compile(r"(?im)\b(?:reuters|afp|epa|ap)\s*/\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"),
    # Subscribe / cookie / advertising chrome
    re.compile(r"(?im)\b(?:subscribe to our newsletter|sign up for our[^.!?\n]*|"
               r"accept (?:all )?cookies|advertisement|read more)\b[^.!?\n]*[.!?]?"),
]

# If a real article is long and cleaning removed most of it, a rule misfired —
# revert to the HTML-stripped baseline instead of shipping a gutted article.
_GUARD_MIN_BASELINE_LEN = 400
_GUARD_MIN_KEEP_RATIO = 0.4


def clean_text(text: str) -> str:
    """Return article text with boilerplate removed.

    Args:
        text: Raw title or body text (may contain HTML / widgets / credits).

    Returns:
        Cleaned, whitespace-normalized text. Never empty unless the input was.
    """
    if not text:
        return ""

    baseline = _WHITESPACE.sub(" ", _HTML_TAG.sub(" ", text)).strip()

    cleaned = baseline
    for pat in _BOILERPLATE:
        cleaned = pat.sub(" ", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()

    # Guard: never annihilate, and never gut a long real article.
    if not cleaned:
        return baseline
    if len(baseline) >= _GUARD_MIN_BASELINE_LEN and len(cleaned) < _GUARD_MIN_KEEP_RATIO * len(baseline):
        return baseline

    return cleaned
