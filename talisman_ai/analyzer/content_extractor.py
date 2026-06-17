"""Main-content extraction / boilerplate removal — multi-signal, deterministic.

The corpus review showed site chrome (share bars, "preferred source on Google",
author rows, related-links) leaking into NER as fake tickers/entities. This
module separates article content from chrome using, in order of strength:

  1. HTML path — if the input still has tags, trafilatura (best-in-class
     DOM main-content extraction).
  2. jusText-style block classification — for delimited plain text: split into
     blocks and drop those with low function-word (stopword) density, high
     link/symbol density, or list-of-proper-nouns shape (nav/share rows carry
     almost no stopwords; real prose carries a steady fraction).
  3. Embedding relevance — for grammatical-but-off-topic chrome (CTAs) that the
     structural pass keeps, drop blocks whose embedding is far from the
     article's on-topic centroid. Uses the SentenceTransformer already loaded
     by the engine (injected); skipped if none provided.

Everything is a pure function of the input plus static tables/models, so the
miner and validator stay byte-identical (validator hard-gates determinism).

Glued boilerplate with no delimiters (e.g. "…saves lives Facebook Twitter
Email") is only partially handled here; the mention-level salience gates in
entity_filter / asset_extractor catch the remainder.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .text_cleaner import clean_text

# ISO 639-1 -> jusText stoplist name.
_JUSTEXT_LANG = {
    "en": "English", "it": "Italian", "ru": "Russian", "fr": "French",
    "es": "Spanish", "de": "German", "pl": "Polish", "pt": "Portuguese",
    "nl": "Dutch", "ro": "Romanian", "tr": "Turkish",
}

_HTML_RE = re.compile(r"<[a-zA-Z/][^>]*>")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
# Symbols typical of nav/share rows and link chrome.
_SYMBOLS = set("|•·›»→/@#")

# Structural thresholds (jusText-style).
_MIN_STOPWORD_RATIO = 0.15   # below this, a multi-word block is boilerplate
_GOOD_MIN_WORDS = 8          # long + stopword-rich => confidently content
_GOOD_STOPWORD_RATIO = 0.20
_MAX_SYMBOL_RATIO = 0.20
_MAX_CAPS_RATIO = 0.65       # share/nav rows are mostly Capitalized brand tokens
# Min cosine to the on-topic anchor. Set LOW on purpose: empirically a genuine
# but topically-distinct article sentence (e.g. "Bitcoin rallied…" under a
# Fed-rates headline) scores ~0.10–0.13, while true chrome ("Follow us on
# Facebook" ~0.00, "Ricevi … su Google" ~ −0.10) sits near/below zero. A low
# bar removes only clearly-unrelated chrome and never a real tangent — recall
# is protected; residual borderline CTAs are caught downstream by the
# wordfreq + financial-cue asset gate.
_REL_THRESHOLD = 0.05

_stoplist_cache: Dict[str, set] = {}


def _stoplist(language: str) -> set:
    name = _JUSTEXT_LANG.get(language, "English")
    if name not in _stoplist_cache:
        try:
            import justext
            _stoplist_cache[name] = {w.lower() for w in justext.get_stoplist(name)}
        except Exception:
            _stoplist_cache[name] = set()
    return _stoplist_cache[name]


def looks_like_html(text: str) -> bool:
    return bool(text) and bool(_HTML_RE.search(text))


def _block_features(block: str, language: str) -> Dict[str, float]:
    words = _WORD_RE.findall(block)
    n = len(words)
    if n == 0:
        return {"n_words": 0, "stopword_ratio": 0.0, "symbol_ratio": 0.0, "caps_ratio": 0.0}
    stops = _stoplist(language)
    n_stop = sum(1 for w in words if w.lower() in stops)
    n_caps = sum(1 for w in words if w[:1].isupper())
    n_sym = sum(1 for ch in block if ch in _SYMBOLS)
    return {
        "n_words": n,
        "stopword_ratio": n_stop / n,
        "symbol_ratio": n_sym / max(1, len(block)) * 10,  # scaled per-10-chars
        "caps_ratio": n_caps / n,
    }


def _segment(text: str) -> List[str]:
    """Split into candidate blocks on newlines and sentence terminators."""
    blocks: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Further split very long lines into sentences so a trailing share row
        # that follows a period becomes its own block.
        parts = re.split(r"(?<=[.!?])\s+", line)
        blocks.extend(p.strip() for p in parts if p.strip())
    return blocks


def _cosine(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


class ContentExtractor:
    """Extracts article main content from HTML or plain text."""

    def __init__(self, embedder=None):
        self.embedder = embedder

    def extract(self, text: str, language: str = "en", title: Optional[str] = None) -> str:
        if not text or not text.strip():
            return ""

        if looks_like_html(text):
            html_out = self._extract_html(text)
            if html_out:
                return html_out  # trafilatura already removed chrome

        # Plain-text path: structural classification + optional relevance.
        text = clean_text(text)  # cheap regex pre-strip of known widgets
        blocks = _segment(text)
        if not blocks:
            return text

        labels = [self._structural_label(b, language) for b in blocks]
        labels = self._smooth(labels)
        if self.embedder is not None:
            labels = self._relevance_pass(blocks, labels, title)

        kept = [b for b, lab in zip(blocks, labels) if lab != "bad"]
        return " ".join(kept).strip() or text

    # -- HTML ---------------------------------------------------------------
    def _extract_html(self, html: str) -> Optional[str]:
        try:
            import trafilatura
            out = trafilatura.extract(html, include_comments=False,
                                      include_tables=False, no_fallback=False)
            return (out or "").strip() or None
        except Exception:
            return None

    # -- structural (jusText-style) ----------------------------------------
    def _structural_label(self, block: str, language: str) -> str:
        f = _block_features(block, language)
        n = f["n_words"]
        if n == 0:
            return "bad"
        if n <= 2:
            # Tiny fragments: keep only if stopword-bearing (likely real clause start)
            return "near" if f["stopword_ratio"] > 0 else "bad"
        if f["symbol_ratio"] > _MAX_SYMBOL_RATIO:
            return "bad"
        if f["stopword_ratio"] < _MIN_STOPWORD_RATIO and f["caps_ratio"] > _MAX_CAPS_RATIO:
            return "bad"  # nav/share row: proper nouns, no function words
        if f["stopword_ratio"] < _MIN_STOPWORD_RATIO:
            return "bad"
        if n >= _GOOD_MIN_WORDS and f["stopword_ratio"] >= _GOOD_STOPWORD_RATIO:
            return "good"
        return "near"

    def _smooth(self, labels: List[str]) -> List[str]:
        """Context smoothing: a 'near' block becomes good/bad like its nearest
        confident neighbor (jusText's second pass)."""
        out = list(labels)
        for i, lab in enumerate(labels):
            if lab != "near":
                continue
            prev = next((labels[j] for j in range(i - 1, -1, -1) if labels[j] != "near"), None)
            nxt = next((labels[j] for j in range(i + 1, len(labels)) if labels[j] != "near"), None)
            out[i] = "good" if "good" in (prev, nxt) else "bad" if (prev or nxt) == "bad" else "good"
        return out

    # -- embedding relevance ------------------------------------------------
    def _relevance_pass(self, blocks: List[str], labels: List[str],
                        title: Optional[str] = None) -> List[str]:
        good_idx = [i for i, l in enumerate(labels) if l == "good"]
        if not good_idx:
            return labels
        vecs = {i: self._enc(blocks[i]) for i in good_idx}
        tvec = self._enc(title) if title and title.strip() else None
        if tvec is None and len(good_idx) < 2:
            return labels  # nothing authoritative to judge against

        out = list(labels)
        for i in good_idx:
            # Leave-one-out: the topic anchor is the title blended with the mean
            # of the OTHER content blocks, so a block can't vouch for itself.
            others = [vecs[j] for j in good_idx if j != i]
            parts = []
            if others:
                dim = len(others[0])
                parts.append([sum(v[k] for v in others) / len(others) for k in range(dim)])
            if tvec is not None:
                parts.append(tvec)
            anchor = [sum(p[k] for p in parts) / len(parts) for k in range(len(parts[0]))]
            if _cosine(vecs[i], anchor) < _REL_THRESHOLD:
                out[i] = "bad"  # grammatical but off-topic chrome (CTA)
        return out

    def _enc(self, text: str) -> List[float]:
        v = self.embedder.encode(text, normalize_embeddings=True)
        return list(v)

    def _centroid(self, blocks: List[str]) -> List[float]:
        vecs = [self._enc(b) for b in blocks]
        dim = len(vecs[0])
        return [sum(v[k] for v in vecs) / len(vecs) for k in range(dim)]
