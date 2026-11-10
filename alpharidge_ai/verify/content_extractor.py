"""Main-content extraction: site chrome (share bars, author rows, related links) removed.

  1. HTML: hidden text stripped, then trafilatura's DOM main-content extraction.
  2. Delimited plain text: jusText-style block classification; blocks with low
     function-word density, high symbol density or a list-of-proper-nouns shape are
     dropped (nav and share rows carry almost no stopwords; prose carries a steady share).

A pure function of the input and static tables, so the API and validators read the same
text from the same page.
"""

from __future__ import annotations

import re
from html import unescape
from typing import Dict, List, Optional, Tuple

from .markup import search_upto, sub_upto
from .text_cleaner import clean_text

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

_stoplist_cache: Optional[set] = None

# Text a reader never sees.
_HIDDEN = ("//*[@hidden] | //*[translate(@aria-hidden, 'TRUE', 'true')='true'] | //template"
           " | //noscript")
# 0, 0.0, .0 and the like.
_ZERO = re.compile(r"0*\.?0+(?![.\d])")
_FAR = re.compile(r"-\d{4,}(?:\.\d+)?px|-\d{3,}(?:\.\d+)?r?em")
_TINY = re.compile(r"[01](?:\.0+)?(?:px)?")
_CLIP = re.compile(r"rect\((?:\s*[01](?:px)?\s*,?){4}\)")
_CLIP_PATH = re.compile(r"inset\(\s*50%|circle\(\s*0*\.?0+(?:px|%|r?em)?\s*(?:\)|at\s)")
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_SELECTOR = re.compile(r"([a-z][a-z0-9]*)?([.#])([\w-]+)")


def _decls(style: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """A declaration block's normal and !important values."""
    out: Tuple[Dict[str, str], Dict[str, str]] = ({}, {})
    for part in style.split(";"):
        prop, sep, value = part.partition(":")
        if sep:
            value = value.lower()
            out["!important" in value][prop.strip().lower()] = \
                value.replace("!important", "").strip()
    return out


def _rules(css: str) -> List[Tuple[str, str, Optional[str], Tuple[dict, dict]]]:
    """Top-level single-class or single-id rules of a stylesheet, in order."""
    css, out, i = sub_upto(_CSS_COMMENT, "", css, "*/"), [], 0
    while True:
        j = css.find("{", i)
        if j < 0:
            return out
        depth, k = 1, j + 1
        while k < len(css) and depth:
            depth += {"{": 1, "}": -1}.get(css[k], 0)
            k += 1
        head, body, i = css[i:j].rsplit(";", 1)[-1].strip(), css[j + 1:k - 1], k
        if head.startswith("@") or "{" in body:
            continue
        decls = _decls(body)
        for sel in head.split(","):
            m = _SELECTOR.fullmatch(sel.strip().lower())
            if m:
                out.append((m.group(2), m.group(3), m.group(1), decls))


class _Styles:
    """Each element's declarations, in cascade order: matching page stylesheet rules by
    specificity then source order, then its inline style; !important values after those."""

    def __init__(self, rules):
        self.rules: Dict[Tuple[str, str], list] = {}
        for n, (kind, name, tag, decls) in enumerate(rules):
            spec = (100 if kind == "#" else 10) + (tag is not None)
            self.rules.setdefault((kind, name), []).append((spec, n, tag, decls))
        self.cache: dict = {}

    def of(self, node) -> Dict[str, str]:
        if node in self.cache:
            return self.cache[node]
        found = []
        if self.rules:
            keys = [(".", c) for c in (node.get("class") or "").lower().split()]
            keys += [("#", node.get("id").lower())] if node.get("id") else []
            found = sorted((spec, n, decls) for key in set(keys)
                           for spec, n, tag, decls in self.rules.get(key, ())
                           if tag is None or tag == node.tag)
        layers = [decls for _, _, decls in found]
        if node.get("style"):
            layers.append(_decls(node.get("style")))
        out: Dict[str, str] = {}
        for important in (0, 1):
            for decls in layers:
                out.update(decls[important])
        self.cache[node] = out
        return out


def _hides(d: Dict[str, str]) -> bool:
    if d.get("display") == "none" or d.get("visibility") in ("hidden", "collapse"):
        return True
    if _ZERO.match(d.get("opacity", "")):
        return True
    if d.get("position") in ("absolute", "fixed") and any(
            _FAR.fullmatch(d.get(k, "")) for k in ("left", "top", "right")):
        return True
    if _FAR.fullmatch(d.get("text-indent", "")) or _CLIP.match(d.get("clip", "")) \
            or _CLIP_PATH.match(d.get("clip-path", "")):
        return True
    return d.get("overflow") == "hidden" and any(
        _TINY.fullmatch(d.get(k, "")) for k in ("width", "height")) and not any(
        k.startswith("padding") and re.search(r"[1-9]", v) for k, v in d.items())


def _sized(d: Dict[str, str]) -> bool:
    size = d.get("font-size", "")
    return bool(re.match(r"[.\d]", size)) and not _ZERO.match(size)


def _hidden(node, styles: _Styles) -> bool:
    """A style that hides the element: a zero font size counts only when nothing inside
    sets its own."""
    d = styles.of(node)
    if not d:
        return False
    if _hides(d):
        return True
    return bool(_ZERO.match(d.get("font-size", ""))) and not any(
        _sized(styles.of(n)) for n in node.iterdescendants() if isinstance(n.tag, str))


_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>", re.I)


def strip_hidden(html: str) -> str:
    """The page without hidden elements, scripts and comments; empty if it cannot be
    parsed."""
    try:
        import lxml.html
        root = lxml.html.fromstring(_XML_DECL.sub("", html))
    except Exception:
        return ""
    styles = _Styles([r for sheet in root.xpath("//style")
                      if (sheet.get("media") or "all").strip().lower() in ("all", "screen")
                      for r in _rules(sheet.text_content())])
    found = root.xpath(_HIDDEN + " | //script | //style | //comment()")
    found += [n for n in root.iter() if isinstance(n.tag, str) and _hidden(n, styles)]
    for node in found:
        if node.getparent() is not None:
            node.drop_tree()
    # Namespaced inline tags (inline XBRL in filings) keep their text.
    for node in [n for n in root.iter() if isinstance(n.tag, str) and ":" in n.tag]:
        if node.getparent() is not None:
            node.drop_tag()
    return lxml.html.tostring(root, encoding="unicode")


_SPACES = re.compile(r"[^\S\n]+")
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def tidy(text: str) -> str:
    """Extracted text with HTML entities decoded, straight quotes and single spaces."""
    lines = unescape(text).translate(_QUOTES).split("\n")
    return "\n".join(_SPACES.sub(" ", line).strip() for line in lines)


def _stoplist() -> set:
    """jusText's English stoplist (pages are checked in English only)."""
    global _stoplist_cache
    if _stoplist_cache is None:
        try:
            import justext
            _stoplist_cache = {w.lower() for w in justext.get_stoplist("English")}
        except Exception:
            _stoplist_cache = set()
    return _stoplist_cache


def looks_like_html(text: str) -> bool:
    return bool(text) and bool(search_upto(_HTML_RE, text, ">"))


def _block_features(block: str) -> Dict[str, float]:
    words = _WORD_RE.findall(block)
    n = len(words)
    if n == 0:
        return {"n_words": 0, "stopword_ratio": 0.0, "symbol_ratio": 0.0, "caps_ratio": 0.0}
    stops = _stoplist()
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


class ContentExtractor:
    """Extracts article main content from HTML or plain text."""

    def extract(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        if looks_like_html(text):
            text = strip_hidden(text)
            if not text.strip():
                return ""
            html_out = self._extract_html(text)
            if html_out:
                return html_out  # trafilatura already removed chrome

        # Plain-text path: structural classification.
        text = clean_text(text)  # cheap regex pre-strip of known widgets
        blocks = _segment(text)
        if not blocks:
            return text

        labels = [self._structural_label(b) for b in blocks]
        labels = self._smooth(labels)

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
    def _structural_label(self, block: str) -> str:
        f = _block_features(block)
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
