"""The deterministic floor: what every article is checked against, with no model.

Pure Python and regex. Every validator must reach the same verdict on the same inputs,
so nothing here samples, times out, or calls a model. Results feed both the audit and
the volume credit: an article that fails the floor earns nothing.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from rapidfuzz import fuzz

# Grounding tolerances.
VALUE_REL_TOLERANCE = 0.005
QUOTE_MATCH_RATIO = 0.85
QUOTE_WINDOW_SLACK = 20
QUOTE_MAX_CHARS = 1000
QUOTE_MAX_OVERLAP = 0.50
SPAN_MIN_CHARS = 12
SPAN_MIN_ADJACENT_WORDS = 3
TEXT_STATS_TOLERANCE = 0.01

_MAGNITUDES = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mn": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "t": 1e12, "tn": 1e12, "trillion": 1e12,
}

_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}
_CURRENCY_CODES = {"usd", "eur", "gbp", "jpy", "cny", "inr", "chf", "cad", "aud", "krw"}
_CURRENCY_PROXIMITY = 12

_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-", "‑": "-",
    " ": " ", " ": " ", " ": " ", "​": "",
    "…": "...",
}

_NUMBER_RE = re.compile(r"[-+]?\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_TRAILING_PUNCT = ".,;:!?'\"-)]}"


# ---- normalisation ----------------------------------------------------------------

@dataclass(frozen=True)
class Normalized:
    """Normalised text plus the offset map back into the original."""
    text: str
    offsets: Tuple[int, ...]

    def to_original(self, start: int, end: int) -> Tuple[int, int]:
        """Map a span in normalised space back to original character offsets."""
        if not self.offsets:
            return (0, 0)
        start = max(0, min(start, len(self.offsets) - 1))
        end = max(start + 1, min(end, len(self.offsets)))
        first = self.offsets[start]
        last = self.offsets[end - 1] + 1
        return (first, last)


def normalize(text: Optional[str]) -> Normalized:
    """NFKC, punctuation folding, lowercase, whitespace collapse, offsets preserved."""
    if not text:
        return Normalized("", ())

    out: List[str] = []
    offs: List[int] = []
    prev_space = True  # strips leading whitespace

    for i, ch in enumerate(text):
        mapped = _PUNCT_MAP.get(ch)
        if mapped is None:
            mapped = unicodedata.normalize("NFKC", ch)
        if not mapped:
            continue
        mapped = mapped.lower()
        for c in mapped:
            if c.isspace():
                if prev_space:
                    continue
                out.append(" ")
                offs.append(i)
                prev_space = True
            else:
                out.append(c)
                offs.append(i)
                prev_space = False

    while out and out[-1] == " ":
        out.pop()
        offs.pop()
    return Normalized("".join(out), tuple(offs))


def _strip_edges(s: str) -> str:
    return s.strip().strip(_TRAILING_PUNCT).strip()


# ---- number parsing ---------------------------------------------------------------

@dataclass(frozen=True)
class ParsedNumber:
    value: float
    unit: str          # "pct" | "currency:XXX" | "count" | "none"
    offset: int


def _magnitude_after(text: str, end: int) -> Tuple[float, int]:
    """Magnitude word within the next two tokens, and where it ends."""
    tail = text[end:end + 32]
    tokens = list(_WORD_RE.finditer(tail))[:2]
    for tok in tokens:
        word = tok.group(0)
        if word in _MAGNITUDES:
            return _MAGNITUDES[word], end + tok.end()
    return 1.0, end


def _currency_near(text: str, start: int, end: int) -> Optional[str]:
    left = text[max(0, start - _CURRENCY_PROXIMITY):start]
    right = text[end:end + _CURRENCY_PROXIMITY]
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in left or sym in right:
            return code
    for word in _WORD_RE.findall(left) + _WORD_RE.findall(right):
        if word in _CURRENCY_CODES:
            return word.upper()
    return None


def parse_numbers(normalized_text: str) -> List[ParsedNumber]:
    """Every numeric token in the text, with its magnitude, unit class and offset."""
    found: List[ParsedNumber] = []
    for m in _NUMBER_RE.finditer(normalized_text):
        raw = m.group(0).replace(",", "").replace(" ", "")
        try:
            value = float(raw)
        except ValueError:
            continue

        mag, mag_end = _magnitude_after(normalized_text, m.end())
        value *= mag

        after = normalized_text[mag_end:mag_end + 2]
        if after.startswith("%") or after.strip().startswith("%"):
            unit = "pct"
        else:
            code = _currency_near(normalized_text, m.start(), mag_end)
            unit = f"currency:{code}" if code else "count"

        found.append(ParsedNumber(value=value, unit=unit, offset=m.start()))
    return found


def _unit_class(unit: Optional[str]) -> str:
    u = (unit or "").strip().lower()
    if not u:
        return "none"
    if u in ("%", "pct", "percent", "percentage", "bps", "basis_points"):
        return "pct"
    if u in _CURRENCY_CODES:
        return f"currency:{u.upper()}"
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in u:
            return f"currency:{code}"
    return "count"


def _units_compatible(claim_unit: str, text_unit: str) -> bool:
    """A percentage needs a percent token, a currency needs a currency marker."""
    if claim_unit == "pct":
        return text_unit == "pct"
    if claim_unit.startswith("currency:"):
        if not text_unit.startswith("currency:"):
            return False
        return claim_unit == text_unit or text_unit == "currency:"
    return text_unit in ("count", "none") or not text_unit.startswith("currency:")


def _stated_precision(value: float) -> float:
    """Half a unit in the claim's own last significant place."""
    s = repr(float(value))
    if "e" in s or "E" in s:
        return abs(value) * VALUE_REL_TOLERANCE
    if "." in s:
        decimals = len(s.split(".")[1].rstrip("0"))
        return 0.5 * (10.0 ** -decimals) if decimals else 0.5
    trailing = len(s) - len(s.rstrip("0"))
    return 0.5 * (10.0 ** trailing)


def value_grounded(claim_value: float, text_value: float) -> bool:
    """A claim matches a text number within tolerance, or at its own stated precision."""
    if text_value == claim_value:
        return True
    rel = abs(claim_value - text_value) / max(abs(text_value), 1e-9)
    if rel <= VALUE_REL_TOLERANCE:
        return True
    return abs(claim_value - text_value) <= _stated_precision(claim_value)


# ---- claims -----------------------------------------------------------------------

def ground_claim(claim, numbers: Sequence[ParsedNumber]) -> bool:
    """True when some parsed text number matches the claim in value and unit class."""
    try:
        value = float(getattr(claim, "value", None))
    except (TypeError, ValueError):
        return False
    unit = _unit_class(getattr(claim, "unit", None))
    for n in numbers:
        if _units_compatible(unit, n.unit) and value_grounded(value, n.value):
            return True
    return False


# ---- quotes -----------------------------------------------------------------------

@dataclass(frozen=True)
class AlignedQuote:
    start: int          # original-text offsets of the maximal aligned match
    end: int
    ratio: float


def align_quote(article: Normalized, quote_text: str,
                start_offset: Optional[int], end_offset: Optional[int]) -> Optional[AlignedQuote]:
    """Locate a quote in the article and return its maximal match in original offsets.

    The miner's offsets say where to look, never what the answer is: the returned span
    is the maximal match this validator finds, so two miners quoting the same sentence
    key to the same span.
    """
    if not quote_text or len(quote_text) > QUOTE_MAX_CHARS or not article.text:
        return None

    needle = normalize(quote_text).text
    if not needle:
        return None

    search_lo, search_hi = 0, len(article.text)
    if start_offset is not None and end_offset is not None:
        try:
            lo = int(start_offset)
            hi = int(end_offset)
        except (TypeError, ValueError):
            lo = hi = None
        if lo is not None and hi is not None and hi > lo:
            n_lo = _original_to_normalized(article, lo)
            n_hi = _original_to_normalized(article, hi)
            search_lo = max(0, n_lo - QUOTE_WINDOW_SLACK)
            search_hi = min(len(article.text), n_hi + QUOTE_WINDOW_SLACK)
            if search_hi - search_lo < len(needle):
                search_hi = min(len(article.text), search_lo + len(needle) + QUOTE_WINDOW_SLACK)

    window = article.text[search_lo:search_hi]
    if not window:
        return None

    exact = window.find(needle)
    if exact >= 0:
        s = search_lo + exact
        o_start, o_end = article.to_original(s, s + len(needle))
        return AlignedQuote(o_start, o_end, 1.0)

    if len(needle) > len(window):
        return None

    hit = fuzz.partial_ratio_alignment(needle, window)
    if hit is None or hit.score / 100.0 < QUOTE_MATCH_RATIO:
        return None

    s = search_lo + hit.dest_start
    e = search_lo + hit.dest_end
    if e <= s:
        return None
    o_start, o_end = article.to_original(s, e)
    return AlignedQuote(o_start, o_end, hit.score / 100.0)


def _original_to_normalized(article: Normalized, original_offset: int) -> int:
    """First normalised index at or after an original-text offset."""
    lo, hi = 0, len(article.offsets)
    while lo < hi:
        mid = (lo + hi) // 2
        if article.offsets[mid] < original_offset:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _overlap_fraction(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if hi <= lo:
        return 0.0
    shorter = min(a[1] - a[0], b[1] - b[0])
    return (hi - lo) / max(shorter, 1)


# ---- evidence spans ---------------------------------------------------------------

def span_supported(article: Normalized, span: Optional[str],
                   surface_form: Optional[str] = None) -> bool:
    """An evidence span must be in the article and carry enough context to be evidence.

    A bare ticker repeated as its own evidence proves nothing, so a short span has to
    bring adjacent words with it.
    """
    if not span:
        return False
    needle = normalize(span).text
    if not needle or needle not in article.text:
        return False
    if len(needle) >= SPAN_MIN_CHARS:
        return True

    words = _WORD_RE.findall(needle)
    surface = normalize(surface_form).text if surface_form else ""
    if surface:
        extra = [w for w in words if w not in _WORD_RE.findall(surface)]
        return len(extra) >= SPAN_MIN_ADJACENT_WORDS
    return len(words) >= SPAN_MIN_ADJACENT_WORDS + 1


# ---- proof of read ----------------------------------------------------------------

def content_hash(text: Optional[str]) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def text_stats(text: Optional[str]) -> Dict[str, int]:
    t = text or ""
    return {
        "chars": len(t),
        "words": len(_WORD_RE.findall(t)),
        "sentences": len([s for s in re.split(r"[.!?]+", t) if s.strip()]),
    }


def stats_within_tolerance(claimed: Optional[Dict[str, int]],
                           actual: Dict[str, int]) -> bool:
    if not claimed:
        return False
    for key, real in actual.items():
        if key not in claimed:
            return False
        try:
            got = float(claimed[key])
        except (TypeError, ValueError):
            return False
        if abs(got - real) > max(1.0, real * TEXT_STATS_TOLERANCE):
            return False
    return True


# ---- the floor --------------------------------------------------------------------

@dataclass
class FloorResult:
    floor_pass: bool
    reason: str = ""
    grounded: Set[int] = field(default_factory=set)
    ungrounded: Set[int] = field(default_factory=set)
    aligned_quotes: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    rejected_quotes: Set[int] = field(default_factory=set)
    span_failures: Set[int] = field(default_factory=set)

    @property
    def quote_keys(self) -> Set[Tuple[int, int]]:
        return set(self.aligned_quotes.values())


def evaluate(intel, article_text: str, *,
             claimed_hash: Optional[str] = None,
             claimed_stats: Optional[Dict[str, int]] = None,
             claim_cap: int = 40) -> FloorResult:
    """Run the floor over one submission. Deterministic, no model, no network.

    Proof of read fails the article outright. Everything else is per-item: an
    ungrounded claim scores nothing, but it does not void the article.
    """
    article = normalize(article_text)
    if not article.text:
        return FloorResult(False, "empty_article")

    if claimed_hash is not None and claimed_hash != content_hash(article_text):
        return FloorResult(False, "content_hash_mismatch")
    if claimed_stats is not None and not stats_within_tolerance(claimed_stats,
                                                               text_stats(article_text)):
        return FloorResult(False, "text_stats_mismatch")

    result = FloorResult(True, "ok")
    numbers = parse_numbers(article.text)

    for i, claim in enumerate(list(getattr(intel, "numeric_claims", None) or [])[:claim_cap]):
        (result.grounded if ground_claim(claim, numbers) else result.ungrounded).add(i)

    accepted: List[Tuple[int, int]] = []
    for i, quote in enumerate(list(getattr(intel, "quotes", None) or [])[:claim_cap]):
        hit = align_quote(article, getattr(quote, "text", None),
                          getattr(quote, "start_offset", None),
                          getattr(quote, "end_offset", None))
        if hit is None:
            result.rejected_quotes.add(i)
            continue
        span = (hit.start, hit.end)
        if any(_overlap_fraction(span, prev) > QUOTE_MAX_OVERLAP for prev in accepted):
            result.rejected_quotes.add(i)
            continue
        accepted.append(span)
        result.aligned_quotes[i] = span

    for i, asset in enumerate(getattr(intel, "assets", None) or []):
        evidence = getattr(asset, "evidence_span", None) or getattr(asset, "evidence", None)
        if evidence is None:
            continue
        surface = getattr(asset, "symbol", None) or getattr(asset, "name", None)
        if not span_supported(article, evidence, surface):
            result.span_failures.add(i)

    return result
