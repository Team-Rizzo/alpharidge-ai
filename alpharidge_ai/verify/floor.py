"""Deterministic grounding: does a page state this figure.

Pure Python and regex. The API and every validator must reach the same verdict on the
same inputs, so nothing here samples, times out, or calls a model.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Grounding tolerance.
VALUE_REL_TOLERANCE = 0.005

_MAGNITUDES = {
    "k": 1e3, "thousand": 1e3, "thousands": 1e3,
    "m": 1e6, "mn": 1e6, "million": 1e6, "millions": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9, "billions": 1e9,
    "t": 1e12, "tn": 1e12, "trillion": 1e12, "trillions": 1e12,
    # Indian numbering, ordinary in financial copy and absent from the short scale.
    "lakh": 1e5, "lakhs": 1e5, "crore": 1e7, "crores": 1e7,
}

BASIS_POINT = 0.01  # of a percentage point

_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}
_CURRENCY_CODES = {"usd", "eur", "gbp", "jpy", "cny", "inr", "chf", "cad", "aud", "krw",
                   "nzd", "hkd", "sgd", "twd", "mxn", "brl"}
# Dollars of other countries: "C$1.26 billion", "A$40 million".
_DOLLARS = {"us": "USD", "c": "CAD", "ca": "CAD", "a": "AUD", "au": "AUD", "nz": "NZD",
            "hk": "HKD", "s": "SGD", "sg": "SGD", "nt": "TWD", "mx": "MXN", "r": "BRL"}
_DOLLAR_PREFIX = re.compile(r"(?<![a-z])(" + "|".join(sorted(_DOLLARS, key=len, reverse=True))
                            + r")$")
_CURRENCY_PROXIMITY = 12

_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-", "‑": "-",
    " ": " ", " ": " ", " ": " ", "​": "",
    "…": "...", "\u00ad": "",
}

# A numeric run: digits with any mix of grouping and decimal marks. The marks are
# interpreted afterwards, because their meaning depends on the writer's locale.
_NUMBER_RE = re.compile(r"(?:(?<![\w%])[-+])?\d[\d.,\u00a0 ]*\d|(?:(?<![\w%])[-+])?\d")

# Number words.
_WORD_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_WORD_SCALES = {"hundred": 100, "thousand": 1e3, "million": 1e6, "billion": 1e9,
                "trillion": 1e12}
_JOINERS = {"and", "-"}

_CURRENCY_WORDS = {
    "dollar": "USD", "dollars": "USD", "usd": "USD",
    "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "pound": "GBP", "pounds": "GBP", "sterling": "GBP",
    "yen": "JPY", "yuan": "CNY", "renminbi": "CNY", "won": "KRW",
    "real": "BRL", "reais": "BRL", "franc": "CHF", "francs": "CHF",
}
_WORD_RE = re.compile(r"\w+", re.UNICODE)


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


def normalize(text: Optional[str], lines: bool = False) -> Normalized:
    """NFKC, punctuation folding, lowercase, whitespace collapse, offsets preserved. With
    `lines`, a run of whitespace with a line break in it collapses to a line break."""
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
                brk = lines and c in "\n\r"
                if prev_space:
                    if brk and out:
                        out[-1] = "\n"
                    continue
                out.append("\n" if brk else " ")
                offs.append(i)
                prev_space = True
            else:
                out.append(c)
                offs.append(i)
                prev_space = False

    while out and out[-1] in " \n":
        out.pop()
        offs.pop()
    return Normalized("".join(out), tuple(offs))


# ---- number parsing ---------------------------------------------------------------

@dataclass(frozen=True)
class ParsedNumber:
    value: float
    unit: str          # "pct" | "currency:XXX" | "count" | "none"
    offset: int
    # False when the value was inferred rather than read: the second reading of an
    # ambiguous grouping mark, or a table's scale applied.
    literal: bool = True
    # A step size: basis points or percentage points ("cut by 25 basis points").
    step: bool = False


def _magnitude_after(text: str, end: int) -> Tuple[float, int]:
    """Magnitude word within the next two tokens on the same line, and where it ends. A
    one-letter scale counts only as the first token, so "8 a.m." is not 8 million."""
    tail = text[end:end + 32].split("\n", 1)[0]
    tokens = list(_WORD_RE.finditer(tail))[:2]
    for i, tok in enumerate(tokens):
        word = tok.group(0)
        if len(word) == 1 and i:
            continue
        if word in _MAGNITUDES:
            return _MAGNITUDES[word], end + tok.end()
    return 1.0, end


def _currency_near(text: str, start: int, end: int) -> Optional[str]:
    """The currency marked nearest the figure: "€7.6 billion ($8.5 billion)" is in euros,
    "C$1.26 billion" and "$1.26 billion CAD" in Canadian dollars."""
    left = text[max(0, start - _CURRENCY_PROXIMITY):start]
    right = text[end:end + _CURRENCY_PROXIMITY]
    found = []  # (distance, side, code, written as a word)
    for side, part in ((0, left), (1, right)):
        for i, ch in enumerate(part):
            code = _CURRENCY_SYMBOLS.get(ch)
            if ch == "$":
                prefix = _DOLLAR_PREFIX.search(part, 0, i)
                code = _DOLLARS[prefix.group(1)] if prefix else "$"
            if code:
                found.append((len(part) - i - 1 if side == 0 else i, side, code, False))
        for m in _WORD_RE.finditer(part):
            w = m.group(0)
            code = w.upper() if w in _CURRENCY_CODES else _CURRENCY_WORDS.get(w)
            if code:
                found.append((len(part) - m.end() if side == 0 else m.start(), side, code, True))
    if not found:
        return None
    code = min(found)[2]
    if code == "$":
        after = min((f for f in found if f[1] == 1), default=None)
        dollar = after is not None and after[3] and after[2] in _DOLLARS.values()
        code = after[2] if dollar else "USD"
    return code


def _readings(token: str) -> List[float]:
    """Every plausible value for a numeric run.

    `1,203` is one thousand two hundred and three to an English writer and one point
    two zero three to a French one, and the text does not say which, so both are
    emitted. Readings beyond the first are marked inferred.
    """
    raw = token.strip().replace("\u00a0", " ").replace(" ", "")
    sign = -1.0 if raw.startswith("-") else 1.0
    raw = raw.lstrip("+-")
    if not raw or not raw[0].isdigit():
        return []

    dots, commas = raw.count("."), raw.count(",")
    out: List[float] = []

    def add(value: Optional[float]) -> None:
        if value is not None and value == value and value not in out:
            out.append(value)

    def grouped(text: str, group: str, decimal: str) -> Optional[float]:
        """Read `text` treating `group` as the thousands mark and `decimal` as the point."""
        body, _, fraction = (text.rpartition(decimal) if decimal and decimal in text
                             else (text, "", ""))
        body = body or text
        if decimal and decimal in text and (not fraction.isdigit() or not body):
            return None
        parts = body.split(group) if group else [body]
        if len(parts) > 1:
            if not parts[0] or len(parts[0]) > 3:
                return None
            if any(len(p) != 3 or not p.isdigit() for p in parts[1:]):
                return None
        if not "".join(parts).isdigit():
            return None
        try:
            return float("".join(parts) + ("." + fraction if fraction else ""))
        except ValueError:
            return None

    if dots and commas:
        # The rightmost mark is the decimal point; the other groups.
        if raw.rfind(".") > raw.rfind(","):
            add(grouped(raw, ",", "."))
        else:
            add(grouped(raw, ".", ","))
    elif commas:
        add(grouped(raw, ",", ""))          # grouping: 1,203 -> 1203
        add(grouped(raw, "", ","))          # decimal:  17,99 -> 17.99
    elif dots:
        add(grouped(raw, "", "."))          # decimal:  17.99 -> 17.99
        add(grouped(raw, ".", ""))          # grouping: 1.203 -> 1203
    else:
        try:
            add(float(raw))
        except ValueError:
            return []

    return [sign * v for v in out]


def _word_numbers(normalized_text: str) -> List[Tuple[float, int, int]]:
    """Spelled-out numbers, as (value, start, end) over the normalised text."""
    tokens = list(_WORD_RE.finditer(normalized_text))
    found: List[Tuple[float, int, int]] = []
    i = 0
    while i < len(tokens):
        word = tokens[i].group(0)
        if word not in _WORD_UNITS and word not in _WORD_SCALES:
            i += 1
            continue

        total = current = 0.0
        used = False
        saw_unit = False
        start = tokens[i].start()
        end = tokens[i].end()
        j = i
        while j < len(tokens):
            w = tokens[j].group(0)
            if w in _WORD_UNITS:
                current += _WORD_UNITS[w]
                used = saw_unit = True
            elif w in _WORD_SCALES:
                # A scale word with nothing in front of it is a figure of speech, not a
                # number. Reading "billion" alone as 1e9 would ground that exact claim
                # against any article that merely uses the word.
                if not saw_unit:
                    break
                scale = _WORD_SCALES[w]
                if scale == 100:
                    current *= 100
                else:
                    total += current * scale
                    current = 0.0
                used = True
            elif w in _JOINERS and used:
                pass                         # "two hundred and five"
            else:
                break
            end = tokens[j].end()
            j += 1

        value = total + current
        if used and value:
            found.append((value, start, end))
        i = max(j, i + 1)
    return found


# "4-1/2", "4 1/4", "1/4": the halves, quarters and eighths rate statements use.
_FRACTION_RE = re.compile(
    r"(?<![\d/.])(?:(?P<whole>\d{1,3})[-\s])?(?P<num>[1-7])/(?P<den>[248])(?![\d/])")
_PERCENT_AFTER = re.compile(r"\s?(?:%|(?:percent(?:age points?)?|per cent|pct|pp)(?!\w))")
_BPS_AFTER = re.compile(r"\s?(?:basis[-\s]points?|bps|bp)(?!\w)")
_PP_AFTER = re.compile(r"\s?(?:percentage[-\s]points?|pp)(?!\w)")
# "a quarter point", "half a percentage point", "three-quarters of a point".
_STEP_WORDS = re.compile(
    r"(?<!\w)(?P<n>quarter|half|three[-\s]quarters)(?:[-\s]+(?:of\s+)?an?)?[-\s]+"
    r"(?:percentage[-\s]+)?points?(?!\w)")
_STEP_SIZES = {"quarter": 0.25, "half": 0.5}
# Cents read as dollars only for a per-share figure: "26 cents a share", "EPS of 26
# cents", "(26) cents".
_CENTS_AFTER = re.compile(r"\)?\s?(?:cents?|¢)(?!\w)")
_PER_SHARE_AFTER = re.compile(r"\s+(?:a|per|each)\s+(?:diluted\s+|basic\s+)?share(?!\w)")
_PER_SHARE_BEFORE = re.compile(r"(?:per\s+(?:diluted\s+|basic\s+)?share|eps)\s*"
                               r"(?::|of|was|were|is|at)?\s*\(?$")
# A table's stated scale: "(in millions, except per share amounts)", "($ in thousands)",
# "in millions, except per share data".
_SCALE_HEAD = re.compile(r"(?:\(\s*(?:(?P<sym>[$€£¥])\s*|(?:dollars|amounts)\s+)?|(?<!\w)(?=in\s+"
                         r"(?:thousands|millions|billions),?\s+except\s))in\s+"
                         r"(?P<mag>thousands|millions|billions)(?:\s+of\s+(?P<cur>\w+))?(?!\w)")
_SCALE_REACH = 3000


def _scale_regions(text: str) -> List[Tuple[int, int, float, Optional[str]]]:
    heads = list(_SCALE_HEAD.finditer(text))
    out = []
    for i, m in enumerate(heads):
        nxt = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        cur = (_CURRENCY_SYMBOLS.get(m.group("sym") or "")
               or _CURRENCY_WORDS.get(m.group("cur") or ""))
        end = min(m.end() + _SCALE_REACH, nxt)
        out.append((m.end(), end, _MAGNITUDES[m.group("mag")], cur))
    return out


def parse_numbers(normalized_text: str) -> List[ParsedNumber]:
    """Every numeric token in the text, with its magnitude, unit class and offset."""
    found: List[ParsedNumber] = []
    text = normalized_text
    regions = _scale_regions(text)

    def emit(value: float, start: int, end: int, literal: bool = True) -> None:
        mag, mag_end = _magnitude_after(text, end)
        value *= mag
        step = bool(_PP_AFTER.match(text, mag_end))
        paren = start > 0 and text[start - 1] == "(" and text.startswith(")", end)
        if _PERCENT_AFTER.match(text, mag_end):
            unit = "pct"
        elif _BPS_AFTER.match(text, mag_end):
            value, unit, step = value * BASIS_POINT, "pct", True
        else:
            code = _currency_near(text, start, mag_end)
            unit = f"currency:{code}" if code else "count"
        found.append(ParsedNumber(value=value, unit=unit, offset=start, literal=literal,
                                  step=step))
        if mag == 1.0 and unit != "pct":
            region = next((r for r in regions if r[0] <= start < r[1]), None)
            if region:
                own = unit if unit.startswith("currency:") else None
                cur = own or (f"currency:{region[3]}" if region[3] else None)
                if cur is None:
                    sym = [c for c in text[region[0]:start] if c in _CURRENCY_SYMBOLS]
                    cur = sym and f"currency:{_CURRENCY_SYMBOLS[sym[-1]]}"
                found.append(ParsedNumber(value=value * region[2], unit=cur or "count",
                                          offset=start, literal=False))
        cents = mag == 1.0 and _CENTS_AFTER.match(text, end)
        if cents and (paren or _PER_SHARE_AFTER.match(text, cents.end())
                      or _PER_SHARE_BEFORE.search(text, max(0, start - 30), start)):
            found.append(ParsedNumber(value=value / 100, unit="currency:USD", offset=start,
                                      literal=literal))

    for m in _NUMBER_RE.finditer(normalized_text):
        for i, value in enumerate(_readings(m.group(0))):
            emit(value, m.start(), m.end(), literal=(i == 0))

    for value, start, end in _word_numbers(normalized_text):
        emit(value, start, end)

    for m in _FRACTION_RE.finditer(normalized_text):
        num, den = int(m.group("num")), int(m.group("den"))
        if num < den:
            emit(int(m.group("whole") or 0) + num / den, m.start(), m.end())

    for m in _STEP_WORDS.finditer(normalized_text):
        found.append(ParsedNumber(value=_STEP_SIZES.get(m.group("n"), 0.75), unit="pct",
                                  offset=m.start(), step=True))

    return found


def unit_magnitude(unit: Optional[str]) -> float:
    """The scale named inside a unit string.

    A figure may be split between the fields, `value: 25.7` with `unit: "million euros"`,
    while the page says 25,700,000, so the claim is scaled by its own unit before matching.

    A basis point is a hundredth of a percentage point, and both sit in the same unit
    class, so the scale has to be carried here or the two compare equal.
    """
    text = (unit or "").lower().replace("_", " ")
    scale = 1.0
    for word in _WORD_RE.findall(text):
        if word in _MAGNITUDES:
            scale *= _MAGNITUDES[word]
    words = set(_WORD_RE.findall(text))
    if words & {"bps", "bp"} or "basis point" in text:
        scale *= BASIS_POINT
    return scale


def unit_class(unit: Optional[str]) -> str:
    u = (unit or "").strip().lower()
    if not u:
        return "none"
    if u.replace("_", " ") in ("%", "pct", "percent", "percentage", "bps", "bp",
                              "basis point", "basis points"):
        return "pct"
    if u in _CURRENCY_CODES:
        return f"currency:{u.upper()}"
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in u:
            return f"currency:{code}"
    # "million euros" names a scale and a currency; the currency is the unit.
    for word in _WORD_RE.findall(u):
        if word in _CURRENCY_CODES:
            return f"currency:{word.upper()}"
        if word in _CURRENCY_WORDS:
            return f"currency:{_CURRENCY_WORDS[word]}"
    return "count"


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

def visible(fact, numbers: Sequence[ParsedNumber]) -> bool:
    """The figure's size is a number the page prints, in any standard format: grouping and
    decimal marks, scale words, a table's scale heading, percent or per cent, basis or
    percentage points, cents a share; with or without a currency mark, but a US dollar
    figure is not a number marked in another currency. A zero is left to the judge."""
    v = getattr(fact, "value", None)
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v == 0:
        return True
    size = abs(v) * unit_magnitude(fact.unit)
    usd = (fact.unit or "").startswith("usd")
    return any(value_grounded(size, abs(n.value)) and not (
        usd and n.unit.startswith("currency:") and n.unit != "currency:USD") for n in numbers)


# ---- hashing ----------------------------------------------------------------------

def content_hash(text: Optional[str]) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
