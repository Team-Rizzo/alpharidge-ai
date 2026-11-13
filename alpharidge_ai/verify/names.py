"""Company names written shorter or longer: initials, joined words and short company words
("FHLBank Chicago" for Federal Home Loan Bank of Chicago). For finding pages and filers, never
for the check a claim's page runs."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Sequence, Tuple

from alpharidge_ai.verify import floor

# Short and long forms of words in company names.
_WORD_FORMS = {"corp": "corporation", "co": "company", "cos": "companies",
               "inc": "incorporated", "ltd": "limited", "intl": "international",
               "natl": "national", "bros": "brothers", "hldgs": "holdings", "grp": "group",
               "mfg": "manufacturing", "svcs": "services", "assn": "association",
               "fin": "financial", "tech": "technologies", "univ": "university"}
_FILLER = frozenset("of the and for".split())
_TAIL = frozenset("incorporated corporation company companies limited plc llc lp sa ag nv se "
                  "group holdings holding".split())
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z][a-z])")
_TOKEN = re.compile(r"[^\W_]+")


def name_words(name: str) -> Tuple[str, ...]:
    """A name as words, lowercase, camel case split ("NorthStrive"), "&" as "and", short
    forms of company words spelled out and a leading "the" dropped."""
    text = floor.normalize(_CAMEL.sub(" ", name or "")).text.replace("&", " and ")
    words = [_WORD_FORMS.get(w, w) for w in _TOKEN.findall(text.replace("'", ""))]
    return tuple(words[1:] if len(words) > 1 and words[0] == "the" else words)


def _initials(run) -> set:
    out = {"".join(w[0] for w in run)}
    core = [w for w in run if w not in _FILLER]
    if core:
        out.add("".join(w[0] for w in core))
    return out


def _covers(token: str, run) -> bool:
    if token == "".join(run):
        return True
    if len(run) < 2 or run[0] in _FILLER:
        return False
    return token in _initials(run) or any(
        token == i + run[-1] for i in _initials(run[:-1]))


def _abbreviates(short, long, open_end: bool) -> bool:
    """Each short word stands for a run of long words in order: the run joined, its initials,
    or its initials before its last word ("FHLBank"); filler words may be skipped, and the
    long name's trailing company words left out. With `open_end` the long words may go on."""
    short = tuple(w for w in short if w not in _FILLER)
    while short and short[-1] in _TAIL:
        short = short[:-1]

    @lru_cache(maxsize=None)
    def go(i: int, j: int) -> bool:
        if i == len(short):
            return open_end or all(w in _TAIL or w in _FILLER for w in long[j:])
        if j >= len(long):
            return False
        if long[j] in _FILLER and go(i, j + 1):
            return True
        return any(_covers(short[i], long[j:k]) and go(i + 1, k)
                   for k in range(j + 1, min(len(long), j + len(short[i])) + 1))

    return bool(short) and go(0, 0)


def abbreviates(short: str, long: str) -> bool:
    """`short` is `long` written shorter: initials, a joined or shortened form, or without
    its company suffix ("FHLBank Chicago" for Federal Home Loan Bank of Chicago, "SST VI" for
    Strategic Storage Trust VI, Inc.)."""
    return _abbreviates(name_words(short), name_words(long), False)


def _starts(words, j: int) -> set:
    """The forms a run of words from `j` on may take as one word."""
    out = set()
    for k in range(j + 1, min(len(words), j + 6) + 1):
        run = words[j:k]
        out.add("".join(run))
        if len(run) > 1 and run[0] not in _FILLER:
            out |= _initials(run) | {i + run[-1] for i in _initials(run[:-1])}
    return out


@lru_cache(maxsize=8192)
def _name_parts(name: str):
    mine = name_words(name)
    core = [w for w in mine if w not in _FILLER]
    if not core:
        return None
    return mine, core[0], frozenset(_starts(mine, 0)), sum(map(len, core)) + len(mine)


def named_in(words: Sequence[str], name: str) -> bool:
    """A run of `words` (a text's `name_words`) names `name`: as written, shorter or longer."""
    parts = _name_parts(name)
    if parts is None:
        return False
    mine, first, heads, reach = parts
    words = tuple(words)
    for s, w in enumerate(words):
        if w[0] != first[0]:
            continue
        if any(_covers(first, words[s:k]) for k in range(s + 1, min(len(words), s + len(first))
                                                          + 1)) and _abbreviates(
                mine, words[s:s + reach], True):
            return True
        if w != first and w in heads and any(_abbreviates(words[s:t], mine, False)
                              for t in range(s + 1, min(len(words), s + len(mine)) + 1)):
            return True
    return False
