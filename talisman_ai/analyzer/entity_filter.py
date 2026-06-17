"""Confidence gating, blocklist filtering, and offset-overlap dedup for NER.

This is the grounding stage of the pipeline. It takes raw NER candidates from
the (language-routed) candidate generator and produces a clean, deduplicated
entity set:

* **Confidence gates** — a lone model must clear a high bar; a span corroborated
  by >=2 models may clear a lower one. Replaces the old hardcoded 0.9.
* **Blocklists** — per-language stopwords, days, and months, plus numeric /
  single-char / punctuation garbage. Backstops §3.3 ("domenica" -> location).
* **Offset-overlap dedup** — clusters candidates by character span overlap and
  keeps the longest, highest-confidence representative. Replaces the broken
  substring dedup that dropped "China" when "Bank of China" was present.

Pure Python and order-independent => deterministic (validator-safe).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Candidate:
    """A raw NER candidate span before grounding."""
    text: str
    start: int
    end: int
    label: str
    model: str
    score: float


# Below this, a candidate is never considered (pure noise floor).
_FLOOR = 0.30
# A single-model span must clear this to survive.
_SINGLETON_MIN = 0.60
# A span seen by >=2 distinct models only needs this.
_AGREEMENT_MIN = 0.40

_DAYS = {
    "en": {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"},
    "it": {"lunedì", "lunedi", "martedì", "martedi", "mercoledì", "mercoledi",
           "giovedì", "giovedi", "venerdì", "venerdi", "sabato", "domenica"},
    "ru": {"понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"},
}
_MONTHS = {
    "en": {"january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"},
    "it": {"gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
           "agosto", "settembre", "ottobre", "novembre", "dicembre"},
    "ru": {"январь", "февраль", "март", "апрель", "май", "июнь", "июль",
           "август", "сентябрь", "октябрь", "ноябрь", "декабрь"},
}
# Generic high-frequency words that are never useful financial entities.
_STOPWORDS = {
    "en": {"the", "and", "for", "with", "from", "this", "that", "live", "today",
           "news", "update", "updates", "video", "photo", "image"},
    "it": {"diretta", "oggi", "notizie", "video", "foto", "ieri", "domani"},
    "ru": {"сегодня", "новости", "видео", "фото"},
}


def in_salient_range(start: int, end: int, salient_ranges: List[Tuple[int, int]]) -> bool:
    """True if [start, end) lies within any salient (verb-bearing) sentence."""
    return any(s <= start and end <= e for s, e in salient_ranges)


def apply_salience(candidates: List[Candidate],
                   salient_ranges: List[Tuple[int, int]]) -> List[Candidate]:
    """Dependency-parse salience gate: keep only candidates inside a sentence
    that contains a finite verb. Drops entities stranded in verbless fragments
    (share bars / nav rows / photo-credit lists). No-op when no salient range
    was found, so we never nuke an entire (e.g. headline-only) document."""
    if not salient_ranges:
        return candidates
    return [c for c in candidates if in_salient_range(c.start, c.end, salient_ranges)]


class EntityFilter:
    """Grounds raw NER candidates into a clean, deduplicated entity set."""

    def __init__(self, floor: float = _FLOOR, singleton_min: float = _SINGLETON_MIN,
                 agreement_min: float = _AGREEMENT_MIN):
        self.floor = floor
        self.singleton_min = singleton_min
        self.agreement_min = agreement_min

    def _blocked(self, text: str, language: str) -> bool:
        t = text.strip()
        if len(t) < 2:
            return True
        # No alphabetic character => numeric / punctuation garbage.
        if not any(ch.isalpha() for ch in t):
            return True
        low = t.lower()
        for table in (_DAYS, _MONTHS, _STOPWORDS):
            if low in table.get(language, set()) or low in table.get("en", set()):
                return True
        return False

    def filter(self, candidates: List[Candidate], language: str = "en") -> List[Candidate]:
        """Return grounded, deduplicated candidates sorted by start offset."""
        # 1. Drop noise floor + blocklist hits.
        kept = [c for c in candidates
                if c.score >= self.floor and not self._blocked(c.text, language)]
        if not kept:
            return []

        # 2. Cluster by character-offset overlap (sweep over sorted spans).
        ordered = sorted(kept, key=lambda c: (c.start, -c.end))
        clusters: List[List[Candidate]] = []
        cur: List[Candidate] = []
        cur_end = -1
        for c in ordered:
            if cur and c.start < cur_end:  # overlaps current cluster
                cur.append(c)
                cur_end = max(cur_end, c.end)
            else:
                if cur:
                    clusters.append(cur)
                cur = [c]
                cur_end = c.end
        if cur:
            clusters.append(cur)

        # 3. Per cluster: confidence gate by single-model vs. agreement, then
        #    pick the longest / highest-confidence representative.
        out: List[Candidate] = []
        for cl in clusters:
            models = {c.model for c in cl}
            best_score = max(c.score for c in cl)
            threshold = self.agreement_min if len(models) >= 2 else self.singleton_min
            if best_score < threshold:
                continue
            rep = max(cl, key=lambda c: (c.score, c.end - c.start, c.text))
            out.append(rep)

        out.sort(key=lambda c: c.start)
        return out
