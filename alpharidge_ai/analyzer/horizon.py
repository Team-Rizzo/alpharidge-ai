"""Independent per-asset horizon outlooks via temporal evidence partitioning.

The old pipeline copied `direction` into short/medium/long_term_outlook, so the
three horizons carried no independent signal. This module derives them from the
text instead: each of an asset's mention sentences is bucketed by its
time-framing cue (short / medium / long), and the already-computed FinABSA votes
are aggregated *per bucket*. So "the stock fell today but its long-term strategy
is intact" can yield short=bearish, long=bullish — genuinely independent, even
opposite-sign — at zero extra model cost (votes are reused from score_assets).

The per-sentence temporal cue is the *deterministic* catalyst signal: it does
not depend on the non-deterministic LLM Call 1 (temporal_focus/forward_event_type),
so horizons stay verifiable across the consensus boundary.

Honesty rule: differentiate only when there is real temporal evidence. If an
asset has no explicit short/long cue, the horizons mirror `direction` rather
than fabricate a decay — we don't manufacture horizon signal we can't see.
`coverage` (fraction of assets that got differentiated horizons) is reported by
the eval so we know how often the mechanism actually fires.

Thresholds/cues here are a first cut pending calibration on the horizon gold set
(Gate 2). If rule-based coverage/accuracy is insufficient, escalate to a learned
per-asset horizon head.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from alpharidge_ai.analyzer.aspect_sentiment import aggregate_votes

HORIZONS = ("short", "medium", "long")

# 7-class ordinal scale (bearish -> bullish), used for decay-toward-neutral backfill.
_SCALE = ["very_bearish", "bearish", "slightly_bearish", "neutral",
          "slightly_bullish", "bullish", "very_bullish"]
_MID = 3


def _sign(s: str) -> int:
    if s not in _SCALE:
        return 0
    i = _SCALE.index(s)
    return (i > _MID) - (i < _MID)


def reconcile_direction_with_horizons(direction: str, short: str, medium: str, long: str) -> str:
    """Resolve a direction that contradicts ALL of its horizons.

    Per-asset `direction` is deterministic FinABSA; horizons may be overridden by
    the LLM's semantic read. When FinABSA says bullish but every horizon is
    bearish/neutral (the URI case), the output is incoherent and FinABSA is the
    less-trustworthy signal — replace direction with the strongest opposing
    horizon. Untouched when any horizon agrees in sign, when direction is neutral,
    or when horizons are merely neutral (no sign opposition). Deterministic."""
    ds = _sign(direction)
    if ds == 0:
        return direction
    horizons = [short, medium, long]
    signs = [_sign(h) for h in horizons]
    if any(s == ds for s in signs):
        return direction          # at least one horizon agrees -> coherent enough
    opposing = [h for h in horizons if _sign(h) == -ds]
    if not opposing:
        return direction          # only neutrals, no contradiction
    return max(opposing, key=lambda h: abs(_SCALE.index(h) - _MID))


def decay_one_step(direction: str) -> str:
    """Move one ordinal step toward neutral (used to backfill an empty horizon)."""
    if direction not in _SCALE:
        return "neutral"
    i = _SCALE.index(direction)
    if i < _MID:
        return _SCALE[i + 1]
    if i > _MID:
        return _SCALE[i - 1]
    return "neutral"


# Time-framing cues. Word-boundary matched, lowercased. Longer/structural phrases
# win ties (see bucket_sentence).
_SHORT_CUES = [
    r"today", r"this morning", r"this afternoon", r"intraday", r"this week",
    r"this session", r"pre-?market", r"after-?hours", r"overnight",
    r"on the day", r"in early trading", r"midday", r"at the open",
    r"at the close", r"right now", r"currently", r"so far today",
]
_LONG_CUES = [
    r"long-?term", r"in the long run", r"next year", r"coming years",
    r"over the next \w+ years", r"by 20[2-9]\d", r"multi-?year", r"decade",
    r"secular", r"structural(?:ly)?", r"strateg(?:y|ic)", r"transformation",
    r"roadmap", r"pipeline", r"for years to come", r"over time",
]
_MEDIUM_CUES = [
    r"next quarter", r"coming quarters", r"this year", r"full-?year",
    r"fiscal year", r"guidance", r"outlook", r"forecast",
    r"in the coming months", r"over the coming months", r"near-?term",
]

_SHORT_RE = [re.compile(rf"\b{c}\b", re.I) for c in _SHORT_CUES]
_LONG_RE = [re.compile(rf"\b{c}\b", re.I) for c in _LONG_CUES]
_MEDIUM_RE = [re.compile(rf"\b{c}\b", re.I) for c in _MEDIUM_CUES]


def _count(res, text):
    return sum(1 for r in res if r.search(text))


def bucket_sentence(sentence: str) -> str:
    """Assign a sentence to short / medium / long by its strongest temporal cue.

    Ties / no-cue default to 'medium' (most financial commentary is medium-horizon).
    Long (structural) and short (immediate) cues outrank the medium default; on a
    long-vs-short tie, long wins (the structural framing is the more deliberate
    signal). A sentence genuinely spanning horizons is a known limitation of
    sentence-level bucketing — a learned span-level head is the Plan-B fix.
    """
    s = sentence or ""
    n_short = _count(_SHORT_RE, s)
    n_long = _count(_LONG_RE, s)
    n_med = _count(_MEDIUM_RE, s)
    best = max(n_short, n_long, n_med)
    if best == 0:
        return "medium"
    if n_long == best:
        return "long"
    if n_short == best:
        return "short"
    return "medium"


def project_horizons(mentions: List[str], votes: List[int],
                     overall_direction: str) -> Tuple[str, str, str]:
    """Return (short, medium, long) 7-class outlooks from bucketed mention votes.

    `votes[i]` is the FinABSA vote for `mentions[i]` (reused from score_assets).
    """
    buckets = [bucket_sentence(m) for m in mentions]
    # Honesty rule: no explicit near/far evidence -> don't fabricate differentiation.
    if not any(b in ("short", "long") for b in buckets):
        return overall_direction, overall_direction, overall_direction

    populated = {}
    for h in HORIZONS:
        hv = [v for v, b in zip(votes, buckets) if b == h]
        if hv:
            populated[h] = aggregate_votes(hv)[0]

    # Backfill an empty horizon by decaying the medium view (or the overall
    # direction if medium itself is empty) one step toward neutral.
    base = populated.get("medium") or overall_direction
    result = {h: populated.get(h) or decay_one_step(base) for h in HORIZONS}
    return result["short"], result["medium"], result["long"]
