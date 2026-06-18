"""Target-aware per-asset aspect sentiment.

Replaces the former whole-sentence FinBERT majority vote
(``_finbert_asset_sentiments``, now removed). That method scored an entire mention sentence
and could not attribute polarity to a *specific* asset — in a sentence like
"Microsoft shares slid as the broader Nasdaq pulled back" it emitted one label
for the whole sentence, so the subject (MSFT) washed to neutral while a
cleanly-negative throwaway index got the bearish label. Majority-voting hedged
financial prose then amplified neutral.

This module uses **FinABSA** (``amphora/FinABSA``), a financial target-aware
aspect-based sentiment model. The target asset is marked with ``[TGT]`` and the
model classifies the polarity *toward that target*. FinABSA won the FinEntity
bake-off decisively (macro-F1 0.84 vs 0.70 for general DeBERTa-ABSA and 0.68 for
the whole-sentence FinBERT baseline; see ``eval/scripts/bakeoff_aspect.py``).

FinABSA is highly confident per sentence (near one-hot), so per-mention
probabilities carry no useful gradation. Instead the 7-class ``Sentiment`` scale
and magnitude/confidence are derived by **voting across an asset's mention
sentences**: three bearish mentions read very_bearish, one reads bearish, a
bull/bear split reads neutral. This is what stops consistent weak signal from
collapsing to neutral.

Determinism (consensus boundary): pinned model id, eval()/no_grad, greedy
decoding (num_beams=1, do_sample=False), integer vote arithmetic. Identical
output on miner and validator given the same model + device class.

Drop-in contract: ``score_assets(...)`` returns the same list-of-dict shape that
``_build_assets`` already consumes (ticker/direction/magnitude/confidence/
short_term/medium_term/long_term/causal_driver/evidence_spans).

Horizons (short/medium/long) are produced by ``horizon.project_horizons`` via
temporal evidence partitioning (Gate 2): each mention is bucketed by its
time-framing cue and FinABSA-voted per bucket, so the three horizons carry
independent signal (and can disagree in sign) instead of mirroring `direction`.
The rule thresholds there are pending calibration against the horizon gold set.
"""
from __future__ import annotations

import re
import threading
from typing import Iterable, List, Optional

# Pinned for consensus. Winner of the FinEntity bake-off
# (eval/scripts/bakeoff_aspect.py). Swap here if that decision changes.
ABSA_MODEL_ID = "amphora/FinABSA"

# Bound CPU latency: an asset rarely needs more than a few mention sentences to
# decide direction, and more mentions only slow the (bit-exact) CPU path.
MAX_MENTIONS_PER_ASSET = 6

# Below this confidence, a PRIMARY-subject asset's FinABSA direction is too weak to
# trust (few mentions or near-even split), so it defers to the article-level
# sentiment. Tuned so a lone noisy vote among neutral mentions defers while any
# genuine 2+ -mention agreement keeps its own direction.
PRIMARY_MIN_CONFIDENCE = 0.15

# Coreference: in a single-primary-subject article, these generic referents point
# at the subject ("The company raised guidance", "Shares surged") — the sentences
# that carry sentiment but never name the asset. Each phrase is itself a valid
# FinABSA target (present in the sentence), so [TGT] lands on it and the polarity
# is read toward the company. Gated to a SINGLE primary subject so "the company"
# is unambiguous; multi-subject articles keep explicit-name matching only.
#
# Only entity-of-investment referents (the company / the stock / the shares) —
# NOT actors like "the board"/"the CEO"/"management". FinABSA scores sentiment
# TOWARD its target, and "the board approved a buyback" reads neutral toward the
# board even though it's bullish for the stock, so actor referents only dilute.
GENERIC_REFERENTS = {
    "the company", "the firm", "the group", "the business", "the maker",
    "the stock", "the shares", "the share price", "shares",
}

_LABEL_VAL = {"positive": 1, "negative": -1, "neutral": 0}


def vote_to_sentiment(net: int, n: int, agree: int) -> tuple[str, float]:
    """Map a mention vote tally to a (7-class Sentiment value, magnitude).

    net   = (#bullish - #bearish) across an asset's mention sentences
    n     = total mentions scored
    agree = max(#bullish, #bearish) — evidence behind the dominant sign

    Gradation is evidence-aware: a single decisive mention reaches
    bullish/bearish but NOT very_* (which needs >=2 agreeing mentions), and a
    minority signal among many mentions reads slightly_*.
    """
    if n <= 0 or net == 0:
        return "neutral", 0.0
    strength = abs(net) / n  # in (0, 1]
    sign = "bullish" if net > 0 else "bearish"
    if strength >= 0.75 and agree >= 2:
        return f"very_{sign}", min(1.0, strength)
    if strength >= 0.40:
        return sign, strength
    return f"slightly_{sign}", strength


def aggregate_votes(votes: List[int]) -> tuple[str, float, float]:
    """Reduce a list of {+1,0,-1} mention votes to (direction, magnitude, confidence).

    Shared by per-asset direction (all mentions) and per-horizon projection
    (a temporal bucket's mentions), so both come from one FinABSA pass.
    """
    n = len(votes)
    if n == 0:
        return "neutral", 0.0, 0.0
    pos = votes.count(1)
    neg = votes.count(-1)
    net = pos - neg
    agree = max(pos, neg)
    direction, magnitude = vote_to_sentiment(net, n, agree)
    consensus = agree / n
    # Confidence must reward EVIDENCE VOLUME, not just agreement — otherwise a lone
    # unanimous mention (consensus 1.0) outranks six corroborating-but-mixed ones.
    # `volume` saturates at 4 agreeing mentions; a single mention is capped well
    # below certainty regardless of how decisive its polarity reads.
    volume = min(1.0, agree / 4.0)
    confidence = max(0.0, min(1.0, consensus * volume * (0.7 + 0.3 * magnitude)))
    return direction, magnitude, confidence


class AspectSentimentScorer:
    """Lazy, thread-safe, deterministic FinABSA target-aware scorer."""

    # FinABSA emits "...sentence isNEGATIVE ." with no space before the label,
    # so a \b-anchored match misses it. Plain substring search over the 3 labels.
    _TEMPLATE_RE = re.compile(r"POSITIVE|NEGATIVE|NEUTRAL")

    def __init__(self, model_id: str = ABSA_MODEL_ID, device: Optional[str] = None,
                 max_len: int = 256):
        self.model_id = model_id
        self._device = device
        self._max_len = max_len
        self._tok = None
        self._model = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = dev
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            self._model = (AutoModelForSeq2SeqLM
                           .from_pretrained(self.model_id).to(dev).eval())

    def _decode_vote(self, ids) -> int:
        gen = self._tok.decode(ids, skip_special_tokens=True).upper()
        m = self._TEMPLATE_RE.search(gen)
        word = m.group(0).lower() if m else "neutral"
        return _LABEL_VAL.get(word, 0)

    def _label(self, text: str, target: str) -> int:
        """Polarity toward `target` in `text` as a vote in {+1, 0, -1}."""
        return self.label_many([(text, target)])[0]

    def label_many(self, pairs: List[tuple], batch_size: int = 16) -> List[int]:
        """Batched polarity for many (text, target) pairs -> votes in {+1,0,-1}.

        Batching matters on the bit-exact CPU path: per-call overhead dominates,
        so collapsing an article's mentions into a few padded batches is the main
        latency lever. Greedy decoding keeps it deterministic.
        """
        import torch
        if not pairs:
            return []
        self._ensure()
        masked = [re.sub(re.escape(t), "[TGT]", txt, flags=re.IGNORECASE)
                  for txt, t in pairs]
        votes: List[int] = []
        for i in range(0, len(masked), batch_size):
            chunk = masked[i:i + batch_size]
            enc = self._tok(chunk, return_tensors="pt", truncation=True,
                            max_length=self._max_len, padding=True).to(self._device)
            with torch.no_grad():
                out = self._model.generate(**enc, max_new_tokens=16,
                                           num_beams=1, do_sample=False)
            votes.extend(self._decode_vote(out[j]) for j in range(out.shape[0]))
        return votes

    def score_mentions(self, mentions: List[str], target: str):
        """Vote across an asset's mention sentences.

        Returns (direction_7class, magnitude, confidence, evidence_spans).
        """
        if not mentions:
            return "neutral", 0.0, 0.0, []
        mentions = mentions[:MAX_MENTIONS_PER_ASSET]
        votes = self.label_many([(s, target) for s in mentions])
        direction, magnitude, confidence = aggregate_votes(votes)
        net = votes.count(1) - votes.count(-1)
        # evidence: prefer sentences that carried the winning sign
        win = 1 if net > 0 else -1 if net < 0 else 0
        spans = [s for s, v in zip(mentions, votes) if v == win][:3] or mentions[:1]
        return direction, magnitude, confidence, spans


def _asset_forms(e) -> set:
    forms = set()
    tk = getattr(e, "ticker", None)
    if tk:
        forms.add(tk)
    name = getattr(e, "canonical_name", None)
    if name:
        forms.add(name)
    for extra in (getattr(e, "surface_forms", None) or []):
        if extra:
            forms.add(extra)
    return forms


def _best_target_form(forms: set) -> str:
    """Prefer a human name over a bare ticker as the ABSA target string —
    ABSA was trained on names/phrases, and tickers rarely appear in prose."""
    named = [f for f in forms if not f.isupper() or " " in f]
    return max(named or list(forms), key=len) if forms else ""


def _target_in_sentence(sentence: str, forms, fallback: str = "") -> str:
    """The longest form actually PRESENT in this sentence.

    FinABSA marks the target with [TGT] via substring replacement; if the chosen
    target string isn't in the sentence, nothing is marked and the model returns
    a generic (usually neutral) label. So the target must be picked per-mention
    from the forms that occur in *that* sentence — not one global longest form.
    """
    present = [f for f in forms
               if re.search(rf"\b{re.escape(f)}\b", sentence, re.IGNORECASE)]
    return max(present, key=len) if present else fallback


def _mention_sentences(sentences: Iterable[str], forms: Iterable[str]) -> List[str]:
    pats = [re.compile(rf"\b{re.escape(f)}\b", re.IGNORECASE)
            for f in forms if f and len(f) >= 2]
    return [s for s in sentences if any(p.search(s) for p in pats)]


def score_assets(resolved_assets, ner_result, scorer: AspectSentimentScorer,
                 fallback_direction: str = "neutral") -> List[dict]:
    """Drop-in replacement for the former `_finbert_asset_sentiments`.

    Reuses the sentence pool the NER engine already produced
    (`ner_result.sentence_sentiments[*]["text"]`). Horizons currently mirror
    `direction` (see module docstring).
    """
    from talisman_ai.analyzer.horizon import project_horizons  # lazy: avoid import cycle

    scored = getattr(ner_result, "sentence_sentiments", None) or []
    sentences = [s.get("text") or "" for s in scored if s.get("text")]

    # A single dominant subject lets us resolve generic referents ("the company")
    # to it; with 0 or >1 primary subjects we can't, so coref expansion is off.
    primaries = [e for e in resolved_assets
                 if getattr(e, "ticker", None) and getattr(e, "is_primary_subject", False)]
    single_primary = primaries[0] if len(primaries) == 1 else None

    out = []
    for e in resolved_assets:
        tk = getattr(e, "ticker", None)
        if not tk:
            continue
        forms = _asset_forms(e)
        target = _best_target_form(forms)  # the asset's own name (never a referent)
        # For the lone primary subject, also match generic-referent sentences.
        match_forms = forms | GENERIC_REFERENTS if e is single_primary else forms
        mentions = _mention_sentences(sentences, match_forms)[:MAX_MENTIONS_PER_ASSET]
        if mentions and target:
            # one FinABSA pass; reused for overall direction AND per-horizon buckets.
            # Per-mention target = a form present in that sentence, so [TGT] always lands.
            votes = scorer.label_many(
                [(m, _target_in_sentence(m, match_forms, fallback=target)) for m in mentions])
            direction, magnitude, conf = aggregate_votes(votes)
            net = votes.count(1) - votes.count(-1)
            win = 1 if net > 0 else -1 if net < 0 else 0
            spans = [s for s, v in zip(mentions, votes) if v == win][:3] or mentions[:1]
            if getattr(e, "is_primary_subject", False) and conf < PRIMARY_MIN_CONFIDENCE:
                # Weak/conflicting per-asset evidence on the article's OWN subject
                # (e.g. one mis-scored "Bull-Trap" headline among neutral mentions):
                # the article-level read aggregates more signal, so defer to it
                # rather than emit a noisy minority direction.
                direction = short = medium = long = fallback_direction
                magnitude, conf = 0.4, max(conf, 0.3)
                driver = (f"FinABSA inconclusive ({len(mentions)} mention(s), low consensus); "
                          f"used article-level sentiment")
            else:
                short, medium, long = project_horizons(mentions, votes, direction)
                driver = f"FinABSA target-aware over {len(mentions)} mention sentence(s)"
        else:
            direction = short = medium = long = fallback_direction
            magnitude, conf, spans = 0.4, 0.3, []
            driver = "No asset-specific sentence; fell back to article-level sentiment"
        out.append({
            "ticker": tk,
            "direction": direction,
            "magnitude": magnitude,
            "confidence": conf,
            "short_term": short,
            "medium_term": medium,
            "long_term": long,
            "causal_driver": driver,
            "evidence_spans": spans,
        })
    return out
