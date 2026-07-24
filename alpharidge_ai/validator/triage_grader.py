"""Validator-side grading of miner triage claims (schema v3).

Design rule (load-bearing): HARD penalties attach only to deterministic
evidence — proof-of-read mismatches, gazetteer-backed false negatives, and
gazetteer-backed positive canaries. Verdicts that rest only on an LLM opinion
are always SOFT (small EMA weight, never a single-shot park), so audit-model
mistakes degrade a miner's reputation slowly enough for the clean-batch signal
to dominate. Cheat strategies are still caught fast because every profitable
cheat necessarily trips deterministic evidence.

Grading order per batch (cheap -> expensive):
  1. proof-of-read on every article (deterministic; failures -> integrity path)
  2. canary checks (known-positive / known-negative plants)
  3. random audit of claimed-irrelevant articles (gazetteer first, LLM second)
Step 4 — the full reference analysis of sampled claimed-relevant articles —
stays in the existing validation pipeline; its false-positive verdicts are fed
back through `fp_soft_event()`.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from alpharidge_ai.triage import (
    LABEL_BORDERLINE,
    LABEL_IRRELEVANT,
    LABEL_RELEVANT,
    extract_triage,
    verify_proof_of_read,
)


@dataclass
class TriageConfig:
    audit_irrelevant_n: int = 1
    borderline_cap: int = 3
    hard_weight: float = 2.0
    soft_weight: float = 0.4
    clean_weight: float = 1.0
    hard_llm_verdicts: bool = False   # served-config tightening lever; default LLM = soft
    canary_ttl_s: float = 6 * 3600.0
    canary_max_exposures: int = 30


@dataclass
class TriageEvent:
    kind: str        # "hard" | "soft"
    code: str
    article_id: int


@dataclass
class TriageGradeResult:
    events: List[TriageEvent] = field(default_factory=list)
    proof_failures: List[int] = field(default_factory=list)   # -> existing integrity path
    relevant_ids: List[int] = field(default_factory=list)     # claimed relevant, feed deep validation
    borderline_ids: List[int] = field(default_factory=list)
    retire_candidate_ids: List[int] = field(default_factory=list)
    v2_grace: bool = False

    def observations(self, cfg: TriageConfig) -> List[Tuple[float, float]]:
        """(score, weight) pairs for the triage reputation EMA."""
        if self.v2_grace:
            return []
        if not self.events and not self.proof_failures:
            return [(1.0, cfg.clean_weight)]
        obs: List[Tuple[float, float]] = [
            (0.0, cfg.hard_weight if e.kind == "hard" else cfg.soft_weight)
            for e in self.events
        ]
        obs.extend((0.0, cfg.hard_weight) for _ in self.proof_failures)
        return obs


def fp_soft_event(article_id: int) -> TriageEvent:
    """A sampled claimed-relevant article whose reference analysis showed it to
    be non-relevant junk (spam-flagging). Reference-adjudicated -> soft."""
    return TriageEvent("soft", "triage_false_positive", article_id)


def grade_batch(
    items: List[dict],
    canary_labels: Dict[int, Tuple[str, bool]],
    det_relevant: Callable[[dict], bool],
    llm_relevant: Callable[[dict], Optional[bool]],
    rng,
    cfg: TriageConfig,
) -> TriageGradeResult:
    """Grade the triage layer of one returned miner batch.

    items: [{"article_id", "title", "body", "analysis_data"}] — every article
      the validator dispatched, in the miner's returned form.
    canary_labels: article_id -> ("pos"|"neg", deterministic) for planted canaries.
    det_relevant(item): deterministic gazetteer check (R1) on the validator's copy.
    llm_relevant(item): audit-LLM relevance verdict; None = unavailable (no event).
    """
    res = TriageGradeResult()

    records: Dict[int, Optional[dict]] = {}
    errors: Dict[int, Optional[str]] = {}
    for it in items:
        rec, err = extract_triage(it.get("analysis_data"))
        records[it["article_id"]] = rec
        errors[it["article_id"]] = err

    if not any(rec for rec in records.values()) and not any(errors.values()):
        # Pre-v3 miner: no triage anywhere. Grace path — existing v2 grading only.
        res.v2_grace = True
        return res

    # 1. proof-of-read: required on every article once the miner speaks v3.
    for it in items:
        aid = it["article_id"]
        proof = (it.get("analysis_data") or {}).get("proof_of_read") \
            if isinstance(it.get("analysis_data"), dict) else None
        if not verify_proof_of_read(proof, it.get("title") or "", it.get("body") or ""):
            res.proof_failures.append(aid)

    # Labels; malformed/missing records on a v3 miner are soft protocol
    # violations and are treated as irrelevant claims so they stay auditable.
    labels: Dict[int, str] = {}
    n_border = 0
    for it in items:
        aid = it["article_id"]
        rec = records[aid]
        if rec is None:
            res.events.append(TriageEvent("soft", "triage_malformed", aid))
            labels[aid] = LABEL_IRRELEVANT
            continue
        label = rec["label"]
        if label == LABEL_BORDERLINE:
            n_border += 1
            if n_border > cfg.borderline_cap:
                label = LABEL_IRRELEVANT   # excess borderline is an irrelevant claim
        labels[aid] = label

    for it in items:
        aid = it["article_id"]
        if labels[aid] == LABEL_RELEVANT:
            res.relevant_ids.append(aid)
        elif labels[aid] == LABEL_BORDERLINE:
            res.borderline_ids.append(aid)

    # 2. canary checks
    for it in items:
        aid = it["article_id"]
        if aid not in canary_labels or aid in res.proof_failures:
            continue
        kind, deterministic = canary_labels[aid]
        if kind == "pos" and labels[aid] == LABEL_IRRELEVANT:
            severity = "hard" if deterministic else "soft"
            res.events.append(TriageEvent(severity, "canary_pos_missed", aid))
        elif kind == "neg" and labels[aid] == LABEL_RELEVANT:
            res.events.append(TriageEvent("soft", "canary_neg_flagged", aid))

    # 3. random audit of claimed-irrelevant (non-canary, proof-passing)
    auditable = [
        it for it in items
        if labels[it["article_id"]] == LABEL_IRRELEVANT
        and it["article_id"] not in canary_labels
        and it["article_id"] not in res.proof_failures
    ]
    audited_ids = set()
    for it in rng.sample(auditable, min(cfg.audit_irrelevant_n, len(auditable))):
        aid = it["article_id"]
        audited_ids.add(aid)
        if det_relevant(it):
            res.events.append(TriageEvent("hard", "false_negative_deterministic", aid))
            continue
        verdict = llm_relevant(it)
        if verdict is True:
            severity = "hard" if cfg.hard_llm_verdicts else "soft"
            res.events.append(TriageEvent(severity, "false_negative_llm", aid))

    contradicted = {e.article_id for e in res.events}
    res.retire_candidate_ids = [
        it["article_id"] for it in auditable
        if it["article_id"] not in contradicted
    ] + [
        it["article_id"] for it in items
        if labels[it["article_id"]] == LABEL_IRRELEVANT
        and it["article_id"] in canary_labels
        and canary_labels[it["article_id"]][0] == "neg"
        and it["article_id"] not in contradicted
    ]
    return res


class CanaryPool:
    """Validator-local pool of pre-labeled canary articles.

    Positives: articles with a deterministic gazetteer hit (free, unambiguous).
    Negatives: no gazetteer hit AND two self-consistent audit-LLM passes said
    non-economic (caller enforces the double-pass before add()).
    Entries expire after ttl seconds or max_exposures uses.
    """

    def __init__(self, cfg: TriageConfig, state_path: Optional[str] = None,
                 now: Callable[[], float] = time.time):
        self._cfg = cfg
        self._now = now
        self._state_path = state_path
        # article_id -> {kind, deterministic, born, exposures}
        self._entries: Dict[int, dict] = {}
        if state_path and os.path.exists(state_path):
            try:
                with open(state_path) as f:
                    raw = json.load(f)
                self._entries = {int(k): v for k, v in raw.items()}
            except (ValueError, OSError):
                self._entries = {}

    def add(self, article_id: int, kind: str, deterministic: bool) -> None:
        assert kind in ("pos", "neg")
        self._entries[article_id] = {
            "kind": kind, "deterministic": bool(deterministic),
            "born": self._now(), "exposures": 0,
        }

    def _alive(self, e: dict) -> bool:
        return (self._now() - e["born"] < self._cfg.canary_ttl_s
                and e["exposures"] < self._cfg.canary_max_exposures)

    def prune(self) -> None:
        self._entries = {k: v for k, v in self._entries.items() if self._alive(v)}

    def size(self, kind: str) -> int:
        return sum(1 for e in self._entries.values() if e["kind"] == kind and self._alive(e))

    def draw(self, kind: str, rng) -> Optional[int]:
        """Pick a live canary of the given kind and count the exposure."""
        alive = [aid for aid, e in self._entries.items()
                 if e["kind"] == kind and self._alive(e)]
        if not alive:
            return None
        aid = rng.choice(alive)
        self._entries[aid]["exposures"] += 1
        return aid

    def label_of(self, article_id: int) -> Optional[Tuple[str, bool]]:
        e = self._entries.get(article_id)
        if e is None:
            return None
        return e["kind"], e["deterministic"]

    def save(self) -> None:
        if not self._state_path:
            return
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({str(k): v for k, v in self._entries.items()}, f)
        os.replace(tmp, self._state_path)
