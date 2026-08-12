"""Validator-side grading of miner triage claims (schema v3).

Hard events come from deterministic checks; LLM-adjudicated events are soft.
Grading order per batch: proof-of-read, canary checks, audits of not-relevant
claims. Sampled claimed-relevant articles are validated by the existing deep
pipeline, which reports false positives via `fp_soft_event()`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from alpharidge_ai.triage import (
    FLAG_VALUABLE,
    LABEL_BORDERLINE,
    LABEL_IRRELEVANT,
    LABEL_RELEVANT,
    analysis_indicates_value,
    extract_triage,
    verify_proof_of_read,
)


@dataclass
class TriageConfig:
    """Design constants — not runtime configuration. Changing one is a
    released code change."""
    audit_irrelevant_n: int = 1
    borderline_cap: int = 3
    hard_weight: float = 2.0
    soft_weight: float = 0.4
    clean_weight: float = 1.0
    canary_ttl_s: float = 6 * 3600.0
    canary_max_exposures: int = 30
    canary_pos_rate: float = 0.7
    canary_neg_rate: float = 0.7
    fee_points: float = 0.2
    rel_point_mult: int = 6
    audit_min_confidence: float = 0.75
    neg_pool_target: int = 12
    neg_mint_budget: int = 3
    overlap_k: int = 3                # assignees per article once enforced
    verification_ttl_s: float = 900.0


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
    canary_ids: List[int] = field(default_factory=list)
    # Borderline resolved by the analyze-then-flag lane:
    borderline_valuable_ids: List[int] = field(default_factory=list)
    borderline_discard_ids: List[int] = field(default_factory=list)
    grace: bool = False   # pre-triage batch before enforcement; legacy grading applies

    def observations(self, cfg: TriageConfig,
                     clean_article_id: Optional[int] = None
                     ) -> List[Tuple[int, float, float]]:
        """(article_id, score, weight) triples for the reputation EMA."""
        if self.grace:
            return []
        if not self.events and not self.proof_failures:
            return ([] if clean_article_id is None
                    else [(int(clean_article_id), 1.0, cfg.clean_weight)])
        obs: List[Tuple[int, float, float]] = [
            (e.article_id, 0.0, cfg.hard_weight if e.kind == "hard" else cfg.soft_weight)
            for e in self.events
        ]
        obs.extend((aid, 0.0, cfg.hard_weight) for aid in self.proof_failures)
        return obs


def fp_soft_event(article_id: int) -> TriageEvent:
    """Sampled claimed-relevant article the reference analysis found non-relevant."""
    return TriageEvent("soft", "triage_false_positive", article_id)


def grade_batch(
    items: List[dict],
    canary_labels: Dict[int, Tuple[str, bool]],
    det_relevant: Callable[[dict], bool],
    llm_relevant: Callable[[dict], Optional[bool]],
    stage_label: Callable[[dict], str],
    rng,
    cfg: TriageConfig,
    enforced: bool,
) -> TriageGradeResult:
    """Grade the triage layer of one returned miner batch.

    items: [{"article_id", "title", "body", "analysis_data"}] per dispatched article.
    canary_labels: article_id -> ("pos"|"neg", deterministic).
    det_relevant(item): gazetteer check on the validator's copy.
    llm_relevant(item): audit-LLM verdict; None = no verdict.
    stage_label(item): reference TriageStage label on the validator's copy.
    """
    res = TriageGradeResult(canary_ids=list(canary_labels))

    records: Dict[int, Optional[dict]] = {}
    errors = False
    for it in items:
        rec, err = extract_triage(it.get("analysis_data"))
        records[it["article_id"]] = rec
        errors = errors or err is not None

    if not enforced and not errors and not any(records.values()):
        # Pre-triage batch before the cutover: legacy grading applies.
        res.grace = True
        return res

    # 1. proof-of-read: required on every article.
    for it in items:
        aid = it["article_id"]
        proof = (it.get("analysis_data") or {}).get("proof_of_read") \
            if isinstance(it.get("analysis_data"), dict) else None
        if not verify_proof_of_read(proof, it.get("title") or "", it.get("body") or ""):
            res.proof_failures.append(aid)

    # Malformed/missing records are soft violations, treated as irrelevant claims.
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

    analysed: Dict[int, bool] = {
        it["article_id"]: (isinstance(it.get("analysis_data"), dict)
                           and bool(it["analysis_data"].get("event_fingerprint")))
        for it in items}

    for it in items:
        aid = it["article_id"]
        rec = records[aid]
        if labels[aid] == LABEL_RELEVANT:
            res.relevant_ids.append(aid)
        elif labels[aid] == LABEL_BORDERLINE:
            flag = (rec or {}).get("flag")
            if analysed[aid] and flag:
                # Analyze-then-flag lane. The flag must match the miner's own
                # submitted analysis.
                if (flag == FLAG_VALUABLE) != analysis_indicates_value(it.get("analysis_data")):
                    res.events.append(TriageEvent("soft", "borderline_flag_mismatch", aid))
                elif flag == FLAG_VALUABLE:
                    res.borderline_valuable_ids.append(aid)
                elif stage_label(it) == LABEL_IRRELEVANT:
                    # Discarded as junk, and the reference stage saw no
                    # ambiguity either: the borderline claim was unwarranted.
                    res.events.append(TriageEvent("soft", "borderline_unwarranted", aid))
                else:
                    res.borderline_discard_ids.append(aid)
            else:
                res.borderline_ids.append(aid)

    # A non-relevant label must not carry an analysis payload, except a
    # flagged borderline (the analyze-then-flag lane).
    for it in items:
        aid = it["article_id"]
        flagged_borderline = (labels[aid] == LABEL_BORDERLINE
                              and (records[aid] or {}).get("flag"))
        if (labels[aid] != LABEL_RELEVANT and aid not in canary_labels
                and not flagged_borderline and analysed[aid]):
            res.events.append(TriageEvent("soft", "analysis_on_nonrelevant", aid))

    # 2. canary checks (borderline does not exempt a canary).
    for it in items:
        aid = it["article_id"]
        if aid not in canary_labels or aid in res.proof_failures:
            continue
        kind, deterministic = canary_labels[aid]
        if kind == "pos" and labels[aid] != LABEL_RELEVANT:
            severity = "hard" if deterministic else "soft"
            res.events.append(TriageEvent(severity, "canary_pos_missed", aid))
        elif kind == "neg" and labels[aid] == LABEL_RELEVANT:
            res.events.append(TriageEvent("soft", "canary_neg_flagged", aid))

    # 3. gazetteer audit on every not-relevant claim; LLM audit on a sample
    # of irrelevant claims only. Flagged-valuable borderline is exempt (its
    # analysis is kept and deep-validated like a relevant claim).
    not_relevant = [
        it for it in items
        if labels[it["article_id"]] in (LABEL_IRRELEVANT, LABEL_BORDERLINE)
        and it["article_id"] not in res.borderline_valuable_ids
        and it["article_id"] not in canary_labels
        and it["article_id"] not in res.proof_failures
    ]
    det_clean = []
    for it in not_relevant:
        if det_relevant(it):
            res.events.append(TriageEvent(
                "hard", "false_negative_deterministic", it["article_id"]))
        elif labels[it["article_id"]] == LABEL_IRRELEVANT:
            det_clean.append(it)
    for it in rng.sample(det_clean, min(cfg.audit_irrelevant_n, len(det_clean))):
        if llm_relevant(it) is True:
            res.events.append(TriageEvent("soft", "false_negative_llm",
                                          it["article_id"]))

    contradicted = {e.article_id for e in res.events}
    res.borderline_valuable_ids = [
        a for a in res.borderline_valuable_ids if a not in contradicted]
    res.borderline_discard_ids = [
        a for a in res.borderline_discard_ids if a not in contradicted]
    res.retire_candidate_ids = [
        it["article_id"] for it in not_relevant
        if it["article_id"] not in contradicted
        and labels[it["article_id"]] == LABEL_IRRELEVANT
    ] + [
        it["article_id"] for it in items
        if labels[it["article_id"]] == LABEL_IRRELEVANT
        and it["article_id"] in canary_labels
        and canary_labels[it["article_id"]][0] == "neg"
        and it["article_id"] not in contradicted
    ]
    return res


class CanaryPool:
    """In-memory pool of pre-labeled canary articles. Entries expire after
    ttl seconds or max_exposures uses. Deliberately not persisted — the pool
    refills within a dispatch tick, and disk state can desync from the
    in-memory article payloads."""

    def __init__(self, cfg: TriageConfig, now: Callable[[], float] = time.time):
        self._cfg = cfg
        self._now = now
        # article_id -> {kind, deterministic, born, exposures}
        self._entries: Dict[int, dict] = {}

    def add(self, article_id: int, kind: str, deterministic: bool) -> None:
        """Register a canary. Re-adding a known id is a no-op."""
        assert kind in ("pos", "neg")
        if article_id in self._entries:
            return
        self._entries[article_id] = {
            "kind": kind, "deterministic": bool(deterministic),
            "born": self._now(), "exposures": 0,
        }

    def _alive(self, e: dict) -> bool:
        return (self._now() - e["born"] < self._cfg.canary_ttl_s
                and e["exposures"] < self._cfg.canary_max_exposures)

    def prune(self) -> None:
        self._entries = {k: v for k, v in self._entries.items() if self._alive(v)}

    def ids(self) -> set:
        return set(self._entries)

    def size(self, kind: str) -> int:
        return sum(1 for e in self._entries.values() if e["kind"] == kind and self._alive(e))

    def draw(self, kind: str, rng, available=None, charge: int = 1) -> Optional[int]:
        """Pick a live canary and count the exposure. `available` restricts
        the draw to injectable ids; `charge` is how many miners will see it."""
        alive = [aid for aid, e in self._entries.items()
                 if e["kind"] == kind and self._alive(e)
                 and (available is None or aid in available)]
        if not alive:
            return None
        aid = rng.choice(alive)
        self._entries[aid]["exposures"] += max(1, int(charge))
        return aid

    def label_of(self, article_id: int) -> Optional[Tuple[str, bool]]:
        e = self._entries.get(article_id)
        if e is None:
            return None
        return e["kind"], e["deterministic"]
