"""Durable, consensus-safe reputation state.

Reputation is a substantive-weighted recency EMA of per-article graded scores, kept per
hotkey. To keep both validators identical, per-article observations are broadcast and the
UNION (local + received) is applied at a delayed epoch close in a DETERMINISTIC order
(sort by article_id, then source) — EMA is order-dependent, so arrival order must not
matter. Mirrors the reward/penalty broadcast-store pattern (delayed application, keep a
few epochs, persist to JSON).

State is authoritative on disk; losing it resets all history, so it must be persisted and
backed up like the reward store.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from alpharidge_ai import config
from alpharidge_ai.validator import reputation as rep


def _default_path() -> Path:
    return Path(getattr(config, "REPUTATION_STATE_LOCATION",
                        str(Path(__file__).resolve().parent.parent / ".reputation_state.json")))


# one observation: (article_id, graded, weight)
Obs = Tuple[int, float, float]


@dataclass
class ReputationStore:
    path: Path = field(default_factory=_default_path)
    keep_epochs: int = 4

    # durable per-hotkey state: hotkey -> {"r": reputation, "n": sample count}
    state: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # pending observations: epoch -> sender_hotkey -> target_hotkey -> [Obs]
    obs: Dict[int, Dict[str, Dict[str, List[Obs]]]] = field(default_factory=dict)
    finalized: List[int] = field(default_factory=list)

    def load(self) -> None:
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text())
            self.state = {str(k): {"r": float(v["r"]), "n": int(v.get("n", 0))}
                          for k, v in (data.get("state") or {}).items()}
            self.finalized = [int(e) for e in (data.get("finalized") or [])][-64:]
            raw = data.get("obs") or {}
            self.obs = {int(e): {s: {t: [tuple(o) for o in lst] for t, lst in tgts.items()}
                                 for s, tgts in senders.items()}
                        for e, senders in raw.items()}
        except Exception:
            pass

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "state": self.state,
            "finalized": self.finalized[-64:],
            "obs": self.obs,
        }))
        tmp.replace(self.path)

    # ---- ingest ----
    def _add(self, epoch: int, sender: str, target: str, o: Obs) -> None:
        aid, g, w = int(o[0]), float(o[1]), float(o[2])
        if not (0.0 <= g <= 1.0) or not (0.0 < w <= 10.0):  # bounds guard
            return
        self.obs.setdefault(epoch, {}).setdefault(sender, {}).setdefault(target, []).append((aid, g, w))

    def record_local(self, epoch: int, self_hotkey: str, target: str, article_id: int,
                     graded: float, weight: float) -> None:
        """Own observation — buffered for aggregation and for broadcast (via export)."""
        self._add(epoch, self_hotkey, target, (article_id, graded, weight))

    def ingest(self, sender: str, epoch: int, targets: Dict[str, List[Obs]]) -> None:
        """A peer validator's observations for an epoch."""
        if epoch in self.finalized:
            return
        for target, lst in (targets or {}).items():
            for o in lst:
                self._add(epoch, sender, target, o)

    def export(self, epoch: int, self_hotkey: str) -> Dict[str, List[Obs]]:
        """Own observations for `epoch`, to broadcast to peers."""
        return dict((self.obs.get(epoch, {}) or {}).get(self_hotkey, {}))

    # ---- finalize (delayed) ----
    def finalize(self, epoch: int, alpha: float = None) -> None:
        """Apply the union of all senders' observations for `epoch` to the EMA in a
        deterministic order. Call for a delayed epoch (e.g. E-2) so broadcasts have settled."""
        if epoch in self.finalized or epoch not in self.obs:
            return
        alpha = rep.ALPHA if alpha is None else alpha
        # union per target: dedup identical (sender, obs), then order by (article_id, sender)
        per_target: Dict[str, List[Tuple[int, str, float, float]]] = {}
        for sender, targets in self.obs[epoch].items():
            for target, lst in targets.items():
                seen = set()
                for aid, g, w in lst:
                    key = (aid, sender)
                    if key in seen:
                        continue
                    seen.add(key)
                    per_target.setdefault(target, []).append((aid, sender, g, w))
        for target, rows in per_target.items():
            rows.sort(key=lambda x: (x[0], x[1]))  # (article_id, sender) — deterministic
            st = self.state.setdefault(target, {"r": rep.PRIOR, "n": 0})
            for _aid, _sender, g, w in rows:
                st["r"] = rep.update(st["r"], g, w, alpha)
                st["n"] += 1
        self.finalized.append(epoch)
        del self.obs[epoch]
        self._prune()
        self.save()

    def _prune(self) -> None:
        for e in sorted(self.obs)[:-self.keep_epochs]:
            del self.obs[e]
        self.finalized = self.finalized[-64:]

    # ---- read ----
    def reputation(self, hotkey: str) -> float:
        return self.state.get(hotkey, {}).get("r", rep.PRIOR)

    def samples(self, hotkey: str) -> int:
        return int(self.state.get(hotkey, {}).get("n", 0))

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Per-hotkey {r, n} for telemetry / emission."""
        return {k: dict(v) for k, v in self.state.items()}
