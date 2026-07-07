"""Enum rarity weights for the info-weighted enum component.

Rolling per-class frequency of the validator's own reference analyses -> weight
-log2(p): agreement on a common class is worth little, agreement on a rare class a lot.
Local per-validator state (does not need cross-validator identity — broadcast-aggregate
carries the resulting scores, not the table). Uniform weights until warmed up.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict

from alpharidge_ai import config
from alpharidge_ai.validator.quality import _SCORED_ENUMS, _ev

_FLOOR_P = 0.01
_MIN_SAMPLES = 200  # below this, return uniform (weights unstable on tiny n)


def _default_path() -> Path:
    return Path(getattr(config, "RARITY_STATE_LOCATION",
                        str(Path(__file__).resolve().parent.parent / ".rarity_state.json")))


class RarityTable:
    def __init__(self, path: Path = None):
        self.path = path or _default_path()
        self.counts: Dict[str, Dict[str, int]] = {}
        self.n = 0

    def load(self) -> None:
        try:
            if self.path.exists():
                d = json.loads(self.path.read_text())
                self.counts = {k: {c: int(v) for c, v in cc.items()}
                               for k, cc in (d.get("counts") or {}).items()}
                self.n = int(d.get("n", 0))
        except Exception:
            pass

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"counts": self.counts, "n": self.n}))
        tmp.replace(self.path)

    def observe(self, ref: dict) -> None:
        """Count the enum values of one reference analysis (model_dump dict)."""
        for f in _SCORED_ENUMS:
            if f in ref:
                self.counts.setdefault(f, {})
                cls = _ev(ref[f])
                self.counts[f][cls] = self.counts[f].get(cls, 0) + 1
        self.n += 1

    def weights(self) -> Dict[str, Dict[str, float]]:
        """rarity[field][class] = -log2(max(floor, p)). Empty (=> uniform) until warmed up."""
        if self.n < _MIN_SAMPLES:
            return {}
        out: Dict[str, Dict[str, float]] = {}
        for f, cc in self.counts.items():
            total = sum(cc.values()) or 1
            out[f] = {cls: -math.log2(max(_FLOOR_P, n / total)) for cls, n in cc.items()}
        return out
