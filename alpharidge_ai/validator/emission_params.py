"""Where the emission curve's values come from.

One place, so the gating path, the snapshot and the dashboard cannot drift apart. A
published profile wins; without one, the served config applies and behaviour is
unchanged.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional

from alpharidge_ai import config


@dataclass(frozen=True)
class EmissionParams:
    midpoint: float
    gain: float
    ceiling: float
    bonus_start: float
    bonus_full: float
    n_min: int
    source: str

    def as_args(self):
        """Positional arguments for reputation.emission()."""
        return (self.midpoint, self.gain, self.ceiling, self.bonus_start,
                self.bonus_full)


def _from_config() -> EmissionParams:
    return EmissionParams(
        midpoint=float(getattr(config, "EMISSION_MIDPOINT", 0.59)),
        gain=float(getattr(config, "EMISSION_GAIN", 100.0)),
        ceiling=float(getattr(config, "EMISSION_BONUS_CEILING", 0.0)),
        bonus_start=float(getattr(config, "EMISSION_BONUS_START", 0.63)),
        bonus_full=float(getattr(config, "EMISSION_BONUS_FULL", 0.75)),
        n_min=int(getattr(config, "EMISSION_N_MIN", 0)),
        source="config",
    )


def resolve(profile=None) -> EmissionParams:
    """The curve in force. Falls back to served config when no profile is active."""
    if profile is None:
        return _from_config()
    e = profile.emission
    return EmissionParams(
        midpoint=float(e.midpoint), gain=float(e.gain), ceiling=float(e.ceiling),
        bonus_start=float(e.bonus_start), bonus_full=float(e.bonus_full),
        n_min=int(e.n_min), source=f"profile v{profile.version}",
    )


def live_median(snapshot, n_min: int, registered) -> Optional[float]:
    """Median reputation among registered hotkeys with enough observations to count.

    Reported, never applied. The midpoint is meant to sit near the field, and this is
    how far it has drifted; moving it is a publish, so that every validator moves at
    the same block rather than each drifting on its own.

    `registered` is the hotkey set currently on the metagraph, and it is required rather
    than optional. The store keeps every hotkey it has ever seen, so a median taken over
    it describes a population several times the size of the field being paid — mostly
    hotkeys that left, with their reputation frozen wherever it stood when they did.
    """
    allowed = {str(h) for h in (registered or ())}
    values = [float(v.get("r", 0.0)) for k, v in (snapshot or {}).items()
              if isinstance(v, dict) and int(v.get("n", 0)) >= int(n_min)
              and str(k) in allowed]
    return statistics.median(values) if values else None
