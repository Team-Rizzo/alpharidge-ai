"""Exit a validator that has stopped working, so its process manager starts it again."""
from __future__ import annotations

from typing import Optional

# A step (ledger pull, weights) normally takes well under a minute.
STEP_STALL_S = 300
# A judge pull may take up to its own limit, plus the poll interval.
JUDGE_STALL_S = 1200


def stall(now: float, last_step: float, last_pull: Optional[float]) -> Optional[str]:
    """Why the validator counts as stalled, or None. `last_pull` is None when it does not
    judge."""
    if now - last_step > STEP_STALL_S:
        return f"no validator step for {now - last_step:.0f}s"
    if last_pull is not None and now - last_pull > JUDGE_STALL_S:
        return f"no judge pull finished for {now - last_pull:.0f}s"
    return None
