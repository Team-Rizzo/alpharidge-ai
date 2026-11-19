"""Checks that this validator prices claims as the API does.

The API serves its pay settings on /config/subnet. They never reach the epoch's points, which
come from the signed leaves. When they differ from this validator's, re-verify's price check
(`reverify.paid_right`) fails on honest claims, and an epoch whose window holds scoop debt fails
the debt check (`unbound_debt`) and keeps the last weights. So each config read compares them,
logs an alarm and records the difference (`pay_differs`): while it stands, re-verify posts a
price mismatch as unreachable, never as a disagreement that would hold an honest claim.

The API also serves the pay table switch (which of this code's tables applies from when). That
is applied, not compared: it names a table, never values.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

from alpharidge_ai.schema.types import PaySettingError, pay_settings, pay_switch, set_pay_switch

_MATCH = "the API and every validator must use the same values"
# The alarm line while the API's last served pay settings differ from this validator's.
_differs: Optional[str] = None


def pay_differs() -> Optional[str]:
    """The alarm line while the API's last served pay settings differ from this validator's;
    None when they match or the API serves none."""
    return _differs


def _shown(value) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def pay_line(own: Optional[Dict[str, float]] = None) -> str:
    """This validator's pay settings, for the start-up log."""
    own = pay_settings() if own is None else own
    values = " ".join(f"{k}={_shown(v)}" for k, v in own.items())
    return f"[CONFIG] pay settings: {values} ({_MATCH})"


def _same(a: float, b) -> bool:
    try:
        return math.isclose(a, float(b), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def pay_alarm(served: dict, own: Optional[Dict[str, float]] = None) -> Optional[str]:
    """The alarm line when the API's served pay settings differ from this validator's, else
    None. A value the API does not serve is not compared."""
    own = pay_settings() if own is None else own
    differ = [f"{k} here {_shown(v)}, API {_shown(served[k])}"
              for k, v in own.items() if k in served and not _same(v, served[k])]
    if not differ:
        return None
    return ("[CONFIG] ALARM pay settings differ from the API's: " + "; ".join(differ)
            + ". Until they match, re-verify cannot check paid claims' prices (it posts a "
            "price mismatch as unreachable, pay_settings_differ, not as a disagreement) and an "
            "epoch with scoop debt in its window fails the debt check (unbound_debt) and keeps "
            f"the last weights; {_MATCH} (EL_EVENT_PAY_X, EL_SCOOP_PREMIUM, EL_SCOOP_MISS_DEBT_X "
            "in .vali_env and the API's .env)")


def apply_switch(served) -> Optional[tuple]:
    """Apply the served pay table switch; a bad one is logged and the last known one stays.
    Returns the switch in force."""
    import bittensor as bt

    before = pay_switch()
    try:
        now = set_pay_switch(served)
    except PaySettingError as e:
        bt.logging.error(f"[CONFIG] ALARM the API serves a pay switch this validator refuses ({e}); "
                         f"keeping {before}")
        return before
    if now != before:
        bt.logging.info(f"[CONFIG] pay table switch: {now} (from API)")
    return now


def check_pay(served) -> Optional[bool]:
    """Log an alarm and record it (`pay_differs`) when the served pay settings differ from
    this validator's. True when they match, False when they differ, None when the API serves
    none."""
    import bittensor as bt

    global _differs
    if not isinstance(served, dict) or not served:
        _differs = None
        bt.logging.warning("[CONFIG] the API serves no pay settings on /config/subnet; this "
                           "validator cannot check that its pay matches the API's")
        return None
    _differs = pay_alarm(served)
    if _differs:
        bt.logging.error(_differs)
        return False
    return True
