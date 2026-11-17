import os

# Tests price at unit pay values (event 1, premium 5, debt 2); the release's own values are
# checked in test_scoop_pay.
for _k, _v in {"EVENT_PAY_X": "1", "SCOOP_PREMIUM": "5", "SCOOP_MISS_DEBT_X": "2"}.items():
    os.environ.setdefault(f"EL_{_k}", _v)
