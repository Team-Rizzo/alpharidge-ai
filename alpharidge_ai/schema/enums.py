"""Shared vocabularies for claims, events and facts. Categories and event types are data:
`data/categories.json` and `data/types.json`."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

_DATA = Path(__file__).with_name("data")
CATEGORIES = json.loads((_DATA / "categories.json").read_text(encoding="utf-8"))


class ClaimKind(str, Enum):
    EVENT = "event"
    FACT = "fact"
    STAGE = "stage"


def _vocabulary(name: str, values) -> type:
    return Enum(name, [(v.upper(), v) for v in values], type=str, module=__name__,
                qualname=name)


Category = _vocabulary("Category", CATEGORIES)


class IdScheme(str, Enum):
    WIKIDATA = "wikidata"
    LEI = "lei"
    CIK = "cik"
    OTHER = "other"


EventType = _vocabulary(
    "EventType", json.loads((_DATA / "types.json").read_text(encoding="utf-8")))


class Stage(str, Enum):
    ANNOUNCED = "announced"
    SCHEDULED = "scheduled"
    FILED = "filed"
    PENDING = "pending"
    APPROVED = "approved"
    RULED = "ruled"
    APPEALED = "appealed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class NumericUnit(str, Enum):
    PERCENT = "percent"
    PERCENT_CHANGE = "percent_change"
    BASIS_POINTS = "basis_points"
    USD_BILLIONS = "usd_billions"
    USD_MILLIONS = "usd_millions"
    USD_TRILLIONS = "usd_trillions"
    THOUSANDS = "thousands"
    MILLIONS = "millions"
    RATIO = "ratio"
    INDEX_POINTS = "index_points"
    USD_PER_UNIT = "usd_per_unit"
    COUNT = "count"
    OTHER = "other"
