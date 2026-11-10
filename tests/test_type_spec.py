import json
from types import SimpleNamespace

from alpharidge_ai.schema import (
    OPEN_TYPES, TYPE_SPEC, Category, EventType, Stage, has_required, headline,
)
from alpharidge_ai.schema.types import TYPES_PATH

MARKET = {"monetary_policy", "economic_data", "trading_halt", "ipo_listing", "delisting",
          "index_change"}
POLITICS = {"election_result", "legislation_passed", "appointment", "sanction_or_tariff"}
SPORTS = {"match_result", "transfer", "injury", "award"}
TECHNOLOGY = {"product_launch", "funding", "partnership", "security_incident", "breakthrough",
              "outage", "vulnerability", "tech_regulation", "infrastructure_investment"}


def test_every_type_has_a_spec():
    assert set(TYPE_SPEC) == set(EventType)
    with open(TYPES_PATH) as f:
        assert set(json.load(f)) == {t.value for t in EventType}
    for spec in TYPE_SPEC.values():
        assert spec.group and spec.window_days > 0
        assert set(spec.stage_keywords) <= set(spec.all_stages)


def test_categories_and_no_price_moves():
    by_cat = {c: {t.value for t, s in TYPE_SPEC.items() if s.category is c} for c in Category}
    assert MARKET < by_cat[Category.FINANCE]
    assert by_cat[Category.POLITICS] == POLITICS
    assert by_cat[Category.SPORTS] == SPORTS
    assert by_cat[Category.TECHNOLOGY] == TECHNOLOGY
    assert not any("price" in t.value or "move" in t.value for t in EventType)
    moves = ("surge", "plunge", "rally", "tumble", "soar", "slump", "share price", "stock price")
    for t in MARKET:
        assert not any(m in k for k in TYPE_SPEC[EventType(t)].keywords for m in moves)


def test_open_types_have_a_price_tier():
    tiers = {t.value: TYPE_SPEC[t].tier for t in OPEN_TYPES}
    assert {t for t, v in tiers.items() if v == "high"} == {
        "merger_acquisition", "layoffs", "bankruptcy", "security_incident", "funding", "transfer",
        "breakthrough", "outage"}
    assert set(tiers.values()) == {"low", "high"}


def test_open_types_define_what_the_judge_is_asked():
    for t in OPEN_TYPES:
        spec = TYPE_SPEC[t]
        assert spec.definition and set(spec.fields) <= set(spec.slots), t
    assert TYPE_SPEC[EventType("merger_acquisition")].subject_slot == "target"
    assert TYPE_SPEC[EventType("personnel")].name_slots == ("person",)


def test_headline_is_built_from_fields():
    h = headline("merger_acquisition", ["Acme", "Widget Co", "Acme"], Stage.COMPLETED)
    assert h == "Acme and Widget Co: acquisition completed"
    assert headline("earnings", ["Acme"], Stage.ANNOUNCED) == "Acme: results"


def test_required_entries_take_any_of_their_slots():
    def ok(event_type, facts=(), roles=()):
        return has_required(SimpleNamespace(
            event_type=event_type, stage=None, actors=[SimpleNamespace(role=r) for r in roles],
            facts=[SimpleNamespace(field=f) for f in facts]))
    assert ok("earnings", ["period", "revenue"]) and ok("earnings", ["period", "eps"])
    assert not ok("earnings", ["period"]) and not ok("earnings", ["eps"])
    assert ok("merger_acquisition", roles=["acquirer", "target"])
    assert not ok("merger_acquisition", roles=["acquirer"])
    assert ok("security_incident")


def test_a_new_type_in_a_new_category_needs_data_only(tmp_path):
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    import alpharidge_ai
    pkg = tmp_path / "alpharidge_ai"
    shutil.copytree(Path(alpharidge_ai.__file__).parent, pkg,
                    ignore=shutil.ignore_patterns("__pycache__"))
    data = pkg / "schema" / "data"
    types = json.loads((data / "types.json").read_text())
    types["product_recall"] = {"category": "consumer", "group": "product_recall",
                               "window_days": 3, "tier": "low", "slots": ["product"],
                               "actor_roles": {"company": "company", "regulator": "regulator"},
                               "subject_roles": ["company"]}
    (data / "types.json").write_text(json.dumps(types))
    cats = json.loads((data / "categories.json").read_text())
    (data / "categories.json").write_text(json.dumps([*cats, "consumer"]))
    official = json.loads((pkg / "verify" / "data" / "official.json").read_text())
    official["sites"].append("recalls.example.gov/notices/")
    official["date_labels"].append("recall date")
    (pkg / "verify" / "data" / "official.json").write_text(json.dumps(official))
    code = (
        "from datetime import datetime, timezone\n"
        "from alpharidge_ai.schema import TYPE_SPEC, Category, EventType, category_of\n"
        "from alpharidge_ai.verify.dates import publish_date\n"
        "t = EventType('product_recall')\n"
        "assert category_of(t) is Category.CONSUMER\n"
        "assert TYPE_SPEC[t].subject_roles == ('company',)\n"
        "html = '<p>Recall Date: October 22, 2026</p><p>Acme recalls toasters.</p>'\n"
        "got = publish_date(html, 'https://www.recalls.example.gov/notices/1')\n"
        "assert got == (datetime(2026, 10, 22, tzinfo=timezone.utc), True), got\n")
    run = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True,
                         text=True, env={"PYTHONPATH": str(tmp_path)})
    assert run.returncode == 0, run.stderr


def test_every_qualifier_value_passes_the_intake_text_check():
    from alpharidge_ai.schema import models
    for event_type, spec in TYPE_SPEC.items():
        for name, field in spec.fields.items():
            for value in (field.get("values") or {}):
                assert models._TEXT.fullmatch(value), (event_type, name, value)
