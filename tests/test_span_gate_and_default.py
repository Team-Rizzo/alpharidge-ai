"""The evidence-span gate has a consequence, and floor gating is on unless disabled."""

import types

import pytest

from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.oracle import floor, runner
from tests.test_profile_client import valid

TEXT = "NVDA rose sharply after the quarterly report was published today."


def asset(symbol, spans):
    return types.SimpleNamespace(symbol=symbol, evidence_spans=list(spans))


def entity(name, spans):
    return types.SimpleNamespace(canonical_name=name, evidence_spans=list(spans))


def intel(assets=(), entities=()):
    return types.SimpleNamespace(numeric_claims=[], quotes=[], assets=list(assets),
                                 entities=list(entities))


# ---- the gate now bites ------------------------------------------------------------

def test_an_asset_without_usable_evidence_is_not_scored():
    """E18: a failed span fails that asset, not the article."""
    sub = intel([asset("NVDA", ["NVDA"]), asset("AMD", ["NVDA rose sharply after"])])
    result = floor.evaluate(sub, TEXT)
    assert result.unevidenced_assets == {"nvda"}
    assert runner._judgment_fields(sub, result)["assets"] == ["AMD"]


def test_the_article_itself_still_passes():
    sub = intel([asset("NVDA", ["NVDA"])])
    assert floor.evaluate(sub, TEXT).floor_pass


def test_entities_are_gated_the_same_way():
    sub = intel(entities=[entity("Nvidia", ["Nvidia"]),
                          entity("Report", ["quarterly report was published"])])
    result = floor.evaluate(sub, TEXT)
    assert "nvidia" in result.unevidenced_assets | result.unevidenced_entities
    assert runner._judgment_fields(sub, result)["entities"] == ["report"]


def test_an_item_with_no_spans_at_all_is_left_alone():
    """Absent evidence is not failed evidence; only a span that does not hold counts."""
    sub = intel([asset("NVDA", [])])
    result = floor.evaluate(sub, TEXT)
    assert result.unevidenced_assets == set()
    assert runner._judgment_fields(sub, result)["assets"] == ["NVDA"]


def test_scoring_without_a_floor_result_is_unchanged():
    sub = intel([asset("NVDA", ["NVDA"])])
    assert runner._judgment_fields(sub, None)["assets"] == ["NVDA"]


def test_a_dropped_asset_lowers_keeper_agreement():
    from alpharidge_ai.mechanism import scoring
    sub = intel([asset("NVDA", ["NVDA"]), asset("AMD", ["NVDA rose sharply after"])])
    result = floor.evaluate(sub, TEXT)
    grader = {"assets": ["NVDA", "AMD"]}
    gated = scoring.keeper_agreement(runner._judgment_fields(sub, result), grader)
    ungated = scoring.keeper_agreement(runner._judgment_fields(sub, None), grader)
    assert gated < ungated


# ---- floor gating defaults on ------------------------------------------------------

def test_floor_gating_is_on_when_a_profile_does_not_mention_it():
    """It withholds pay for work that failed a check. There is no operating state where
    paying for that is wanted, so it does not wait for the flip."""
    assert mp.parse(valid()).settlement.floor_gating is True


def test_it_can_still_be_turned_off_explicitly():
    raw = valid()
    raw["settlement"]["floor_gating"] = False
    assert mp.parse(raw).settlement.floor_gating is False


def test_the_switches_that_change_economics_stay_off():
    p = mp.parse(valid())
    assert p.settlement.live is False
    assert p.oracle.live is False
    assert p.rations.dispatch is False


def test_it_is_still_rejected_when_it_is_not_a_boolean():
    raw = valid()
    raw["settlement"]["floor_gating"] = "yes"
    with pytest.raises(mp.ProfileError):
        mp.parse(raw)
