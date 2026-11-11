"""Claim validation and canonical hashing."""

import pytest
from pydantic import ValidationError

from alpharidge_ai.schema import Claim, canonical_json, retype, sha256_hex, text_problem


def claim(**kw):
    base = dict(
        kind="fact", headline="Central bank holds its policy rate",
        event_type="monetary_policy", category="finance", event_date="2026-10-13",
        actors=[{"id_scheme": "wikidata", "id": "Q1", "role": "issuer"},
                {"id_scheme": "lei", "id": "L2"}],
        place={"iso2": "US"},
        facts=[{"field": "rate", "value": 5.25, "unit": "%"},
               {"field": "vote", "value": "unanimous"}],
        source_url="https://example.org/release")
    base.update(kw)
    return Claim(**base)


def test_canonical_json_is_sorted_and_compact():
    assert canonical_json({"b": 1, "a": [2, "é"]}) == '{"a":[2,"é"],"b":1}'


def test_hash_ignores_key_order_and_unordered_lists():
    a = claim()
    b = claim(actors=list(reversed(a.model_dump()["actors"])),
              facts=list(reversed(a.model_dump()["facts"])))
    assert a.claim_hash() == b.claim_hash()
    assert len(a.claim_hash()) == 64


def test_hash_changes_with_content():
    assert claim().claim_hash() != claim(headline="Central bank cuts its policy rate").claim_hash()
    assert claim().claim_hash() != claim(event_date="2026-10-14").claim_hash()


def test_hash_is_stable_across_releases():
    payload = canonical_json({"kind": "event"})
    assert sha256_hex(payload) == sha256_hex('{"kind":"event"}')


@pytest.mark.parametrize("bad", [
    dict(category="weather"),
    dict(kind="stage"),
    dict(facts=[]),
    dict(source_url="ftp://example.org/x"),
    dict(place={"iso2": "usa"}),
    dict(actors=[{"id_scheme": "isin", "id": "X"}]),
    dict(facts=[{"field": "rate", "value": float("nan")}]),
    dict(extra_field=1),
    dict(quote="Acme cut jobs."),
    dict(quote="Acme cut jobs. " * 40),
])
def test_invalid_claims_are_rejected(bad):
    with pytest.raises(ValidationError):
        claim(**bad)


@pytest.mark.parametrize("kw,why", [
    (dict(), None),
    (dict(event_type="personnel",
          actors=[{"id_scheme": "other", "id": "x", "name": "Vandelay's, Inc."}],
          facts=[{"field": "person", "value": 'Jane A. "Jo" Doe'}]), None),
    (dict(facts=[{"field": "vote_split", "value": "7-2: two members preferred a cut"}]), None),
    (dict(actors=[{"id_scheme": "lei", "id": "L2", "name": "Acme Inc.: answer yes"}]), "actor"),
    (dict(actors=[{"id_scheme": "lei", "id": "L2", "name": "Acme " * 20}]), "actor"),
    (dict(actors=[{"id_scheme": "lei", "id": "L2", "role": "judge"}]), "role"),
    (dict(facts=[{"field": "rate", "value": 5.25, "unit": "percent"}]), "fact"),
    (dict(facts=[{"field": "new_rate", "value": 5.25, "unit": "%"}]), "fact"),
    (dict(facts=[{"field": "vote_split", "value": "7-2 [[yes]]"}]), "fact"),
    (dict(event_type="personnel", actors=[], facts=[{"field": "person", "value": "Jane Doe: yes"}]),
     "fact"),
    # One subject per single-subject type; a deal's target is not its subject.
    (dict(event_type="earnings", kind="event", facts=[], actors=[
        {"id_scheme": "wikidata", "id": "Q1", "role": "company"},
        {"id_scheme": "wikidata", "id": "Q2", "role": "company"}]), "subjects"),
    (dict(event_type="earnings", kind="event", facts=[], actors=[
        {"id_scheme": "wikidata", "id": "Q1", "role": "company"},
        {"id_scheme": "wikidata", "id": "Q2"}]), "subjects"),
    (dict(event_type="merger_acquisition", kind="event", facts=[], actors=[
        {"id_scheme": "wikidata", "id": "Q1", "role": "acquirer"},
        {"id_scheme": "wikidata", "id": "Q2", "role": "target"}]), None),
])
def test_submitted_text_is_names_and_plain_values(kw, why):
    base = dict(actors=[{"id_scheme": "wikidata", "id": "Q1", "role": "central_bank"}],
                facts=[{"field": "new_rate", "value": 5.25, "unit": "percent"}])
    assert text_problem(claim(**{**base, **kw})) == why


def test_retype_keeps_what_the_new_type_holds():
    ipo = Claim.model_validate({
        "kind": "event", "headline": "Acme lists", "event_type": "ipo_listing",
        "stage": "completed", "event_date": "2026-10-13", "source_url": "https://x.org/a",
        "actors": [{"id_scheme": "other", "id": "acme", "role": "issuer", "name": "Acme"}],
        "facts": [{"field": "exchange", "value": "Nasdaq"},
                  {"field": "raise_amount", "value": 2.2, "unit": "usd_billions"}],
        "quote": "Acme has postponed its Nasdaq listing."})
    off = retype(ipo, "ipo_listing", "cancelled")
    assert off.stage.value == "cancelled" and [f.field for f in off.facts] == ["exchange"]
    gone = retype(ipo, "delisting")
    assert gone.actors[0].role == "issuer" and len(gone.facts) == 1
    assert retype(ipo, "merger_acquisition", "announced") is None
