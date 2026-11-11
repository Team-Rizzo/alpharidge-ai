"""Claim submission: the signed body hash covers the exact bytes sent."""

from types import SimpleNamespace

import httpx
import pytest
from bittensor_wallet import Keypair
from pydantic import ValidationError

from alpharidge_ai.ledger import request_auth as ra
from alpharidge_ai.ledger.api_client import AlpharidgeAPIClient
from alpharidge_ai.schema import Claim, canonical_body, category_of

KP = Keypair.create_from_seed("0x" + "33" * 32)

CLAIM = Claim(kind="event", headline="Acme agrees to buy Widget Co", event_type="merger_acquisition",
              category="finance", actors=[{"id_scheme": "wikidata", "id": "Q1", "name": "Acme"}],
              event_date="2026-10-13", facts=[{"field": "deal_value", "value": 1.2, "unit": "usd_billions"}],
              source_url="https://example.org/a")


@pytest.mark.asyncio
async def test_submit_claim_signs_the_canonical_bytes():
    seen = {}

    def handler(request: httpx.Request):
        seen["request"] = request
        return httpx.Response(202, json={"id": "1", "status": "queued"})

    client = AlpharidgeAPIClient("http://api", SimpleNamespace(hotkey=KP), max_retries=1)
    client._client = httpx.AsyncClient(base_url="http://api", transport=httpx.MockTransport(handler))
    assert (await client.submit_claim(CLAIM))["id"] == "1"
    req = seen["request"]
    assert req.method == "POST" and req.url.path == "/claims"
    assert req.content == canonical_body(CLAIM)
    h = req.headers
    assert h["X-Auth-Body-SHA256"] == ra.body_sha256(req.content)
    msg = ra.auth_message("POST", "/claims", h["X-Auth-Body-SHA256"], int(h["X-Auth-Timestamp"]),
                          h["X-Auth-Nonce"])
    assert h["X-Auth-Message"] == msg
    assert KP.verify(msg.encode(), bytes.fromhex(h["X-Auth-Signature"]))
    await client.close()


def test_category_comes_from_the_type():
    assert category_of("monetary_policy").value == "finance"
    assert category_of("transfer").value == "sports"
    assert category_of("merger_acquisition").value == "finance"
    assert category_of("sanction_or_tariff").value == "politics"


def test_event_claim_cannot_name_an_event():
    with pytest.raises(ValidationError):
        Claim(**{**CLAIM.model_dump(), "event_id": "7"})
