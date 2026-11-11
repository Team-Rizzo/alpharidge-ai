import httpx
from bittensor_wallet import Keypair

from alpharidge_ai.ledger import request_auth as ra

KP = Keypair.create_from_seed("0x" + "22" * 32)


def test_signature_binds_method_path_body_timestamp_and_nonce():
    h = ra.signed_headers(KP, "post", "http://api/claims?x=1", b'{"a":1}', timestamp=100, nonce="ab")
    msg = ra.auth_message("POST", "/claims?x=1", ra.body_sha256(b'{"a":1}'), 100, "ab")
    assert h["X-Auth-Message"] == msg and h["X-Auth-Nonce"] == "ab"
    assert h["X-Auth-Body-SHA256"] == ra.body_sha256(b'{"a":1}')
    assert KP.verify(msg.encode(), bytes.fromhex(h["X-Auth-Signature"]))


def test_fresh_nonce_per_request():
    a = ra.signed_headers(KP, "GET", "http://api/x")
    b = ra.signed_headers(KP, "GET", "http://api/x")
    assert a["X-Auth-Nonce"] != b["X-Auth-Nonce"]


def test_target_matches_what_httpx_sends():
    req = httpx.Client(base_url="http://api").build_request("GET", "/ledger/points", params={"epoch": 7})
    assert ra.request_target(str(req.url)) == "/ledger/points?epoch=7"


def test_message_matches_the_api_format():
    assert ra.auth_message("post", "/claims?x=1", "ab", 100, "cd") == \
        "alpharidge-ai-auth:v2|POST|/claims?x=1|ab|100|cd"
