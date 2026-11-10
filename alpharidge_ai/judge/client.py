"""The page judge a listed validator runs: the model the operator names, or the scripted judge
for tests and local runs."""
from __future__ import annotations

import json
from typing import Optional
from urllib.parse import urlsplit

import httpx

from alpharidge_ai.judge import prompt

JUDGE_TIMEOUT_S = 10
# Estimated cost of one call, used until a response reports its own.
JUDGE_EST_USD = 0.00015

# The scripted judge's answers per check, as the API's verdict reports them.
ANSWERS = ("yes", "no", "unclear")
CHECKS = ("event", "status", "timing", "subject", "no_injection")
_P = {"yes": 0.95, "no": 0.05, "unclear": 0.5}


class ModelJudge:
    """A model that scores each question as a probability, on the page with the miner's quote
    marked. `judge_page` returns the response's answers as given."""

    free = False
    est_usd = JUDGE_EST_USD

    def __init__(self, api_key: str, url: str, model: str, timeout_s: Optional[float] = None,
                 transport: Optional[httpx.BaseTransport] = None):
        self.url, self.model = url, model
        self.client = httpx.Client(
            timeout=timeout_s or JUDGE_TIMEOUT_S, transport=transport,
            headers={"authorization": f"Bearer {api_key}", "content-type": "application/json"})
        self.last_cost: Optional[float] = None
        self.served: Optional[str] = None

    def judge_page(self, claim, text, published=None, types=None) -> dict:
        self.last_cost = self.served = None
        sent = prompt.judge_body(claim, text, published, self.model, types)
        r = self.client.post(self.url, json=sent)
        r.raise_for_status()
        body = r.json()
        cost = (body.get("usage") or {}).get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
            self.last_cost = float(cost)
        if isinstance(body.get("model"), str):
            self.served = body["model"]
        out = body.get("answers")
        return out if isinstance(out, dict) else {}


class ScriptedJudge:
    """Deterministic, free page judge for tests and the local end-to-end run. A check is yes unless
    `answers` sets it, or `script` does for a source URL that contains one of its keys;
    subject is no when a subject's name is not on the page. The answers come back in the
    model's shape, each yes at probability `yes`."""

    free = True
    model = "scripted"
    est_usd = JUDGE_EST_USD
    last_cost = 0.0

    def __init__(self, answers: Optional[dict] = None, script: Optional[dict] = None,
                 yes: float = _P["yes"]):
        self.answers = answers or {}
        self.script = script or {}
        self.yes = yes
        self.calls = 0

    def checks(self, claim, text) -> dict:
        """The answer per check, per claimed field, the reason and the type pick."""
        from alpharidge_ai.verify import floor
        from alpharidge_ai.verify.check import find, name_forms
        norm = floor.normalize(text).text
        named = all(any(find(norm, form) for form in name_forms(a.name))
                    for a in claim.actors if a.name)
        out = {c: "yes" for c in CHECKS}
        out.update(subject="yes" if named else "no", not_news_reason="none",
                   fields=[{"field": f.field, "verdict": "yes"} for f in claim.facts])
        out.update(self.answers)
        for part, answers in self.script.items():
            if part in claim.source_url:
                out.update(answers)
        return out

    def judge_page(self, claim, text, published=None, types=None) -> dict:
        self.calls += 1
        return self.raw(claim, self.checks(claim, text))

    def raw(self, claim, got: dict) -> dict:
        questions, want, carry = prompt.judge_questions(claim)
        p = {**_P, "yes": self.yes}
        fields = {str(f.get("field")): f.get("verdict") for f in got.get("fields") or ()
                  if isinstance(f, dict)}
        check = {"event": "event", "status": "status", "timing": "timing",
                 "inject": "no_injection"}
        answer = {n: got.get(check[n], "unclear") if n in check else
                  got.get("subject", "unclear") if n[0] in "aw" or n == "office" else "yes"
                  for n in want}
        for field, verdict in fields.items():
            for n in carry.get(field, ()):
                if n[0] in "qdt" and n[1:].isdigit():
                    answer[n] = verdict
        reason = got.get("not_news_reason")
        out = {}
        for n, w in want.items():
            a = answer[n] if answer[n] in ANSWERS else "unclear"
            if questions[n]["type"] == "noul":
                out[n] = {"type": "noul", "noul": p[a]}
                continue
            other = [o for o in questions[n]["criteria"] if o != w]
            losing = {c for c, r in prompt.NOT_NEWS.get(n, {}).items() if r == reason}
            pick = w if a != "no" else next((o for o in other if o in losing), other[0])
            probs = {w: p[a]}
            if pick != w:
                probs[pick] = round(1 - p[a], 6)
            out[n] = {"type": "choice", "choice": pick, "probabilities": probs}
        ta = got.get("type_answer")
        if isinstance(ta, dict) and isinstance(ta.get("kind"), str):
            out["kind"] = {"type": "choice", "choice": ta["kind"],
                           "probabilities": {ta["kind"]: ta.get("p", p["yes"])}}
        return out


def from_env(env) -> Optional[object]:
    """The judge a validator's env names (EL_JUDGE model, the default, or scripted), or None
    when it is not set up. The model judge's key is EL_JUDGE_API_KEY or API_KEY. Raises on a
    scripted judge without EL_ALLOW_SCRIPTED_JUDGE."""
    kind = (env.get("EL_JUDGE") or "model").strip()
    if kind == "scripted":
        if (env.get("EL_ALLOW_SCRIPTED_JUDGE") or "").strip().lower() != "true":
            raise RuntimeError("EL_JUDGE=scripted is for tests and local runs: "
                               "set EL_ALLOW_SCRIPTED_JUDGE=true")
        path = (env.get("EL_JUDGE_SCRIPT") or "").strip()
        script = json.load(open(path, encoding="utf-8")) if path else None
        yes = float(env.get("EL_JUDGE_SCRIPTED_YES") or _P["yes"])
        return ScriptedJudge(script=script, yes=yes)
    if kind != "model":
        raise RuntimeError(f"Unknown EL_JUDGE {kind!r}: use model or scripted")
    key, url, model = (str(env.get(k) or "").strip()
                       for k in ("EL_JUDGE_API_KEY", "EL_JUDGE_URL", "EL_JUDGE_MODEL"))
    key = key or str(env.get("API_KEY") or "").strip()
    if not (key and url and model):
        return None
    if urlsplit(url).scheme not in ("http", "https"):
        raise RuntimeError("EL_JUDGE_URL must be an http(s) URL")
    return ModelJudge(key, url, model)
