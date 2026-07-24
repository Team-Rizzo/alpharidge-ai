"""Audit-LLM relevance verdict for triage false-negative adjudication.

One cheap tool call over the headline plus a short body slice — orders of
magnitude cheaper than the full ArticleIntelligence reference analysis, which
is what makes it affordable to run on claimed-irrelevant articles.

Deliberately asymmetric. A `True` verdict costs a miner reputation, so it is
returned only when the model is both confident and explicit; anything else —
low confidence, a malformed reply, a transport error — returns None, which the
grader treats as "no event". Mistakes by this model therefore cost us recall
in auditing, never a wrongly punished miner. Combined with the grader keeping
LLM verdicts on the soft path by default, an audit model that is merely decent
cannot meaningfully harm an honest miner.
"""
from __future__ import annotations

import json
from typing import Optional

import bittensor as bt

_TOOL = {
    "type": "function",
    "function": {
        "name": "judge_market_relevance",
        "description": "Judge whether a news article is market-relevant.",
        "parameters": {
            "type": "object",
            "properties": {
                "relevant": {
                    "type": "boolean",
                    "description": (
                        "True only if the article concerns (a) a specific tradeable "
                        "asset — company/equity, cryptocurrency, commodity, currency "
                        "pair, or sovereign debt — or (b) a macroeconomic or economic "
                        "policy event tied to a named economy: monetary policy, "
                        "inflation/employment/GDP data, fiscal, trade, tariff or "
                        "sanctions policy, or a supply shock with plausible market "
                        "transmission. General news, sports, entertainment, crime, "
                        "lifestyle, local human-interest and religion are NOT relevant."
                    ),
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in the judgement, 0.0 to 1.0.",
                },
                "reason": {"type": "string", "description": "Brief justification."},
            },
            "required": ["relevant", "confidence", "reason"],
        },
    },
}

_PROMPT = """Judge whether this news article is market-relevant for a financial \
intelligence feed.

TITLE: {title}

BODY: {body}

Answer with the judge_market_relevance tool. Be strict: most general news is \
NOT market-relevant. Only say relevant when a tradeable asset or a named-economy \
macro/policy event is genuinely the subject of the article."""


class TriageAuditor:
    """Wraps an OpenAI-compatible client for triage relevance verdicts."""

    def __init__(self, client, model: str, min_confidence: float = 0.75,
                 body_chars: int = 1500):
        self._client = client
        self._model = model
        self._min_confidence = min_confidence
        self._body_chars = body_chars

    def relevance_verdict(self, title: str, body: str) -> Optional[bool]:
        """True = confidently relevant, False = confidently not, None = no verdict."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": _PROMPT.format(
                    title=(title or "")[:300],
                    body=(body or "")[:self._body_chars])}],
                tools=[_TOOL],
                tool_choice={"type": "function",
                             "function": {"name": "judge_market_relevance"}},
                temperature=0,
                max_tokens=200,
            )
            calls = response.choices[0].message.tool_calls
            if not calls:
                return None
            payload = json.loads(calls[0].function.arguments)
            confidence = float(payload.get("confidence", 0.0))
            if confidence < self._min_confidence:
                return None
            return bool(payload["relevant"])
        except Exception as e:
            bt.logging.debug(f"[TRIAGE_AUDIT] verdict unavailable: {e}")
            return None
