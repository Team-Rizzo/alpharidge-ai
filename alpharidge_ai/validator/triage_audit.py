"""Audit-LLM relevance verdict for triage adjudication.

Returns a verdict only when the model is confident; low confidence, malformed
replies, and transport errors all return None (no event).
"""
from __future__ import annotations

import json
import re
from typing import Optional

import bittensor as bt

from alpharidge_ai.utils import llm_spend

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
                    "description": (
                        "How sure you are of your `relevant` answer, 0.0 to 1.0 "
                        "(1.0 = certain). Not the probability that the article is relevant."),
                },
                "reason": {"type": "string", "description": "Brief justification."},
            },
            "required": ["relevant", "confidence"],
        },
    },
}

# Two framings of the same question.
_PROMPTS = {
    "strict": """Judge whether this news article is market-relevant for a financial \
intelligence feed.

TITLE: {title}

BODY: {body}

Answer with the judge_market_relevance tool. Be strict: most general news is \
NOT market-relevant. Only say relevant when a tradeable asset or a named-economy \
macro/policy event is genuinely the subject of the article.""",
    "lenient": """Judge whether this news article has any plausible connection to \
financial markets. The article may be in any language.

TITLE: {title}

BODY: {body}

Answer with the judge_market_relevance tool. Say relevant if the article involves, \
even briefly, a named company or brand, a listed security, an industry or sector, a \
commodity, energy or food supply, a currency, interest rates, prices, trade, \
regulation, or the economic policy or economic conditions of a country or region. \
Say not relevant only when it clearly has none of these: for example sports results, \
entertainment, crime, weather without economic effect, religion, or local community \
news.""",
}


_LISTING = re.compile(
    r"<[A-Z0-9]{1,6}\.[A-Z]{1,3}>"
    r"|\b(?:NYSE|NASDAQ|Nasdaq|AMEX|TSXV?|LSE|HKEX|SEHK|SGX|ASX|NSE|BSE|TSE|TYO|KRX|KOSDAQ"
    r"|SSE|SZSE|XETRA|Xetra|Euronext|SIX|OTC(?:QX|QB)?|B3|BMV|JSE|TASE|MOEX)\s*:\s*[A-Z0-9.]{1,10}"
    r"|\$[A-Z]{2,5}\b")


def names_a_listing(title: str, body: str) -> bool:
    """Ticker or exchange notation anywhere in the article."""
    return bool(_LISTING.search(f"{title or ''} {body or ''}"))


class TriageAuditor:
    """Wraps an OpenAI-compatible client for triage relevance verdicts."""

    def __init__(self, client, model: str, min_confidence: float = 0.75,
                 body_chars: int = 1500):
        self._client = client
        self._model = model
        self._min_confidence = min_confidence
        self._body_chars = body_chars

    def clearly_irrelevant(self, title: str, body: str) -> bool:
        """Irrelevant beyond reasonable doubt: no listing named, and both the strict
        and the lenient reading say so with confidence."""
        if names_a_listing(title, body):
            return False
        return (self.relevance_verdict(title, body, framing="strict") is False
                and self.relevance_verdict(title, body, framing="lenient") is False)

    def relevance_verdict(self, title: str, body: str,
                          framing: str = "strict") -> Optional[bool]:
        """True = confidently relevant, False = confidently not, None = no verdict."""
        if llm_spend.paused():
            return None
        try:
            extra = llm_spend.request_body(self._client)
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": _PROMPTS[framing].format(
                    title=(title or "")[:300],
                    body=(body or "")[:self._body_chars])}],
                tools=[_TOOL],
                tool_choice={"type": "function",
                             "function": {"name": "judge_market_relevance"}},
                temperature=0,
                max_tokens=1000,
                **({"extra_body": extra} if extra else {}),
            )
            llm_spend.record("triage_audit", self._model, response)
            calls = response.choices[0].message.tool_calls
            if not calls:
                return None
            payload = json.loads(calls[0].function.arguments)
            confidence = float(payload.get("confidence", 0.0))
            if confidence < self._min_confidence:
                return None
            return bool(payload["relevant"])
        except Exception as e:
            if llm_spend.is_key_limit(e):
                llm_spend.pause(e)
                return None
            bt.logging.warning(f"[TRIAGE_AUDIT] verdict unavailable: {e}")
            return None
