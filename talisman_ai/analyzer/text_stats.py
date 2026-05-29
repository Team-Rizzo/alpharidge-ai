"""
Deterministic text statistics computation for ML feature extraction.

All functions are pure — no LLM calls, no randomness. Given identical input
text, miner and validator produce byte-identical TextStatistics objects.
This places these features in Tier 2 validation (must match exactly).
"""

from __future__ import annotations

import math
import re
from typing import Optional

from talisman_ai.models.article_intelligence import Sentiment, TextStatistics


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_SPLIT = re.compile(r"\S+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_NUMERIC_TOKEN = re.compile(r"[\$€£¥]?\d[\d,]*\.?\d*[%]?")
_QUOTE_BLOCK = re.compile(r'"[^"]{10,}"')
_TICKER_PATTERN = re.compile(r"\$[A-Z]{1,6}\b")
_LINK_PATTERN = re.compile(r"https?://\S+")
_SUBHEADING_PATTERN = re.compile(r"^#{1,4}\s|^[A-Z][A-Za-z\s]{3,50}:$|^[A-Z][A-Za-z\s]{3,50}\n={3,}", re.MULTILINE)
_TABLE_PATTERN = re.compile(r"\|.*\|.*\|")
_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`]+`")

_HEDGING_WORDS = frozenset([
    "might", "could", "may", "possibly", "perhaps", "likely", "unlikely",
    "reportedly", "allegedly", "apparently", "seems", "appears", "suggests",
    "potential", "potentially", "estimated", "approximately", "sources say",
    "according to sources", "expected to", "speculated", "rumored",
])

_CERTAINTY_WORDS = frozenset([
    "will", "confirmed", "definitive", "definitively", "certain", "certainly",
    "guaranteed", "absolutely", "undoubtedly", "clearly", "obviously",
    "announced", "officially", "verified", "proven",
])

_CLICKBAIT_PATTERNS = [
    re.compile(r"you won't believe", re.IGNORECASE),
    re.compile(r"this is (huge|massive|insane|crazy)", re.IGNORECASE),
    re.compile(r"here'?s why", re.IGNORECASE),
    re.compile(r"\d+ (things|reasons|ways)", re.IGNORECASE),
    re.compile(r"(shocking|stunning|jaw.dropping)", re.IGNORECASE),
    re.compile(r"what .+ doesn't want you to know", re.IGNORECASE),
    re.compile(r"(🚀|💰|🔥|⚡){2,}"),
    re.compile(r"!!+"),
    re.compile(r"\b(BREAKING|URGENT|ALERT)\b"),
]


def _count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:'\"")
    if len(word) <= 2:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _flesch_reading_ease(total_words: int, total_sentences: int, total_syllables: int) -> Optional[float]:
    if total_words == 0 or total_sentences == 0:
        return None
    score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    return round(max(0.0, min(100.0, score)), 2)


def _flesch_kincaid_grade(total_words: int, total_sentences: int, total_syllables: int) -> Optional[float]:
    if total_words == 0 or total_sentences == 0:
        return None
    grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    return round(max(0.0, grade), 2)


def _word_density(text_lower: str, word_set: frozenset, total_words: int) -> float:
    if total_words == 0:
        return 0.0
    words = _WORD_SPLIT.findall(text_lower)
    count = sum(1 for w in words if w.strip(".,!?;:'\"") in word_set)
    return round(min(1.0, count / total_words), 4)


def _title_sentiment(title: str) -> Sentiment:
    t = title.lower()
    bullish_terms = ["surge", "soar", "rally", "jump", "gain", "rise", "bull", "record high", "ath", "boom", "breakout"]
    bearish_terms = ["crash", "plunge", "drop", "fall", "sink", "bear", "collapse", "slump", "tumble", "selloff", "dump"]
    bull_count = sum(1 for term in bullish_terms if term in t)
    bear_count = sum(1 for term in bearish_terms if term in t)
    if bull_count > bear_count:
        return Sentiment.BULLISH if bull_count == 1 else Sentiment.VERY_BULLISH
    if bear_count > bull_count:
        return Sentiment.BEARISH if bear_count == 1 else Sentiment.VERY_BEARISH
    return Sentiment.NEUTRAL


def compute_text_stats(title: str, body: str) -> TextStatistics:
    """Compute all text statistics from title and body text.

    Every computation is deterministic — no randomness, no external calls.
    """
    full_text = body or ""
    text_lower = full_text.lower()

    # Length
    char_count = len(full_text)
    words = _WORD_SPLIT.findall(full_text)
    word_count = len(words)
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(full_text) if s.strip()]
    sentence_count = max(len(sentences), 1) if full_text else 0
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(full_text) if p.strip()]
    paragraph_count = max(len(paragraphs), 1) if full_text else 0

    # Readability
    avg_sentence_length = round(word_count / max(sentence_count, 1), 2)
    total_chars_in_words = sum(len(w) for w in words)
    avg_word_length = round(total_chars_in_words / max(word_count, 1), 2)
    total_syllables = sum(_count_syllables(w) for w in words)
    flesch_re = _flesch_reading_ease(word_count, sentence_count, total_syllables)
    flesch_kg = _flesch_kincaid_grade(word_count, sentence_count, total_syllables)

    # Density
    numeric_tokens = _NUMERIC_TOKEN.findall(full_text)
    numeric_density = round(min(1.0, len(numeric_tokens) / max(word_count, 1)), 4)

    quote_blocks = _QUOTE_BLOCK.findall(full_text)
    quoted_chars = sum(len(q) for q in quote_blocks)
    quote_density = round(min(1.0, quoted_chars / max(char_count, 1)), 4)

    tickers = _TICKER_PATTERN.findall(full_text)
    ticker_mention_count = len(tickers)
    unique_ticker_count = len(set(t.upper() for t in tickers))

    link_count = len(_LINK_PATTERN.findall(full_text))
    image_count = full_text.count("![") + full_text.lower().count("<img")

    # Structure
    has_table = bool(_TABLE_PATTERN.search(full_text))
    has_chart_image = any(kw in text_lower for kw in ["chart", "graph", "figure", "diagram"])
    has_code_block = bool(_CODE_BLOCK_PATTERN.search(full_text))
    subheading_count = len(_SUBHEADING_PATTERN.findall(full_text))

    # Title features
    title_words = _WORD_SPLIT.findall(title)
    title_word_count = len(title_words)
    title_has_number = bool(re.search(r"\d", title))
    title_has_question = "?" in title
    t_sentiment = _title_sentiment(title)

    # Language features
    hedging = _word_density(text_lower, _HEDGING_WORDS, word_count)
    certainty = _word_density(text_lower, _CERTAINTY_WORDS, word_count)

    clickbait_hits = sum(1 for p in _CLICKBAIT_PATTERNS if p.search(title))
    clickbait_score = round(min(1.0, clickbait_hits / 3.0), 4)

    return TextStatistics(
        char_count=char_count,
        word_count=word_count,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count,
        avg_sentence_length=avg_sentence_length,
        avg_word_length=avg_word_length,
        flesch_reading_ease=flesch_re,
        flesch_kincaid_grade=flesch_kg,
        numeric_density=numeric_density,
        quote_density=quote_density,
        named_entity_density=0.0,  # set later by entity extractor
        ticker_mention_count=ticker_mention_count,
        unique_ticker_count=unique_ticker_count,
        link_count=link_count,
        image_count=image_count,
        has_table=has_table,
        has_chart_image=has_chart_image,
        has_code_block=has_code_block,
        subheading_count=subheading_count,
        title_word_count=title_word_count,
        title_has_number=title_has_number,
        title_has_question=title_has_question,
        title_sentiment=t_sentiment,
        hedging_score=hedging,
        certainty_score=certainty,
        clickbait_score=clickbait_score,
        language="en",
    )
