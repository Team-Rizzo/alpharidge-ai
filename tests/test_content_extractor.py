"""Tests for content_extractor — main-content / boilerplate separation.

Combines: trafilatura for HTML inputs, a jusText-style stopword/link/length
block classifier for delimited plain text, and (optionally) embedding relevance
to the article topic for off-topic grammatical chrome. Deterministic.
"""

import math

from alpharidge_ai.analyzer.content_extractor import (
    ContentExtractor, looks_like_html, _block_features,
)


class FakeEmbedder:
    """Deterministic stub: vector keyed on whether text is 'on-topic'.

    On-topic blocks (contain 'fed'/'rates'/'market') -> [1,0]; off-topic
    (CTA chrome) -> [0,1]. Lets us test embedding-relevance gating without
    loading a real model.
    """
    def encode(self, text, normalize_embeddings=True):
        t = text.lower()
        on = any(w in t for w in ("fed", "rate", "rates", "market", "stocks"))
        return [1.0, 0.0] if on else [0.0, 1.0]


def test_looks_like_html():
    assert looks_like_html('<p>hi</p>') is True
    assert looks_like_html("Just plain text, no tags.") is False


def test_html_path_uses_trafilatura():
    html = ("<html><body><nav>Home About Contact</nav>"
            "<article><p>The Federal Reserve cut interest rates by 25 basis points "
            "on Wednesday, sending the S&amp;P 500 to a record high.</p></article>"
            "<footer>Follow us on Facebook Twitter</footer></body></html>")
    out = ContentExtractor().extract(html, "en")
    assert "Federal Reserve cut interest rates" in out
    assert "Facebook" not in out
    assert "Home About Contact" not in out


def test_removes_share_bar_block():
    text = ("The Federal Reserve cut interest rates on Wednesday, a move investors "
            "had widely expected after months of cooling inflation.\n"
            "Copy link Facebook X Reddit Pinterest Flipboard Share this article")
    out = ContentExtractor().extract(text, "en")
    assert "Federal Reserve cut interest rates" in out
    assert "Reddit" not in out and "Pinterest" not in out


def test_removes_social_nav_row():
    text = ("General Motors confirmed the new safety technology saves lives in a "
            "statement issued by the company on Thursday afternoon.\n"
            "Facebook Twitter Email WhatsApp Gmail")
    out = ContentExtractor().extract(text, "en")
    assert "General Motors confirmed" in out
    assert "WhatsApp" not in out and "Gmail" not in out


def test_keeps_real_prose():
    text = ("Nvidia reported record quarterly revenue of forty-two billion dollars "
            "as demand for its data-center chips continued to surge.")
    out = ContentExtractor().extract(text, "en")
    assert "Nvidia reported record quarterly revenue" in out


def test_embedding_relevance_drops_offtopic_cta():
    # Grammatical CTA with stopwords that the structural pass would keep, but
    # embedding relevance flags as off-topic.
    text = ("The Fed cut rates and stocks rallied on the news of lower borrowing costs.\n"
            "Choose our newsletter as your preferred daily companion and never miss out.")
    ce = ContentExtractor(embedder=FakeEmbedder())
    out = ce.extract(text, "en", title="Fed cuts rates, stock market rallies")
    assert "Fed cut rates" in out
    assert "preferred daily companion" not in out


def test_deterministic():
    text = "Real sentence about the market.\nFacebook Twitter Email Share"
    ce = ContentExtractor()
    assert ce.extract(text, "en") == ce.extract(text, "en")


def test_block_features_stopword_ratio():
    f_prose = _block_features("the fund rallied as investors bought the dip", "en")
    f_nav = _block_features("Facebook Twitter Email WhatsApp Gmail", "en")
    assert f_prose["stopword_ratio"] > f_nav["stopword_ratio"]
    assert f_nav["stopword_ratio"] < 0.15


def test_empty_and_short_safe():
    ce = ContentExtractor()
    assert ce.extract("", "en") == ""
    assert isinstance(ce.extract("Hi.", "en"), str)
