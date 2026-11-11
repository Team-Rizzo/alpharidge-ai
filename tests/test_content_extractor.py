"""Main-content extraction: trafilatura for HTML, a jusText-style block classifier for
delimited plain text. Deterministic."""

from alpharidge_ai.verify.content_extractor import (
    ContentExtractor, looks_like_html, _block_features,
)


def test_looks_like_html():
    assert looks_like_html('<p>hi</p>') is True
    assert looks_like_html("Just plain text, no tags.") is False


def test_html_path_uses_trafilatura():
    html = ("<html><body><nav>Home About Contact</nav>"
            "<article><p>The Federal Reserve cut interest rates by 25 basis points "
            "on Wednesday, sending the S&amp;P 500 to a record high.</p></article>"
            "<footer>Follow us on Facebook Twitter</footer></body></html>")
    out = ContentExtractor().extract(html)
    assert "Federal Reserve cut interest rates" in out
    assert "Facebook" not in out
    assert "Home About Contact" not in out


def test_removes_share_bar_block():
    text = ("The Federal Reserve cut interest rates on Wednesday, a move investors "
            "had widely expected after months of cooling inflation.\n"
            "Copy link Facebook X Reddit Pinterest Flipboard Share this article")
    out = ContentExtractor().extract(text)
    assert "Federal Reserve cut interest rates" in out
    assert "Reddit" not in out and "Pinterest" not in out


def test_removes_social_nav_row():
    text = ("Stark Motors confirmed the new safety technology saves lives in a "
            "statement issued by the company on Thursday afternoon.\n"
            "Facebook Twitter Email WhatsApp Gmail")
    out = ContentExtractor().extract(text)
    assert "Stark Motors confirmed" in out
    assert "WhatsApp" not in out and "Gmail" not in out


def test_keeps_real_prose():
    text = ("Acme reported record quarterly revenue of forty-two billion dollars "
            "as demand for its data-center chips continued to surge.")
    out = ContentExtractor().extract(text)
    assert "Acme reported record quarterly revenue" in out


def test_deterministic():
    text = "Real sentence about the market.\nFacebook Twitter Email Share"
    ce = ContentExtractor()
    assert ce.extract(text) == ce.extract(text)


def test_block_features_stopword_ratio():
    f_prose = _block_features("the fund rallied as investors bought the dip")
    f_nav = _block_features("Facebook Twitter Email WhatsApp Gmail")
    assert f_prose["stopword_ratio"] > f_nav["stopword_ratio"]
    assert f_nav["stopword_ratio"] < 0.15


def test_empty_and_short_safe():
    ce = ContentExtractor()
    assert ce.extract("") == ""
    assert isinstance(ce.extract("Hi."), str)


def test_filing_text_keeps_inline_tags_entities_and_sized_children():
    from alpharidge_ai.verify.fetch import FetchConfig, read_page
    body = ("<p>On October&#160;20, 2026, <ix:nonNumeric name='dei:EntityRegistrantName'>Acme "
            "Holdings plc</ix:nonNumeric> (the &#8220;Company&#8221;) committed to a "
            "restructuring plan affecting about 1,400 roles across its operations.</p>")
    html = (f"<html><head><title>8-K</title></head><body><div style='font-size:0'>"
            f"<div style='font-size:10pt'>{body}</div></div>"
            f"<div style='font-size:0'>Hidden text</div></body></html>")
    text = read_page("u", "u", 200, html.encode(), html, FetchConfig()).text
    assert 'On October 20, 2026, Acme Holdings plc (the "Company") committed' in text
    assert "Hidden" not in text


def test_page_stylesheet_rules_and_offscreen_styles_hide_text():
    from alpharidge_ai.verify.content_extractor import strip_hidden
    import lxml.html
    css = ("/* x */ .a{color:red} .b, p.c {display: none} #d{position:absolute;"
           "left:-9999px} .e{clip:rect(0 0 0 0)} .f{width:1px;height:1px;overflow:hidden}"
           " @media (max-width:600px){.g{display:none}} div.c{display:none}")
    html = (f"<html><head><style>{css}</style></head><body><p class='a'>Shown one.</p>"
            "<p class='x b'>Gone b.</p><p class='c'>Gone c.</p><div id='d'>Gone d.</div>"
            "<span class='e'>Gone e.</span><span class='f'>Gone f.</span>"
            "<p class='g'>Shown g.</p><span class='c'>Shown c.</span>"
            "<p class='b' style='display:block'>Shown b.</p>"
            "<p style='position:fixed;top:-2000px'>Gone inline.</p></body></html>")
    text = lxml.html.fromstring(strip_hidden(html)).text_content()
    assert "Gone" not in text
    assert all(f"Shown {k}." in text for k in ("one", "g", "c", "b"))
    # A zero radius clips everything; a responsive embed box takes its size from padding.
    html = ("<p style='clip-path:circle(0)'>Gone r.</p>"
            "<p style='clip-path:circle(0.5em at 50% 50%)'>Shown r.</p>"
            "<div style='height:0;padding-bottom:56.25%;overflow:hidden'>Shown e.</div>")
    text = lxml.html.fromstring(strip_hidden(html)).text_content()
    assert "Gone" not in text and "Shown r." in text and "Shown e." in text
    # Later rules and !important win whatever the class order; print-only sheets are out.
    html = ("<style>.off{display:none} .on{display:block} .k{display:none!important}</style>"
            "<style media='print'>.p{display:none}</style><p class='on off'>Shown o.</p>"
            "<p class='k' style='display:block'>Gone k.</p><p class='p'>Shown p.</p>")
    text = lxml.html.fromstring(strip_hidden(html)).text_content()
    assert "Gone" not in text and "Shown o." in text and "Shown p." in text
