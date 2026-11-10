"""text_cleaner strips page boilerplate before the shared check reads the text."""

from alpharidge_ai.verify.text_cleaner import clean_text


def test_strips_html_tags():
    assert "<" not in clean_text("<p>Acme rose 3%</p>")
    assert "Acme rose 3%" in clean_text("<p>Acme rose 3%</p>")


def test_removes_follow_us_widget():
    out = clean_text("Markets fell sharply today. Follow us on Google and Facebook for updates.")
    assert "Markets fell sharply today." in out
    assert "facebook" not in out.lower()
    assert "follow us on google" not in out.lower()


def test_removes_share_on_widget():
    out = clean_text("The deal closed. Share on Twitter, Facebook, LinkedIn.")
    assert "The deal closed." in out
    assert "twitter" not in out.lower()


def test_removes_getty_photo_credit():
    out = clean_text("The president addressed the nation. (Photo: Getty Images)")
    assert "getty" not in out.lower()
    assert "The president addressed the nation." in out


def test_removes_photo_by_credit():
    out = clean_text("Crowds gathered downtown. Photo by Jane Doe / Reuters.")
    assert "jane doe" not in out.lower()
    assert "Crowds gathered downtown." in out


def test_keeps_clean_financial_text_intact():
    src = ("The Federal Reserve cut interest rates by 25 basis points on Wednesday, "
           "sending the S&P 500 to a record high as investors cheered the decision.")
    out = clean_text(src)
    assert "Federal Reserve" in out
    assert "S&P 500" in out
    assert "record high" in out


def test_total_boilerplate_does_not_annihilate():
    # An all-boilerplate input is kept rather than emptied.
    assert clean_text("Advertisement") == "Advertisement"


def test_long_clean_article_not_over_stripped():
    src = " ".join(["Acme reported record revenue of $42 billion this quarter."] * 12)
    out = clean_text(src)
    assert len(out) >= 0.6 * len(src)


def test_deterministic():
    src = "News. Follow us on Google. (Photo: Getty Images)"
    assert clean_text(src) == clean_text(src)
