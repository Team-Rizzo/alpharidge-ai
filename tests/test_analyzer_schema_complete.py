"""Every list the extraction tool is read for is a list it must return."""

from alpharidge_ai.analyzer.article_intelligence_analyzer import EXTRACT_CLASSIFY_TOOL


def _params():
    return EXTRACT_CLASSIFY_TOOL["function"]["parameters"]


def test_array_fields_are_required():
    params = _params()
    arrays = {k for k, v in params["properties"].items() if v.get("type") == "array"}
    assert {"numeric_claims", "quotes", "additional_tickers", "economic_data"} <= arrays
    assert arrays <= set(params["required"])


def test_required_fields_are_declared():
    params = _params()
    assert set(params["required"]) <= set(params["properties"])
