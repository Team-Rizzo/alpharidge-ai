"""Every property the grader is read for is a property it must return.

An array left out of `required` may be omitted by the model, and an omitted array
arrives downstream indistinguishable from an empty one, so the miner is compared
against a reference that was never produced.
"""

from alpharidge_ai.oracle import grader


def test_every_declared_property_is_required():
    fn = grader.JUDGMENT_TOOL["function"]["parameters"]
    assert set(fn["properties"]) == set(fn["required"])


def test_arrays_are_required():
    fn = grader.JUDGMENT_TOOL["function"]["parameters"]
    arrays = {k for k, v in fn["properties"].items() if v.get("type") == "array"}
    assert arrays, "the judgment tool is expected to return list-valued fields"
    assert arrays <= set(fn["required"])
