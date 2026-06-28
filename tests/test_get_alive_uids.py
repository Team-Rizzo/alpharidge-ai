"""Regression test for get_alive_uids (deserialize=True response handling).

Guards the bug where the code read `response.synapse.is_alive` / `response.axon.uid`
on a deserialized IsAlive synapse (which has neither). uid must be paired by index.
"""

import asyncio

from alpharidge_ai.utils.uids import get_alive_uids


class _TI:
    def __init__(self, status=200):
        self.status_code = status


class _Resp:
    """Mimics a deserialized IsAlive synapse: is_alive field + .dendrite TerminalInfo,
    and deliberately NO .synapse and NO .axon.uid."""
    def __init__(self, is_alive, status=200):
        self.is_alive = is_alive
        self.dendrite = _TI(status)


class _Axon:
    def __init__(self, serving):
        self.is_serving = serving


class _N:
    def __init__(self, n):
        self._n = n
    def item(self):
        return self._n


class _Meta:
    def __init__(self, serving_flags):
        self.axons = [_Axon(s) for s in serving_flags]
        self.n = _N(len(serving_flags))


class _Dendrite:
    def __init__(self, responses):
        self._responses = responses
        self.sent_axons = None
    async def forward(self, axons, synapse, timeout, deserialize):
        self.sent_axons = axons
        return self._responses


def test_returns_alive_uids_paired_by_index():
    # serving uids = 0, 2, 3 (uid 1 not serving). Responses, in send order:
    #   uid0 alive, uid2 dead, uid3 alive  -> expect [0, 3]
    meta = _Meta([True, False, True, True])
    dendrite = _Dendrite([_Resp(True), _Resp(False), _Resp(True)])
    out = asyncio.run(get_alive_uids(meta, dendrite))
    assert out == [0, 3]
    assert len(dendrite.sent_axons) == 3   # only serving axons pinged


def test_non_200_is_not_alive():
    meta = _Meta([True, True])
    # both report is_alive True but one had a non-200 dendrite status
    dendrite = _Dendrite([_Resp(True, status=200), _Resp(True, status=503)])
    out = asyncio.run(get_alive_uids(meta, dendrite))
    assert out == [0]


def test_no_serving_axons_returns_empty():
    meta = _Meta([False, False])
    dendrite = _Dendrite([])
    assert asyncio.run(get_alive_uids(meta, dendrite)) == []
