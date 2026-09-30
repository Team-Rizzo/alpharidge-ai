"""Canary checks run beside dispatch, never in front of it."""

import collections
import threading
import time

import tests.test_triage_e2e as e2e

TRIAGE_CFG = e2e.validator_module.TRIAGE_CFG


class Stage:
    def evaluate(self, title, body):
        return {"label": "irrelevant" if title.startswith("junk") else "relevant"}, None, None


class Auditor:
    def __init__(self, delay=0.0, verdict=False, fail=False):
        self.delay, self.verdict, self.fail = delay, verdict, fail
        self.calls = 0
        self.lock = threading.Lock()

    def relevance_verdict(self, title, body, framing="strict"):
        with self.lock:
            self.calls += 1
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("provider down")
        return self.verdict(title) if callable(self.verdict) else self.verdict


def _art(aid, junk=True):
    return e2e.NewsArticleForScoring(id=aid, url=f"http://x/{aid}", title=("junk " if junk else "news ") + str(aid),
                                     content="body text", source="test")


def _validator(auditor):
    v = e2e.HarnessValidator()
    v._triage_stage = Stage()
    v._triage_auditor = auditor
    return v


def _wait(v, timeout=10.0):
    end = time.time() + timeout
    while v._mint_future is not None and not v._mint_future.done() and time.time() < end:
        time.sleep(0.01)


def test_a_slow_check_does_not_hold_up_dispatch():
    v = _validator(Auditor(delay=1.0))
    tick = [_art(i) for i in range(1, 41)]
    t0 = time.time()
    out = v._canary_tick(tick)
    assert time.time() - t0 < 0.2
    held = {int(a.id) for a in v._mint_future.picks}
    assert len(held) == TRIAGE_CFG.neg_mint_scan
    assert not held & {int(a.id) for a in out}
    assert len(out) == 40 - len(held)
    t0 = time.time()
    v._canary_tick([_art(i) for i in range(100, 110)])
    assert time.time() - t0 < 0.2
    _wait(v)


def test_every_article_is_dispatched_or_minted_exactly_once():
    v = _validator(Auditor(verdict=lambda title: False if int(title.split()[1]) % 3 == 0 else True))
    sent, next_id = [], 1
    for _ in range(30):
        tick = [_art(next_id + j, junk=(j % 2 == 0)) for j in range(20)]
        next_id += 20
        sent += [int(a.id) for a in v._canary_tick(tick)]
        _wait(v)
    v._canary_tick([])
    minted = set(v._canary_articles)
    assert len(sent) == len(set(sent))
    assert not minted & set(sent)
    assert set(sent) | minted == set(range(1, next_id))


def test_minted_canaries_reach_the_pool():
    v = _validator(Auditor(verdict=False))
    v._canary_tick([_art(i) for i in range(1, 21)])
    _wait(v)
    v._canary_tick([])
    assert v._canary_pool.size("neg", TRIAGE_CFG.canary_fresh_s) == TRIAGE_CFG.neg_mint_budget
    assert all(v._canary_articles[a].analysis is None for a in v._canary_articles)


def test_one_check_at_a_time():
    auditor = Auditor(delay=0.3, verdict=False)
    v = _validator(auditor)
    v._canary_tick([_art(i) for i in range(1, 21)])
    fut = v._mint_future
    second = v._canary_tick([_art(i) for i in range(100, 121)])
    assert v._mint_future is fut
    assert len(second) == 21
    _wait(v)


def test_hourly_ceiling(monkeypatch):
    monkeypatch.setattr(TRIAGE_CFG, "neg_mint_checks_per_hour", 6)
    auditor = Auditor(verdict=True)
    v = _validator(auditor)
    for n in range(10):
        v._canary_tick([_art(1000 * (n + 1) + i) for i in range(20)])
        _wait(v)
    v._canary_tick([])
    assert sum(n for _, n in v._mint_checks) == 6
    assert auditor.calls == 6
    v._mint_checks = collections.deque([(time.time() - 3700, 6)])
    v._canary_tick([_art(i) for i in range(90000, 90020)])
    assert v._mint_future is not None
    _wait(v)


def test_a_failed_check_returns_its_articles():
    v = _validator(Auditor(fail=True))
    v._canary_tick([_art(i) for i in range(1, 21)])
    held = {int(a.id) for a in v._mint_future.picks}
    _wait(v)
    out = v._canary_tick([])
    assert {int(a.id) for a in out} == held
    assert v._canary_pool.size("neg") == 0


def test_no_auditor_returns_everything():
    v = _validator(None)
    v._get_triage_auditor = lambda: None
    out = v._canary_tick([_art(i) for i in range(1, 21)])
    assert v._mint_future is None and len(out) == 20


def test_full_pool_starts_no_check():
    v = _validator(Auditor(verdict=False))
    for aid in range(5000, 5000 + TRIAGE_CFG.neg_pool_target):
        v._canary_pool.add(aid, "neg", deterministic=False)
        v._canary_articles[aid] = _art(aid)
    out = v._canary_tick([_art(i) for i in range(1, 21)])
    assert v._mint_future is None and len(out) == 20


def test_live_canaries_are_not_dispatched_as_work():
    v = _validator(Auditor(verdict=False))
    v._canary_pool.add(7, "neg", deterministic=False)
    v._canary_articles[7] = _art(7)
    out = v._canary_tick([_art(7, junk=False)])
    assert 7 not in {int(a.id) for a in out}
    _wait(v)


class _Round(e2e.HarnessValidator):
    _on_articles = e2e.validator_module.Validator._on_articles
    _inject_canaries = e2e.validator_module.Validator._inject_canaries

    def __init__(self, auditor, n_miners=30):
        super().__init__()
        self._triage_stage = Stage()
        self._triage_auditor = auditor
        self.uid = 0
        hks = [f"hk{i}" for i in range(n_miners)]
        self.metagraph = type("M", (), {"hotkeys": hks, "coldkeys": [f"ck{i}" for i in range(n_miners)],
                                        "n": type("N", (), {"item": lambda s: n_miners})()})()
        self._pending_miner_tasks = []
        self._max_pending_miner_tasks = 10 ** 6
        self._liveness = type("L", (), {"is_alive": lambda s, hk: True})()
        self.sent = []
        self._next = 1

    def _refresh_ration_plan(self, *a): pass
    def _prune_verification(self): pass
    def _count_live_served(self, *a): pass
    def _refund_credit(self, *a): pass
    def _track_task(self, task): pass

    def _select_article_targets(self, batches, exclude):
        out = []
        for b in batches:
            self._next = self._next % (len(self.metagraph.hotkeys) - 1) + 1
            out.append((self._next, b))
        return out

    async def _dispatch_article_miner_batch(self, batch, uid):
        self.sent.append((uid, [int(a.id) for a in batch]))


def test_dispatch_rounds_with_a_slow_provider(monkeypatch):
    import asyncio
    monkeypatch.setattr(e2e.config, "ADAPTIVE_BATCH_SIZE_ENABLED", False, raising=False)
    monkeypatch.setattr(e2e.config, "ADAPTIVE_DISPATCH_ENABLED", False, raising=False)
    monkeypatch.setattr(e2e.config, "TRIAGE_ENFORCED", False, raising=False)
    monkeypatch.setattr(e2e.config, "MINER_BATCH_SIZE", 6, raising=False)
    v = _Round(Auditor(delay=0.05, verdict=lambda t: int(t.split()[1]) % 2 == 0))

    async def run():
        waits, next_id = [], 1
        for _ in range(60):
            tick = [_art(next_id + j, junk=(j % 3 != 0)) for j in range(24)]
            next_id += 24
            t0 = time.time()
            await v._on_articles(tick)
            waits.append(time.time() - t0)
            await asyncio.sleep(0.03)
        return waits, next_id
    waits, next_id = asyncio.run(run())
    _wait(v)
    tail = [int(a.id) for a in v._canary_tick([])]
    assert max(waits) < 0.25
    work = [aid for _, b in v.sent for aid in b if aid not in v._canary_articles] + tail
    canaries = [(uid, aid) for uid, b in v.sent for aid in b if aid in v._canary_articles]
    assert len(work) == len(set(work))
    assert canaries, "no canary was ever placed"
    unsent = set(range(1, next_id)) - set(work) - set(v._canary_articles)
    assert len(unsent) == len(canaries)
    per_ck = collections.Counter((v.metagraph.coldkeys[uid], aid) for uid, aid in canaries)
    assert max(per_ck.values()) == 1
    assert max(collections.Counter(aid for _, aid in canaries).values()) <= TRIAGE_CFG.canary_max_exposures
