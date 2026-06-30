"""Unit tests for the adaptive-dispatch pilot metrics line."""

from alpharidge_ai.utils.dispatch_metrics import AdaptiveDispatchMetrics


def _parse(line):
    assert line.startswith("[ADAPTIVE_METRICS]")
    out = {}
    for tok in line.split()[1:]:
        k, v = tok.split("=")
        out[k] = v
    return out


def test_counts_and_rates():
    m = AdaptiveDispatchMetrics()
    for _ in range(10):
        m.incr("dispatched")
    m.incr("ack_ok", 8)
    m.incr("ack_fail", 2)
    m.incr("valid", 6)
    m.incr("invalid", 2)
    m.incr("timeout", 2)
    for hk in ("a", "b", "c"):
        m.mark_scored(hk)

    m.incr("depth_dispatched", 4)
    for hk in ("a", "a", "d"):     # 'a' seen twice -> 2 distinct timed-out miners
        m.mark_timeout(hk)
    m.mark_timeout("")             # ignored

    d = _parse(m.format_line(window_values=[1.0, 2.0, 3.0], live=5, on_cooldown=1))
    assert d["dispatched"] == "10"
    assert d["depth_dispatched"] == "4"
    assert d["timeout"] == "2"             # event count
    assert d["timeout_miners"] == "2"      # distinct set (a, d)
    assert d["distinct_scored"] == "3"
    assert d["ack_fail"] == "2"
    assert d["completion_pct"] == "60.0"     # 6/10
    assert d["ackfail_pct"] == "20.0"        # 2/10
    assert d["timeout_pct"] == "20.0"        # 2/10
    assert d["window_min"] == "1.00"
    assert d["window_max"] == "3.00"
    assert d["window_med"] == "2.00"
    assert d["window_ge2"] == "2"            # windows 2.0 and 3.0
    assert d["window_ge3"] == "1"            # window 3.0
    assert d["window_ge4"] == "0"
    assert d["live"] == "5"
    assert d["on_cooldown"] == "1"


def test_ack_latency_percentiles():
    m = AdaptiveDispatchMetrics()
    for v in [0.1, 0.2, 0.3, 0.4, 5.0]:   # one slow tail
        m.record_ack(v)
    m.record_ack(None)                     # ignored
    d = _parse(m.format_line(window_values=[], live=0, on_cooldown=0))
    assert d["ack_n"] == "5"
    assert d["ack_p50"] == "0.30"          # median
    assert d["ack_p95"] == "5.00"          # tail surfaces


def test_ack_empty_safe():
    m = AdaptiveDispatchMetrics()
    d = _parse(m.format_line(window_values=[], live=0, on_cooldown=0))
    assert d["ack_n"] == "0"
    assert d["ack_p50"] == "0.00"
    assert d["ack_p95"] == "0.00"


def test_empty_is_safe_no_div_zero():
    m = AdaptiveDispatchMetrics()
    d = _parse(m.format_line(window_values=[], live=0, on_cooldown=0))
    assert d["dispatched"] == "0"
    assert d["completion_pct"] == "0.0"
    assert d["timeout_pct"] == "0.0"
    assert d["window_n"] == "0"
    assert d["window_med"] == "0.00"
    assert d["window_ge2"] == "0"
    assert d["depth_dispatched"] == "0"


def test_reset_clears_counts_and_scored():
    m = AdaptiveDispatchMetrics()
    m.incr("dispatched", 5)
    m.mark_scored("a")
    m.record_ack(2.0)
    m.reset()
    d = _parse(m.format_line(window_values=[], live=0, on_cooldown=0))
    assert d["dispatched"] == "0"
    assert d["distinct_scored"] == "0"
    assert d["ack_n"] == "0"


def test_distinct_scored_dedupes():
    m = AdaptiveDispatchMetrics()
    m.mark_scored("a")
    m.mark_scored("a")
    m.mark_scored("b")
    m.mark_scored("")     # ignored
    d = _parse(m.format_line(window_values=[], live=0, on_cooldown=0))
    assert d["distinct_scored"] == "2"
