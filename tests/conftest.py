import pytest


@pytest.fixture(autouse=True)
def _draw_log_dir(tmp_path, monkeypatch):
    from alpharidge_ai import config
    from alpharidge_ai.utils import draw_log
    monkeypatch.setattr(config, "DRAW_LOG_LOCATION", str(tmp_path / "draws"), raising=False)
    monkeypatch.setattr(draw_log, "_state", {"day": None, "fh": None})


@pytest.fixture(autouse=True)
def _llm_calls_not_paused(monkeypatch):
    from alpharidge_ai.utils import llm_spend
    monkeypatch.setattr(llm_spend, "_paused_until", [0.0])
