import json
import time

from alpharidge_ai import config
from alpharidge_ai.utils import draw_log


def test_lines_are_written_at_once(tmp_path):
    draw_log.write("canary", hk="hk", n="ab", canary=5)
    files = list((tmp_path / "draws").glob("draws-*.jsonl"))
    assert len(files) == 1
    row = json.loads(files[0].read_text().splitlines()[0])
    assert row["kind"] == "canary" and row["canary"] == 5


def test_old_days_are_removed(tmp_path, monkeypatch):
    folder = tmp_path / "draws"
    folder.mkdir()
    old = time.strftime("%Y%m%d", time.gmtime(time.time() - 30 * 86400))
    recent = time.strftime("%Y%m%d", time.gmtime(time.time() - 2 * 86400))
    (folder / f"draws-{old}.jsonl").write_text("{}\n")
    (folder / f"draws-{recent}.jsonl").write_text("{}\n")
    monkeypatch.setattr(config, "DRAW_LOG_KEEP_DAYS", 14, raising=False)
    draw_log.write("picks", hk="hk", n="cd")
    names = {f.name for f in folder.glob("*.jsonl")}
    assert f"draws-{old}.jsonl" not in names and f"draws-{recent}.jsonl" in names
