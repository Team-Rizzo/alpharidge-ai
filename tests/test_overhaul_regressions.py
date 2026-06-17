"""
Regression harness tests for the article-intelligence overhaul.
"""
import json
import os
import subprocess
import sys
import pytest

EVAL = "/home/rizzo/talisman/eval"


def test_overhaul_bench_runs_on_smoke(tmp_path):
    # 3-row smoke gold built from the existing sample
    src = "/home/rizzo/talisman/eval/eval/data/gold_z-ai_glm-5.1.jsonl"
    smoke = tmp_path / "smoke.jsonl"
    with open(src) as f, open(smoke, "w") as o:
        for i, line in zip(range(3), f):
            o.write(line)
    r = subprocess.run(
        [sys.executable, "scripts/overhaul_bench.py",
         "--gold", str(smoke), "--limit", "3", "--fields", "assets", "entities"],
        cwd=EVAL, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "assets" in r.stdout and "entities" in r.stdout
