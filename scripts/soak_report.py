"""Relevance-judge report from the draw log and validator log. Read-only; run from the alpharidge-ai folder.

    python scripts/soak_report.py [--since-hours 24] [--draws .triage_draws] [--log ~/.pm2/logs/alpharidge.validator-out.log]
"""
import argparse, collections, glob, json, os, re, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--since-hours", type=float, default=24)
ap.add_argument("--draws", default=".triage_draws")
ap.add_argument("--log", default=os.path.expanduser("~/.pm2/logs/alpharidge.validator-out.log"))
ap.add_argument("--threshold", type=float, default=0.05)
a = ap.parse_args()
cutoff = time.time() - a.since_hours * 3600

per = collections.defaultdict(lambda: [0, 0, 0, 0])      # judged, non-market, batches, would-zero
last_rate = {}
for f in sorted(glob.glob(os.path.join(a.draws, "draws-*.jsonl"))):
    for line in open(f, encoding="utf-8"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        j = d.get("judged")
        if d.get("t", 0) < cutoff or not j:
            continue
        p = per[d.get("hk")]
        p[0] += j.get("n", 0); p[1] += len(j.get("non_market") or []); p[2] += 1; p[3] += int(bool(j.get("zeroed")))
        last_rate[d.get("hk")] = j.get("rate")

rows = [(hk, v) for hk, v in per.items() if v[0] >= 20]
rates = sorted(v[1] / v[0] for _, v in rows)
q = lambda f: rates[min(len(rates) - 1, int(f * len(rates)))] if rates else float("nan")
tot = [sum(v[i] for v in per.values()) for i in range(4)]
print(f"window: last {a.since_hours:.0f} h | hotkeys judged: {len(per)} ({len(rows)} with >= 20 judged claims)")
print(f"claims judged: {tot[0]} | judged non-market: {tot[1]} ({100 * tot[1] / max(1, tot[0]):.1f}%)")
print(f"batches with judging: {tot[2]} | batches that would be zeroed: {tot[3]} ({100 * tot[3] / max(1, tot[2]):.1f}%)")
print(f"per-hotkey non-market share p10/p50/p90/p95: {q(.1):.3f} / {q(.5):.3f} / {q(.9):.3f} / {q(.95):.3f}")
off = sum(1 for hk, r in last_rate.items() if r is not None and r > a.threshold)
print(f"hotkeys currently above the {a.threshold:.0%} threshold (would be repeat offenders): {off} of {len(last_rate)}")
print("lowest-rate hotkeys (>= 20 judged):")
for hk, v in sorted(rows, key=lambda x: x[1][1] / x[1][0])[:10]:
    print(f"  {hk[:12]}  judged {v[0]:4d}  non-market {100 * v[1] / v[0]:5.1f}%")

# hourly shadow lines, judge timing, and LLM spend
shadow, spend = [], []
pat = re.compile(r"\[TRIAGE_SHADOW\] last hour (.*)")
try:
    for line in open(a.log, encoding="utf-8", errors="replace"):
        m = pat.search(line)
        if m:
            shadow.append(dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv))
        if "[LLM_SPEND]" in line:
            spend.append(line.strip()[-200:])
except FileNotFoundError:
    pass
if shadow:
    js = [float(s.get("judge_s_per_batch", 0)) for s in shadow]
    un = sum(int(s.get("unjudged", 0)) for s in shadow); cl = sum(int(s.get("claims", 0)) for s in shadow)
    print(f"[TRIAGE_SHADOW] hours logged: {len(shadow)} | judge seconds per batch median {statistics.median(js):.2f} max {max(js):.2f}"
          f" | unjudged {un} of {cl} claims ({100 * un / max(1, cl):.1f}%)"
          f" | F1 hits {sum(int(s.get('f1_hits', 0)) for s in shadow)} | index-page hits {sum(int(s.get('index_hits', 0)) for s in shadow)}")
print("last LLM spend lines:"); [print("  " + s) for s in spend[-3:]]
