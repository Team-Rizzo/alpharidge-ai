"""Large-scale before/after on the real local-stack corpus.

The DB already holds the OLD pipeline's stored output (news_article_analysis.
analysis_data: assets[], entities[], detected_language) next to the raw article
text. We re-run the NEW NER pipeline on the same raw text and compare:

  * language distribution      (OLD was hardcoded "en")
  * junk-ticker volume         (GOOGL/TGT/META/XAU/V/USO/ADA from boilerplate)
  * total asset mentions        (precision proxy: fewer spurious tickers)
  * entity noise leaks          (days/boilerplate)

Usage:
  CUDA_VISIBLE_DEVICES="" python -m tests.corpus_benchmark [N]
"""

from __future__ import annotations

import json
import sys
import time

import psycopg2

from alpharidge_ai.analyzer.asset_extractor import _is_ambiguous_word

# Tickers the review flagged as boilerplate/dictionary-word artifacts.
JUNK_TICKERS = {"GOOGL", "TGT", "META", "XAU", "V", "USO", "ADA", "CL", "KC", "PA",
                "SQ", "IXIC", "ON", "IT", "GC"}

DB = dict(host="127.0.0.1", port=5433, dbname="talisman",
          user="talisman", password="talisman_dev")


def fetch(n: int):
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("""
        select a.id, a.title, a.content, an.analysis_data
        from news_article_analysis an
        join news_articles a on a.id = an.article_id
        where an.analysis_data is not null
          and length(coalesce(a.content,'')) > 200
        order by an.analyzed_at desc
        limit %s
    """, (n,))
    rows = cur.fetchall()
    conn.close()
    return rows


def old_assets(ad):
    out = {}
    for a in ad.get("assets", []) or []:
        tk = (a.get("ticker") or "").upper()
        if tk:
            out[tk] = a.get("evidence_spans", []) or []
    return out


def is_junk_evidence(spans):
    """OLD asset is artifact-like if its only evidence is ambiguous words and
    no cashtag is present."""
    toks = [s for s in spans if s]
    if not toks:
        return True
    if any("$" in s for s in toks):
        return False
    return all(_is_ambiguous_word(t.replace("$", "")) for t in toks)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    print(f"[corpus] fetching {n} articles...", flush=True)
    rows = fetch(n)
    print(f"[corpus] got {len(rows)} articles", flush=True)

    from alpharidge_ai.analyzer.ner_fusion import NERFusionEngine
    t0 = time.time()
    eng = NERFusionEngine(enable_refined=False)  # refined = canonical names only
    print(f"[corpus] engine ready in {time.time()-t0:.0f}s; running...", flush=True)

    agg = {
        "n": len(rows),
        "old_lang": {}, "new_lang": {},
        "old_asset_mentions": 0, "new_asset_mentions": 0,
        "old_junk_mentions": 0, "new_junk_mentions": 0,
        "old_ticker_counts": {}, "new_ticker_counts": {},
        "articles_old_nonen_actually": 0,
    }
    t1 = time.time()
    for i, (aid, title, content, ad) in enumerate(rows, 1):
        oa = old_assets(ad)
        old_lang = (ad.get("detected_language") or "?")
        agg["old_lang"][old_lang] = agg["old_lang"].get(old_lang, 0) + 1
        for tk, spans in oa.items():
            agg["old_asset_mentions"] += 1
            agg["old_ticker_counts"][tk] = agg["old_ticker_counts"].get(tk, 0) + 1
            if tk in JUNK_TICKERS and is_junk_evidence(spans):
                agg["old_junk_mentions"] += 1

        # Match the analyzer's 3000-char truncation for a fair before/after.
        ner = eng.extract_and_resolve(title or "", (content or "")[:3000])
        new_lang = ner.detected_language
        agg["new_lang"][new_lang] = agg["new_lang"].get(new_lang, 0) + 1
        if old_lang == "en" and new_lang != "en":
            agg["articles_old_nonen_actually"] += 1
        for a in ner.resolved_assets:
            if a.ticker:
                tk = a.ticker.upper()
                agg["new_asset_mentions"] += 1
                agg["new_ticker_counts"][tk] = agg["new_ticker_counts"].get(tk, 0) + 1
                if tk in JUNK_TICKERS:
                    agg["new_junk_mentions"] += 1

        if i % 25 == 0:
            print(f"[corpus] {i}/{len(rows)}  ({(time.time()-t1)/i:.1f}s/article)", flush=True)

    json.dump(agg, open("/tmp/corpus_benchmark_result.json", "w"), indent=2)
    _report(agg)


def _report(agg):
    print("\n================ CORPUS BEFORE/AFTER ================")
    print(f"articles: {agg['n']}")
    print(f"\nLanguage — OLD: {agg['old_lang']}")
    print(f"Language — NEW: {agg['new_lang']}")
    print(f"Articles OLD called 'en' but NEW detects non-English: {agg['articles_old_nonen_actually']}")
    print(f"\nAsset mentions — OLD: {agg['old_asset_mentions']}  NEW: {agg['new_asset_mentions']}")
    print(f"Junk-ticker mentions — OLD: {agg['old_junk_mentions']}  NEW: {agg['new_junk_mentions']}")

    def top(d, k=12):
        return sorted(d.items(), key=lambda x: -x[1])[:k]
    print(f"\nTop OLD tickers: {top(agg['old_ticker_counts'])}")
    print(f"Top NEW tickers: {top(agg['new_ticker_counts'])}")
    for tk in ["GOOGL", "TGT", "META", "XAU", "V", "USO", "ADA"]:
        o = agg["old_ticker_counts"].get(tk, 0)
        nw = agg["new_ticker_counts"].get(tk, 0)
        print(f"  {tk:6} OLD={o:4}  NEW={nw:4}")


if __name__ == "__main__":
    main()
