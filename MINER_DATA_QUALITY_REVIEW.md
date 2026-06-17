# SN45 Miner Output — Data Quality Review

**Date:** 2026-06-17
**Reviewer:** Claude (manual inspection of live local-stack output)
**Scope:** The V2 `ArticleIntelligence` JSON the miner produces and the validator stores in
`news_article_analysis.analysis_data` (JSONB), compared field-by-field against the original article text.
**Model under test:** `deepseek/deepseek-v4-flash` (OpenRouter) + local NER fusion
(spaCy `en_core_web_trf`, GLiNER, Flair, ReFinED, FinBERT) running on GPU.

> TL;DR — The **narrative intelligence is genuinely good** (summaries, relevance/impact
> classification, event fingerprinting, and asset→ticker mapping *when the article is actually
> financial*). The **structured extraction is noisy**: the asset extractor emits ticker false
> positives from substring matches, per-asset sentiment is overconfident, NER is noisy on
> non‑English text, and language detection is broken. A downstream consumer that filters on
> `impact_potential != negligible` and trusts the summary fields gets real value; one that trusts
> raw `assets`/`entities` at face value gets garbage mixed in.

---

## 1. What samples I looked at

Two rounds, 10 articles total, across English / Italian / Russian and across both ingest paths
(RSS headline‑only vs. CC‑News full body).

**Round 1 — sorted by most extracted assets** (to stress the structured fields). This happened to
surface only Italian articles, which biased my first read — see the correction in §5:

| # | article_id | source | topic | assets | entities |
|---|-----------|--------|-------|-------:|---------:|
| 1 | 22537 | upday.com (IT) | Trump leads 16 US billionaires to China (trade) — **financial/geopolitical** | 9 | 15 |
| 2 | 22932 | zazoom.it | "Be My Sunshine" Turkish soap-opera replay — **non‑financial** | 3 | 7 |
| 3 | 22941 | zazoom.it | Italian TV schedule changes — **non‑financial** | 3 | 15 |
| 4 | 23035 | zazoom.it | Jazz festival opening — **non‑financial** | 3 | 15 |

**Round 2 — deliberately non‑Italian, mixing RSS and full‑body CC‑News:**

| # | article_id | source | path | topic | assets | impact |
|---|-----------|--------|------|-------|-------:|--------|
| 5 | 8170 | reuters | RSS (title only) | Trump's Iran war weighs on G7 economies — **financial/geopolitical** | 2 | high |
| 6 | 8172 | reuters | RSS (title only) | Messi's feats no surprise (football) — non‑financial | 2 | negligible |
| 7 | 8229 | reuters | RSS (title only) | Pauline Hanson on multiculturalism (AU politics) | 2 | negligible |
| 8 | 22545 | va.gov | CC‑News (5000 ch) | VA electronic health‑record modernization — **English, full body** | 1 | low |
| 9 | 22640 | vesti.ru | CC‑News (1552 ch) | Pentagon $2B General Dynamics submarine deal — **Russian, full body** | 1 | low |
| 10 | 22648/22637 | vesti.ru | CC‑News | Russian general news | 0 | negligible |

---

## 2. What's good ✅

**2.1 Summarization (`chart_summary`) — the strongest output, and it translates.**
Every sample produced an accurate, well‑written **English** headline / one‑liner / `what_changed` /
`context_paragraph`, even from Italian source text. Examples (source → output headline):
- Art 22537: *"Trump Leads 16 US Billionaires to China in High-Stakes Trade Push"* — context paragraph correctly names Boeing, Blackstone, Tesla, Wall Street banks.
- Art 22932: *"Turkish Soap Opera 'Be My Sunshine' Episode Replay Streams on Mediaset."*
- Art 23035: *"Makaya McCraven Opens Jazz Festival at Teatro Asioli in Correggio."*

**2.2 Relevance / impact gating works.** The three non‑financial articles were all correctly tagged
`content_type: other`, `market_analysis_type: none`, `impact_potential: negligible`,
`overall_sentiment: neutral`, `primary_sector_symbol: OTHER`. The financial one got
`impact_potential: high`, `overall_sentiment: slightly_bullish (0.25)`, `event_type: geopolitical`.
The system *knows* what is market‑relevant.

**2.3 Asset→ticker mapping is strong on genuinely financial text.** Art 22537 correctly mapped
Goldman Sachs→`GS`, Nvidia→`NVDA`, Tesla→`TSLA`, Boeing→`BA`, Visa→`V`, Mastercard→`MA`,
BlackRock→`BLK`, Citigroup→`C` — all genuinely in the article.

**2.4 Event fingerprinting is accurate.** Art 22537: `event_type: geopolitical`,
`event_title: "Trump leads 16 US billionaires to China for trade negotiations"`,
`semantic_fingerprint: [trump_xi_meeting, us-china_trade, billionaire_delegation, …]`. Useful for dedup/clustering.

---

## 3. What's bad ❌

**3.1 Asset extraction emits false positives via loose substring/keyword matching.**
This is the **#1 issue, and it is universal — every source type, every language**, not an Italian
quirk. The `evidence_spans` reveal the mechanism. It fires on HTML attributes, tracking URLs,
acronyms, proper names, and common dictionary words:

| Article (topic) | Bogus ticker | `evidence_spans` | Reality |
|---|---|---|---|
| **Reuters — every RSS item** | `TGT` (Target) | `target="_blank"` | the **HTML attribute** in the summary |
| **Reuters — every RSS item** | `GOOGL` | `news.google.com` | the **Google News tracking URL**, not the company |
| vesti.ru (Russian sub deal) | `SQ` (Block Inc.) | `["block"]` | *"Virginia **Block** VI"* submarine class |
| va.gov (health records) | `MS` (Morgan Stanley) | `["MS"]` | the acronym **MS** |
| Soap opera | `ADA` (Cardano) | `["ada"]` | the character **Ada** Masal |
| Soap opera / jazz | `GOOGL`, `META` | `["google"]`, `["facebook"]` | *"follow us on Google / Facebook"* widget |
| TV schedule | `V` (Visa) | `["V"]` | a lone capital letter **V** |
| Jazz festival | `XAU` (Gold) | `["gold"]` | the word **gold** |
| Trump/China (financial!) | `USO` (US Oil Fund) | `["uso"]` | Italian word **uso** ("use") |

The Reuters case is the worst: **`GOOGL` + `TGT` are attached to *every* Google‑News‑sourced RSS
article** (a footballer story, an Australian‑politics story, …) purely from the HTML wrapper. Any
"who's being talked about" aggregation over this data is dominated by that artifact.

**3.2 Per‑asset sentiment is overconfident and thinly grounded.** In art 22537, Visa is "bullish over
13 mention sentences" and Citigroup "neutral over 19 mention sentences" in an article that names each
**once**. `is_primary_subject: true` is set for **all 9** assets — the flag does not discriminate.
Assets with zero mention sentences fall back to `direction: neutral, magnitude 0.5, confidence 0.5`,
i.e. no real signal but still emitted.

**3.3 NER is noisy on non‑English / boilerplate text.** Italian fragments became entities:
`"domenica"` (Sunday) → `location`; `"CHIUDONO 4"` ("4 close") → `organization`;
`"Diretta Tennis"` → `organization`; plus photo‑credit / social‑widget leakage
(`Getty Images`, `Google`, `Facebook`).

**3.4 Language detection is wrong.** Every sample is Italian but `detected_language: "en"` and
`text_stats.language: "en"`. This both mislabels the data and lets non‑English articles through filters
that assume English.

**3.5 `narrative_keywords` occasionally hallucinate.** The Trump/China trade article got
`["global-trade-war", "us-crypto-regulation", "institutional-crypto-adoption"]` — the two crypto tags
are unrelated to the article (looks like selection from a fixed taxonomy without grounding).

**3.6 (Minor) `contagion_links` / `inferred_impacts` are generic.** They are deterministic expansions
from a static `dependency_graph.json` (e.g. `NVDA → AMD (direct_competitor)`, `NVDA → TSM
(foundry_partner)`), not derived from the article. Useful as second‑order hints but identical for any
article that mentions NVDA/TSLA; don't mistake them for article‑specific signal.

**3.7 RSS items have NO article body — the miner analyzes titles only.** For every RSS‑path article
(reuters, bloomberg, cnbc, wsj, …, the *majority* of the corpus), `news_articles.content` is **empty**
and `summary` is the raw Google‑News HTML wrapper:
`<a href="https://news.google.com/rss/articles/CBMi…">Title</a>&nbsp;&nbsp;<font>Reuters</font>`.
So the entire deep pipeline (NER + 2 LLM calls + embeddings, ~15 s) runs on **just the headline**, and
the HTML wrapper is what feeds §3.1's `TGT`/`GOOGL` false positives. This is the single highest‑value
fix: RSS ingestion must resolve the Google‑News redirect to the real article URL and scrape the body,
and the system should **refuse to run full analysis on a summary/title‑only record**.
*(Contributing cause in this run: the scraper was started with `--no-scrape`, which skips body
fetching — but the guard against summary‑only analysis should exist regardless.)*

---

## 4. What needs changing (recommendations, roughly in priority order)

1. **Tighten the asset extractor** (`talisman_ai/analyzer/asset_extractor.py`). Require word‑boundary
   matches, a minimum token length (kill 1‑letter `V`, 3‑letter dictionary‑word collisions like
   `ADA`/`USO`/`XAU`), and ignore matches inside known boilerplate ("follow us on Google/Facebook",
   photo credits). Consider gating ticker emission on `impact_potential != negligible`.
2. **Fix language detection** and drop / down‑rank non‑English (or route to a multilingual path
   explicitly). Today `detected_language` is effectively hardcoded to `en`.
3. **Ground per‑asset sentiment.** Only emit a non‑neutral direction when there's a real mention
   sentence; stop setting `is_primary_subject: true` for every asset (derive it from the summary /
   primary subject, not from being mentioned at all).
4. **De‑noise NER** for non‑financial entity spam — filter common adjectives/days, require a minimum
   confidence, and strip site boilerplate before NER.
5. **Don't analyze obvious noise.** Samples 2–4 are TV listings / soaps / concerts that are correctly
   flagged `negligible` *after* paying ~15 s of GPU analysis. A cheap pre‑filter (source allow‑list,
   language, keyword) would save most of that compute.
6. **Ground `narrative_keywords`** in the article instead of selecting from a global taxonomy.

What I would **not** change: the `chart_summary`, `event_fingerprint`, `impact_potential`, and overall
sentiment fields are good — these are the parts worth building on.

---

## 5. Data‑hygiene finding (corpus composition) — *corrected after round 2*

My round‑1 read over‑weighted the Italian aggregator; that was a **sampling artifact** of sorting by
"most assets." The analyzed corpus is actually **dominated by mainstream English financial sources**:
seeking_alpha (85), investing_com (77), reuters (74), S&P Global (66), WSJ (58), bloomberg (54),
cnbc (52), FT (51), yahoo_finance (50), zacks (44), motley_fool (41)… `zazoom.it` is only ~30. So the
"non‑financial Italian" problem is real but **minor in volume**.

The bigger, higher‑volume problems are: (a) **RSS articles have no body** (§3.7) — most of that
financial corpus is being analyzed on headlines only; (b) **language detection is hardcoded‑ish to
`en`** (§3.4) even for Russian/Italian full‑body articles; and (c) the **universal asset false
positives** (§3.1). Run the §6.4 corpus queries to quantify each.

---

## 6. How to reproduce this review yourself

Everything below runs against the local stack (Postgres on `127.0.0.1:5433`, db `talisman`).
Use the `talisman_ai` conda env so the JSONB decodes cleanly.

### 6.1 Pull samples (original article + full analysis) to disk

```bash
/home/rizzo/miniconda3/envs/talisman_ai/bin/python - <<'PY'
import psycopg2, json
conn = psycopg2.connect(host="127.0.0.1", port=5433, dbname="talisman",
                        user="talisman", password="talisman_dev")
cur = conn.cursor()
# Richest outputs first (most assets); flip the ORDER BY to sample differently.
cur.execute("""
  select a.id, a.source, a.title, a.content, an.analysis_data
  from news_article_analysis an
  join news_articles a on a.id = an.article_id
  where an.analysis_data is not null
    and length(coalesce(a.content,'')) > 400
  order by jsonb_array_length(coalesce(an.analysis_data->'assets','[]'::jsonb)) desc nulls last,
           an.analyzed_at desc
  limit 4
""")
for i,(aid,src,title,content,ad) in enumerate(cur.fetchall(), 1):
    json.dump({"article_id":aid,"source":src,"title":title,
               "original_content":(content or "")[:2500],"analysis_data":ad},
              open(f"/tmp/sample_{i}_art{aid}.json","w"), indent=2, default=str)
    print(f"sample {i}: art {aid} {src} assets={len(ad.get('assets',[]))} entities={len(ad.get('entities',[]))}")
PY
```

### 6.2 Strip the 384‑dim embeddings so the JSON is readable

```bash
/home/rizzo/miniconda3/envs/talisman_ai/bin/python - <<'PY'
import json, glob
def strip(o):
    if isinstance(o, dict):
        return {k:strip(v) for k,v in o.items() if 'embedding' not in k.lower()}
    if isinstance(o, list):
        if len(o) > 50 and all(isinstance(x,(int,float)) for x in o): return f"<{len(o)} floats omitted>"
        return [strip(x) for x in o]
    return o
for p in glob.glob("/tmp/sample_*.json"):
    if p.endswith("_compact.json"): continue
    d = json.load(open(p)); d["analysis_data"] = strip(d["analysis_data"])
    json.dump(d, open(p.replace(".json","_compact.json"),"w"), indent=2, default=str)
print("wrote *_compact.json")
PY
```

Then open `/tmp/sample_*_compact.json` and read each alongside its `original_content`.

### 6.3 Fields worth checking against the source text

- `chart_summary.{headline,one_liner,what_changed,context_paragraph}` — does it match the article?
- `content_type`, `market_analysis_type`, `impact_potential`, `overall_sentiment` — relevance correct?
- `assets[*].{ticker,evidence_spans,direction,is_primary_subject}` — **inspect `evidence_spans`**; that
  is where the false positives are obvious.
- `entities[*].{name,entity_type}` — NER noise (adjectives, boilerplate, days of week).
- `detected_language` vs. the actual language of `original_content`.
- `narrative_keywords` — grounded in the article, or taxonomy hallucination?
- Ignore `*_embedding`, and treat `contagion_links` / `inferred_impacts` as static graph expansions.

### 6.4 Quick corpus‑level sanity queries

```sql
-- How much of the analyzed corpus is market-relevant vs noise?
select analysis_data->>'impact_potential' as impact, count(*)
from news_article_analysis where analysis_data is not null group by 1 order by 2 desc;

-- Claimed language distribution (expect it to be ~all 'en' even when it isn't)
select analysis_data->>'detected_language' as lang, count(*)
from news_article_analysis where analysis_data is not null group by 1 order by 2 desc;

-- Most frequently extracted tickers (eyeball for junk like V, ADA, XAU, USO on non-financial text)
select a.ticker, count(*) from news_article_analysis an,
       jsonb_to_recordset(an.analysis_data->'assets') as a(ticker text)
where an.analysis_data is not null group by 1 order by 2 desc limit 30;
```

---

*Caveat: this is a 4‑article spot check chosen to maximize structured‑field coverage, not a random
sample. Conclusions about prevalence (§5) should be confirmed with the corpus‑level queries in §6.4.*
