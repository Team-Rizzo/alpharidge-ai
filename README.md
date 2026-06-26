# Alpharidge AI: The Perception Subnet powering AlphaRidge

## What this powers

Subnet 45 is the news analysis engine behind **[AlphaRidge](https://alpharidge.ai)**, a real-time market intelligence platform that reads the conversation across **every market** (equities, FX, crypto, commodities and indices) and turns it into structured, screenable signals.

AlphaRidge reads **1,000+ news sources** plus X and Telegram, and scores every article, post and message into a structured read: sentiment, impact, market outlook, content type, signal quality, and the asset it maps to. Those scores power the product surfaces.

**SN45 is where that scoring happens.** Every item the product shows was read, analyzed, and independently cross-checked by the subnet first. That cross-checking is the network itself: each item is scored by multiple independent models, consensus becomes the read, and where they disagree it surfaces as lower confidence.

| Layer      | What it does                                          | Delivered by         |
|------------|------------------------------------------------------|----------------------|
| Perception | Read & analyze every news article, post and message    | **SN45** (this repo) |
| Product    | Chart, Screener, Sentinel, Alerts & API              | AlphaRidge           |

![architecture phase 1](./architecture_p1.png)

---

## Phase Roadmap

| Phase              | Focus                    | Goal |
|--------------------|--------------------------|------|
| ✅ Phase 1 (current) | News & social analysis   | Read and score 1,000+ news sources plus X and Telegram into structured signals across every market |
| 🔜 Phase 2          | Chain & market data      | Fuse on-chain money flow and market data with the sentiment read |
| 🔜 Phase 3          | Predictive signals       | Forward-looking indicators and custom signals, surfaced to users through AlphaRidge |

---

## Overview

Alpharidge AI (Subnet 45) continuously analyzes the news and social streams that move **every market** (equities, FX, crypto, commodities and indices) across news articles, X posts, and Telegram messages.

Miners don't just classify an item. They run a full **intelligence pipeline** over each one, extracting multi-asset sentiment, named entities (Wikidata-linked), economic data points, cross-market contagion chains, chart summaries, narrative tags, and semantic embeddings. Validators independently re-run the same pipeline and verify the result before it earns rewards.

A coordination API leases work items to validators, stores completed analysis in Postgres, and runs the **cross-article intelligence** layer on top, clustering articles into real-world events and matching them to long-running market narratives. The result is the structured, verified signal feed that powers [AlphaRidge](https://alpharidge.ai).

---

## How It Works

### Miner

Miners receive work batches from validators over the Bittensor network (`ArticleBatch` for news, `TweetBatch` for X, `TelegramBatch` for Telegram) and run a staged analysis pipeline over each item. For news articles, the **Article Intelligence** pipeline runs four stages:

1. **Deterministic + NER (~1s):** text statistics, keyword asset extraction (200+ tracked assets), and a 4-engine NER fusion (spaCy + GLiNER + Flair + ReFinED) that resolves entities to Wikidata IDs and tradeable tickers, plus FinBERT sentence-level sentiment.
2. **LLM "Extract & Classify":** classification enums, economic data points, quotes, and an event fingerprint, with the NER results passed in as hints.
3. **LLM "Reason & Summarize":** per-asset sentiment with causal drivers, cross-market contagion chains, chart summaries, and narrative keywords, built from a compact fact sheet rather than the raw article.
4. **Embeddings:** title, body, and narrative vectors (`all-MiniLM-L6-v2`, 384-d, L2-normalized) for downstream clustering and matching.

The result is a rich, structured analysis (dozens of fields across many feature groups) returned to the validator for verification. Posts from X and Telegram run a lighter classification path.

---

### Validation

The validator independently re-runs the same analyzer on a sampled subset of each miner batch and compares results through a **multi-tier validation** ladder. If a batch fails any tier it is labeled INVALID and discarded; only a fully passing batch earns rewards.

- **Tier 1 (exact enum match):** categorical fields (content type, sentiment, impact, urgency, primary-asset sentiment direction, …) must match exactly.
- **Tier 2 (deterministic match):** content hash and text statistics (word/sentence/char counts, ticker mentions) must be byte-identical.
- **Tier 2.5 (embedding verification):** title-embedding cosine ≥ 0.90 and narrative-embedding cosine ≥ 0.80, with dimension/L2-norm sanity checks.
- **Tier 3 (weighted composite):** near-deterministic fields (asset extraction, asset sentiment, entities, economic data, event fingerprint, contagion, narrative keywords) are scored with Jaccard / text similarity and must clear a weighted threshold.

Batch validation also includes cross-article adversarial checks, such as flagging miners that send near-identical embeddings across different articles.

---

## Rewards

Rewards are tracked as **epoch-bucketed points** inside validators:

- ✅ **Accepted item**: miner earns points
- ❌ **Rejected / timed-out item**: miner accrues a penalty for that epoch

Each validator pools both signals across itself and its peers:

- its **local** points/penalties, plus
- **validator↔validator broadcasts** (compact, signed epoch snapshots that peers verify before pooling)

For a delayed epoch window (typically **E-2**), a miner keeps its reward only when its **pooled points exceed its pooled penalties**; otherwise it is zeroed for that epoch. This means the occasional routine penalty (a timeout or a single mismatch) no longer wipes out an otherwise net-positive miner, and only net-negative (low-quality or cheating) miners are zeroed. Weights are then set on-chain proportional to each surviving miner's points.

> **Emission economics:** miners are paid for **completed, verified work**. Every accepted item earns emission, and the reward is sized to cover the real cost of running the analysis **plus a healthy multiplier on top**. Do good work and you're guaranteed to come out ahead.

---

## Architecture

```

┌──────────────┐        ┌───────────┐        ┌──────────────┐
│ API Server   │  --->  │ Validator │  --->  │   Miner      │
│ (lease queue)│        │           │        │ (analysis)   │
└──────┬───────┘        └─────┬─────┘        └──────┬───────┘
       │                      │                     │
       │                      │  Batch + analysis   │
       │  completed analysis  │<────────────────────┘
       │<─────────────────────┤
       │                      │
       v                      v
┌──────────────────┐   ┌──────────────┐
│ Cross-article    │   │ Set Weights  │
│ intelligence     │   │  (on-chain)  │
│ (events +        │   └──────────────┘
│  narratives)     │
└──────────────────┘

```

After a batch passes validation, the validator returns the verified analysis to the API. The API stores it and runs the cross-article layer: **event clustering** (grouping items about the same real-world occurrence via fingerprint → content hash → embedding cosine → title overlap) and **narrative matching** (slug → embedding → keyword signals against long-running market themes), with background jobs maintaining narrative lifecycle, centroid drift, and auto-discovery.

---

## Project Structure

```
alpharidge-ai/
├── neurons/                          # Miner and validator nodes
│   ├── miner.py                      # Miner entry point
│   └── validator.py                  # Validator entry point
└── alpharidge_ai/                    # Core library
    ├── protocol.py                   # Bittensor synapses (ArticleBatch, TweetBatch, TelegramBatch, ...)
    ├── config.py                     # Configuration
    ├── models/                       # Data models (article_intelligence.py, reward.py)
    ├── analyzer/                     # Analysis pipeline
    │   ├── article_intelligence_analyzer.py   # Staged article pipeline orchestrator
    │   ├── ner_fusion.py             # spaCy + GLiNER + Flair + ReFinED + FinBERT fusion
    │   ├── asset_extractor.py        # Keyword multi-asset extraction
    │   ├── text_stats.py             # Deterministic text features
    │   ├── scoring.py                # Multi-tier validation / scoring
    │   ├── llm_cache.py              # TTL cache for deterministic LLM calls
    │   └── data/                     # Asset/narrative/contagion/entity reference data
    ├── validator/                    # Validation, grading, reward + penalty broadcasts
    └── utils/                        # Utility functions
```

---

## Configuration

Before running miners or validators, you need to set up your environment configuration files. Template files are provided that you must rename and fill in with your credentials.

Most variables ship with sensible defaults in the template, so for a standard setup you only need to set `API_KEY`.

### Miner Configuration (`.miner_env`)

Copy `.miner_env_tmpl` to `.miner_env` and configure the following variables:

| Variable | Description |
|----------|-------------|
| `MODEL` | LLM model identifier for analysis (pre-filled, e.g. `deepseek/deepseek-v4-flash`) |
| `API_KEY` | **OpenRouter API key (`sk-or-...`)**, get one at https://openrouter.ai/keys |
| `LLM_BASE` | Base URL for the LLM API (pre-filled: `https://openrouter.ai/api/v1`) |
| `MINER_API_URL` | Coordination API base URL (pre-filled: `https://api.alpharidge.ai`) |
| `BATCH_HTTP_TIMEOUT` | HTTP timeout in seconds for API requests (default: `30.0`) |

### Validator Configuration (`.vali_env`)

Copy `.vali_env_tmpl` to `.vali_env` and configure the following variables:

| Variable | Description |
|----------|-------------|
| `MODEL` | LLM model identifier for re-analysis (pre-filled, e.g. `deepseek/deepseek-v4-flash`) |
| `API_KEY` | **OpenRouter API key (`sk-or-...`)**, get one at https://openrouter.ai/keys |
| `LLM_BASE` | Base URL for the LLM API (pre-filled: `https://openrouter.ai/api/v1`) |
| `MINER_API_URL` | Coordination API base URL (pre-filled: `https://api.alpharidge.ai`) |
| `BATCH_HTTP_TIMEOUT` | HTTP timeout in seconds for API requests (default: `30.0`) |
| `VALIDATION_POLL_SECONDS` | Seconds between polling for new validations (default: `10`) |
| `VALIDATION_FETCH_LIMIT` | Items fetched per poll cycle, split into miner batches (default: `24`) |
| `VALIDATION_MAX_WORKERS` | Max concurrent validation threads making LLM calls (default: `8`) |
| `MINER_SEND_TIMEOUT` | Dispatch (validator → miner) dendrite timeout in seconds (default: `6`) |
| `SCORES_BLOCK_INTERVAL` | Blocks between fetching scores from the API (default: `100`) |
| `LLM_CACHE_TTL` / `LLM_CACHE_MAX_SIZE` | TTL (s) and max size for the in-memory LLM result cache (defaults: `300` / `1024`) |

> The verifiable-points settings (`API_ATTESTATION_PUBKEY`, `ENFORCE_SIGNED_ATTESTATIONS`, `DEEP_VERIFY_SAMPLE_RATE`) have safe defaults baked into `config.py`, so you normally don't need to set them.

---

## Running on Mainnet

### Hardware

Miners and validators both run the analyzer, so they share the same requirements:

| Component | Requirement |
|-----------|-------------|
| GPU       | ≥ 8 GB VRAM |
| RAM       | ≥ 16 GB |
| Disk      | ≥ 60 GB free (~44 GB of models download on first run) |
| CPU       | 16 cores recommended (8 minimum) |

### Install

**Get the code** (in a Python 3.12 venv):

```bash
git clone https://github.com/Team-Rizzo/alpharidge-ai.git    # note: the repo name has changed
cd alpharidge-ai
python3.12 -m venv .venv && source .venv/bin/activate
```

**Quick (recommended):**

```bash
./install.sh                 # CUDA 12.8 default; TORCH_INDEX=https://download.pytorch.org/whl/cuXXX ./install.sh for another driver
```

**Manual (equivalent, if you'd rather not use the script):**

```bash
python -m pip install --upgrade pip setuptools wheel

# 1. PyTorch: match the CUDA build to your driver (cu128 = CUDA 12.x; see https://pytorch.org):
pip install "torch>=2" --index-url https://download.pytorch.org/whl/cu128

# 2. The rest of the stack (the spaCy en_core_web_trf model is pinned in requirements.txt):
pip install -r requirements.txt
pip install -e .

# 3. ReFinED (Amazon entity-linker) is NOT on PyPI, install from GitHub with --no-deps
#    so it doesn't downgrade torch/transformers, then its small runtime deps:
pip install --no-deps "git+https://github.com/amazon-science/ReFinED.git@V1"
pip install ujson nltk Unidecode lmdb prettyprint
```

### Run Miner

```bash
cp .miner_env_tmpl .miner_env
# Switch from Chutes to OpenRouter: edit .miner_env → set API_KEY (OpenRouter sk-or-...).
# Get a key at https://openrouter.ai/keys. MODEL / LLM_BASE are pre-filled.
.venv/bin/python -m neurons.miner \
  --netuid 45 \
  --wallet.name your_coldkey_here \
  --wallet.hotkey your_hotkey_here \
  --logging.info
```

*Optional: Add `--axon.external_port` and `--axon.external_ip`

### Run Validator

```bash
cp .vali_env_tmpl .vali_env
# Switch from Chutes to OpenRouter: edit .vali_env → set API_KEY (OpenRouter sk-or-...).
# Get a key at https://openrouter.ai/keys. MODEL / LLM_BASE / MINER_API_URL are pre-filled.
.venv/bin/python -m neurons.validator \
  --netuid 45 \
  --subtensor.network finney \
  --wallet.name <your wallet> \
  --wallet.hotkey <your hotkey> \
  --logging.info
```

*Optional*: Run the validator under PM2 with the auto-updater:

```bash
python3 scripts/start_validator.py --pm2_name sn45vali -- --netuid 45 --logging.info
```

If you run into a pip error like “packages do not match the hashes…”, it can be caused by a stale pip wheel cache.
Try:

```bash
.venv/bin/python -m pip cache purge
```

---

## License

MIT
