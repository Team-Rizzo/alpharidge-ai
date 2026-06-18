"""Data-integrity guards for the shipped asset universe files.

The Sandisk(NAND)->Micron(MU) misattribution had its root cause in the *data*:
generic product/sector terms ("nand", "dram", "hbm") were listed as Micron
`unique_identifiers`, so the whole NAND-flash category resolved to MU. These guards
run over the shipped JSON so the (large, hand/▒tool-built) universe — and any future
expansion — cannot reintroduce the pattern. Category/theme words belong in
`thematic_tags`, never in `unique_identifiers`.
"""
import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "talisman_ai" / "analyzer" / "data"
ASSET_FILES = ["assets_traditional.json", "assets_expanded.json", "assets_sp1500.json"]

# Generic product / sector / theme terms that identify a CATEGORY, not a company.
# A unique_identifier is supposed to pick out exactly one asset; these never can.
GENERIC_TERMS = {
    # memory / semiconductor product categories
    "nand", "dram", "hbm", "ssd", "gpu", "cpu", "chip", "chips", "foundry",
    "lithography", "wafer", "memory", "semiconductor", "semiconductors",
    "processor", "processors", "node", "nodes", "fab", "fabs",
    # broad themes
    "ai", "artificial intelligence", "cloud", "saas", "ev", "electric vehicle",
    "evs", "5g", "iot", "blockchain", "fintech", "biotech", "data center",
    "data centers", "cybersecurity", "quantum", "robotics", "software", "hardware",
    # finance / sector generic
    "bank", "banking", "insurance", "retail", "energy", "oil", "gas", "pharma",
    "airline", "automaker", "streaming", "social media", "search", "e-commerce",
    "ecommerce", "payments", "ride-sharing", "telecom", "utility",
    # crypto generic
    "stablecoin", "defi", "layer 1", "layer 2", "l1", "l2", "smart contract",
    "nft", "meme coin", "exchange", "token", "proof of stake", "proof of work",
    "altcoin", "dao",
}


def _load(fname):
    return json.loads((DATA_DIR / fname).read_text())


def _iter_assets():
    for fname in ASSET_FILES:
        for a in _load(fname):
            yield fname, a


def _ticker(a):
    return a.get("symbol") or a.get("ticker")


def test_no_generic_terms_in_unique_identifiers():
    offenders = []
    for fname, a in _iter_assets():
        for u in (a.get("unique_identifiers") or []):
            if u.strip().lower() in GENERIC_TERMS:
                offenders.append(f"{_ticker(a)} ({fname}): {u!r}")
    assert not offenders, (
        "Generic category terms must live in thematic_tags, not unique_identifiers "
        "(they misattribute a whole category to one company):\n  " + "\n  ".join(offenders))


def test_no_famous_person_name_identifiers():
    """Person names are unreliable asset keys (a CEO is quoted across many
    unrelated stories) — they belong to no asset. Guards against reintroduction."""
    BANNED = {"elon musk", "warren buffett", "buffett", "tim cook", "jeff bezos",
              "mark zuckerberg", "jensen huang", "jamie dimon", "michael saylor",
              "satya nadella", "sundar pichai", "bob iger"}
    offenders = []
    for _, a in _iter_assets():
        for u in (a.get("unique_identifiers") or []):
            if u.lower().strip() in BANNED:
                offenders.append(f"{_ticker(a)}: {u!r}")
    assert not offenders, "Person-name identifiers must be removed:\n  " + "\n  ".join(offenders)


def test_no_overly_common_single_word_identifiers():
    """A single-token unique_identifier that is a very common English word (incl.
    plurals/inflections, by wordfreq) collides with ordinary prose ("regions",
    "minerals"). Those tickers must resolve via the multiword name + cashtag +
    ticker instead. Scoped to the generated file (hand-curated entries may keep
    deliberate exceptions)."""
    from wordfreq import zipf_frequency
    offenders = []
    for a in _load("assets_sp1500.json"):
        for u in (a.get("unique_identifiers") or []):
            if " " not in u and zipf_frequency(u.lower(), "en") >= 4.0:
                offenders.append(f"{_ticker(a)}: {u!r} (zipf {zipf_frequency(u.lower(),'en'):.2f})")
    assert not offenders, "Overly-common single-word identifiers:\n  " + "\n  ".join(offenders)


def test_unique_identifiers_are_not_shared_across_equities():
    """The same unique_identifier must not point at two different equities.

    (Crypto protocol/token pairs e.g. DAI/MKR can legitimately share a protocol
    name, so this guard is scoped to the equities file where it indicates a bug.)
    """
    owners = {}
    for a in _load("assets_traditional.json"):
        for u in (a.get("unique_identifiers") or []):
            owners.setdefault(u.strip().lower(), set()).add(_ticker(a))
    shared = {u: sorted(v) for u, v in owners.items() if len(v) > 1}
    assert not shared, f"unique_identifiers shared across equities: {shared}"
