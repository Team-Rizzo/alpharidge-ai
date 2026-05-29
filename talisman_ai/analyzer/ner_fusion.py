"""
NER Fusion Engine — combines 4 NER models + entity resolution + sentiment.

Loads all models once at startup, processes articles in ~0.9s:
- spaCy trf: people, orgs, money, percentages, dates
- GLiNER: crypto, indices, financial institutions, economic indicators
- Flair OntoNotes: high-confidence ORG/PERSON/MONEY
- ReFinED: Wikidata entity linking (resolves ambiguity)
- FinBERT: sentence-level sentiment
- Override dict: hardcoded financial entity mappings
- Asset registry: keyword-based ticker extraction
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ner_fusion")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class ResolvedEntity:
    """A fully resolved entity with canonical identity."""
    text: str
    canonical_name: str
    entity_type: str
    ticker: Optional[str] = None
    asset_class: Optional[str] = None
    wikidata_qid: Optional[str] = None
    role: Optional[str] = None
    confidence: float = 1.0
    source: str = "unknown"
    start: int = 0
    end: int = 0
    sentiment_toward: Optional[str] = None


@dataclass
class NERResult:
    """Complete NER extraction result for an article."""
    resolved_assets: List[ResolvedEntity] = field(default_factory=list)
    resolved_entities: List[ResolvedEntity] = field(default_factory=list)
    money_values: List[dict] = field(default_factory=list)
    percentages: List[dict] = field(default_factory=list)
    dates: List[dict] = field(default_factory=list)
    sentence_sentiments: List[dict] = field(default_factory=list)


def _load_json(filename: str) -> dict:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


class NERFusionEngine:
    """Loads all NER models at init, processes articles via extract_and_resolve()."""

    def __init__(self, enable_refined: bool = True, enable_flair: bool = True,
                 enable_gliner: bool = True, enable_finbert: bool = True):
        self._overrides = _load_json("financial_overrides.json").get("entities", {})
        self._qid_map = _load_json("wikidata_ticker_map.json").get("entities", {})

        from .asset_extractor import AssetExtractor
        self._asset_extractor = AssetExtractor()

        # spaCy — always loaded (lightweight, fast)
        import spacy
        logger.info("[NER] Loading spaCy en_core_web_trf...")
        self._nlp = spacy.load("en_core_web_trf")
        logger.info("[NER] spaCy loaded")

        # GLiNER
        self._gliner = None
        if enable_gliner:
            try:
                from gliner import GLiNER
                logger.info("[NER] Loading GLiNER...")
                self._gliner = GLiNER.from_pretrained("urchade/gliner_base")
                logger.info("[NER] GLiNER loaded")
            except Exception as e:
                logger.warning(f"[NER] GLiNER failed to load: {e}")

        # Flair
        self._flair_tagger = None
        if enable_flair:
            try:
                from flair.models import SequenceTagger
                logger.info("[NER] Loading Flair OntoNotes...")
                self._flair_tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
                logger.info("[NER] Flair loaded")
            except Exception as e:
                logger.warning(f"[NER] Flair failed to load: {e}")

        # ReFinED
        self._refined = None
        if enable_refined:
            try:
                from transformers import AutoTokenizer
                _orig = AutoTokenizer.from_pretrained.__func__
                def _patched(cls, *args, **kwargs):
                    kwargs.pop("add_special_tokens", None)
                    return _orig(cls, *args, **kwargs)
                AutoTokenizer.from_pretrained = classmethod(_patched)

                from refined.inference.processor import Refined
                logger.info("[NER] Loading ReFinED aida_model...")
                self._refined = Refined.from_pretrained(model_name="aida_model", entity_set="wikidata")
                logger.info("[NER] ReFinED loaded")
            except Exception as e:
                logger.warning(f"[NER] ReFinED failed to load: {e}")

        # FinBERT
        self._finbert = None
        if enable_finbert:
            try:
                from transformers import pipeline
                logger.info("[NER] Loading FinBERT...")
                self._finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
                logger.info("[NER] FinBERT loaded")
            except Exception as e:
                logger.warning(f"[NER] FinBERT failed to load: {e}")

        # SentenceTransformer — for article embeddings
        self._embedder = None
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("[NER] Loading SentenceTransformer all-MiniLM-L6-v2...")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("[NER] SentenceTransformer loaded")
        except Exception as e:
            logger.warning(f"[NER] SentenceTransformer failed to load: {e}")

        logger.info(f"[NER] Fusion engine ready: spaCy=✓ GLiNER={'✓' if self._gliner else '✗'} "
                    f"Flair={'✓' if self._flair_tagger else '✗'} ReFinED={'✓' if self._refined else '✗'} "
                    f"FinBERT={'✓' if self._finbert else '✗'} "
                    f"Embedder={'✓' if self._embedder else '✗'} "
                    f"overrides={len(self._overrides)} qid_map={len(self._qid_map)}")

    def encode_text(self, text: str) -> Optional[List[float]]:
        """Encode text into a 384-dim L2-normalized embedding vector."""
        if self._embedder is None or not text or not text.strip():
            return None
        return self._embedder.encode(text.strip(), normalize_embeddings=True).tolist()

    def extract_and_resolve(self, title: str, body: str) -> NERResult:
        """Full NER extraction + entity resolution pipeline."""
        full_text = f"{title}\n{body}"
        result = NERResult()

        # 1. Keyword asset extraction
        asset_matches = self._asset_extractor.extract_assets(title, body)
        for m in asset_matches:
            result.resolved_assets.append(ResolvedEntity(
                text=m.evidence_spans[0] if m.evidence_spans else m.ticker,
                canonical_name=m.asset_name,
                entity_type="asset",
                ticker=m.ticker,
                asset_class=m.asset_class,
                confidence=m.disambiguation_confidence,
                source="keyword",
            ))

        # 2. spaCy NER
        raw_entities: List[Tuple[str, str, str, float, int, int]] = []  # text, label, source, conf, start, end
        doc = self._nlp(full_text)
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "NORP", "FAC", "EVENT", "PRODUCT"):
                raw_entities.append((ent.text, ent.label_, "spacy", 0.9, ent.start_char, ent.end_char))
            elif ent.label_ == "MONEY":
                result.money_values.append({"text": ent.text, "start": ent.start_char, "end": ent.end_char})
            elif ent.label_ == "PERCENT":
                result.percentages.append({"text": ent.text, "start": ent.start_char, "end": ent.end_char})
            elif ent.label_ == "DATE":
                result.dates.append({"text": ent.text, "start": ent.start_char, "end": ent.end_char})

        # 3. GLiNER
        if self._gliner:
            gliner_labels = [
                "person", "company", "organization", "financial institution",
                "cryptocurrency", "stock ticker", "index", "commodity",
                "government body", "regulatory body", "economic indicator",
            ]
            try:
                gl_entities = self._gliner.predict_entities(full_text, gliner_labels, threshold=0.4)
                for ge in gl_entities:
                    raw_entities.append((ge["text"], ge["label"], "gliner", ge["score"], ge["start"], ge["end"]))
            except Exception as e:
                logger.debug(f"[NER] GLiNER predict failed: {e}")

        # 4. Flair
        if self._flair_tagger:
            try:
                from flair.data import Sentence
                flair_sent = Sentence(full_text)
                self._flair_tagger.predict(flair_sent)
                for ent in flair_sent.get_spans("ner"):
                    if ent.tag in ("PERSON", "ORG", "MONEY", "PERCENT"):
                        raw_entities.append((ent.text, ent.tag, "flair", ent.score, ent.start_position, ent.end_position))
            except Exception as e:
                logger.debug(f"[NER] Flair predict failed: {e}")

        # 5. Deduplicate overlapping spans
        seen_tickers = {e.ticker for e in result.resolved_assets if e.ticker}
        deduped = self._dedup_entities(raw_entities)

        # 6. Resolve each entity
        for text, label, source, conf, start, end in deduped:
            resolved = self._resolve_entity(text, label, source, conf, start, end)
            if resolved:
                if resolved.ticker and resolved.ticker not in seen_tickers:
                    seen_tickers.add(resolved.ticker)
                    result.resolved_assets.append(resolved)
                elif resolved.entity_type != "asset":
                    result.resolved_entities.append(resolved)

        # 7. FinBERT sentence sentiments
        if self._finbert:
            sentences = [s.strip() for s in re.split(r"[.!?]+", full_text) if len(s.strip()) > 20]
            for sent in sentences[:20]:
                try:
                    fb_result = self._finbert(sent, truncation=True, max_length=512)[0]
                    label = fb_result["label"].lower()
                    direction = "bullish" if label == "positive" else "bearish" if label == "negative" else "neutral"
                    result.sentence_sentiments.append({
                        "text": sent[:200],
                        "sentiment": direction,
                        "score": fb_result["score"],
                    })
                except Exception:
                    pass

        return result

    def _dedup_entities(self, raw: list) -> list:
        """Deduplicate entities by text overlap. Keep highest confidence per span."""
        if not raw:
            return []
        sorted_ents = sorted(raw, key=lambda x: (-x[3], x[0].lower()))
        seen_texts = set()
        result = []
        for text, label, source, conf, start, end in sorted_ents:
            key = text.lower().strip()
            if key in seen_texts or len(key) < 2:
                continue
            # Skip if a substring of an already-seen entity
            if any(key in s or s in key for s in seen_texts):
                continue
            seen_texts.add(key)
            result.append((text, label, source, conf, start, end))
        return result

    def _resolve_entity(self, text: str, label: str, source: str,
                        conf: float, start: int, end: int) -> Optional[ResolvedEntity]:
        """Resolve a raw NER entity to canonical form."""
        key = text.lower().strip()

        # Step 1: Override dict
        if key in self._overrides:
            info = self._overrides[key]
            return ResolvedEntity(
                text=text, canonical_name=info["canonical"],
                entity_type=info["type"],
                ticker=info.get("ticker"),
                asset_class=info.get("asset_class"),
                wikidata_qid=info.get("wikidata"),
                role=info.get("role"),
                confidence=1.0, source="override",
                start=start, end=end,
            )

        # Step 2: ReFinED entity linking
        if self._refined:
            try:
                spans = self._refined.process_text(text)
                for span in spans:
                    if span.predicted_entity and span.predicted_entity.wikidata_entity_id:
                        qid = span.predicted_entity.wikidata_entity_id
                        if qid in self._qid_map:
                            info = self._qid_map[qid]
                            return ResolvedEntity(
                                text=text,
                                canonical_name=info.get("name", span.predicted_entity.wikipedia_entity_title or text),
                                entity_type=info.get("type", "unknown"),
                                ticker=info.get("ticker"),
                                asset_class=info.get("asset_class"),
                                wikidata_qid=qid,
                                role=info.get("role"),
                                confidence=0.9, source="refined",
                                start=start, end=end,
                            )
                        # QID found but not in our map — use Wikipedia title
                        wiki_title = span.predicted_entity.wikipedia_entity_title
                        if wiki_title:
                            etype = self._infer_type_from_label(label)
                            return ResolvedEntity(
                                text=text, canonical_name=wiki_title,
                                entity_type=etype, wikidata_qid=qid,
                                confidence=0.7, source="refined",
                                start=start, end=end,
                            )
            except Exception:
                pass

        # Step 3: Map NER label to our type system
        etype = self._infer_type_from_label(label)
        return ResolvedEntity(
            text=text, canonical_name=text, entity_type=etype,
            confidence=conf, source=source, start=start, end=end,
        )

    def _infer_type_from_label(self, label: str) -> str:
        """Map NER labels to our entity type system."""
        label_map = {
            "PERSON": "person", "person": "person",
            "ORG": "organization", "company": "organization",
            "organization": "organization", "financial institution": "organization",
            "GPE": "location", "NORP": "government",
            "government body": "government", "regulatory body": "regulatory_body",
            "cryptocurrency": "asset", "stock ticker": "asset",
            "index": "asset", "commodity": "asset",
            "economic indicator": "metric",
            "MONEY": "metric", "PERCENT": "metric",
            "PRODUCT": "product", "EVENT": "event",
        }
        return label_map.get(label, "organization")
