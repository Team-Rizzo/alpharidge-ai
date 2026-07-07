"""Multilingual sentence embedder for grounding/faithfulness similarity.

Added alongside the analyzer's existing all-MiniLM-L6-v2 (which the deterministic gates
keep). Exposes .sim(a, b) -> cosine in [-1,1] on L2-normalized encodings, with a small
cache. Loaded lazily so import is cheap.
"""
from __future__ import annotations
import numpy as np

_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class MultilingualEmbedder:
    def __init__(self, model: str = _MODEL, cache_size: int = 4096):
        self._model_name = model
        self._m = None
        self._cache: dict = {}
        self._cache_size = cache_size

    def _model(self):
        if self._m is None:
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(self._model_name)
        return self._m

    def _enc(self, text: str):
        text = (text or "").strip()
        if not text:
            return None
        v = self._cache.get(text)
        if v is None:
            v = self._model().encode(text, normalize_embeddings=True).astype(np.float32)
            if len(self._cache) >= self._cache_size:
                self._cache.clear()
            self._cache[text] = v
        return v

    def sim(self, a: str, b: str) -> float:
        ea, eb = self._enc(a), self._enc(b)
        if ea is None or eb is None:
            return 0.0
        return float(np.dot(ea, eb))
