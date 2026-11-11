"""Text embeddings for semantic search, novelty and momentum.

* `fastembed` (default): sentence-transformers/all-MiniLM-L6-v2 via ONNX Runtime
  on CPU, 384-d, ~90 MB model downloaded once into MODEL_DIR. No PyTorch.
* `hash`: dependency-free hashed bag-of-words vectors. Used in tests/CI, and a
  usable fallback when the model cannot be downloaded.
* `off`: keyword search only; novelty/momentum fall back to neutral values.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .config import Settings

log = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class FastEmbedder:
    def __init__(self, settings: Settings):
        from fastembed import TextEmbedding  # heavy import, only when used

        self.name = settings.embedding_model
        self.batch = settings.embedding_batch
        settings.model_dir.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(
            model_name=settings.embedding_model,
            cache_dir=str(settings.model_dir),
            threads=settings.embedding_threads,
        )
        self._is_bge = "bge" in self.name.lower()
        self._lock = threading.Lock()  # ONNX session is not re-entrant enough for web use

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        with self._lock:
            vecs = list(self._model.embed(list(texts), batch_size=self.batch))
        return _normalise(np.vstack(vecs)) if vecs else np.zeros((0, 0), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        prefix = BGE_QUERY_PREFIX if self._is_bge else ""
        with self._lock:
            vec = next(iter(self._model.embed([prefix + text])))
        return _normalise(vec)


_WORD = re.compile(r"[a-z][a-z0-9\-]+")


class HashEmbedder:
    """Hashed unigram+bigram vectors (stable across runs, no model download)."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        words = _WORD.findall(text.lower())
        feats = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return _normalise(np.vstack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self._vec(text))


_EMBEDDER: Embedder | None = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder(settings: Settings) -> Embedder | None:
    """Process-wide embedder (None when EMBEDDINGS=off)."""
    global _EMBEDDER
    kind = settings.embeddings.lower()
    if kind in ("off", "none", "0", "false"):
        return None
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            if kind == "hash":
                _EMBEDDER = HashEmbedder()
            else:
                _EMBEDDER = FastEmbedder(settings)
                log.info("loaded embedding model %s", _EMBEDDER.name)
        return _EMBEDDER


def embedder_name(settings: Settings) -> str | None:
    kind = settings.embeddings.lower()
    if kind in ("off", "none", "0", "false"):
        return None
    return HashEmbedder().name if kind == "hash" else settings.embedding_model
