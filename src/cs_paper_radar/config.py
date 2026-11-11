"""Runtime configuration, read from environment variables (12-factor style).

Every setting has a sensible default so `cs-paper-radar` works out of the box;
override in `.env` (Docker Compose) or the shell.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in _env(name, default).split(",") if x.strip()]


# Default weights of the composite "frontier score" (see docs/scoring.md).
DEFAULT_WEIGHTS = {
    "venue": 0.20,      # accepted / to appear at a recognised venue
    "artifacts": 0.15,  # code, data, models or project page released
    "momentum": 0.20,   # its research neighbourhood is growing this week
    "novelty": 0.20,    # far from everything published in the previous months
    "breadth": 0.10,    # cross-listed across several categories
    "substance": 0.10,  # full paper (pages/figures/journal ref), not a stub
    "team": 0.05,       # collaboration size (weak, capped)
}


def _weights() -> dict[str, float]:
    """`SCORE_WEIGHTS="venue=0.3,novelty=0.3"` overrides individual weights."""
    weights = dict(DEFAULT_WEIGHTS)
    for item in _env_list("SCORE_WEIGHTS", ""):
        key, _, value = item.partition("=")
        if key.strip() in weights and value:
            weights[key.strip()] = float(value)
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")))
    sources: list[str] = field(default_factory=lambda: _env_list("SOURCES", "arxiv"))

    # arXiv
    arxiv_categories: list[str] = field(
        default_factory=lambda: _env_list(
            "ARXIV_CATEGORIES", "cs.*,eess.SY,eess.SP,eess.AS,eess.IV"
        )
    )
    arxiv_api_url: str = field(
        default_factory=lambda: _env("ARXIV_API_URL", "https://export.arxiv.org/api/query")
    )
    # One request covers a whole weekday (~900 papers, ~2 s); arXiv allows up to 2000.
    arxiv_page_size: int = field(default_factory=lambda: _env_int("ARXIV_PAGE_SIZE", 1000))
    arxiv_delay_s: float = field(default_factory=lambda: _env_float("ARXIV_DELAY_S", 3.5))
    arxiv_timeout_s: float = field(default_factory=lambda: _env_float("ARXIV_TIMEOUT_S", 90))
    # Days re-fetched by each daily run: the API publishes a day's submissions
    # only after arXiv's evening announcement, late announcements happen, and
    # authors add acceptance notes / code links in new versions during the week.
    update_lookback_days: int = field(default_factory=lambda: _env_int("UPDATE_LOOKBACK_DAYS", 8))
    # Days further back that are re-fetched when incomplete (fewer stored than arXiv reports).
    heal_window_days: int = field(default_factory=lambda: _env_int("HEAL_WINDOW_DAYS", 60))
    heal_max_days: int = field(default_factory=lambda: _env_int("HEAL_MAX_DAYS", 10))

    # Embeddings (semantic search, novelty, momentum). EMBEDDINGS=off -> keyword only.
    embeddings: str = field(default_factory=lambda: _env("EMBEDDINGS", "fastembed"))
    # all-MiniLM-L6-v2: 384-d, ~24 abstracts/s on 4 Skylake vCPUs (bge-small: ~5/s).
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_threads: int = field(default_factory=lambda: _env_int("EMBEDDING_THREADS", 4))
    embedding_batch: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH", 16))

    # Ranking
    top_n: int = field(default_factory=lambda: _env_int("TOP_N", 20))
    top_max_per_category: int = field(
        default_factory=lambda: _env_int("TOP_MAX_PER_CATEGORY", 4)
    )
    # Weekly picks only from papers whose *primary* category is CS/EESS; cross-lists
    # from physics, math, biology... still appear in daily listings and topics.
    top_cs_primary_only: bool = field(
        default_factory=lambda: _env("TOP_CS_PRIMARY_ONLY", "true").lower() in ("1", "true", "yes")
    )
    novelty_lookback_days: int = field(
        default_factory=lambda: _env_int("NOVELTY_LOOKBACK_DAYS", 90)
    )
    momentum_baseline_weeks: int = field(
        default_factory=lambda: _env_int("MOMENTUM_BASELINE_WEEKS", 4)
    )
    weights: dict[str, float] = field(default_factory=_weights)

    # Web
    vector_cache_days: int = field(default_factory=lambda: _env_int("VECTOR_CACHE_DAYS", 400))
    site_name: str = field(default_factory=lambda: _env("SITE_NAME", "CS Paper Radar"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "radar.sqlite3"

    @property
    def model_dir(self) -> Path:
        return Path(_env("MODEL_DIR", str(self.data_dir / "models")))


def get_settings() -> Settings:
    return Settings()
