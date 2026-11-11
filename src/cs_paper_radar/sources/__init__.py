"""Paper sources. Each source turns a date range into `Paper` records.

Add a new source (Semantic Scholar, OpenReview, ACL Anthology, ...) by
implementing `Source` and registering it in `SOURCES`.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import Settings
from .base import Source

_REGISTRY: dict[str, Callable[[Settings], Source]] = {}


def register(name: str):
    def deco(factory: Callable[[Settings], Source]):
        _REGISTRY[name] = factory
        return factory

    return deco


def get_source(name: str, settings: Settings) -> Source:
    # Import side effect registers the built-in sources.
    from . import arxiv  # noqa: F401

    if name not in _REGISTRY:
        raise KeyError(f"unknown source {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](settings)


def available_sources() -> list[str]:
    from . import arxiv  # noqa: F401

    return sorted(_REGISTRY)
