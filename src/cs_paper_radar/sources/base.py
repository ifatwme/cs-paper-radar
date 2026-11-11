"""Source interface."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import Protocol

from ..models import Paper


class Source(Protocol):
    name: str

    def fetch_day(self, day: date) -> Iterator[Paper]:
        """Yield every paper first submitted on `day` (UTC)."""
        ...
