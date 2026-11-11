"""Source-independent paper record."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Paper:
    """One paper as delivered by any source (arXiv today, others later).

    `uid` is globally unique across sources: "<source>:<native id>",
    e.g. "arxiv:2611.12557".
    """

    source: str
    native_id: str
    title: str
    abstract: str
    authors: list[str]
    published: datetime            # first version (v1) submission time, UTC
    updated: datetime | None = None
    version: int = 1
    primary_category: str = ""
    categories: list[str] = field(default_factory=list)
    comment: str = ""
    journal_ref: str = ""
    doi: str = ""
    abs_url: str = ""
    pdf_url: str = ""

    @property
    def uid(self) -> str:
        return f"{self.source}:{self.native_id}"

    @property
    def embedding_text(self) -> str:
        return f"{self.title}. {self.abstract}"

    @property
    def withdrawn(self) -> bool:
        comment = self.comment.lower()
        head = self.abstract[:200].lower()
        return (
            comment.startswith("withdrawn")
            or "paper has been withdrawn" in comment
            or "paper has been withdrawn" in head
            or head.startswith("withdrawn")
        )
