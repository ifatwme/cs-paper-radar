"""Venue quality tiers from the CORE / ICORE conference ranking.

The CSV export of https://portal.core.edu.au/conf-ranks/ (free) is downloaded
into DATA_DIR and refreshed every 30 days. Rows look like:
    id,title,acronym,source,rank,primary?,FoR codes...
"""

from __future__ import annotations

import csv
import io
import logging
import re
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

CORE_URL = ("https://portal.core.edu.au/conf-ranks/?search=&by=all&source={source}"
            "&sort=arank&page=1&do=Export")
CORE_SOURCES = ("ICORE2026", "CORE2023")

TIER_VALUE = {"A*": 1.0, "A": 0.85, "B": 0.65, "C": 0.45}
# Journals are not ranked by CORE; these families are well-established in CS/CE.
JOURNAL_PATTERNS = [
    (re.compile(r"\b(IEEE|ACM) Transactions\b|\bTransactions on\b", re.I), 0.85, "transactions journal"),
    (re.compile(r"\bJournal of Machine Learning Research\b|\bJMLR\b|\bTMLR\b|Transactions on Machine Learning Research", re.I), 0.85, "ML journal"),
    (re.compile(r"\bNature\b|\bScience\b|\bPNAS\b|Proceedings of the National Academy", re.I), 1.0, "multidisciplinary journal"),
    (re.compile(r"\bCommunications of the ACM\b|\bACM Computing Surveys\b|\bProceedings of the IEEE\b", re.I), 0.85, "flagship journal"),
    (re.compile(r"\bJournal\b|\bLetters\b|\bMagazine\b|\bReview\b", re.I), 0.55, "journal"),
]
_NORM = re.compile(r"[^a-z0-9 ]+")
_YEAR = re.compile(r"\b(19|20)\d{2}\b|\b\d+(st|nd|rd|th)\b", re.I)
_FILLER = re.compile(r"\b(proceedings|of|the|annual|international|conference|on|in|ieee|acm|cvf|"
                     r"symposium|workshops?|and|for)\b")


def _norm(text: str) -> str:
    return " ".join(_FILLER.sub(" ", _NORM.sub(" ", _YEAR.sub(" ", text.lower()))).split())


class VenueRanker:
    def __init__(self, rows: list[tuple[str, str, str]]):
        self.by_acronym: dict[str, str] = {}
        self.by_title: dict[str, str] = {}
        for title, acronym, rank in rows:
            rank = rank.strip()
            if rank not in TIER_VALUE:
                continue
            if acronym:
                self.by_acronym.setdefault(acronym.strip().upper(), rank)
            if title:
                self.by_title.setdefault(_norm(title), rank)
        self.size = len(self.by_acronym)

    def tier(self, venue: str) -> tuple[str | None, float | None, str]:
        """Return (tier, value, how) for a venue name or acronym; (None, None, '') if unknown."""
        if not venue:
            return None, None, ""
        v = venue.strip()
        # 1. acronyms inside the string ("NeurIPS 2026", "Proc. of CCS '26")
        for tok in re.findall(r"[A-Za-z][A-Za-z&\-]{1,15}", v):
            rank = self.by_acronym.get(tok.upper())
            if rank and sum(c.isupper() for c in tok) >= 2:
                return rank, TIER_VALUE[rank], f"CORE {rank} ({tok})"
        # 2. full title
        n = _norm(v)
        if n in self.by_title:
            rank = self.by_title[n]
            return rank, TIER_VALUE[rank], f"CORE {rank}"
        for pattern, value, label in JOURNAL_PATTERNS:
            if pattern.search(v):
                return None, value, label
        return None, None, ""


def _download(path: Path) -> bool:
    rows = []
    for source in CORE_SOURCES:
        try:
            req = urllib.request.Request(CORE_URL.format(source=source),
                                         headers={"User-Agent": "cs-paper-radar"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode("utf-8", "replace")
            rows = list(csv.reader(io.StringIO(text)))
            if len(rows) > 100:
                break
        except Exception as exc:
            log.warning("CORE download (%s) failed: %s", source, exc)
    if len(rows) <= 100:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return True


@lru_cache(maxsize=4)
def load_ranker(data_dir: str) -> VenueRanker:
    path = Path(data_dir) / "core_ranks.csv"
    stale = not path.exists() or time.time() - path.stat().st_mtime > 30 * 86400
    if stale and not _download(path) and not path.exists():
        log.warning("CORE ranking unavailable; venue tiers fall back to journal patterns only")
        return VenueRanker([])
    with open(path, newline="") as fh:
        rows = [(r[1], r[2], r[4]) for r in csv.reader(fh) if len(r) > 4]
    return VenueRanker(rows)
