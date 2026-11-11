"""arXiv API source (https://info.arxiv.org/help/api/user-manual.html).

One query per UTC day:
    (cat:cs.* OR cat:eess.SY ...) AND submittedDate:[YYYYMMDD0000 TO YYYYMMDD2359]
paged with `start`/`max_results`, respecting arXiv's "one request every 3 s".
The API occasionally returns an empty page in the middle of a result set;
such pages are retried with backoff before giving up.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import date, datetime, timezone

from .. import __version__
from ..config import Settings
from ..models import Paper
from . import register

log = logging.getLogger(__name__)

NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
}
_ID_RE = re.compile(r"arxiv\.org/abs/(?P<id>.+?)(?:v(?P<v>\d+))?$")
_WS = re.compile(r"\s+")


class ArxivError(RuntimeError):
    pass


def _text(el: ET.Element | None) -> str:
    return _WS.sub(" ", el.text or "").strip() if el is not None else ""


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def parse_entry(entry: ET.Element) -> Paper | None:
    raw_id = _text(entry.find("a:id", NS))
    m = _ID_RE.search(raw_id)
    if not m:
        return None
    native_id, version = m.group("id"), int(m.group("v") or 1)
    primary = entry.find("arxiv:primary_category", NS)
    categories = [c.get("term", "") for c in entry.findall("a:category", NS)]
    # Keep only arXiv taxonomy codes (the feed may include ACM/MSC classes).
    categories = [c for c in dict.fromkeys(categories) if re.fullmatch(r"[a-z\-]+(\.[A-Za-z\-]+)?", c)]
    pdf_url = ""
    for link in entry.findall("a:link", NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")
    published = _text(entry.find("a:published", NS))
    updated = _text(entry.find("a:updated", NS))
    return Paper(
        source="arxiv",
        native_id=native_id,
        version=version,
        title=_text(entry.find("a:title", NS)),
        abstract=_text(entry.find("a:summary", NS)),
        authors=[_text(a.find("a:name", NS)) for a in entry.findall("a:author", NS)],
        published=_dt(published),
        updated=_dt(updated) if updated else None,
        primary_category=primary.get("term", "") if primary is not None else (categories[:1] or [""])[0],
        categories=categories,
        comment=_text(entry.find("arxiv:comment", NS)),
        journal_ref=_text(entry.find("arxiv:journal_ref", NS)),
        doi=_text(entry.find("arxiv:doi", NS)),
        abs_url=f"https://arxiv.org/abs/{native_id}",
        pdf_url=pdf_url or f"https://arxiv.org/pdf/{native_id}",
    )


def parse_feed(xml_bytes: bytes) -> tuple[int, list[Paper]]:
    """Return (totalResults, papers) for one Atom page."""
    root = ET.fromstring(xml_bytes)
    total = int(_text(root.find("os:totalResults", NS)) or 0)
    papers = [p for e in root.findall("a:entry", NS) if (p := parse_entry(e))]
    return total, papers


@register("arxiv")
class ArxivSource:
    name = "arxiv"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._last_request = 0.0

    def category_query(self) -> str:
        cats = " OR ".join(f"cat:{c}" for c in self.settings.arxiv_categories)
        return f"({cats})"

    def _get(self, params: dict[str, str | int]) -> bytes:
        url = f"{self.settings.arxiv_api_url}?{urllib.parse.urlencode(params)}"
        wait = self.settings.arxiv_delay_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(
            url, headers={"User-Agent": f"cs-paper-radar/{__version__} (+https://github.com/ifatwme/cs-paper-radar)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.settings.arxiv_timeout_s) as resp:
                return resp.read()
        finally:
            self._last_request = time.monotonic()

    def _page(self, query: str, start: int, size: int, expect_entries: bool) -> tuple[int, list[Paper]]:
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                total, papers = parse_feed(
                    self._get(
                        {
                            "search_query": query,
                            "start": start,
                            "max_results": size,
                            "sortBy": "submittedDate",
                            "sortOrder": "ascending",
                        }
                    )
                )
                if papers or not expect_entries:
                    return total, papers
                last_error = ArxivError(f"empty page at start={start}")
            except (urllib.error.URLError, TimeoutError, ET.ParseError, OSError) as exc:
                last_error = exc
            backoff = min(60, 5 * 2**attempt)
            log.warning("arXiv request failed (%s); retry %d in %ds", last_error, attempt + 1, backoff)
            time.sleep(backoff)
        raise ArxivError(f"arXiv API failed for start={start}: {last_error}")

    def fetch_day(self, day: date) -> Iterator[Paper]:
        stamp = day.strftime("%Y%m%d")
        query = f"{self.category_query()} AND submittedDate:[{stamp}0000 TO {stamp}2359]"
        size = self.settings.arxiv_page_size
        total, papers = self._page(query, 0, size, expect_entries=False)
        log.info("arXiv %s: %d papers", day, total)
        seen = 0
        start = 0
        while True:
            yield from papers
            seen += len(papers)
            start += size
            if start >= total or not papers:
                break
            _, papers = self._page(query, start, size, expect_entries=True)
        if seen < total:
            log.warning("arXiv %s: got %d of %d papers", day, seen, total)
