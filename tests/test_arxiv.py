from pathlib import Path

import pytest

from cs_paper_radar.config import Settings
from cs_paper_radar.sources import arxiv as arxiv_mod
from cs_paper_radar.sources.arxiv import ArxivError, ArxivSource, parse_feed

FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_cs_cr.xml"


def test_parse_real_feed():
    total, papers = parse_feed(FIXTURE.read_bytes())
    assert total == 41
    assert len(papers) == 6
    for p in papers:
        assert p.source == "arxiv"
        assert p.uid == f"arxiv:{p.native_id}"
        assert "v" not in p.native_id.split(".")[-1]
        assert p.title and p.abstract and p.authors
        assert "cs.CR" in p.categories
        assert p.primary_category in p.categories
        assert p.published.tzinfo is not None
        assert p.published.date().isoformat() == "2026-11-02"
        assert p.abs_url.startswith("https://arxiv.org/abs/")
        assert "  " not in p.title


def test_query_uses_configured_categories(monkeypatch):
    monkeypatch.setenv("ARXIV_CATEGORIES", "cs.*,eess.SY")
    src = ArxivSource(Settings())
    assert src.category_query() == "(cat:cs.* OR cat:eess.SY)"


def _feed(total: int, ids: list[str]) -> bytes:
    entries = "".join(
        f"""<entry><id>http://arxiv.org/abs/{i}v2</id><title>T {i}</title><summary>S</summary>
        <published>2026-11-02T10:00:00Z</published><updated>2026-11-03T10:00:00Z</updated>
        <author><name>A</name></author><arxiv:primary_category term="cs.CR"/>
        <category term="cs.CR"/><category term="68T05"/></entry>"""
        for i in ids
    )
    return f"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <opensearch:totalResults>{total}</opensearch:totalResults>{entries}</feed>""".encode()


def test_pagination_and_empty_page_retry(monkeypatch):
    monkeypatch.setenv("ARXIV_PAGE_SIZE", "2")
    monkeypatch.setattr(arxiv_mod.time, "sleep", lambda s: None)
    responses = iter([
        _feed(3, ["2611.00001", "2611.00002"]),
        _feed(3, []),                       # flaky empty page -> retried
        _feed(3, ["2611.00003"]),
    ])
    src = ArxivSource(Settings())
    monkeypatch.setattr(src, "_get", lambda params: next(responses))
    from datetime import date

    papers = list(src.fetch_day(date(2026, 11, 2)))
    assert [p.native_id for p in papers] == ["2611.00001", "2611.00002", "2611.00003"]
    assert papers[0].version == 2
    assert papers[0].categories == ["cs.CR"]  # MSC class dropped


def test_persistent_failure_raises(monkeypatch):
    monkeypatch.setenv("ARXIV_PAGE_SIZE", "2")
    monkeypatch.setattr(arxiv_mod.time, "sleep", lambda s: None)
    calls = iter([_feed(3, ["2611.00001", "2611.00002"])] + [_feed(3, [])] * 10)
    src = ArxivSource(Settings())
    monkeypatch.setattr(src, "_get", lambda params: next(calls))
    from datetime import date

    with pytest.raises(ArxivError):
        list(src.fetch_day(date(2026, 11, 2)))
