from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone

import pytest

from cs_paper_radar.config import Settings
from cs_paper_radar.models import Paper
from cs_paper_radar.sources import register

TOPICS = {
    "cs.CR": ["network intrusion detection", "malware traffic", "firewall evasion", "botnet",
              "anomaly detection on network flows", "zero-day exploits"],
    "cs.LG": ["graph neural networks", "contrastive learning", "optimizer", "generalization bounds",
              "mixture of experts", "transformer training"],
    "cs.NI": ["6G wireless", "edge caching", "congestion control", "satellite networks",
              "network slicing", "routing protocols"],
    "cs.SE": ["unit test generation", "code review", "static analysis", "program repair",
              "fuzzing", "software supply chain"],
}
FILLER = ("We study {a} and {b}. Our method combines {a} with {b} and is evaluated on public "
          "benchmarks, improving accuracy and robustness over strong baselines across many settings. "
          "We analyse failure cases and discuss limitations of {a} in realistic deployments.")


def make_papers(start: date, days: int, per_day: int = 30, seed: int = 1) -> list[Paper]:
    rng = random.Random(seed)
    papers = []
    serial = 0
    for d in range(days):
        day = start + timedelta(days=d)
        late = d >= days - 7
        for _ in range(per_day):
            cat = rng.choice(list(TOPICS))
            if late and rng.random() < 0.35:
                cat = "cs.CR"  # security gets hot in the last week
            a, b = rng.sample(TOPICS[cat], 2)
            serial += 1
            comment = rng.choice([
                "", "12 pages, 5 figures", "Accepted at USENIX Security 2026", "Under review",
                "4 pages, workshop paper", "Code: https://github.com/example/repo",
            ])
            cats = [cat] + rng.sample([c for c in TOPICS if c != cat], rng.randint(0, 2))
            papers.append(Paper(
                source="fake",
                native_id=f"{day:%y%m}.{serial:05d}",
                title=f"On {a} for {b}",
                abstract=FILLER.format(a=a, b=b),
                authors=[f"Author {i}" for i in range(rng.randint(1, 9))],
                published=datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc),
                primary_category=cat,
                categories=cats,
                comment=comment,
                abs_url=f"https://example.org/{serial}",
                pdf_url=f"https://example.org/{serial}.pdf",
            ))
    return papers


START = date(2026, 9, 7)  # a Monday
DAYS = 42


class FakeSource:
    name = "fake"
    _cache: dict[date, list[Paper]] = {}

    def __init__(self, settings):
        if not FakeSource._cache:
            for p in make_papers(START, DAYS):
                FakeSource._cache.setdefault(p.published.date(), []).append(p)

    def fetch_day(self, day):
        yield from FakeSource._cache.get(day, [])


register("fake")(FakeSource)


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDINGS", "hash")
    monkeypatch.setenv("SOURCES", "fake")
    import cs_paper_radar.embed as embed

    monkeypatch.setattr(embed, "_EMBEDDER", None)
    return Settings()


@pytest.fixture
def populated(settings) -> Settings:
    from cs_paper_radar import pipeline

    pipeline.backfill(settings, START, START + timedelta(days=DAYS - 1))
    return settings
