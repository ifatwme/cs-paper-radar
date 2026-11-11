from datetime import timedelta

from fastapi.testclient import TestClient

from cs_paper_radar.db import Database, iso_week
from cs_paper_radar.topics import TopicEngine, VectorIndex

from .conftest import DAYS, START


def test_pipeline_populates_everything(populated):
    db = Database(populated.db_path)
    stats = db.stats()
    assert stats["papers"] == DAYS * 30
    assert stats["embedded"] == stats["papers"]
    assert stats["scored"] == stats["papers"]
    assert stats["last_run"]["status"] == "ok"

    last_week = iso_week(START + timedelta(days=DAYS - 1))
    picks = db.weekly_picks(last_week)
    assert len(picks) >= 8
    totals = [p["total"] for p in picks]
    assert all(0 <= t <= 1 for t in totals)
    # diversity cap is respected
    from collections import Counter

    assert max(Counter(p["primary_category"] for p in picks).values()) <= populated.top_max_per_category
    # corpus signals were computed (not the neutral fallback) once history exists
    assert any(p["novelty"] != 0.5 for p in picks)
    assert any(p["momentum"] != 0.5 for p in picks)


def test_upsert_is_idempotent(populated):
    from cs_paper_radar import pipeline

    stats = pipeline.backfill(populated, START, START + timedelta(days=2))
    assert stats["ingest"]["inserted"] == 0
    assert stats["embedded"] == 0


def test_topic_report(populated):
    db = Database(populated.db_path)
    engine = TopicEngine(populated, db, VectorIndex(populated, db))
    rep = engine.report("network intrusion detection", days=14)
    assert rep["matched_window"] > 0
    assert rep["frontier"] and rep["new"]
    assert all("intrusion" in (h["paper"]["title"] + h["paper"]["abstract"]).lower()
               for h in rep["frontier"] if h["keyword"])
    assert rep["categories"][0]["category"] == "cs.CR"
    assert len(rep["trend"]) >= 6
    assert rep["emerging"]["terms"]


def test_web_pages_and_api(populated):
    from cs_paper_radar.web.app import create_app

    client = TestClient(create_app(populated))
    for path in ["/", "/daily", "/weekly", "/topic", "/topic?q=network+security", "/categories",
                 "/about", "/feed/weekly.xml", "/healthz", "/api/docs"]:
        r = client.get(path)
        assert r.status_code == 200, path

    week = client.get("/api/weekly?n=10").json()
    assert week["n"] == 10
    uid = week["papers"][0]["uid"]
    assert client.get(f"/paper/{uid}").status_code == 200
    detail = client.get(f"/api/papers/{uid}").json()
    assert detail["score"]["total"] == week["papers"][0]["score"]["total"]
    assert len(detail["similar"]) > 0

    daily = client.get("/api/daily", params={"date": START.isoformat(), "cat": "cs.CR"}).json()
    assert daily["count"] > 0 and all("cs.CR" in p["categories"] for p in daily["papers"])

    topic = client.get("/api/topic", params={"q": "fuzzing", "days": 30}).json()
    assert topic["frontier"] and "paper" not in topic["frontier"][0]
    assert client.get("/api/search", params={"q": "botnet", "days": 60}).json()["results"]
    assert client.get("/weekly?week=2026-W99x").status_code in (400, 422)
    assert client.get("/paper/arxiv:nope").status_code == 404
    assert client.get("/api/papers/arxiv:nope").status_code == 404
