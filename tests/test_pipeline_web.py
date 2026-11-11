from datetime import timedelta

from fastapi.testclient import TestClient

from cs_paper_radar.db import Database, iso_week, week_bounds
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

    # proportional selection: each area's share of the picks follows its share of papers
    from cs_paper_radar.scoring import area_quotas, area_slug

    week_papers = db.papers_between(*[d.isoformat() for d in week_bounds(last_week)])
    sizes = Counter(area_slug(p) for p in week_papers)
    quotas = area_quotas(dict(sizes), len(picks))
    got = Counter(area_slug(p) for p in picks)
    assert all(abs(got[a] - q) <= 1 for a, q in quotas.items()), (got, quotas)
    # corpus signals were computed once history exists
    assert all(p["novelty"] is not None and p["momentum"] is not None for p in picks)
    # without enrichment, external signals are "not measured", not zero
    assert all(p["citations"] is None and p["community"] is None for p in picks)
    assert all(0 < p["coverage"] < 1 for p in picks)


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
                 "/about", "/feed/weekly.xml", "/healthz", "/api/docs", "/api/fairness"]:
        r = client.get(path)
        assert r.status_code == 200, path

    week = client.get("/api/weekly?n=10").json()
    assert week["n"] == 10
    uid = week["papers"][0]["uid"]
    assert client.get(f"/paper/{uid}").status_code == 200
    detail = client.get(f"/api/papers/{uid}").json()
    assert detail["score"]["total"] == week["papers"][0]["score"]["total"]
    assert set(detail["score"]["signals"]) >= {"citations", "venue", "novelty"}
    assert "github_stars" in detail["metrics"]
    fair = client.get("/api/fairness", params={"week": iso_week(START + timedelta(days=DAYS - 8))}).json()
    assert fair["areas"] and abs(sum(a["paper_share"] for a in fair["areas"]) - 1) < 0.01
    assert len(detail["similar"]) > 0

    daily = client.get("/api/daily", params={"date": START.isoformat(), "cat": "cs.CR"}).json()
    assert daily["count"] > 0 and all("cs.CR" in p["categories"] for p in daily["papers"])

    topic = client.get("/api/topic", params={"q": "fuzzing", "days": 30}).json()
    assert topic["frontier"] and "paper" not in topic["frontier"][0]
    assert client.get("/api/search", params={"q": "botnet", "days": 60}).json()["results"]
    assert client.get("/weekly?week=2026-W99x").status_code in (400, 422)
    assert client.get("/paper/arxiv:nope").status_code == 404
    assert client.get("/api/papers/arxiv:nope").status_code == 404


def test_heal_refetches_missing_and_partial_days(populated):
    from cs_paper_radar import pipeline

    db = Database(populated.db_path)
    gap = (START + timedelta(days=10)).isoformat()
    partial = (START + timedelta(days=12)).isoformat()
    with db.session() as conn:
        conn.execute("DELETE FROM papers WHERE pub_date = ?", (gap,))
        conn.execute("DELETE FROM fetch_log WHERE day = ?", (gap,))
        # simulate a run that stored 5 of 30 papers
        conn.execute("UPDATE fetch_log SET reported = 30, fetched = 5 WHERE day = ?", (partial,))
    assert set(db.incomplete_days("fake", START.isoformat(), (START + timedelta(days=DAYS - 1)).isoformat())) == {gap, partial}

    out = pipeline.heal(populated, db, START + timedelta(days=DAYS))
    assert gap in out["days"] and partial in out["days"]
    assert out["inserted"] == 30
    assert db.incomplete_days("fake", START.isoformat(), (START + timedelta(days=DAYS - 1)).isoformat()) == []


def test_enrichment_feeds_cohort_normalised_signals(populated, monkeypatch):
    """Mocked Semantic Scholar / Hugging Face / GitHub responses flow into the
    scores; zero citations mean 'no evidence' (0), missing papers stay unmeasured."""
    from cs_paper_radar import enrich, pipeline
    from cs_paper_radar.enrich.http import PoliteClient

    db = Database(populated.db_path)
    last = START + timedelta(days=DAYS - 1)
    week = iso_week(last)
    papers = db.papers_between((last - timedelta(days=6)).isoformat(), last.isoformat())
    cited = {p["native_id"]: i % 4 for i, p in enumerate(papers)}  # 0..3 citations
    first = papers[0]["native_id"]

    def fake_request(self, method, url, params=None, body=None, headers=None):
        if "semanticscholar" in url:
            return [{"paperId": f"s2-{i}", "citationCount": cited.get(i.split(":")[1], 0),
                     "influentialCitationCount": 0, "venue": "", "publicationVenue": None,
                     "journal": None, "authors": [{"hIndex": 7}]}
                    if i.split(":")[1] in cited else None for i in body["ids"]]
        if "huggingface" in url:
            if params["date"] == last.isoformat() and params["p"] == 0:
                return [{"paper": {"id": first, "upvotes": 42}, "numComments": 3}]
            return []
        if "api.github.com" in url:
            return {"stargazers_count": 250}
        raise AssertionError(url)

    monkeypatch.setattr(PoliteClient, "request", fake_request)
    monkeypatch.setattr(PoliteClient, "_wait", lambda self, host: None)
    monkeypatch.setenv("ENRICHERS", "semantic_scholar,huggingface,github")
    monkeypatch.setenv("HF_DAYS", "3")
    from cs_paper_radar.config import Settings

    settings = Settings()
    out = enrich.run_enrichment(settings, db, today=last + timedelta(days=2))
    assert out["semantic_scholar"]["found"] >= len(papers)
    assert out["huggingface"]["matched"] == 1
    pipeline.score(settings, db, [week])

    scored = {p["native_id"]: p for p in db.papers_between((last - timedelta(days=6)).isoformat(), last.isoformat())}
    zero = [p for nid, p in scored.items() if cited[nid] == 0]
    three = [p for nid, p in scored.items() if cited[nid] == 3]
    assert all(p["citations"] == 0 for p in zero)
    assert min(p["citations"] for p in three) > max(p["citations"] for p in zero)
    assert scored[first]["n_upvotes"] == 42 and scored[first]["max_h_index"] == 7
    # upvotes exist for 1 paper only -> below the coverage threshold -> not measured
    assert scored[first]["community"] is None
    assert all(p["coverage"] > 0.5 for p in scored.values())
