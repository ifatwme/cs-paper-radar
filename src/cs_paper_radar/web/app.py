"""FastAPI web app: HTML pages + JSON API."""

from __future__ import annotations

import hashlib
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
from xml.sax.saxutils import escape

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import __version__
from ..config import Settings, get_settings
from ..db import SIGNALS, Database, iso_week, week_bounds
from ..scoring import fairness_report, select_weekly
from ..taxonomy import (
    AREA_SLUGS,
    AREAS,
    CATEGORY_NAMES,
    SLUG_OF_AREA,
    area_of,
    category_name,
    home_category,
)
from ..topics import PRESET_GROUPS, PRESET_TOPICS, TopicEngine, VectorIndex

log = logging.getLogger(__name__)
HERE = Path(__file__).parent

SIGNAL_LABELS = {
    "citations": "Citations",
    "community": "Upvotes",
    "adoption": "Code stars",
    "venue": "Venue",
    "artifacts": "Code/data",
    "novelty": "Novelty",
    "momentum": "Momentum",
    "substance": "Substance",
    "breadth": "Breadth",
    "team": "Team",
}


def sparkline(values: list[float], width: int = 220, height: int = 40) -> str:
    if not values:
        return ""
    hi = max(values) or 1.0
    step = width / max(len(values) - 1, 1)
    pts = [f"{i * step:.1f},{height - 2 - (v / hi) * (height - 4):.1f}" for i, v in enumerate(values)]
    return " ".join(pts)


def _week_label(week: str) -> str:
    start, end = week_bounds(week)
    return f"{start:%b %d} – {end:%b %d, %Y}"


def _shift_week(week: str, delta: int) -> str:
    start, _ = week_bounds(week)
    return iso_week(start + timedelta(weeks=delta))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    db = Database(settings.db_path)
    db.init()
    index = VectorIndex(settings, db)
    engine = TopicEngine(settings, db, index)

    app = FastAPI(
        title=settings.site_name,
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    env = templates.env
    # Content hash for cache-busting static assets (nginx caches /static for days).
    asset_version = hashlib.sha1(
        b"".join((HERE / "static" / f).read_bytes() for f in ("style.css", "app.js", "theme.js"))
    ).hexdigest()[:10]
    env.globals.update(
        asset_version=asset_version,
        top_n=settings.top_n,
        site_name=settings.site_name,
        version=__version__,
        signals=SIGNALS,
        signal_labels=SIGNAL_LABELS,
        weights=settings.weights,
        category_name=category_name,
        presets=PRESET_TOPICS,
        preset_groups=PRESET_GROUPS,
        area_slugs=AREA_SLUGS,
        quote=quote,
    )
    env.filters["sparkline"] = sparkline
    env.filters["pct"] = lambda v: f"{(v or 0) * 100:.0f}%"
    env.filters["week_label"] = _week_label

    def render(request: Request, name: str, **ctx) -> HTMLResponse:
        ctx.setdefault("stats", db.stats())
        return templates.TemplateResponse(request, name, ctx)

    def latest_date() -> str:
        return db.date_range()[1] or datetime.now(timezone.utc).date().isoformat()

    def latest_week() -> str:
        return iso_week(date.fromisoformat(latest_date()))

    def decorate(papers: list[dict]) -> list[dict]:
        """Attach the paper's CS/CE home category and research area."""
        for p in papers:
            home = home_category(p["primary_category"], p["categories"])
            p["home"] = home
            p["area"] = area_of(home)
            p["area_slug"] = SLUG_OF_AREA.get(p["area"], "other")
        return papers

    def area_counts(papers: list[dict]) -> list[dict]:
        counts: dict[str, int] = {}
        for p in papers:
            counts[p["area_slug"]] = counts.get(p["area_slug"], 0) + 1
        return [{"slug": s, "name": n, "n": counts.get(s, 0)} for s, n in AREA_SLUGS.items()]

    _weekly_cache: dict[tuple, list[dict]] = {}

    def weekly_top(week: str, n: int, category: str | None = None, area: str | None = None) -> list[dict]:
        """Top-N of a week, selected with the same rule as the stored picks (cached
        per data version)."""
        key = (week, n, category, area, db.data_version())
        if key in _weekly_cache:
            return _weekly_cache[key]
        start, end = week_bounds(week)
        papers = decorate(db.papers_between(start.isoformat(), end.isoformat(), category, scored_only=True))
        if area:
            papers = [p for p in papers if p["area_slug"] == area]
        if settings.top_cs_primary_only:
            papers = [p for p in papers if p["primary_category"] in CATEGORY_NAMES]
        vectors = {p["uid"]: v for p in papers[: n * 8] if (v := index.vector(p["uid"])) is not None}
        scores = {p["uid"]: p["total"] for p in papers}
        by_uid = {p["uid"]: p for p in papers}
        # Filtered views (one category/area) are plain rankings; the full week uses
        # the configured selection rule (proportional area slots by default).
        picks = select_weekly(settings, papers, scores, n, vectors,
                              proportional=None if not (category or area) else False,
                              cap=0 if (category or area) else None)
        out = [by_uid[r["uid"]] for r in picks]
        if len(_weekly_cache) > 256:
            _weekly_cache.clear()
        _weekly_cache[key] = out
        return out

    # ----------------------------------------------------------------- HTML
    @app.get("/healthz", response_class=PlainTextResponse, include_in_schema=False)
    def healthz() -> str:
        return "ok"

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(request: Request):
        last = latest_date()
        week = latest_week()
        top = weekly_top(week, settings.top_n)
        start = (date.fromisoformat(last) - timedelta(days=41)).isoformat()
        daily = db.daily_counts(start, last)
        wk_start, wk_end = week_bounds(week)
        week_days = len(db.daily_counts(wk_start.isoformat(), wk_end.isoformat()))
        week_papers = decorate(db.papers_between(wk_start.isoformat(), wk_end.isoformat()))
        movers = sorted((r for r in category_table() if r["last7"] >= 20),
                        key=lambda r: r["growth"], reverse=True)
        return render(
            request, "home.html", latest=last, week=week, top=top, daily=daily,
            week_total=len(week_papers), prev_week=_shift_week(week, -1), week_days=week_days,
            today_count=next((d["n"] for d in daily if d["pub_date"] == last), 0),
            areas=area_counts(week_papers), risers=movers[:5],
        )

    @app.get("/daily", response_class=HTMLResponse, include_in_schema=False)
    def daily(request: Request, date_: str | None = Query(None, alias="date"),
              cat: str | None = None, area: str | None = None, sort: str = "area"):
        day = date_ or latest_date()
        try:
            d = date.fromisoformat(day)
        except ValueError:
            raise HTTPException(400, "date must look like 2026-11-10") from None
        if sort not in ("area", "score", "time"):
            sort = "area"
        all_day = decorate(db.papers_on(day, cat or None, "time" if sort == "time" else "score"))
        papers = [p for p in all_day if not area or p["area_slug"] == area]
        groups: list[tuple[str, str, list[dict]]] = []
        if sort == "area":
            by_area: dict[str, list[dict]] = {}
            for p in papers:
                by_area.setdefault(p["area_slug"], []).append(p)
            groups = [(s, AREA_SLUGS.get(s, "Other"), by_area[s])
                      for s in [*AREA_SLUGS, "other"] if s in by_area]
        first, last = db.date_range()
        return render(
            request, "daily.html", day=day, cat=cat or "", area=area or "", sort=sort,
            papers=papers, groups=groups, areas=area_counts(all_day), total_day=len(all_day),
            prev_day=(d - timedelta(days=1)).isoformat(), next_day=(d + timedelta(days=1)).isoformat(),
            first_date=first, last_date=last, all_categories=CATEGORY_NAMES,
        )

    @app.get("/weekly", response_class=HTMLResponse, include_in_schema=False)
    def weekly(request: Request, week: str | None = None, n: int = Query(0, ge=0, le=200),
               cat: str | None = None, area: str | None = None):
        week = week or latest_week()
        n = n or settings.top_n
        try:
            start, end = week_bounds(week)
        except ValueError:
            raise HTTPException(400, "week must look like 2026-W46") from None
        papers = weekly_top(week, n, cat or None, area or None)
        total = sum(db.category_counts(start.isoformat(), end.isoformat(), primary_only=True).values())
        days_with_data = len(db.daily_counts(start.isoformat(), end.isoformat()))
        return render(
            request, "weekly.html", week=week, n=n, cat=cat or "", area=area or "", papers=papers,
            total=total, days_with_data=days_with_data, prev_week=_shift_week(week, -1),
            next_week=_shift_week(week, 1), weeks=db.available_weeks(26), latest_week=latest_week(),
            all_categories=CATEGORY_NAMES, max_per_cat=settings.top_max_per_category,
            selection=settings.top_selection,
        )

    @app.get("/topic", response_class=HTMLResponse, include_in_schema=False)
    def topic(request: Request, q: str = Query("", max_length=200), days: int = Query(30, ge=7, le=180)):
        report = engine.report(q.strip(), days=days) if q.strip() else None
        if report:
            decorate([h["paper"] for h in report["new"] + report["frontier"]])
            for c in report["subtopics"]:
                decorate(c["papers"])
        return render(request, "topic.html", q=q.strip(), days=days, report=report)

    @app.get("/paper/{uid:path}", response_class=HTMLResponse, include_in_schema=False)
    def paper(request: Request, uid: str):
        p = db.get_paper(uid)
        if not p:
            raise HTTPException(404, "paper not found")
        decorate([p])
        sims = index.similar(uid, 8)
        similar = db.get_papers([u for u, _ in sims])
        similar_list = decorate([{**similar[u], "similarity": s} for u, s in sims if u in similar])
        return render(request, "paper.html", p=p, similar=similar_list,
                      related_terms=related_terms(p), history=db.metric_history(uid))

    def related_terms(p: dict, k: int = 10) -> list[str]:
        """Topic links for a paper: phrases from its title first (longest first),
        then acronyms, then the remaining phrases; no phrase contained in another."""
        title = p["title"].lower()
        kp = p.get("keyphrases") or []
        ranked = sorted(kp, key=lambda t: (t.lower() not in title, " " not in t, -len(t)))
        out: list[str] = []
        for t in ranked:
            if any(t.lower() in o.lower() or o.lower() in t.lower() for o in out):
                continue
            out.append(t)
            if len(out) == k:
                break
        return out

    @app.get("/categories", response_class=HTMLResponse, include_in_schema=False)
    def categories(request: Request):
        rows = category_table()
        areas = {}
        for r in rows:
            areas.setdefault(r["area"], []).append(r)
        order = list(AREAS) + ["Other"]
        return render(request, "categories.html", areas=[(a, areas[a]) for a in order if a in areas],
                      week=latest_week())

    def category_table() -> list[dict]:
        last = date.fromisoformat(latest_date())
        cur = db.category_counts((last - timedelta(days=6)).isoformat(), last.isoformat())
        base = db.category_counts((last - timedelta(days=34)).isoformat(), (last - timedelta(days=7)).isoformat())
        rows = []
        for code in sorted(set(cur) | set(base)):
            if code not in CATEGORY_NAMES:
                continue
            now, before = cur.get(code, 0), base.get(code, 0) / 4
            rows.append({"category": code, "name": category_name(code), "area": area_of(code),
                         "last7": now, "avg_prev4": round(before, 1),
                         "growth": round(math.log((now + 1) / (before + 1)), 3)})
        return rows

    @app.get("/robots.txt", response_class=PlainTextResponse, include_in_schema=False)
    def robots() -> str:
        # Listings and paper pages are fine to crawl; topic analysis and the API are
        # expensive and change constantly.
        return "User-agent: *\nDisallow: /topic\nDisallow: /api/\nCrawl-delay: 5\n"

    @app.get("/about", response_class=HTMLResponse, include_in_schema=False)
    def about(request: Request):
        return render(request, "about.html", settings=settings,
                      fairness=fairness_report(db, _shift_week(latest_week(), -1)))

    @app.get("/feed/weekly.xml", include_in_schema=False)
    def feed(request: Request):
        week = latest_week()
        papers = weekly_top(week, settings.top_n)
        base = str(request.base_url).rstrip("/")
        items = "".join(
            f"<item><title>{escape(p['title'])}</title><link>{escape(p['abs_url'])}</link>"
            f"<guid isPermaLink=\"false\">{escape(p['uid'])}</guid>"
            f"<category>{escape(p['primary_category'])}</category>"
            f"<description>{escape(p['abstract'][:600])}</description></item>"
            for p in papers
        )
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
            f"<title>{escape(settings.site_name)}: top {settings.top_n} of {week}</title>"
            f"<link>{escape(base)}/weekly?week={week}</link>"
            f"<description>Top computer science papers of the week on arXiv</description>{items}"
            "</channel></rss>"
        )
        return Response(xml, media_type="application/rss+xml")

    # ------------------------------------------------------------------ API
    def slim(p: dict) -> dict:
        out = {k: p.get(k) for k in (
            "uid", "source", "native_id", "version", "title", "abstract", "authors",
            "primary_category", "categories", "comment", "journal_ref", "doi", "abs_url",
            "pdf_url", "published", "pub_date", "week", "tags")}
        out["score"] = None if p.get("total") is None else {
            "total": p["total"], "coverage": p.get("coverage"),
            "signals": {s: p.get(s) for s in SIGNALS}, "evidence": p.get("details") or {}}
        out["metrics"] = {
            "citations": p.get("n_citations"), "influential_citations": p.get("n_influential"),
            "hf_upvotes": p.get("n_upvotes"), "github_repo": p.get("github_repo"),
            "github_stars": p.get("n_stars"), "venue": p.get("s2_venue") or None,
            "max_author_h_index": p.get("max_h_index"),
        }
        return out

    @app.get("/api/stats", tags=["meta"])
    def api_stats():
        return {**db.stats(), "version": __version__, "sources": settings.sources,
                "categories": settings.arxiv_categories, "weights": settings.weights}

    @app.get("/api/daily", tags=["papers"])
    def api_daily(date_: str | None = Query(None, alias="date", description="YYYY-MM-DD (UTC)"),
                  cat: str | None = None, sort: str = "score"):
        day = date_ or latest_date()
        papers = db.papers_on(day, cat, sort if sort in ("score", "time", "category") else "score")
        return {"date": day, "count": len(papers), "papers": [slim(p) for p in papers]}

    @app.get("/api/weekly", tags=["papers"])
    def api_weekly(week: str | None = None, n: int = Query(0, ge=0, le=200), cat: str | None = None):
        week = week or latest_week()
        papers = weekly_top(week, n or settings.top_n, cat)
        return {"week": week, "n": len(papers), "papers": [slim(p) for p in papers]}

    @app.get("/api/topic", tags=["topics"])
    def api_topic(q: str = Query(..., min_length=2, max_length=200), days: int = Query(30, ge=7, le=180),
                  n: int = Query(0, ge=0, le=100)):
        rep = engine.report(q, days=days, n=n or None)

        def hit(h):
            return {"relevance": h["relevance"], "keyword_match": h["keyword"],
                    "semantic_similarity": h["semantic"], **slim(h["paper"])}

        rep["new"] = [hit(h) for h in rep["new"]]
        rep["frontier"] = [hit(h) for h in rep["frontier"]]
        for c in rep["subtopics"]:
            c["papers"] = [slim(p) for p in c["papers"]]
        return rep

    @app.get("/api/search", tags=["topics"])
    def api_search(q: str = Query(..., min_length=2, max_length=200), days: int = Query(30, ge=1, le=400),
                   limit: int = Query(50, ge=1, le=500)):
        end = latest_date()
        start = (date.fromisoformat(end) - timedelta(days=days - 1)).isoformat()
        hits = engine.search(q, start, end, limit=limit)
        papers = db.get_papers([h["uid"] for h in hits])
        return {"query": q, "start": start, "end": end, "results": [
            {"relevance": h["relevance"], **slim(papers[h["uid"]])} for h in hits if h["uid"] in papers]}

    @app.get("/api/papers/{uid:path}", tags=["papers"])
    def api_paper(uid: str):
        p = db.get_paper(uid)
        if not p:
            raise HTTPException(404, "paper not found")
        sims = index.similar(uid, 10)
        return {**slim(p), "similar": [{"uid": u, "similarity": round(s, 4)} for u, s in sims]}

    @app.get("/api/fairness", tags=["meta"])
    def api_fairness(week: str | None = None, top: int = Query(50, ge=10, le=200)):
        """Per research area: share of papers vs. share of the weekly top picks, mean
        measured coverage, and the share of papers for which each signal was measured."""
        return fairness_report(db, week or _shift_week(latest_week(), -1), top)

    @app.get("/api/categories", tags=["meta"])
    def api_categories():
        return {"categories": category_table()}

    @app.exception_handler(404)
    async def not_found(request: Request, exc):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": getattr(exc, "detail", "not found")}, status_code=404)
        return templates.TemplateResponse(request, "error.html", {"message": "Not found", "stats": None},
                                          status_code=404)

    env.globals["urlencode"] = urlencode
    return app
