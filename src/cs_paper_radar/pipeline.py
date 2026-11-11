"""Pipeline: ingest -> embed -> score. Used by the CLI and the daily timer."""

from __future__ import annotations

import fcntl
import logging
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

from .config import Settings
from .db import Database, iso_week
from .embed import embedder_name, get_embedder, text_hash
from .enrich import run_enrichment
from .scoring import score_week
from .sources import get_source
from .text import keyphrases

log = logging.getLogger(__name__)


@contextmanager
def pipeline_lock(settings: Settings):
    """Only one pipeline run at a time (timer vs. manual backfill)."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.data_dir / ".pipeline.lock", "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another pipeline run is in progress") from None
        yield


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def ingest(settings: Settings, db: Database, start: date, end: date) -> dict:
    stats = {"days": 0, "fetched": 0, "inserted": 0, "updated": 0, "failed_days": []}
    for name in settings.sources:
        source = get_source(name, settings)
        for day in daterange(start, end):
            try:
                papers = list(source.fetch_day(day))
            except Exception as exc:  # keep going; the next run retries this day
                log.error("%s %s failed: %s", name, day, exc)
                stats["failed_days"].append(f"{name}:{day}")
                continue
            ins, upd = db.upsert_papers(papers)
            reported = getattr(source, "last_total", None)
            db.log_fetch(name, day.isoformat(), reported, len(papers))
            if reported is not None and len(papers) < reported:
                stats.setdefault("partial_days", []).append(f"{name}:{day}")
            stats["days"] += 1
            stats["fetched"] += len(papers)
            stats["inserted"] += ins
            stats["updated"] += upd
            log.info("%s %s: %d fetched, %d new, %d updated", name, day, len(papers), ins, upd)
    return stats


def embed_missing(settings: Settings, db: Database, chunk: int = 512) -> int:
    model = embedder_name(settings)
    if model is None:
        return 0
    embedder = get_embedder(settings)
    done = 0
    t0 = time.monotonic()
    while True:
        rows = db.missing_embeddings(model, limit=chunk)
        if not rows:
            break
        texts = [f"{r['title']}. {r['abstract']}" for r in rows]
        vecs = embedder.embed_documents(texts)
        db.save_embeddings(model, [(r["uid"], text_hash(t), v) for r, t, v in zip(rows, texts, vecs)])
        done += len(rows)
        rate = done / max(time.monotonic() - t0, 1e-6)
        log.info("embedded %d papers (%.1f/s)", done, rate)
    return done


def extract_keyphrases(db: Database) -> int:
    """Pre-compute key phrases (used by topic reports) for new/changed papers."""
    done = 0
    while rows := db.missing_keyphrases():
        db.save_keyphrases([(r["uid"], sorted(set(keyphrases(f"{r['title']}. {r['abstract']}"))))
                            for r in rows])
        done += len(rows)
    return done


def score(settings: Settings, db: Database, weeks: list[str]) -> dict:
    model = embedder_name(settings)
    out = {}
    for week in sorted(set(weeks)):
        out[week] = score_week(settings, db, model, week)
        log.info("scored %s: %d papers", week, out[week])
    return out


def weeks_between(start: date, end: date) -> list[str]:
    return sorted({iso_week(d) for d in daterange(start, end)})


def heal(settings: Settings, db: Database, before: date) -> dict:
    """Re-fetch older days that are missing or incomplete (failed/partial runs,
    arXiv outages) so gaps close by themselves on later runs."""
    start = before - timedelta(days=settings.heal_window_days)
    end = before - timedelta(days=1)
    first, _ = db.date_range()
    if first:  # never reach back before the first backfilled day
        start = max(start, date.fromisoformat(first))
    days: list[date] = []
    for name in settings.sources:
        days += [date.fromisoformat(d) for d in db.incomplete_days(name, start.isoformat(), end.isoformat())]
    days = sorted(set(days))[: settings.heal_max_days]
    out = {"days": [d.isoformat() for d in days], "inserted": 0}
    for d in days:
        out["inserted"] += ingest(settings, db, d, d)["inserted"]
    return out


def run(settings: Settings, kind: str, start: date, end: date, rescore_from: date | None = None,
        heal_gaps: bool = False, enrich: bool = False) -> dict:
    db = Database(settings.db_path)
    db.init()
    with pipeline_lock(settings):
        run_id = db.start_run(kind)
        stats: dict = {"start": start.isoformat(), "end": end.isoformat()}
        try:
            stats["ingest"] = ingest(settings, db, start, end)
            if heal_gaps:
                stats["heal"] = heal(settings, db, start)
                if stats["heal"]["days"]:
                    rescore_from = min([rescore_from or start] + [date.fromisoformat(d) for d in stats["heal"]["days"]])
            stats["embedded"] = embed_missing(settings, db)
            stats["keyphrases"] = extract_keyphrases(db)
            if enrich:
                stats["enrich"] = run_enrichment(settings, db)
            stats["scored"] = score(settings, db, weeks_between(rescore_from or start, end))
            status = "ok" if not (stats["ingest"]["failed_days"] or stats["ingest"].get("partial_days")) else "partial"
        except BaseException as exc:
            stats["error"] = repr(exc)
            db.finish_run(run_id, "error", stats)
            raise
        db.finish_run(run_id, status, stats)
        return stats


def daily_update(settings: Settings) -> dict:
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=settings.update_lookback_days)
    # Citations, stars and upvotes keep arriving: re-score the last RESCORE_WEEKS weeks.
    rescore_from = min(start, today - timedelta(weeks=settings.rescore_weeks))
    return run(settings, "update", start, today, rescore_from=rescore_from, heal_gaps=True, enrich=True)


def enrich_and_rescore(settings: Settings, weeks: int | None = None) -> dict:
    """Fetch external evidence and re-score, without touching arXiv."""
    db = Database(settings.db_path)
    db.init()
    today = datetime.now(timezone.utc).date()
    with pipeline_lock(settings):
        run_id = db.start_run("enrich")
        stats = {"enrich": run_enrichment(settings, db, today)}
        stats["scored"] = score(settings, db, weeks_between(
            today - timedelta(weeks=weeks or settings.rescore_weeks), today))
        db.finish_run(run_id, "ok", stats)
        return stats


def backfill(settings: Settings, start: date, end: date) -> dict:
    return run(settings, "backfill", start, end)


def rescore(settings: Settings, start: date, end: date) -> dict:
    db = Database(settings.db_path)
    db.init()
    with pipeline_lock(settings):
        run_id = db.start_run("rescore")
        stats = {"embedded": embed_missing(settings, db), "keyphrases": extract_keyphrases(db),
                 "scored": score(settings, db, weeks_between(start, end))}
        db.finish_run(run_id, "ok", stats)
        return stats
