"""Command line: `cs-paper-radar <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone

from . import __version__
from .config import get_settings
from .db import Database


def _date(s: str) -> date:
    return date.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cs-paper-radar", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create the database schema")
    sub.add_parser("update", help="daily job: fetch recent days, embed, score")

    p = sub.add_parser("backfill", help="fetch a date range (inclusive), embed, score")
    p.add_argument("--start", type=_date)
    p.add_argument("--end", type=_date)
    p.add_argument("--days", type=int, help="shortcut: the last N days up to today")

    p = sub.add_parser("rescore", help="recompute scores and weekly picks for a range")
    p.add_argument("--start", type=_date, required=True)
    p.add_argument("--end", type=_date, default=None)

    sub.add_parser("embed", help="embed papers that have no vector yet")
    p = sub.add_parser("enrich", help="fetch citations/upvotes/stars and re-score recent weeks")
    p.add_argument("--weeks", type=int, default=None, help="weeks to re-score (default RESCORE_WEEKS)")
    sub.add_parser("stats", help="print database statistics")

    p = sub.add_parser("top", help="print the top papers of an ISO week")
    p.add_argument("--week", help="e.g. 2026-W46 (default: latest)")
    p.add_argument("-n", type=int, default=None)

    p = sub.add_parser("topic", help="print a topic report as JSON")
    p.add_argument("query")
    p.add_argument("--days", type=int, default=30)

    p = sub.add_parser("serve", help="run the web app")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    db = Database(settings.db_path)

    from . import pipeline

    if args.cmd == "init":
        db.init()
        print(f"initialised {settings.db_path}")
    elif args.cmd == "update":
        print(json.dumps(pipeline.daily_update(settings), indent=2))
    elif args.cmd == "backfill":
        today = datetime.now(timezone.utc).date()
        if args.days:
            start, end = today - timedelta(days=args.days - 1), today
        else:
            if not args.start:
                parser.error("backfill needs --start or --days")
            start, end = args.start, args.end or today
        print(json.dumps(pipeline.backfill(settings, start, end), indent=2))
    elif args.cmd == "rescore":
        end = args.end or datetime.now(timezone.utc).date()
        print(json.dumps(pipeline.rescore(settings, args.start, end), indent=2))
    elif args.cmd == "enrich":
        print(json.dumps(pipeline.enrich_and_rescore(settings, args.weeks), indent=2))
    elif args.cmd == "embed":
        db.init()
        print(pipeline.embed_missing(settings, db))
    elif args.cmd == "stats":
        db.init()
        print(json.dumps(db.stats(), indent=2))
    elif args.cmd == "top":
        db.init()
        week = args.week or (db.available_weeks(1) or [None])[0]
        if not week:
            print("no data yet", file=sys.stderr)
            return 1
        for i, p in enumerate(db.weekly_picks(week)[: args.n or settings.top_n], 1):
            print(f"{i:2d}. {p['total']:.3f} [{p['primary_category']}] {p['title']}  {p['abs_url']}")
    elif args.cmd == "topic":
        from .topics import TopicEngine, VectorIndex

        db.init()
        engine = TopicEngine(settings, db, VectorIndex(settings, db))
        rep = engine.report(args.query, days=args.days)

        def slim(p):
            return {k: p[k] for k in ("uid", "title", "pub_date", "primary_category", "total")}

        rep["new"] = [{**{k: v for k, v in h.items() if k != "paper"}, **slim(h["paper"])} for h in rep["new"][:10]]
        rep["frontier"] = [{**{k: v for k, v in h.items() if k != "paper"}, **slim(h["paper"])} for h in rep["frontier"]]
        for c in rep["subtopics"]:
            c["papers"] = [slim(p) for p in c["papers"]]
        print(json.dumps(rep, indent=2, default=str))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("cs_paper_radar.web.app:create_app", factory=True, host=args.host,
                    port=args.port, proxy_headers=True, forwarded_allow_ips="*",
                    limit_concurrency=64, timeout_keep_alive=5, server_header=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
