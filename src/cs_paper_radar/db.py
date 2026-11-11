"""SQLite storage: papers + full-text index + embeddings + scores.

SQLite (WAL mode) is enough for a single host: the daily job writes, the web
app reads concurrently. FTS5 provides BM25 keyword search.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from .models import Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    uid              TEXT PRIMARY KEY,
    source           TEXT NOT NULL,
    native_id        TEXT NOT NULL,
    version          INTEGER NOT NULL DEFAULT 1,
    title            TEXT NOT NULL,
    abstract         TEXT NOT NULL,
    authors          TEXT NOT NULL,           -- JSON list
    n_authors        INTEGER NOT NULL,
    primary_category TEXT NOT NULL,
    categories       TEXT NOT NULL,           -- JSON list
    comment          TEXT NOT NULL DEFAULT '',
    journal_ref      TEXT NOT NULL DEFAULT '',
    doi              TEXT NOT NULL DEFAULT '',
    abs_url          TEXT NOT NULL DEFAULT '',
    pdf_url          TEXT NOT NULL DEFAULT '',
    published        TEXT NOT NULL,           -- ISO timestamp, UTC
    pub_date         TEXT NOT NULL,           -- YYYY-MM-DD (UTC)
    week             TEXT NOT NULL,           -- ISO week, e.g. 2026-W46
    updated          TEXT,
    withdrawn        INTEGER NOT NULL DEFAULT 0,
    first_seen       TEXT NOT NULL,
    last_seen        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS papers_pub_date ON papers(pub_date);
CREATE INDEX IF NOT EXISTS papers_week ON papers(week);
CREATE INDEX IF NOT EXISTS papers_primary ON papers(primary_category, pub_date);

CREATE TABLE IF NOT EXISTS paper_categories (
    uid      TEXT NOT NULL REFERENCES papers(uid) ON DELETE CASCADE,
    category TEXT NOT NULL,
    PRIMARY KEY (uid, category)
);
CREATE INDEX IF NOT EXISTS paper_categories_cat ON paper_categories(category);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, abstract, authors, comment,
    content='papers', content_rowid='rowid',
    tokenize='porter unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
    INSERT INTO papers_fts(rowid, title, abstract, authors, comment)
    VALUES (new.rowid, new.title, new.abstract, new.authors, new.comment);
END;
CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract, authors, comment)
    VALUES ('delete', old.rowid, old.title, old.abstract, old.authors, old.comment);
END;
CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract, authors, comment)
    VALUES ('delete', old.rowid, old.title, old.abstract, old.authors, old.comment);
    INSERT INTO papers_fts(rowid, title, abstract, authors, comment)
    VALUES (new.rowid, new.title, new.abstract, new.authors, new.comment);
END;

CREATE TABLE IF NOT EXISTS embeddings (
    uid       TEXT PRIMARY KEY REFERENCES papers(uid) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    vec       BLOB NOT NULL                   -- float32, L2-normalised
);

CREATE TABLE IF NOT EXISTS scores (
    uid         TEXT PRIMARY KEY REFERENCES papers(uid) ON DELETE CASCADE,
    total       REAL NOT NULL,
    venue       REAL NOT NULL,
    artifacts   REAL NOT NULL,
    momentum    REAL NOT NULL,
    novelty     REAL NOT NULL,
    breadth     REAL NOT NULL,
    substance   REAL NOT NULL,
    team        REAL NOT NULL,
    tags        TEXT NOT NULL DEFAULT '[]',   -- JSON list of badges
    details     TEXT NOT NULL DEFAULT '{}',   -- JSON: evidence behind each signal
    computed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS scores_total ON scores(total DESC);

CREATE TABLE IF NOT EXISTS weekly_picks (
    week     TEXT NOT NULL,
    rank     INTEGER NOT NULL,
    uid      TEXT NOT NULL REFERENCES papers(uid) ON DELETE CASCADE,
    total    REAL NOT NULL,
    PRIMARY KEY (week, rank)
);

CREATE TABLE IF NOT EXISTS runs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    kind     TEXT NOT NULL,
    started  TEXT NOT NULL,
    finished TEXT,
    status   TEXT NOT NULL,
    stats    TEXT NOT NULL DEFAULT '{}'
);
"""

SIGNALS = ("venue", "artifacts", "momentum", "novelty", "breadth", "substance", "team")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iso_week(d: date) -> str:
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


def week_bounds(week: str) -> tuple[date, date]:
    year, wk = week.split("-W")
    start = date.fromisocalendar(int(year), int(wk), 1)
    return start, date.fromisocalendar(int(year), int(wk), 7)


_FTS_TOKEN = re.compile(r'"[^"]+"|[\w][\w\-\.]*', re.UNICODE)


def build_fts_query(text: str, mode: str = "and") -> str:
    """Turn free text into a safe FTS5 query.

    mode: "and" (all terms), "or" (any term), "near" (all terms within a few
    words of each other, e.g. "network security" will not match "neural
    network ... security").
    """
    tokens = []
    for tok in _FTS_TOKEN.findall(text):
        tok = tok.strip('"').replace('"', "")
        if not tok or tok.upper() in {"AND", "OR", "NOT", "NEAR"}:
            continue
        tokens.append(f'"{tok}"')
    if mode == "near" and len(tokens) > 1:
        return f"NEAR({' '.join(tokens)}, {2 + len(tokens)})"
    joiner = " OR " if mode == "or" else " "
    return joiner.join(tokens)


def row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ("authors", "categories", "tags", "details"):
        if key in d and isinstance(d[key], str):
            d[key] = json.loads(d[key])
    return d


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=60000")
        return conn

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def init(self) -> None:
        with self.session() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ writes
    def upsert_papers(self, papers: Iterable[Paper]) -> tuple[int, int]:
        """Insert new papers, refresh changed ones. Returns (inserted, updated)."""
        now = utcnow()
        inserted = updated = 0
        with self.session() as conn:
            for p in papers:
                pub = p.published.astimezone(timezone.utc)
                values = {
                    "uid": p.uid,
                    "source": p.source,
                    "native_id": p.native_id,
                    "version": p.version,
                    "title": p.title,
                    "abstract": p.abstract,
                    "authors": json.dumps(p.authors, ensure_ascii=False),
                    "n_authors": len(p.authors),
                    "primary_category": p.primary_category,
                    "categories": json.dumps(p.categories),
                    "comment": p.comment,
                    "journal_ref": p.journal_ref,
                    "doi": p.doi,
                    "abs_url": p.abs_url,
                    "pdf_url": p.pdf_url,
                    "published": pub.isoformat(timespec="seconds"),
                    "pub_date": pub.date().isoformat(),
                    "week": iso_week(pub.date()),
                    "updated": p.updated.isoformat(timespec="seconds") if p.updated else None,
                    "withdrawn": int(p.withdrawn),
                    "now": now,
                }
                existing = conn.execute(
                    "SELECT version, title, abstract, comment, journal_ref, categories "
                    "FROM papers WHERE uid = ?",
                    (p.uid,),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """INSERT INTO papers (uid, source, native_id, version, title, abstract,
                        authors, n_authors, primary_category, categories, comment, journal_ref,
                        doi, abs_url, pdf_url, published, pub_date, week, updated, withdrawn,
                        first_seen, last_seen) VALUES (:uid, :source, :native_id, :version,
                        :title, :abstract, :authors, :n_authors, :primary_category, :categories,
                        :comment, :journal_ref, :doi, :abs_url, :pdf_url, :published, :pub_date,
                        :week, :updated, :withdrawn, :now, :now)""",
                        values,
                    )
                    inserted += 1
                elif (
                    existing["version"] != p.version
                    or existing["title"] != p.title
                    or existing["abstract"] != p.abstract
                    or existing["comment"] != p.comment
                    or existing["journal_ref"] != p.journal_ref
                    or existing["categories"] != values["categories"]
                ):
                    conn.execute(
                        """UPDATE papers SET version=:version, title=:title, abstract=:abstract,
                        authors=:authors, n_authors=:n_authors, primary_category=:primary_category,
                        categories=:categories, comment=:comment, journal_ref=:journal_ref,
                        doi=:doi, pdf_url=:pdf_url, updated=:updated, withdrawn=:withdrawn,
                        last_seen=:now WHERE uid=:uid""",
                        values,
                    )
                    updated += 1
                else:
                    conn.execute("UPDATE papers SET last_seen=? WHERE uid=?", (now, p.uid))
                conn.execute("DELETE FROM paper_categories WHERE uid=?", (p.uid,))
                conn.executemany(
                    "INSERT OR IGNORE INTO paper_categories(uid, category) VALUES (?, ?)",
                    [(p.uid, c) for c in (p.categories or [p.primary_category])],
                )
        return inserted, updated

    def save_embeddings(self, model: str, rows: list[tuple[str, str, np.ndarray]]) -> None:
        with self.session() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings(uid, model, text_hash, vec) VALUES (?,?,?,?)",
                [(uid, model, h, v.astype(np.float32).tobytes()) for uid, h, v in rows],
            )

    def save_scores(self, rows: list[dict]) -> None:
        now = utcnow()
        with self.session() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO scores(uid, total, venue, artifacts, momentum, novelty,
                breadth, substance, team, tags, details, computed_at) VALUES (:uid, :total,
                :venue, :artifacts, :momentum, :novelty, :breadth, :substance, :team, :tags,
                :details, :computed_at)""",
                [
                    {
                        **r,
                        "tags": json.dumps(r.get("tags", [])),
                        "details": json.dumps(r.get("details", {}), ensure_ascii=False),
                        "computed_at": now,
                    }
                    for r in rows
                ],
            )

    def save_weekly_picks(self, week: str, picks: list[tuple[str, float]]) -> None:
        with self.session() as conn:
            conn.execute("DELETE FROM weekly_picks WHERE week=?", (week,))
            conn.executemany(
                "INSERT INTO weekly_picks(week, rank, uid, total) VALUES (?,?,?,?)",
                [(week, i + 1, uid, total) for i, (uid, total) in enumerate(picks)],
            )

    def start_run(self, kind: str) -> int:
        with self.session() as conn:
            cur = conn.execute(
                "INSERT INTO runs(kind, started, status) VALUES (?,?, 'running')", (kind, utcnow())
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, stats: dict) -> None:
        with self.session() as conn:
            conn.execute(
                "UPDATE runs SET finished=?, status=?, stats=? WHERE id=?",
                (utcnow(), status, json.dumps(stats), run_id),
            )

    # ------------------------------------------------------------------- reads
    PAPER_COLUMNS = """p.uid, p.source, p.native_id, p.version, p.title, p.abstract, p.authors,
        p.n_authors, p.primary_category, p.categories, p.comment, p.journal_ref, p.doi,
        p.abs_url, p.pdf_url, p.published, p.pub_date, p.week, p.updated, p.withdrawn,
        s.total, s.venue, s.artifacts, s.momentum, s.novelty, s.breadth, s.substance, s.team,
        s.tags, s.details"""

    def _select(self, where: str, params: tuple | dict = (), tail: str = "") -> list[dict]:
        sql = (
            f"SELECT {self.PAPER_COLUMNS} FROM papers p LEFT JOIN scores s ON s.uid = p.uid "
            f"WHERE {where} {tail}"
        )
        with self.session() as conn:
            return [row_to_dict(r) for r in conn.execute(sql, params)]

    def get_paper(self, uid: str) -> dict | None:
        rows = self._select("p.uid = ?", (uid,))
        return rows[0] if rows else None

    def get_papers(self, uids: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for i in range(0, len(uids), 500):
            chunk = uids[i : i + 500]
            marks = ",".join("?" * len(chunk))
            for r in self._select(f"p.uid IN ({marks})", tuple(chunk)):
                out[r["uid"]] = r
        return out

    def papers_on(
        self, day: str, category: str | None = None, sort: str = "category"
    ) -> list[dict]:
        order = {
            "score": "ORDER BY s.total DESC NULLS LAST",
            "time": "ORDER BY p.published DESC",
            "category": "ORDER BY p.primary_category, s.total DESC NULLS LAST",
        }[sort]
        if category:
            return self._select(
                "p.pub_date = ? AND p.uid IN (SELECT uid FROM paper_categories WHERE category=?)",
                (day, category),
                order,
            )
        return self._select("p.pub_date = ?", (day,), order)

    def papers_between(
        self, start: str, end: str, category: str | None = None, scored_only: bool = False
    ) -> list[dict]:
        where = "p.pub_date BETWEEN ? AND ? AND p.withdrawn = 0"
        params: tuple = (start, end)
        if category:
            where += " AND p.uid IN (SELECT uid FROM paper_categories WHERE category=?)"
            params += (category,)
        if scored_only:
            where += " AND s.total IS NOT NULL"
        return self._select(where, params, "ORDER BY s.total DESC NULLS LAST")

    def weekly_picks(self, week: str) -> list[dict]:
        with self.session() as conn:
            rows = conn.execute(
                "SELECT uid, rank FROM weekly_picks WHERE week=? ORDER BY rank", (week,)
            ).fetchall()
        papers = self.get_papers([r["uid"] for r in rows])
        return [papers[r["uid"]] for r in rows if r["uid"] in papers]

    def fts_search(
        self, query: str, start: str | None = None, end: str | None = None, limit: int = 500,
        mode: str = "and",
    ) -> list[tuple[str, float]]:
        """BM25 keyword search -> [(uid, bm25)] (lower bm25 = better)."""
        fts = build_fts_query(query, mode)
        if not fts:
            return []
        sql = (
            "SELECT p.uid, bm25(papers_fts, 10.0, 3.0, 1.0, 0.5) AS rank "
            "FROM papers_fts JOIN papers p ON p.rowid = papers_fts.rowid "
            "WHERE papers_fts MATCH ? AND p.withdrawn = 0"
        )
        params: list = [fts]
        if start:
            sql += " AND p.pub_date >= ?"
            params.append(start)
        if end:
            sql += " AND p.pub_date <= ?"
            params.append(end)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        with self.session() as conn:
            try:
                return [(r["uid"], r["rank"]) for r in conn.execute(sql, params)]
            except sqlite3.OperationalError:
                return []

    def missing_embeddings(self, model: str, limit: int | None = None) -> list[sqlite3.Row]:
        sql = (
            "SELECT p.uid, p.title, p.abstract FROM papers p LEFT JOIN embeddings e "
            "ON e.uid = p.uid AND e.model = ? WHERE e.uid IS NULL ORDER BY p.published"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self.session() as conn:
            return conn.execute(sql, (model,)).fetchall()

    def load_vectors(
        self, model: str, start: str | None = None, end: str | None = None
    ) -> tuple[list[str], list[str], np.ndarray]:
        """Return (uids, pub_dates, matrix[n, d]) for a date window, ordered by date."""
        sql = (
            "SELECT p.uid, p.pub_date, e.vec FROM embeddings e JOIN papers p ON p.uid = e.uid "
            "WHERE e.model = ? AND p.withdrawn = 0"
        )
        params: list = [model]
        if start:
            sql += " AND p.pub_date >= ?"
            params.append(start)
        if end:
            sql += " AND p.pub_date <= ?"
            params.append(end)
        sql += " ORDER BY p.published"
        uids, dates, vecs = [], [], []
        with self.session() as conn:
            for r in conn.execute(sql, params):
                uids.append(r["uid"])
                dates.append(r["pub_date"])
                vecs.append(np.frombuffer(r["vec"], dtype=np.float32))
        matrix = np.vstack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)
        return uids, dates, matrix

    def daily_counts(self, start: str, end: str) -> list[dict]:
        with self.session() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT pub_date, COUNT(*) AS n FROM papers WHERE pub_date BETWEEN ? AND ? "
                    "GROUP BY pub_date ORDER BY pub_date",
                    (start, end),
                )
            ]

    def category_counts(self, start: str, end: str, primary_only: bool = False) -> dict[str, int]:
        with self.session() as conn:
            if primary_only:
                rows = conn.execute(
                    "SELECT primary_category AS category, COUNT(*) AS n FROM papers "
                    "WHERE pub_date BETWEEN ? AND ? GROUP BY primary_category",
                    (start, end),
                )
            else:
                rows = conn.execute(
                    "SELECT c.category, COUNT(*) AS n FROM paper_categories c JOIN papers p "
                    "ON p.uid = c.uid WHERE p.pub_date BETWEEN ? AND ? GROUP BY c.category",
                    (start, end),
                )
            return {r["category"]: r["n"] for r in rows}

    def date_range(self) -> tuple[str | None, str | None]:
        with self.session() as conn:
            r = conn.execute("SELECT MIN(pub_date), MAX(pub_date) FROM papers").fetchone()
        return r[0], r[1]

    def available_dates(self, limit: int = 60) -> list[str]:
        with self.session() as conn:
            return [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT pub_date FROM papers ORDER BY pub_date DESC LIMIT ?", (limit,)
                )
            ]

    def available_weeks(self, limit: int = 52) -> list[str]:
        with self.session() as conn:
            return [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT week FROM papers ORDER BY week DESC LIMIT ?", (limit,)
                )
            ]

    def stats(self) -> dict:
        with self.session() as conn:
            papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            embedded = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            scored = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
            last = conn.execute(
                "SELECT id, kind, started, finished, status, stats FROM runs "
                "WHERE status != 'running' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        first, latest = self.date_range()
        return {
            "papers": papers,
            "embedded": embedded,
            "scored": scored,
            "first_date": first,
            "latest_date": latest,
            "last_run": dict(last) if last else None,
        }

    def data_version(self) -> int:
        """Changes whenever a pipeline run finishes; used to refresh caches."""
        with self.session() as conn:
            r = conn.execute("SELECT COALESCE(MAX(id), 0) FROM runs WHERE finished IS NOT NULL")
            return int(r.fetchone()[0])
