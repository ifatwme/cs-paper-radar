"""External evidence for papers, fetched from free APIs.

| Source | What | Access |
|---|---|---|
| Semantic Scholar Graph API | citations, influential citations, publication venue, author h-index | free; `S2_API_KEY` optional (higher, dedicated rate limit) |
| Hugging Face daily papers | community upvotes and comments, linked GitHub repo | free, no key |
| GitHub REST API | stars of the paper's code repository | free; `GITHUB_TOKEN` optional (60 -> 5000 requests/hour) |
| CORE conference ranking (ICORE) | venue tier A*/A/B/C | free CSV download, cached for 30 days |

Every source runs under a time budget and fails soft: a missing or rate-limited
API leaves the previous values in place and the run continues.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime, timedelta, timezone

from ..config import Settings
from ..db import Database
from .http import BudgetExceeded, HttpError, PoliteClient

log = logging.getLogger(__name__)

_GH_RE = re.compile(r"github\.com/([A-Za-z0-9][A-Za-z0-9\-]{0,38})/([A-Za-z0-9_.\-]{1,100})", re.I)
_GH_RESERVED = {"features", "orgs", "topics", "sponsors", "settings", "marketplace", "about",
                "collections", "trending", "apps", "login", "site", "pricing", "enterprise"}


def extract_github_repo(*texts: str) -> str | None:
    """First github.com/<owner>/<repo> link in the given texts, normalised."""
    for text in texts:
        for owner, repo in _GH_RE.findall(text or ""):
            repo = re.sub(r"(\.git)?[.,;:)\]]*$", "", repo)
            if owner.lower() in _GH_RESERVED or not repo:
                continue
            return f"{owner}/{repo}"
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------- Semantic Scholar
S2_FIELDS = "citationCount,influentialCitationCount,venue,publicationVenue,journal,authors.hIndex"


def enrich_semantic_scholar(settings: Settings, db: Database, today: date) -> dict:
    headers = {"x-api-key": settings.s2_api_key} if settings.s2_api_key else {}
    client = PoliteClient(min_interval=1.1, budget_s=settings.enrich_budget_s, headers=headers)
    max_batches = settings.s2_max_batches_keyed if settings.s2_api_key else settings.s2_max_batches
    stats = {"requested": 0, "found": 0, "batches": 0, "errors": 0}
    for _ in range(max_batches):
        due = db.s2_due(today, limit=500)
        if not due:
            break
        try:
            res = client.post(
                "https://api.semanticscholar.org/graph/v1/paper/batch",
                {"ids": [f"arXiv:{nid}" for _, nid in due]},
                params={"fields": S2_FIELDS},
            )
        except BudgetExceeded:
            log.warning("Semantic Scholar: time budget exhausted")
            break
        except Exception as exc:  # keep the run alive
            log.error("Semantic Scholar batch failed: %s", exc)
            stats["errors"] += 1
            break
        rows = []
        for (uid, _), item in zip(due, res or []):
            if not item:
                rows.append({"uid": uid, "s2_id": None})
                continue
            venue = (item.get("publicationVenue") or {}).get("name") or item.get("venue") or ""
            vtype = (item.get("publicationVenue") or {}).get("type") or ""
            journal = (item.get("journal") or {}).get("name") or ""
            hs = [a.get("hIndex") for a in item.get("authors") or [] if a.get("hIndex") is not None]
            rows.append({
                "uid": uid, "s2_id": item.get("paperId"),
                "citations": item.get("citationCount") or 0,
                "influential": item.get("influentialCitationCount") or 0,
                "s2_venue": venue if venue.lower() not in ("arxiv.org", "arxiv") else "",
                "s2_venue_type": vtype or ("journal" if journal and journal.lower() != "arxiv.org" else ""),
                "max_h_index": max(hs) if hs else None,
            })
            stats["found"] += 1
        db.save_s2(rows)
        stats["requested"] += len(due)
        stats["batches"] += 1
    return stats


# --------------------------------------------------------------- Hugging Face
def enrich_huggingface(settings: Settings, db: Database, today: date) -> dict:
    client = PoliteClient(min_interval=0.5, budget_s=settings.enrich_budget_s)
    stats = {"days": 0, "papers": 0, "matched": 0}
    rows = []
    for back in range(settings.hf_days):
        day = (today - timedelta(days=back)).isoformat()
        page = 0
        while True:
            try:
                items = client.get("https://huggingface.co/api/daily_papers",
                                   {"date": day, "limit": 100, "p": page})
            except BudgetExceeded:
                log.warning("Hugging Face: time budget exhausted")
                return _save_hf(db, rows, stats)
            except Exception as exc:
                log.error("Hugging Face %s failed: %s", day, exc)
                break
            if not isinstance(items, list):
                break
            for it in items:
                paper = it.get("paper") or {}
                pid = paper.get("id")
                if not pid:
                    continue
                rows.append({
                    "uid": f"arxiv:{pid}",
                    "hf_upvotes": int(paper.get("upvotes") or 0),
                    "hf_comments": int(it.get("numComments") or 0),
                    "hf_github": extract_github_repo(paper.get("githubRepo") or ""),
                })
            stats["papers"] += len(items)
            if len(items) < 100 or page >= 4:
                break
            page += 1
        stats["days"] += 1
    return _save_hf(db, rows, stats)


def _save_hf(db: Database, rows: list[dict], stats: dict) -> dict:
    # A paper can be featured on several days; keep its highest counts.
    best: dict[str, dict] = {}
    for r in rows:
        cur = best.get(r["uid"])
        if cur is None or r["hf_upvotes"] > cur["hf_upvotes"]:
            best[r["uid"]] = r
    stats["matched"] = db.save_hf(list(best.values()))
    return stats


# --------------------------------------------------------------- GitHub
def enrich_github(settings: Settings, db: Database, today: date) -> dict:
    stats = {"repos_found": db.extract_repos(extract_github_repo), "checked": 0, "missing": 0}
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    limit = settings.github_max_requests_keyed if settings.github_token else settings.github_max_requests
    client = PoliteClient(min_interval=0.3 if settings.github_token else 1.0,
                          budget_s=settings.enrich_budget_s, headers=headers, retries=2)
    rows = []
    for uid, repo in db.github_due(today, limit=limit):
        try:
            data = client.get(f"https://api.github.com/repos/{repo}")
            rows.append({"uid": uid, "github_stars": int(data.get("stargazers_count") or 0)})
        except HttpError as exc:
            if exc.status in (404, 451):
                rows.append({"uid": uid, "github_stars": None})  # deleted / private / mistyped link
                stats["missing"] += 1
            elif exc.status in (403, 429):
                log.warning("GitHub rate limit reached")
                break
        except BudgetExceeded:
            break
        except Exception as exc:
            log.error("GitHub %s failed: %s", repo, exc)
            break
        stats["checked"] += 1
    db.save_github(rows)
    return stats


def run_enrichment(settings: Settings, db: Database, today: date | None = None) -> dict:
    """Fetch external evidence for due papers. Never raises: errors are logged and
    reported in the stats so scoring can proceed with what is available."""
    today = today or datetime.now(timezone.utc).date()
    out: dict = {}
    for name, fn in (("semantic_scholar", enrich_semantic_scholar),
                     ("huggingface", enrich_huggingface),
                     ("github", enrich_github)):
        if name not in settings.enrichers:
            continue
        t0 = time.monotonic()
        try:
            out[name] = fn(settings, db, today)
        except Exception as exc:
            log.exception("enrichment %s failed", name)
            out[name] = {"error": repr(exc)}
        out[name]["seconds"] = round(time.monotonic() - t0, 1)
        log.info("enrichment %s: %s", name, out[name])
    db.snapshot_metrics(today.isoformat())
    return out
