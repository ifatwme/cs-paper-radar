# CS Paper Radar

**What is computer science working on this week?** CS Paper Radar lists every new arXiv paper in
computer science and computer engineering each day, picks the **top N papers of the week** with a
transparent score, and turns any topic (e.g. *network security*) into a briefing: what's new, what the
frontier papers are, which terms are emerging, how the topic splits into research directions, and
whether it is growing.

- **Daily** — every paper first submitted on a given day, grouped by category, filterable.
- **Weekly top N** (default 20) — composite "frontier score", diversity-capped per category, no near-duplicates.
- **Topics** — hybrid keyword (SQLite FTS5/BM25) + semantic (local sentence embeddings) search, with
  *what's new*, *frontier*, *emerging terms*, *research directions* (clusters) and a weekly *trend*.
- **Paper pages** — score breakdown with the evidence behind every signal, and the closest papers.
- **Categories** — weekly volume and growth for every CS/EESS category.
- **JSON API** + RSS feed of the weekly top list. Interactive docs at `/api/docs`.

Source for now: the [arXiv API](https://info.arxiv.org/help/api/index.html), categories
`cs.*`, `eess.SY`, `eess.SP`, `eess.AS`, `eess.IV` (configurable). Other sources plug in through
`src/cs_paper_radar/sources/` (see [Adding a source](#adding-a-source)).

## How ranking works

arXiv provides no citations, downloads or votes, so new papers are ranked from signals computed from the
paper and from the corpus. Each signal is in [0, 1]; the score is their weighted sum.

| Signal | Default weight | Measures |
|---|---|---|
| Venue | 20% | "Accepted at / to appear in" a recognised venue (NeurIPS, CCS, USENIX Security, SIGCOMM, OSDI, ICSE, STOC, ...), journal ref, DOI. Workshops count less, "under review" little. |
| Code/data | 15% | Released code, data, models or project page (GitHub/Hugging Face/Zenodo links or explicit statements). |
| Momentum | 20% | Papers this week in the paper's embedding neighbourhood vs. the number expected from the previous 4 weeks — its direction is accelerating. Percentile within the week. |
| Novelty | 20% | Distance to the closest paper of the previous 90 days. Percentile within the week. |
| Breadth | 10% | Cross-listed across categories and research areas. |
| Substance | 10% | Full paper (pages, figures, journal version) rather than a short note. |
| Team | 5% | Collaboration size, log-scaled, capped. |

Weights are configurable with `SCORE_WEIGHTS=venue=0.3,novelty=0.25` (renormalised). The weekly list
takes at most `TOP_MAX_PER_CATEGORY` (4) papers per primary category. Details:
[docs/scoring.md](docs/scoring.md).

## Run it

Requirements: Docker with Compose v2.

```bash
cp .env.example .env            # local use: delete the COMPOSE_FILE and TLS_DIR lines
mkdir -p data && sudo chown 10001:10001 data
docker compose build
docker compose run --rm job backfill --days 90   # first load: ~10 min fetching + ~45 min embedding on 4 vCPUs
docker compose up -d                              # http://127.0.0.1:8000
docker compose run --rm job update                # what the daily timer runs
```

Other jobs: `stats`, `top --week 2026-W46`, `topic "network security"`, `rescore --start 2026-10-01`.
Without Docker: `pip install -r requirements.txt -e . && DATA_DIR=./data cs-paper-radar backfill --days 30 && cs-paper-radar serve`.

### Server deployment (what runs on the VPS)

- `deploy/compose.server.yml` adds an `nginx-unprivileged` HTTPS proxy on `HTTPS_PORT` using an
  existing Let's Encrypt certificate (`TLS_DIR`) and hides the app port. Enabled by `COMPOSE_FILE` in `.env`.
- `deploy/systemd/cs-paper-radar.service` keeps the stack up; `cs-paper-radar-update.timer` runs
  `update` at 02:30 and 14:30 UTC (after arXiv's announcement at 20:00 US Eastern). Install both with
  `make install-systemd`.
- Each update re-fetches the last 8 days (late announcements, new versions, acceptance notes and
  code links added during the week) and re-scores the current and previous week. A full run takes
  ~1-2 minutes: one arXiv request per day (page size 1000), embeddings only for new/changed papers.
- **Self-healing**: every fetched day is logged with the count arXiv reported. Days in the last 60 that
  are missing or were stored incompletely (outage, failed page) are re-fetched automatically on the
  next run, then their weeks are re-scored.
- Only one pipeline run at a time (file lock), so a manual backfill and the timer cannot collide.

Containers run as a non-root user with a read-only root filesystem, all capabilities dropped,
`no-new-privileges`, memory/CPU/PID limits and rotated logs.

## Configuration

All settings are environment variables (see `.env.example`):

| Variable | Default | |
|---|---|---|
| `ARXIV_CATEGORIES` | `cs.*,eess.SY,eess.SP,eess.AS,eess.IV` | arXiv scope |
| `TOP_N` | `20` | size of the weekly list (the page also accepts `?n=`) |
| `TOP_MAX_PER_CATEGORY` | `4` | diversity cap in the weekly list |
| `TOP_CS_PRIMARY_ONLY` | `true` | weekly picks only from papers whose primary category is CS/EESS |
| `SCORE_WEIGHTS` | – | per-signal weight overrides |
| `EMBEDDINGS` | `fastembed` | `fastembed` (local ONNX model), `hash` (no model, for tests), `off` (keywords only) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | any fastembed text model |
| `UPDATE_LOOKBACK_DAYS` | `8` | days re-fetched by each update |
| `HEAL_WINDOW_DAYS` / `HEAL_MAX_DAYS` | `60` / `10` | how far back incomplete days are repaired, and how many per run |
| `ARXIV_PAGE_SIZE` / `ARXIV_DELAY_S` | `1000` / `3.5` | request size and politeness delay (arXiv asks for ≥ 3 s) |
| `NOVELTY_LOOKBACK_DAYS` / `MOMENTUM_BASELINE_WEEKS` | `90` / `4` | corpus-signal windows |

## API

| Endpoint | |
|---|---|
| `GET /api/daily?date=YYYY-MM-DD&cat=cs.CR&sort=score` | papers of a day |
| `GET /api/weekly?week=2026-W46&n=20&cat=cs.NI` | top N of a week |
| `GET /api/topic?q=network security&days=30` | topic report (new, frontier, emerging terms, clusters, trend) |
| `GET /api/search?q=...&days=30` | hybrid search |
| `GET /api/papers/arxiv:2611.12557` | one paper, score evidence, nearest neighbours |
| `GET /api/categories`, `GET /api/stats` | volumes and growth, pipeline status |
| `GET /feed/weekly.xml` | RSS of the current weekly top list |

## Architecture

```
arXiv API ──► sources/arxiv.py ──► db.py (SQLite + FTS5) ──► embed.py (fastembed, ONNX, CPU)
                                        │                         │
                                        ▼                         ▼
                                   scoring.py ◄──── corpus signals (novelty, momentum)
                                        │
                     weekly_picks ◄─────┘
                                        ▼
web/app.py (FastAPI + Jinja) ◄── topics.py (hybrid search, emerging terms, clusters, trend)
```

```
src/cs_paper_radar/
  sources/        pluggable sources (arxiv.py; base.py = interface; registry in __init__.py)
  db.py           schema, upserts, queries
  embed.py        embedding backends
  scoring.py      signals, weekly scoring and top-N selection
  topics.py       topic engine
  pipeline.py     ingest -> embed -> score (file lock: one run at a time)
  cli.py          `cs-paper-radar` command
  web/            FastAPI app, templates, CSS
docker/, deploy/  image, nginx, server override, systemd units
tests/            unit + end-to-end tests (synthetic source, hash embeddings, real arXiv XML fixture)
```

## Adding a source

Implement `fetch_day(day) -> Iterator[Paper]` and register it:

```python
from cs_paper_radar.sources import register

@register("openreview")
class OpenReviewSource:
    name = "openreview"
    def __init__(self, settings): ...
    def fetch_day(self, day): yield Paper(source="openreview", native_id=..., ...)
```

then set `SOURCES=arxiv,openreview`. Paper ids are namespaced (`<source>:<id>`). Planned: Semantic
Scholar citations and Hugging Face Papers upvotes as extra score signals, cross-source deduplication by
DOI/title.

## Development

```bash
make test     # pytest in a container (synthetic source + hash embeddings, no network)
make lint
```

Metadata comes from arXiv; thank you to arXiv for use of its open access interoperability. This project
is not affiliated with arXiv.

License: MIT.
