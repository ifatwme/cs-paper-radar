"""Transparent composite "frontier score".

Every paper gets seven signals in [0, 1]; the total is their weighted sum
(weights in `config.DEFAULT_WEIGHTS`, overridable with SCORE_WEIGHTS). Each
signal stores the evidence that produced it, so the UI can explain every rank.

Metadata signals (per paper):
    venue      accepted / to appear at a recognised venue, journal ref, DOI
    artifacts  code, data, models or a project page are released
    breadth    cross-listed in several categories / research areas
    substance  full paper (pages, figures) rather than a short note
    team       collaboration size (weak, capped)

Corpus signals (need embeddings, computed per ISO week):
    novelty    1 - similarity to the closest paper of the previous N days,
               as a percentile within the week
    momentum   how many same-week papers sit in the paper's neighbourhood,
               relative to the expected count from the baseline weeks
               (growth of its research direction), as a percentile
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date, timedelta

import numpy as np

from .config import Settings
from .db import Database, week_bounds
from .taxonomy import area_of

log = logging.getLogger(__name__)

SCORING_VERSION = 1

# --------------------------------------------------------------------- venues
VENUES = [
    # ML / AI
    "NeurIPS", "NIPS", "ICML", "ICLR", "AAAI", "IJCAI", "AISTATS", "UAI", "COLT", "TMLR", "JMLR",
    "ECML", "PKDD", "KDD", "ICDM", "SDM", "WSDM", "CIKM", "MLSys", "CoLLAs", "AAMAS", "ECAI",
    # Vision / graphics / media
    "CVPR", "ICCV", "ECCV", "WACV", "BMVC", "SIGGRAPH", "3DV", "ACM MM", "ACM Multimedia",
    "MICCAI", "ISBI", "ICIP",
    # NLP / IR / web
    "ACL", "EMNLP", "NAACL", "EACL", "COLING", "CoNLL", "LREC", "SIGIR", "RecSys", "WWW",
    "TheWebConf", "Web Conference", "ICWSM",
    # Security & privacy
    "CCS", "USENIX Security", "NDSS", "IEEE S&P", "S&P", "Oakland", "EuroS&P", "Euro S&P",
    "RAID", "ACSAC", "AsiaCCS", "ASIACCS", "PETS", "PoPETs", "ESORICS", "DIMVA", "WiSec",
    "CRYPTO", "EUROCRYPT", "ASIACRYPT", "TCC", "CHES", "SaTML", "TIFS", "TDSC",
    # Systems, networking, architecture
    "OSDI", "SOSP", "EuroSys", "USENIX ATC", "ATC", "NSDI", "SIGCOMM", "MobiCom", "MobiSys",
    "SenSys", "INFOCOM", "IMC", "CoNEXT", "HotNets", "ICDCS", "IPDPS", "PPoPP", "HPDC",
    "Supercomputing", "SC'", "ASPLOS", "ISCA", "MICRO", "HPCA", "DAC", "ICCAD", "DATE", "FPGA",
    "FAST", "SoCC", "Middleware", "DSN", "RTSS", "EMSOFT", "VLDB", "PVLDB", "SIGMOD", "ICDE",
    "PODS", "CIDR", "JSAC", "ToN", "TON", "TWC", "TCOM", "TMC", "TPDS", "Globecom", "GLOBECOM",
    "ICC", "WCNC", "VTC",
    # Software & PL
    "ICSE", "FSE", "ESEC", "ASE", "ISSTA", "PLDI", "POPL", "OOPSLA", "ICFP", "CAV", "TACAS",
    "LICS", "CONCUR", "ECOOP", "MSR", "ICSME", "SANER", "TOSEM", "TSE", "EMSE",
    # Theory
    "STOC", "FOCS", "SODA", "ICALP", "ITCS", "PODC", "SPAA", "ESA", "STACS", "CCC", "ISIT",
    # HCI, robotics, control, signals
    "CHI", "UIST", "CSCW", "IUI", "ICRA", "IROS", "RSS", "CoRL", "HRI", "CDC", "ACC", "ECC",
    "L4DC", "Automatica", "ICASSP", "Interspeech", "INTERSPEECH", "EUSIPCO", "TSP", "TAC",
    "TPAMI", "IJCV", "TIP", "TNNLS", "TKDE",
    # Broad
    "Nature", "Science Robotics", "Science Advances", "PNAS", "Communications of the ACM", "ACM Computing Surveys",
    "IEEE Transactions", "ACM Transactions", "Proceedings of the IEEE",
]
_VENUE_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(re.escape(v) for v in sorted(VENUES, key=len, reverse=True))
    + r")(?![A-Za-z])"
)
_ACCEPTED_RE = re.compile(
    r"\b(accepted|to appear|appear(?:s|ed)? (?:in|at)|published (?:in|at)|camera[- ]ready|"
    r"presented at|in proceedings of|proceedings of|oral|spotlight)\b",
    re.I,
)
_UNDER_REVIEW_RE = re.compile(r"\b(under review|submitted to|in submission|preprint)\b", re.I)
_WORKSHOP_RE = re.compile(r"\bworkshop\b", re.I)


def venue_signal(comment: str, journal_ref: str, doi: str) -> tuple[float, str]:
    text = f"{comment} {journal_ref}"
    venue = _VENUE_RE.search(text)
    accepted = _ACCEPTED_RE.search(comment)
    workshop = _WORKSHOP_RE.search(text)
    if journal_ref:
        return (0.7 if workshop else 0.95), f"journal ref: {journal_ref[:80]}"
    if accepted and venue:
        if workshop:
            return 0.6, f"accepted (workshop): {venue.group(1)}"
        return 1.0, f"accepted: {venue.group(1)}"
    if _UNDER_REVIEW_RE.search(comment):
        return 0.1, "under review / submitted"
    if accepted:
        return (0.5 if workshop else 0.75), f"accepted: {accepted.group(0)}"
    if venue:
        return (0.4 if workshop else 0.6), f"venue mentioned: {venue.group(1)}"
    if doi:
        return 0.6, f"DOI {doi}"
    return 0.0, ""


_ARTIFACT_URL_RE = re.compile(
    r"(github\.com/[\w\-\.]+(?:/[\w\-\.]+)?|gitlab\.com/[\w\-/\.]+|huggingface\.co/[\w\-/\.]+|"
    r"zenodo\.org/[\w\-/\.]+|kaggle\.com/[\w\-/\.]+|codeocean\.com/[\w\-/\.]+|"
    r"osf\.io/[\w\-/\.]+|figshare\.com/[\w\-/\.]+|bitbucket\.org/[\w\-/\.]+)",
    re.I,
)
_ARTIFACT_TEXT_RE = re.compile(
    r"\b(code|implementation|data(?:set)?s?|models?|benchmark|artifacts?|weights|checkpoints?)\b"
    r"[^.]{0,60}\b(publicly available|is available|are available|open[- ]sourced?|released|"
    r"available at|can be found at)\b|\bproject page\b|\bopen[- ]source\b",
    re.I,
)
_ARTIFACT_FUTURE_RE = re.compile(r"\b(will be|to be) (publicly )?(released|available|open[- ]sourced)\b", re.I)


def artifacts_signal(abstract: str, comment: str) -> tuple[float, str]:
    text = f"{abstract} {comment}"
    url = _ARTIFACT_URL_RE.search(text)
    if url:
        return 1.0, url.group(1).rstrip(".")
    if _ARTIFACT_FUTURE_RE.search(text):
        return 0.5, "artifacts promised"
    m = _ARTIFACT_TEXT_RE.search(text)
    if m:
        return 0.8, "artifacts available"
    if re.search(r"https?://", text):
        return 0.6, "project link"
    return 0.0, ""


def breadth_signal(categories: list[str]) -> tuple[float, str]:
    cats = list(dict.fromkeys(categories))
    areas = {area_of(c) for c in cats}
    value = 0.7 * min(1.0, (len(cats) - 1) / 3) + 0.3 * (1.0 if len(areas) > 1 else 0.0)
    return round(value, 4), f"{len(cats)} categories, {len(areas)} areas"


_PAGES_RE = re.compile(r"(\d{1,3})\s*(?:\+\s*\d+\s*)?pages?", re.I)
_FIGS_RE = re.compile(r"(\d{1,3})\s*figures?", re.I)


def substance_signal(comment: str, journal_ref: str, abstract: str) -> tuple[float, str]:
    pages_m = _PAGES_RE.search(comment)
    figs_m = _FIGS_RE.search(comment)
    pages = int(pages_m.group(1)) if pages_m else None
    if journal_ref:
        value, why = 1.0, "journal version"
    elif pages is None:
        value, why = 0.4, "length unknown"
    elif pages >= 8:
        value, why = 1.0, f"{pages} pages"
    elif pages >= 4:
        value, why = 0.65, f"{pages} pages"
    else:
        value, why = 0.3, f"{pages} pages"
    if figs_m and int(figs_m.group(1)) >= 3:
        value = min(1.0, value + 0.1)
        why += f", {figs_m.group(1)} figures"
    if len(abstract) < 400:
        value *= 0.7
        why += ", short abstract"
    return round(value, 4), why


def team_signal(n_authors: int) -> tuple[float, str]:
    value = min(1.0, math.log1p(max(n_authors, 0)) / math.log1p(10))
    return round(value, 4), f"{n_authors} authors"


_TAG_RULES = [
    ("survey", re.compile(r"\b(survey|systematization of knowledge|SoK|a review of|overview of)\b", re.I)),
    ("benchmark", re.compile(r"\b(benchmark|leaderboard)\b", re.I)),
    ("dataset", re.compile(r"\b(we (release|introduce|present|collect|construct) [^.]{0,40}dataset|new dataset)\b", re.I)),
    ("position", re.compile(r"\bposition paper\b|\bwe argue\b", re.I)),
    ("theory", re.compile(r"\b(we prove|theorem|lower bound|upper bound)\b", re.I)),
]


def metadata_signals(paper: dict) -> dict:
    venue, venue_why = venue_signal(paper["comment"], paper["journal_ref"], paper["doi"])
    artifacts, art_why = artifacts_signal(paper["abstract"], paper["comment"])
    breadth, breadth_why = breadth_signal(paper["categories"] or [paper["primary_category"]])
    substance, sub_why = substance_signal(paper["comment"], paper["journal_ref"], paper["abstract"])
    team, team_why = team_signal(paper["n_authors"])
    text = f"{paper['title']} {paper['abstract']}"
    tags = [name for name, rx in _TAG_RULES if rx.search(text)]
    if venue >= 0.95:
        tags.insert(0, "accepted")
    if artifacts >= 0.8:
        tags.append("code/data")
    return {
        "venue": venue,
        "artifacts": artifacts,
        "breadth": breadth,
        "substance": substance,
        "team": team,
        "tags": tags,
        "details": {
            "venue": venue_why,
            "artifacts": art_why,
            "breadth": breadth_why,
            "substance": sub_why,
            "team": team_why,
        },
    }


# ------------------------------------------------------------ corpus signals
def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Map values to (0, 1] by rank; ties share the average rank."""
    n = len(values)
    if n == 0:
        return values
    order = values.argsort(kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    # average ranks of ties
    uniq, inverse = np.unique(values, return_inverse=True)
    if len(uniq) < n:
        sums = np.bincount(inverse, weights=ranks)
        counts = np.bincount(inverse)
        ranks = (sums / counts)[inverse]
    return ranks / n


def _max_sim(x: np.ndarray, pool: np.ndarray, chunk: int = 2048) -> np.ndarray:
    best = np.full(len(x), -1.0, dtype=np.float32)
    for i in range(0, len(pool), chunk):
        sims = x @ pool[i : i + chunk].T
        best = np.maximum(best, sims.max(axis=1))
    return best


def _count_above(x: np.ndarray, pool: np.ndarray, thr: float, chunk: int = 2048) -> np.ndarray:
    counts = np.zeros(len(x), dtype=np.int64)
    for i in range(0, len(pool), chunk):
        counts += (x @ pool[i : i + chunk].T > thr).sum(axis=1)
    return counts


def neighbour_threshold(x: np.ndarray, pool: np.ndarray, quantile: float = 0.998,
                        sample: int = 1500, seed: int = 0) -> float:
    """Similarity above which two papers count as 'same research direction':
    a high quantile of random cross-pair similarities (adapts to the model)."""
    rng = np.random.default_rng(seed)
    a = x[rng.choice(len(x), size=min(sample, len(x)), replace=False)]
    b = pool[rng.choice(len(pool), size=min(sample, len(pool)), replace=False)]
    return float(np.quantile(a @ b.T, quantile))


def corpus_signals(settings: Settings, db: Database, model: str, week: str) -> dict[str, dict]:
    """Novelty and momentum for every embedded paper of `week`."""
    start, end = week_bounds(week)
    hist_start = start - timedelta(days=settings.novelty_lookback_days)
    base_start = start - timedelta(weeks=settings.momentum_baseline_weeks)
    uids, dates, vecs = db.load_vectors(model, hist_start.isoformat(), end.isoformat())
    if len(uids) == 0:
        return {}
    dates_arr = np.array(dates)
    in_week = (dates_arr >= start.isoformat()) & (dates_arr <= end.isoformat())
    x = vecs[in_week]
    week_uids = [u for u, keep in zip(uids, in_week) if keep]
    if len(x) == 0:
        return {}
    out: dict[str, dict] = {u: {"details": {}} for u in week_uids}

    # Novelty: closest paper published on an earlier day within the lookback.
    week_dates = dates_arr[in_week]
    novelty_raw = np.full(len(x), np.nan)
    history_size = int((dates_arr < start.isoformat()).sum())
    if history_size >= 200:
        for day in np.unique(week_dates):
            rows = week_dates == day
            day_start = (date.fromisoformat(day) - timedelta(days=settings.novelty_lookback_days)).isoformat()
            pool_mask = (dates_arr < day) & (dates_arr >= day_start)
            if pool_mask.any():
                novelty_raw[rows] = 1.0 - _max_sim(x[rows], vecs[pool_mask])
        novelty_raw = np.where(np.isnan(novelty_raw), np.nanmedian(novelty_raw), novelty_raw)
        novelty = percentile_rank(novelty_raw)
        for u, n_raw, n in zip(week_uids, novelty_raw, novelty):
            out[u]["novelty"] = round(float(n), 4)
            out[u]["details"]["novelty"] = f"closest earlier paper: similarity {1 - n_raw:.2f}"
    else:
        for u in week_uids:
            out[u]["novelty"] = 0.5
            out[u]["details"]["novelty"] = "not enough history yet (neutral)"

    # Momentum: neighbours this week vs. expected from the baseline weeks.
    base_mask = (dates_arr >= base_start.isoformat()) & (dates_arr < start.isoformat())
    base = vecs[base_mask]
    if len(base) >= 200 and len(x) >= 50:
        thr = neighbour_threshold(x, base)
        c_now = np.maximum(_count_above(x, x, thr) - 1, 0)  # minus self
        c_base = _count_above(x, base, thr)
        expected = c_base * (len(x) / len(base))
        growth = np.log((c_now + 1.0) / (expected + 1.0))
        # A direction needs some mass to count as "moving": damp tiny neighbourhoods.
        raw = growth * np.minimum(1.0, c_now / 3.0) + 0.05 * np.log1p(c_now)
        momentum = percentile_rank(raw)
        for u, m, cn, ex in zip(week_uids, momentum, c_now, expected):
            out[u]["momentum"] = round(float(m), 4)
            out[u]["details"]["momentum"] = (
                f"{int(cn)} close papers this week vs {ex:.1f} expected from the previous "
                f"{settings.momentum_baseline_weeks} weeks"
            )
            out[u]["neighbours_week"] = int(cn)
        log.info("week %s: momentum threshold %.3f, median neighbours %.0f", week, thr, np.median(c_now))
    else:
        for u in week_uids:
            out[u]["momentum"] = 0.5
            out[u]["details"]["momentum"] = "not enough baseline yet (neutral)"
    return out


def score_week(settings: Settings, db: Database, model: str | None, week: str) -> int:
    """(Re)score every paper of an ISO week and snapshot its top-N picks."""
    start, end = week_bounds(week)
    papers = db.papers_between(start.isoformat(), end.isoformat())
    if not papers:
        return 0
    corpus = corpus_signals(settings, db, model, week) if model else {}
    w = settings.weights
    rows = []
    for p in papers:
        meta = metadata_signals(p)
        corp = corpus.get(p["uid"], {})
        signals = {
            "venue": meta["venue"],
            "artifacts": meta["artifacts"],
            "breadth": meta["breadth"],
            "substance": meta["substance"],
            "team": meta["team"],
            "novelty": corp.get("novelty", 0.5),
            "momentum": corp.get("momentum", 0.5),
        }
        details = {**meta["details"], **corp.get("details", {})}
        details.setdefault("novelty", "no embedding (neutral)")
        details.setdefault("momentum", "no embedding (neutral)")
        tags = list(meta["tags"])
        if signals["momentum"] >= 0.95 and corp.get("neighbours_week", 0) >= 5:
            tags.append("hot direction")
        if signals["novelty"] >= 0.97:
            tags.append("unusual")
        total = sum(w[k] * signals[k] for k in w)
        rows.append({"uid": p["uid"], "total": round(total, 4), **signals,
                     "tags": tags, "details": details})
    db.save_scores(rows)
    picks = select_top(
        sorted(rows, key=lambda r: r["total"], reverse=True),
        {p["uid"]: p["primary_category"] for p in papers},
        n=max(settings.top_n, 50),
        max_per_category=settings.top_max_per_category,
        vectors=_vectors_for(db, model, week) if model else None,
    )
    db.save_weekly_picks(week, [(r["uid"], r["total"]) for r in picks])
    return len(rows)


def _vectors_for(db: Database, model: str, week: str) -> dict[str, np.ndarray]:
    start, end = week_bounds(week)
    uids, _, vecs = db.load_vectors(model, start.isoformat(), end.isoformat())
    return dict(zip(uids, vecs))


def select_top(
    ranked: list[dict],
    primary_of: dict[str, str],
    n: int,
    max_per_category: int,
    vectors: dict[str, np.ndarray] | None = None,
    duplicate_sim: float = 0.93,
) -> list[dict]:
    """Greedy top-N with diversity: at most `max_per_category` papers per primary
    category and no near-duplicates (cosine > duplicate_sim)."""
    picked: list[dict] = []
    per_cat: dict[str, int] = {}
    picked_vecs: list[np.ndarray] = []
    for r in ranked:
        cat = primary_of.get(r["uid"], "")
        if max_per_category and per_cat.get(cat, 0) >= max_per_category:
            continue
        v = vectors.get(r["uid"]) if vectors else None
        if v is not None and picked_vecs and float(np.max(np.vstack(picked_vecs) @ v)) > duplicate_sim:
            continue
        picked.append(r)
        per_cat[cat] = per_cat.get(cat, 0) + 1
        if v is not None:
            picked_vecs.append(v)
        if len(picked) >= n:
            break
    return picked
