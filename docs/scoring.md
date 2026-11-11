# Scoring (v2)

Code: `src/cs_paper_radar/scoring.py`, external data: `src/cs_paper_radar/enrich/`,
venue tiers: `src/cs_paper_radar/venues.py`.

## Goals

1. **Reliable** — prefer measured evidence (citations, adoption, verified venues) over guesses.
2. **Honest** — every number on the site traces back to a raw value and its source; signals that
   could not be measured are shown as "not measured", never silently treated as zero or average.
3. **Fair** — compare a paper only with papers of the same age and research area; leave out
   prestige proxies.

## Signals

| Signal | Weight | Source | Raw value | Normalisation |
|---|---|---|---|---|
| citations | 20% | Semantic Scholar Graph API (batch, 500 papers/request) | citations + 2 × influential citations | cohort percentile, 0 citations → 0 |
| community | 10% | Hugging Face `api/daily_papers` (last 30 days) | upvotes | cohort percentile, 0 → 0 |
| adoption | 10% | GitHub REST `repos/{owner}/{repo}` | stars of the linked repository | cohort percentile, 0 → 0 |
| venue | 15% | Semantic Scholar venue, arXiv journal ref, arXiv comment; CORE/ICORE ranking | tier | absolute (see below) |
| artifacts | 10% | abstract + comment | code/data/model links or statements | absolute |
| novelty | 15% | local embeddings | 1 − max cosine similarity to the previous 90 days | cohort percentile |
| momentum | 15% | local embeddings | neighbourhood growth vs. previous 4 weeks | cohort percentile |
| substance | 5% | comment | pages, figures, journal version | absolute |

`total = Σ wᵢ·sᵢ / Σ wᵢ` over the signals measured for that paper. `coverage = Σ wᵢ` (measured
share of the weight) is stored and shown next to every score.

### Cohorts

A cohort is (research area, ISO week). Areas: AI & ML, Language/Vision/Media, Security & Privacy,
Systems & Networking, Software & Programming, Theory & Algorithms, Robotics/Control/Signals,
People & Society (`taxonomy.AREAS`). A paper's area comes from its CS/EESS home category (its primary
category, or its first CS/EESS cross-list when the primary is e.g. `math.OC`). Areas with fewer than
`COHORT_MIN_SIZE` (40) papers in a week are pooled.

Why: in a typical week a security paper with 3 citations can be in the top 5% of its cohort while an
LLM paper needs 15. Raw counts would rank fields, not papers. Comparing within the same week removes
the age advantage of older papers.

### Missing is not zero

For count signals, a cohort uses the signal only if at least `SIGNAL_MIN_COVERAGE` (3%) and at least
3 of its papers have a non-zero value. Otherwise the signal is "not measured" for the whole cohort
(for example Hugging Face upvotes for theory papers, or citations for a week-old cohort) and the
remaining weights are renormalised. Inside a measured cohort, zero means "no evidence yet" and scores 0.

Novelty/momentum are not measured until enough history exists (≥ 200 earlier papers / a 4-week baseline).

### Venue tiers

| Evidence | Value |
|---|---|
| Venue from Semantic Scholar or the arXiv journal ref, ranked by CORE | A* 1.0 · A 0.85 · B 0.65 · C 0.45 |
| Journal families (IEEE/ACM Transactions, JMLR/TMLR, Nature/Science/PNAS, CACM, Proc. IEEE) | 0.85–1.0 |
| Other journals / unranked published venues | 0.55–0.6 |
| Workshop, companion, extended abstract, "Findings", symposium series | × 0.6 |
| "Accepted at / to appear in X" in the arXiv comment (self-reported) | 0.9 × tier |
| Venue named but acceptance not stated | 0.5 × tier |
| "Under review", "submitted to" | 0.05 |
| DOI only | 0.5 |

CORE rankings come from the free CSV export of https://portal.core.edu.au/conf-ranks/ (ICORE2026,
fallback CORE2023), cached in `DATA_DIR/core_ranks.csv` and refreshed every 30 days.

### Not in the score (shown for context)

* **Author h-index** (Semantic Scholar): rewards established groups regardless of the paper (Matthew
  effect).
* **Team size** and **number of cross-listed categories**: not related to quality and easy to game.

## Freshness

| Data | Refresh |
|---|---|
| Semantic Scholar | papers ≤ 14 days old: daily · ≤ 120 days: weekly · older: monthly |
| Hugging Face | the last 30 days of daily papers on every run |
| GitHub stars | repos of papers ≤ 30 days old: every 2 days · ≤ 180 days: every 14 days |
| Scores and weekly picks | the last 13 weeks (`RESCORE_WEEKS`) on every update |

Daily snapshots of citations, upvotes and stars are kept in `metric_history` (papers ≤ 180 days old).
Rankings of recent weeks are therefore provisional and settle as evidence accumulates.

## Limits

* Citations take weeks to appear; the current week is ranked mainly on venue, artifacts, novelty,
  momentum and (for ML) upvotes.
* Hugging Face upvotes reflect one community (ML); cohort normalisation keeps this from leaking into
  other areas, but within ML it still favours papers that are promoted there.
* GitHub stars can be bought or inflated by social media; they are one 10% signal and are
  cohort-normalised.
* Rate limits without keys: Semantic Scholar shared pool (~1 request/s), GitHub 60 requests/hour.
  Set `S2_API_KEY` / `GITHUB_TOKEN` to raise them.
