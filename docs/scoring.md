# Scoring details

Code: `src/cs_paper_radar/scoring.py`. Every paper of an ISO week is (re)scored whenever the week is
touched by an update, so a paper's score can change while its week is open (new papers change the
percentiles; authors add acceptance notes in new versions).

## Metadata signals

**Venue** (comment + journal ref + DOI)

| Evidence | Value |
|---|---|
| journal ref present | 0.95 (0.7 for workshop) |
| accepted/to appear/published + known venue | 1.0 (0.6 for workshop) |
| "under review", "submitted to", "preprint" | 0.1 |
| accepted wording without a known venue | 0.75 (0.5 for workshop) |
| known venue mentioned, no acceptance wording | 0.6 (0.4 for workshop) |
| DOI only | 0.6 |

The venue list covers the main ML/AI, vision, NLP/IR, security & cryptography, systems/networking/
architecture/databases, software engineering & PL, theory, HCI, robotics/control and signal-processing
venues and journals. Matches use word boundaries, so "ACL" does not match "ORACLE".

**Code/data**: 1.0 for a GitHub/GitLab/Hugging Face/Zenodo/Kaggle/OSF/Figshare/Code Ocean link; 0.8 for
an explicit availability statement; 0.6 for any other link; 0.5 for "will be released".

**Breadth**: `0.7 · min(1, (categories − 1) / 3) + 0.3 · [categories span more than one area]`.

**Substance**: 1.0 for ≥ 8 pages or a journal version, 0.65 for 4–7, 0.3 for < 4, 0.4 if unknown;
+0.1 for ≥ 3 figures; ×0.7 for very short abstracts.

**Team**: `min(1, ln(1 + authors) / ln 11)`.

## Corpus signals (embeddings)

Vectors: `sentence-transformers/all-MiniLM-L6-v2` (384-d) on title + abstract, computed locally.

**Novelty**: for each paper, `1 − max cosine similarity` to any paper submitted in the previous 90 days
(strictly earlier days), converted to a percentile within the week. Neutral (0.5) until at least 200
historical papers exist.

**Momentum**: a similarity threshold `t` is set at the 99.8th percentile of random cross-pair
similarities (adapts to the model). For each paper, count same-week papers above `t` (`c_now`) and
baseline papers from the previous 4 weeks above `t`, scaled to one week (`expected`). Raw momentum is
`ln((c_now + 1) / (expected + 1)) · min(1, c_now / 3) + 0.05 · ln(1 + c_now)` — growth of the
neighbourhood, damped for tiny neighbourhoods — converted to a percentile within the week.

## Total and selection

`total = Σ weight_s · signal_s` (weights sum to 1). The weekly list is the greedy top N by total with
at most 4 papers per primary category and no pair with cosine similarity above 0.93. The top 50 are
stored in `weekly_picks`; the page and API can return up to 200 by recomputing from scores.

## Tags

`accepted`, `code/data`, `hot direction` (momentum ≥ 0.95 and ≥ 5 close papers), `unusual`
(novelty ≥ 0.97), and text-pattern tags `survey`, `benchmark`, `dataset`, `position`, `theory`.

## Known limits

The score estimates attention-worthiness from observable signals; it does not read the paper. Venue
detection depends on what authors write in the arXiv comment. Planned additional signals: citation
velocity (Semantic Scholar), community votes (Hugging Face Papers).
