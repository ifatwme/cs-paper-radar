import numpy as np
import pytest

from cs_paper_radar.db import build_fts_query
from cs_paper_radar.scoring import (
    artifacts_signal,
    breadth_signal,
    percentile_rank,
    select_top,
    substance_signal,
    team_signal,
    venue_signal,
)
from cs_paper_radar.text import phrases


@pytest.mark.parametrize(
    "comment,journal,doi,low,high",
    [
        ("Accepted at USENIX Security 2026", "", "", 1.0, 1.0),
        ("To appear in NeurIPS 2026 (spotlight)", "", "", 1.0, 1.0),
        ("Accepted to the NeurIPS 2026 Workshop on Safety", "", "", 0.6, 0.6),
        ("Under review at ICLR 2027", "", "", 0.1, 0.1),
        ("Submitted to CCS", "", "", 0.1, 0.1),
        ("", "IEEE Transactions on Information Forensics and Security 2026", "", 0.95, 0.95),
        ("CVPR 2026", "", "", 0.6, 0.6),
        ("12 pages, 4 figures", "", "", 0.0, 0.0),
        ("", "", "10.1145/123", 0.6, 0.6),
    ],
)
def test_venue_signal(comment, journal, doi, low, high):
    value, why = venue_signal(comment, journal, doi)
    assert low <= value <= high, why


def test_venue_needs_word_boundaries():
    # "ACL" must not match inside "ORACLE", "ATC" inside "BATCH".
    assert venue_signal("ORACLE BATCH processing notes", "", "")[0] == 0.0


def test_artifacts_signal():
    assert artifacts_signal("Code at https://github.com/foo/bar.", "")[0] == 1.0
    assert artifacts_signal("", "Project page and data: huggingface.co/datasets/x")[0] == 1.0
    assert artifacts_signal("The code is publicly available.", "")[0] == 0.8
    assert artifacts_signal("Code will be released upon acceptance.", "")[0] == 0.5
    assert artifacts_signal("We study things.", "")[0] == 0.0


def test_breadth_substance_team():
    assert breadth_signal(["cs.CR"])[0] == 0.0
    assert breadth_signal(["cs.CR", "cs.LG", "cs.NI", "cs.SE"])[0] == pytest.approx(1.0)
    assert substance_signal("12 pages, 5 figures", "", "x" * 800)[0] == 1.0
    assert substance_signal("3 pages", "", "x" * 800)[0] == pytest.approx(0.3)
    assert substance_signal("", "", "x" * 800)[0] == pytest.approx(0.4)
    assert team_signal(1)[0] < team_signal(5)[0] < team_signal(10)[0] == 1.0
    assert team_signal(40)[0] == 1.0


def test_percentile_rank_handles_ties():
    r = percentile_rank(np.array([3.0, 1.0, 2.0, 2.0]))
    assert r[0] == 1.0 and r[1] == 0.25 and r[2] == r[3] == pytest.approx(0.625)


def test_select_top_diversity():
    ranked = [{"uid": f"p{i}", "total": 1 - i / 100} for i in range(10)]
    primary = {f"p{i}": ("cs.LG" if i < 6 else "cs.CR") for i in range(10)}
    picks = select_top(ranked, primary, n=5, max_per_category=2)
    assert [p["uid"] for p in picks] == ["p0", "p1", "p6", "p7"]
    v = np.array([1.0, 0.0], dtype=np.float32)
    vectors = {"p0": v, "p1": v, "p2": np.array([0.0, 1.0], dtype=np.float32)}
    picks = select_top(ranked[:3], primary, n=3, max_per_category=0, vectors=vectors)
    assert [p["uid"] for p in picks] == ["p0", "p2"]  # p1 is a duplicate of p0


def test_fts_query_is_sanitised():
    assert build_fts_query('network "zero trust" AND security') == '"network" "zero trust" "security"'
    assert build_fts_query("a OR b", mode="or") == '"a" OR "b"'
    assert build_fts_query('"; DROP TABLE papers; --') == '"DROP" "TABLE" "papers"'


def test_phrases_skip_stopwords():
    ph = phrases("We propose a novel intrusion detection system for network traffic.")
    assert "intrusion detection" in ph and "network traffic" in ph
    assert not any(p.startswith("propose") or p == "novel" for p in ph)


def test_keyphrases_keep_phrases_and_acronyms():
    from cs_paper_radar.text import keyphrases

    kp = keyphrases("We study LLMs for DDoS detection in 6G networks. The approach is practical and accurate.")
    assert "ddos detection" in kp and "LLM" in kp and "DDoS" in kp and "6G" in kp
    assert "practical" not in kp and "accurate" not in kp


def test_home_category_for_cross_listed_papers():
    from cs_paper_radar.taxonomy import area_of, home_category

    assert home_category("cs.CR", ["cs.CR", "cs.LG"]) == "cs.CR"
    assert home_category("math.OC", ["math.OC", "eess.SY", "cs.LG"]) == "eess.SY"
    assert area_of(home_category("stat.ML", ["stat.ML", "cs.LG"])) == "AI & Machine Learning"
