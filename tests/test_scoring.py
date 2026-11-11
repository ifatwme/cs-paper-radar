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
from cs_paper_radar.venues import VenueRanker

RANKER = VenueRanker([("Usenix Security Symposium", "USENIX-Security", "A*"),
                      ("Neural Information Processing Systems", "NeurIPS", "A*"),
                      ("International Conference on Learning Representations", "ICLR", "A*"),
                      ("Computer Vision and Pattern Recognition", "CVPR", "A*"),
                      ("IEEE International Conference on Communications", "ICC", "B")])


@pytest.mark.parametrize(
    "comment,journal,s2,low,high",
    [
        ("To appear in NeurIPS 2026 (spotlight)", "", "", 0.9, 0.9),       # self-reported A*: 0.9
        ("Accepted to the NeurIPS 2026 Workshop on Safety", "", "", 0.5, 0.6),
        ("Under review at ICLR 2027", "", "", 0.05, 0.05),
        ("Submitted to CCS", "", "", 0.05, 0.05),
        ("", "IEEE Transactions on Information Forensics and Security 2026", "", 0.85, 0.85),
        ("CVPR 2026", "", "", 0.5, 0.5),                                    # mentioned, not stated accepted
        ("12 pages, 4 figures", "", "", 0.0, 0.0),
        ("", "", "Neural Information Processing Systems", 1.0, 1.0),       # verified A*
        ("", "", "IEEE International Conference on Communications", 0.65, 0.65),
        ("", "", "Proceedings of the AAAI Symposium Series", 0.3, 0.4),
    ],
)
def test_venue_signal(comment, journal, s2, low, high):
    value, why = venue_signal(comment, journal, "", s2, "", RANKER)
    assert low <= value <= high, why


def test_venue_needs_word_boundaries():
    # "ACL" must not match inside "ORACLE", "ATC" inside "BATCH".
    assert venue_signal("ORACLE BATCH processing notes", "", "")[0] == 0.0


def test_github_repo_extraction():
    from cs_paper_radar.enrich import extract_github_repo

    assert extract_github_repo("Code: https://github.com/foo/bar-baz.") == "foo/bar-baz"
    assert extract_github_repo("see github.com/foo/bar.git)") == "foo/bar"
    assert extract_github_repo("https://github.com/features/copilot") is None
    assert extract_github_repo("no link") is None


def test_cohort_percentile_zero_means_no_evidence():
    from cs_paper_radar.scoring import cohort_percentile

    r = cohort_percentile(np.array([0, 0, 0, 1, 5]), zero_is_none=True)
    assert list(r[:3]) == [0, 0, 0] and 0 < r[3] < r[4] == 1.0


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


def test_venue_ranker_titles_and_false_friends():
    r = VenueRanker([("Advances in Neural Information Processing Systems (was NIPS)", "NeurIPS", "A*"),
                     ("International Conference on Neural Information Processing", "ICONIP", "B"),
                     ("Usenix Network and Distributed System Security Symposium", "NDSS", "A*"),
                     ("Association for Computational Linguistics", "ACL", "A*")])
    assert r.tier("Neural Information Processing Systems")[0] == "A*"
    assert r.tier("International Conference on Neural Information Processing")[0] == "B"
    assert r.tier("Network and Distributed System Security Symposium")[0] == "A*"
    assert r.tier("Annual Meeting of the Association for Computational Linguistics")[0] == "A*"
    assert r.tier("SN Computer Science")[1] is None          # not "Science"
    assert r.tier("Nature Machine Intelligence")[1] == 1.0


def test_area_quotas_proportional_with_floor():
    from cs_paper_radar.scoring import area_quotas

    assert area_quotas({}, 20) == {}
    q = area_quotas({"ai": 700, "media": 500, "theory": 120, "security": 80, "tiny": 5}, 20)
    assert sum(q.values()) == 20 and "tiny" not in q
    assert q["ai"] > q["media"] > q["theory"] >= q["security"] >= 1


def test_select_top_quotas_release_unfilled_slots():
    ranked = [{"uid": f"a{i}", "total": 0.9 - i / 100} for i in range(6)] + [{"uid": "b0", "total": 0.1}]
    group = {**{f"a{i}": "ai" for i in range(6)}, "b0": "theory"}
    picks = select_top(ranked, {}, 4, 0, group_of=group, quotas={"ai": 2, "theory": 2})
    # theory has one paper only: its spare slot goes to the next best AI paper
    assert [p["uid"] for p in picks] == ["a0", "a1", "a2", "b0"]
