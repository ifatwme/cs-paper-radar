"""Topic intelligence: hybrid search, "what's new", frontier, emerging terms,
sub-directions and trend for any free-text topic (e.g. "network security").
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter
from datetime import date, timedelta

import numpy as np

from .config import Settings
from .db import Database
from .embed import Embedder, embedder_name, get_embedder
from .scoring import select_top
from .taxonomy import category_name
from .text import phrase_counts, phrases

log = logging.getLogger(__name__)

# Curated starting points shown on the home page; any free text works too.
PRESET_TOPICS = [
    "network security",
    "intrusion detection",
    "LLM agents",
    "LLM safety and jailbreaks",
    "retrieval-augmented generation",
    "diffusion models",
    "vision-language models",
    "reinforcement learning from human feedback",
    "federated learning",
    "differential privacy",
    "post-quantum cryptography",
    "software vulnerability detection",
    "program synthesis and code generation",
    "compilers",
    "distributed systems",
    "LLM inference serving",
    "edge computing",
    "6G wireless networks",
    "computer architecture accelerators",
    "quantum computing",
    "robot learning",
    "autonomous driving",
    "graph neural networks",
    "recommender systems",
    "human-computer interaction",
    "formal verification",
    "approximation algorithms",
    "speech recognition",
]


class VectorIndex:
    """In-memory matrix of recent embeddings, refreshed when the data changes."""

    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.model = embedder_name(settings)
        self._lock = threading.Lock()
        self._version = -1
        self._checked = 0.0
        self.uids: list[str] = []
        self.dates = np.array([], dtype="<U10")
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.pos: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def refresh(self, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self._checked < 60:
            return
        with self._lock:
            self._checked = now
            version = self.db.data_version()
            if version == self._version and not force:
                return
            start = (date.today() - timedelta(days=self.settings.vector_cache_days)).isoformat()
            uids, dates, matrix = self.db.load_vectors(self.model, start)
            self.uids, self.dates, self.matrix = uids, np.array(dates), matrix
            self.pos = {u: i for i, u in enumerate(uids)}
            self._version = version
            log.info("vector index: %d vectors (data version %d)", len(uids), version)

    def window(self, start: str, end: str) -> np.ndarray:
        self.refresh()
        if not len(self.uids):
            return np.zeros(0, dtype=bool)
        return (self.dates >= start) & (self.dates <= end)

    def vector(self, uid: str) -> np.ndarray | None:
        self.refresh()
        i = self.pos.get(uid)
        return None if i is None else self.matrix[i]

    def similar(self, uid: str, k: int = 8) -> list[tuple[str, float]]:
        v = self.vector(uid)
        if v is None:
            return []
        sims = self.matrix @ v
        top = np.argsort(-sims)[: k + 1]
        return [(self.uids[i], float(sims[i])) for i in top if self.uids[i] != uid][:k]


class TopicEngine:
    def __init__(self, settings: Settings, db: Database, index: VectorIndex):
        self.settings = settings
        self.db = db
        self.index = index
        self._embedder: Embedder | None = None

    @property
    def embedder(self) -> Embedder | None:
        if self._embedder is None and self.index.enabled:
            self._embedder = get_embedder(self.settings)
        return self._embedder

    # ------------------------------------------------------------- search
    def search(self, query: str, start: str, end: str, limit: int = 3000) -> list[dict]:
        """Hybrid keyword + semantic retrieval.

        Returns [{uid, relevance (0..1), keyword (bool), semantic (float|None)}]
        ordered by reciprocal-rank fusion. A paper belongs to the topic if it
        matches all query terms (BM25) or is semantically close to the query.
        """
        # Terms close together first ("network security" != "neural network ... security");
        # fall back to all-terms-anywhere when that is too strict.
        kw = self.db.fts_search(query, start, end, limit=limit, mode="near")
        if len(kw) < 15:
            kw = self.db.fts_search(query, start, end, limit=limit, mode="and")
        kw_rank = {uid: i for i, (uid, _) in enumerate(kw)}

        sem_rank: dict[str, int] = {}
        sem_sim: dict[str, float] = {}
        if self.embedder is not None:
            mask = self.index.window(start, end)
            if mask.any():
                idx = np.nonzero(mask)[0]
                q = self.embedder.embed_query(query)
                sims = self.index.matrix[idx] @ q
                order = np.argsort(-sims)
                cutoff = self._semantic_cutoff(sims, kw_count=len(kw))
                for r, j in enumerate(order[:limit]):
                    if sims[j] < cutoff:
                        break
                    uid = self.index.uids[idx[j]]
                    sem_rank[uid] = r
                    sem_sim[uid] = float(sims[j])

        fused: dict[str, float] = {}
        for uid, r in kw_rank.items():
            fused[uid] = fused.get(uid, 0.0) + 1.0 / (60 + r)
        for uid, r in sem_rank.items():
            fused[uid] = fused.get(uid, 0.0) + 1.0 / (60 + r)
        if not fused:
            return []
        best = max(fused.values())
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [
            {
                "uid": uid,
                "relevance": round(score / best, 4),
                "keyword": uid in kw_rank,
                "semantic": round(sem_sim[uid], 4) if uid in sem_sim else None,
            }
            for uid, score in ranked
        ]

    def _semantic_cutoff(self, sims: np.ndarray, kw_count: int) -> float:
        """Adaptive similarity cutoff: well above the bulk of the corpus and within a
        band of the best match, so narrow and broad topics both get sensible sets."""
        if len(sims) == 0:
            return 1.0
        top = float(sims.max())
        mean, std = float(sims.mean()), float(sims.std() or 1e-6)
        z_cut = mean + 3.0 * std
        band_cut = top - (0.12 if (self.index.model or "").startswith("BAAI/bge") else 0.25)
        return max(z_cut, band_cut)

    # ------------------------------------------------------------- report
    def report(self, query: str, days: int = 30, end: str | None = None,
               new_days: int = 7, n: int | None = None, trend_weeks: int = 12) -> dict:
        n = n or self.settings.top_n
        latest = end or self.db.date_range()[1] or date.today().isoformat()
        end_d = date.fromisoformat(latest)
        win_start = (end_d - timedelta(days=days - 1)).isoformat()
        new_start = (end_d - timedelta(days=new_days - 1)).isoformat()
        trend_start_d = end_d - timedelta(weeks=trend_weeks) + timedelta(days=1)
        long_start = min(trend_start_d, end_d - timedelta(days=4 * days - 1)).isoformat()

        hits = self.search(query, long_start, latest)
        papers = self.db.get_papers([h["uid"] for h in hits])
        for h in hits:
            h["paper"] = papers.get(h["uid"])
        hits = [h for h in hits if h["paper"]]

        window = [h for h in hits if h["paper"]["pub_date"] >= win_start]
        prior = [h for h in hits if h["paper"]["pub_date"] < win_start]

        new = sorted(
            (h for h in window if h["paper"]["pub_date"] >= new_start),
            key=lambda h: (h["paper"]["published"], h["relevance"]),
            reverse=True,
        )
        # Frontier: best composite score among matching papers, nudged by relevance.
        relevant = list(window)
        for h in relevant:
            total = h["paper"].get("total") or 0.0
            h["frontier"] = round(total * (0.8 + 0.2 * h["relevance"]), 4)
        ranked = sorted(relevant, key=lambda h: h["frontier"], reverse=True)
        vectors = {h["uid"]: v for h in ranked[:400] if (v := self.index.vector(h["uid"])) is not None}
        frontier = select_top(
            [{"uid": h["uid"], **h} for h in ranked], {}, n=n, max_per_category=0, vectors=vectors
        )

        return {
            "query": query,
            "end": latest,
            "window_start": win_start,
            "days": days,
            "new_days": new_days,
            "matched_window": len(window),
            "matched_new": len(new),
            "new": new[:60],
            "frontier": frontier,
            "emerging": self.emerging_terms(window, prior),
            "subtopics": self.subtopics(window),
            "categories": self._category_mix(window),
            "trend": self._trend(hits, trend_start_d, end_d),
            "semantic": self.index.enabled,
        }

    # -------------------------------------------------------- components
    @staticmethod
    def _doc(h: dict) -> str:
        return f"{h['paper']['title']}. {h['paper']['abstract']}"

    def emerging_terms(self, window: list[dict], prior: list[dict], k: int = 15) -> dict:
        """Weighted log-odds with an informative Dirichlet prior (Monroe et al. 2008):
        phrases over-represented in the recent window relative to the topic's past."""
        if len(window) < 5:
            return {"mode": "none", "terms": []}
        recent = phrase_counts(self._doc(h) for h in window)
        if len(prior) >= 20:
            past = phrase_counts(self._doc(h) for h in prior)
            mode = "emerging"
        else:
            past = Counter()
            mode = "characteristic"
        n1, n2 = len(window), max(len(prior), 1)
        out = []
        for term, c1 in recent.items():
            if c1 < 3:
                continue
            c2 = past.get(term, 0)
            if mode == "emerging":
                # z-score of the log odds ratio of a 2x2 table (document frequencies).
                lor = math.log((c1 + 0.5) / (n1 - c1 + 0.5)) - math.log((c2 + 0.5) / (n2 - c2 + 0.5))
                var = 1 / (c1 + 0.5) + 1 / (n1 - c1 + 0.5) + 1 / (c2 + 0.5) + 1 / (n2 - c2 + 0.5)
                z = lor / math.sqrt(var)
            else:
                z = c1 / math.sqrt(n1)
            out.append({"term": term, "recent": c1, "past": c2, "z": round(z, 2),
                        "recent_share": round(c1 / n1, 3), "past_share": round(c2 / n2, 3)})
        out.sort(key=lambda t: t["z"], reverse=True)
        # Drop terms contained in a better-ranked longer/shorter term.
        picked: list[dict] = []
        for t in out:
            if any(t["term"] in p["term"] or p["term"] in t["term"] for p in picked):
                continue
            picked.append(t)
            if len(picked) >= k:
                break
        return {"mode": mode, "terms": picked}

    def subtopics(self, window: list[dict], max_k: int = 6) -> list[dict]:
        """Cluster the topic's recent papers into research directions."""
        rows = [(h, v) for h in window if (v := self.index.vector(h["uid"])) is not None]
        if len(rows) < 20:
            return []
        hits, vecs = zip(*rows)
        x = np.vstack(vecs)
        k = int(min(max_k, max(2, len(x) // 25)))
        labels = kmeans(x, k)
        docs_per_cluster = {c: [self._doc(h) for h, lab in zip(hits, labels) if lab == c] for c in range(k)}
        df_all = phrase_counts(self._doc(h) for h in hits)
        clusters = []
        for c in range(k):
            members = [h for h, lab in zip(hits, labels) if lab == c]
            if not members:
                continue
            df_c = phrase_counts(docs_per_cluster[c])
            size = len(members)
            # c-TF-IDF style: frequent inside, rare outside.
            scored = []
            for term, cnt in df_c.items():
                if cnt < max(2, size * 0.08):
                    continue
                outside = df_all[term] - cnt
                scored.append((cnt / size * math.log(1 + len(x) / (1 + outside + cnt)), term))
            scored.sort(reverse=True)
            label_terms: list[str] = []
            for _, term in scored:
                if any(term in t or t in term for t in label_terms):
                    continue
                label_terms.append(term)
                if len(label_terms) == 4:
                    break
            best = sorted(members, key=lambda h: h["paper"].get("total") or 0, reverse=True)[:3]
            momentum = np.mean([h["paper"].get("momentum") or 0.5 for h in members])
            clusters.append({
                "label": ", ".join(label_terms) or "misc",
                "size": size,
                "momentum": round(float(momentum), 3),
                "papers": [b["paper"] for b in best],
            })
        clusters.sort(key=lambda c: c["size"], reverse=True)
        return clusters

    @staticmethod
    def _category_mix(window: list[dict]) -> list[dict]:
        counts = Counter(h["paper"]["primary_category"] for h in window)
        total = sum(counts.values()) or 1
        return [{"category": c, "name": category_name(c), "n": n, "share": round(n / total, 3)}
                for c, n in counts.most_common(10)]

    def _trend(self, hits: list[dict], start: date, end: date) -> list[dict]:
        """Topic papers per ISO week, and their share of all papers that week."""
        totals = {r["pub_date"]: r["n"] for r in self.db.daily_counts(start.isoformat(), end.isoformat())}
        weeks: dict[str, dict] = {}
        d = start
        while d <= end:
            wk = d - timedelta(days=d.weekday())
            key = wk.isoformat()
            weeks.setdefault(key, {"week_start": key, "n": 0, "total": 0})
            weeks[key]["total"] += totals.get(d.isoformat(), 0)
            d += timedelta(days=1)
        for h in hits:
            pd = date.fromisoformat(h["paper"]["pub_date"])
            if start <= pd <= end:
                key = (pd - timedelta(days=pd.weekday())).isoformat()
                weeks[key]["n"] += 1
        out = list(weeks.values())
        for w in out:
            w["share"] = round(w["n"] / w["total"], 4) if w["total"] else 0.0
        return out


def kmeans(x: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Spherical k-means with k-means++ seeding (x rows are L2-normalised)."""
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = 1.0 - np.max(x @ np.vstack(centers).T, axis=1)
        d = np.clip(d, 0, None) ** 2
        p = d / d.sum() if d.sum() > 0 else None
        centers.append(x[rng.choice(len(x), p=p)])
    c = np.vstack(centers)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        new = np.argmax(x @ c.T, axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            members = x[labels == j]
            if len(members):
                m = members.mean(axis=0)
                c[j] = m / (np.linalg.norm(m) or 1.0)
    return labels


__all__ = ["PRESET_TOPICS", "TopicEngine", "VectorIndex", "kmeans", "phrases"]
