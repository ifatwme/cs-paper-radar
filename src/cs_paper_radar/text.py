"""Light-weight text utilities: tokenising and key-phrase extraction."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

STOPWORDS = frozenset(
    """a about above across after again against all almost along also although always am among
    an and another any are around as at be became because been before being below between both
    but by can cannot could did do does doing done down during each either else enough etc even
    ever every few for from further furthermore had has have having he hence her here hers how
    however i if in into is it its itself just less like made make makes many may might more
    moreover most much must my namely near nearly neither no nor not now of off often on once one
    only onto or other others otherwise our ours out over own per perhaps please quite rather
    really same several shall she should show shows shown since so some still such than that the
    their theirs them then there thereby therefore these they this those though through thus to
    together too toward towards under until up upon us use used uses using various very via was
    we well were what whatever when where whereas whether which while who whole whom whose why
    will with within without would yet you your yours
    paper papers propose proposed proposes approach approaches method methods methodology result
    results work works study studies present presents presented introduce introduces introduced
    novel new based existing state-of-the-art art sota recent recently first second third
    demonstrate demonstrates demonstrated achieve achieves achieved achieving outperform
    outperforms outperforming significantly significant extensive experiments experiment
    experimental evaluation evaluate evaluated evaluations performance performances task tasks
    problem problems framework frameworks model models data dataset datasets show however
    effective effectively efficient efficiently improve improves improved improvement
    improvements address addresses provide provides provided enable enables enabling different
    including include includes large small high low key main existing across code available
    github https http com www org two three four five set sets well-known way ways large-scale
    real-world given due able allows allow allowing furthermore specifically particular
    particularly respectively previous prior related further additionally overall finally
    remains remain challenging challenge challenges important crucial critical leverage
    leverages leveraging study investigated investigate investigates analysis analyze
    analyzes analyse understanding novel-ly comprehensive superior substantial substantially
    consistently consistent strong stronger robust compared compare comparing comparison
    potential promising various multiple single new lead leads leading
    times faster fewer increasingly alone highest average best better higher lower
    percentage points point jointly final
    """.split()
)

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]|[A-Za-z]")
_LATEX = re.compile(r"\$[^$]*\$|\\[a-zA-Z]+(\{[^}]*\})?")


def tokens(text: str) -> list[str]:
    text = _LATEX.sub(" ", text)
    return [t.lower() for t in _TOKEN.findall(text)]


def phrases(text: str, max_n: int = 3) -> list[str]:
    """Candidate key phrases: 1-3 grams that neither start nor end with a stopword,
    with no stopword inside, no pure numbers and no 1-2 letter words."""
    out: list[str] = []
    for sentence in _SENT.split(text):
        toks = tokens(sentence)
        run: list[str] = []
        for w in toks + [""]:  # sentinel flushes the last run
            if w and len(w) >= 3 and w not in STOPWORDS:
                run.append(w)
                continue
            # emit every n-gram inside a run of content words
            for i in range(len(run)):
                if len(run[i]) >= 4:
                    out.append(run[i])
                for n in range(2, max_n + 1):
                    if i + n <= len(run):
                        out.append(" ".join(run[i : i + n]))
            run = []
    return out


_SENT = re.compile(r"[.;:!?()\[\],]")


def phrase_counts(docs: Iterable[str], max_n: int = 3) -> Counter:
    """Document frequency of each phrase (counted once per document)."""
    df: Counter = Counter()
    for doc in docs:
        df.update(set(phrases(doc, max_n)))
    return df


_ACRONYM = re.compile(r"\b(?=[A-Za-z0-9\-]*[A-Z][A-Za-z0-9\-]*[A-Z0-9])[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]\b|\b\d+G\b")
_GENERIC = frozenset("""ai ml dl sota api gpu gpus cpu cpus llm-based ii iii iv""".split())


def keyphrases(text: str, max_n: int = 3) -> list[str]:
    """Informative terms for summaries: 2-3 word phrases plus acronyms and
    alphanumeric names (LLM, DDoS, IoT, 6G, GPT-4o). Plain single words such as
    "practical" or "accurate" are too generic to describe a research direction."""
    out = [p for p in phrases(text, max_n) if " " in p]
    for m in _ACRONYM.findall(_LATEX.sub(" ", text)):
        low = m.lower()
        if len(m) >= 2 and low not in STOPWORDS and low not in _GENERIC and not m.isdigit():
            if m.endswith("s") and m[:-1].isupper() and len(m) > 2:
                m = m[:-1]  # LLMs -> LLM
            out.append(m)
    return out


def keyphrase_counts(docs: Iterable[str], max_n: int = 3) -> Counter:
    df: Counter = Counter()
    for doc in docs:
        df.update(set(keyphrases(doc, max_n)))
    return df
