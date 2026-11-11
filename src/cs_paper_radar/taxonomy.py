"""arXiv category taxonomy for computer science and computer engineering."""

from __future__ import annotations

CATEGORY_NAMES: dict[str, str] = {
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering, Finance, and Science",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation and Language",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CY": "Computers and Society",
    "cs.DB": "Databases",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DS": "Data Structures and Algorithms",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages and Automata Theory",
    "cs.GL": "General Literature",
    "cs.GR": "Graphics",
    "cs.GT": "Computer Science and Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic in Computer Science",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.MS": "Mathematical Software",
    "cs.NA": "Numerical Analysis",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.NI": "Networking and Internet Architecture",
    "cs.OH": "Other Computer Science",
    "cs.OS": "Operating Systems",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SC": "Symbolic Computation",
    "cs.SD": "Sound",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social and Information Networks",
    "cs.SY": "Systems and Control",
    "eess.AS": "Audio and Speech Processing",
    "eess.IV": "Image and Video Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems and Control",
}

# Coarse areas used to group the daily listing and the category page.
AREAS: dict[str, list[str]] = {
    "AI & Machine Learning": ["cs.AI", "cs.LG", "cs.NE", "cs.MA"],
    "Language, Vision & Media": ["cs.CL", "cs.CV", "cs.IR", "cs.MM", "cs.GR", "cs.SD",
                                 "eess.AS", "eess.IV"],
    "Security & Privacy": ["cs.CR"],
    "Systems & Networking": ["cs.DC", "cs.NI", "cs.OS", "cs.PF", "cs.DB", "cs.AR", "cs.ET"],
    "Software & Programming": ["cs.SE", "cs.PL", "cs.LO", "cs.FL", "cs.MS"],
    "Theory & Algorithms": ["cs.DS", "cs.CC", "cs.CG", "cs.DM", "cs.GT", "cs.IT", "cs.NA",
                            "cs.SC"],
    "Robotics, Control & Signals": ["cs.RO", "cs.SY", "eess.SY", "eess.SP"],
    "People & Society": ["cs.HC", "cs.CY", "cs.SI", "cs.DL", "cs.CE", "cs.GL", "cs.OH"],
}

AREA_OF: dict[str, str] = {cat: area for area, cats in AREAS.items() for cat in cats}


def category_name(code: str) -> str:
    return CATEGORY_NAMES.get(code, code)


def area_of(code: str) -> str:
    if code in AREA_OF:
        return AREA_OF[code]
    return "Other"


def home_category(primary: str, categories: list[str]) -> str:
    """The paper's CS/CE home: its primary category, or - for papers whose primary
    is outside CS/EESS (math.OC, stat.ML, q-bio...) - the first CS/EESS cross-list."""
    if primary in CATEGORY_NAMES:
        return primary
    for c in categories or []:
        if c in CATEGORY_NAMES:
            return c
    return primary


AREA_SLUGS: dict[str, str] = {
    "ai": "AI & Machine Learning",
    "media": "Language, Vision & Media",
    "security": "Security & Privacy",
    "systems": "Systems & Networking",
    "software": "Software & Programming",
    "theory": "Theory & Algorithms",
    "robotics": "Robotics, Control & Signals",
    "society": "People & Society",
}
SLUG_OF_AREA = {v: k for k, v in AREA_SLUGS.items()}


def in_scope(code: str, patterns: list[str]) -> bool:
    """`patterns` like ["cs.*", "eess.SY"] -> does `code` match any of them?"""
    for pattern in patterns:
        if pattern.endswith(".*") and code.startswith(pattern[:-1]):
            return True
        if code == pattern:
            return True
    return False
