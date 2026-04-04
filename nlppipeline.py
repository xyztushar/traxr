"""
TRAXR — NLP Pipeline
PDF text extraction, skill identification, evidence detection, and signal analysis.
Pure Python — no heavy NLP dependencies.
"""

from __future__ import annotations

import re
from io import BytesIO
from dataclasses import dataclass, field

from pypdf import PdfReader
from skilltaxonomy import SKILL_TAXONOMY, SKILL_ALIASES, resolve_alias


# ── Data Structures ──────────────────────────────────────

@dataclass
class ProfileSignals:
    claimed_skills: list[str] = field(default_factory=list)
    supported_skills: list[str] = field(default_factory=list)
    projects: list[dict] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    experience_months: int = 0
    has_internship: bool = False

    def to_dict(self) -> dict:
        return {
            "claimed_skills": self.claimed_skills,
            "supported_skills": self.supported_skills,
            "projects": self.projects,
            "certifications": self.certifications,
            "links": self.links,
            "education": self.education,
            "experience_months": self.experience_months,
            "has_internship": self.has_internship,
        }


# ── PDF Extraction ───────────────────────────────────────

def extract_pdf_text(uploaded_file) -> str:
    """Extract text from uploaded PDF. Returns empty string on failure."""
    try:
        reader = PdfReader(BytesIO(uploaded_file.read()))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        uploaded_file.seek(0)
        return text if len(text.strip()) >= 30 else ""
    except Exception:
        return ""


# ── Tokenisation ─────────────────────────────────────────

def _tokenize_ngrams(text: str) -> set[str]:
    """Generate 1-, 2-, and 3-grams from lowercased text."""
    text = text.lower()
    words = re.findall(r"[a-z0-9#+./]+", text)
    tokens = set(words)
    for i in range(len(words) - 1):
        tokens.add(f"{words[i]} {words[i + 1]}")
    for i in range(len(words) - 2):
        tokens.add(f"{words[i]} {words[i + 1]} {words[i + 2]}")
    return tokens


# ── Skill Extraction ─────────────────────────────────────

def extract_skills(text: str) -> list[str]:
    """Extract canonical skills found in text via taxonomy + alias matching."""
    tokens = _tokenize_ngrams(text)
    found: set[str] = set()
    for skills in SKILL_TAXONOMY.values():
        for skill in skills:
            if skill in tokens:
                found.add(skill)
    for alias, canonical in SKILL_ALIASES.items():
        if alias in tokens:
            found.add(canonical)
    return sorted(found)


_EVIDENCE_RE = [
    re.compile(r"\d+[%kKmM+]"),
    re.compile(
        r"built|developed|implemented|created|designed|deployed|reduced|"
        r"increased|improved|optimized|achieved|launched|automated",
        re.IGNORECASE,
    ),
    re.compile(r"github\.com|gitlab\.com|deployed|live|production", re.IGNORECASE),
    re.compile(r"project|application|app|platform|system|dashboard|service", re.IGNORECASE),
]


def _find_supported_skills(text: str, claimed: list[str]) -> list[str]:
    """A skill is 'supported' if it appears near evidence (projects, metrics, verbs)."""
    lines = text.lower().split("\n")
    supported: set[str] = set()
    for skill in claimed:
        sl = skill.lower()
        for i, line in enumerate(lines):
            if sl not in line and resolve_alias(sl) not in line:
                continue
            ctx = " ".join(lines[max(0, i - 2) : i + 3])
            if any(p.search(ctx) for p in _EVIDENCE_RE):
                supported.add(skill)
                break
    return sorted(supported)


# ── Profile Signal Extraction ─────────────────────────────

def extract_profile_signals(text: str) -> ProfileSignals:
    """Extract structured signals from resume text."""
    sig = ProfileSignals()
    sig.claimed_skills = extract_skills(text)
    sig.supported_skills = _find_supported_skills(text, sig.claimed_skills)
    sig.projects = _extract_projects(text)
    sig.certifications = _extract_certifications(text)
    sig.links = _extract_links(text)
    sig.education = _extract_education(text)
    sig.experience_months = _estimate_experience(text)
    sig.has_internship = bool(re.search(r"intern", text, re.IGNORECASE))
    return sig


# ── Section extractors ───────────────────────────────────

_SECTION_HEADS = re.compile(
    r"^(?:EXPERIENCE|EDUCATION|CERTIF|SKILLS|AWARDS?|ACHIEVE|PROJECTS?|"
    r"TECHNICAL|SUMMARY|OBJECTIVE|INTERESTS|HOBBIES|ACTIVITIES|PUBLICATIONS|REFERENCES)",
    re.IGNORECASE,
)


def _extract_projects(text: str) -> list[dict]:
    lines = text.strip().split("\n")
    projects: list[dict] = []
    in_section = False
    cur: dict | None = None

    for raw in lines:
        line = raw.strip()
        if re.match(r"^(?:PROJECTS?|PERSONAL PROJECTS?|KEY PROJECTS?)", line, re.IGNORECASE):
            in_section = True
            continue
        if in_section and _SECTION_HEADS.match(line):
            if cur:
                projects.append(cur)
            in_section = False
            continue
        if not in_section:
            continue

        # Heuristic: project header = line without bullet, often contains a dash
        is_bullet = line.startswith(("•", "-", "*", "–"))
        if not is_bullet and len(line) > 8 and not line.startswith("http"):
            if cur:
                projects.append(cur)
            cur = {"name": line.split("(")[0].strip(), "tech": [], "description": "",
                   "has_metrics": False, "has_link": False}
        elif cur and line:
            cur["description"] += " " + line
            if re.search(r"\d+[%kKmM+]|\d{2,}", line):
                cur["has_metrics"] = True
            if re.search(r"github\.com|gitlab|\.io|\.app|deployed|live", line, re.IGNORECASE):
                cur["has_link"] = True
            cur["tech"] = sorted(set(cur["tech"] + extract_skills(line)))

    if cur:
        projects.append(cur)
    return projects


def _extract_certifications(text: str) -> list[str]:
    certs: list[str] = []
    in_sec = False
    for line in text.strip().split("\n"):
        s = line.strip()
        if re.match(r"^CERTIF", s, re.IGNORECASE):
            in_sec = True
            continue
        if in_sec and _SECTION_HEADS.match(s):
            in_sec = False
            continue
        if in_sec and s:
            c = re.sub(r"^[•\-*]\s*", "", s).strip()
            if len(c) > 3:
                certs.append(c)
    return certs


def _extract_links(text: str) -> list[str]:
    pattern = re.compile(
        r"(?:https?://)?(?:www\.)?(?:github\.com|gitlab\.com|linkedin\.com|"
        r"bitbucket\.org|[a-z0-9-]+\.(?:io|app|dev|vercel\.app|netlify\.app|herokuapp\.com))"
        r"[^\s,;)]*",
        re.IGNORECASE,
    )
    return list(set(pattern.findall(text)))


def _extract_education(text: str) -> list[str]:
    edu: list[str] = []
    in_sec = False
    for line in text.strip().split("\n"):
        s = line.strip()
        if re.match(r"^EDUCATION", s, re.IGNORECASE):
            in_sec = True
            continue
        if in_sec and _SECTION_HEADS.match(s):
            in_sec = False
            continue
        if in_sec and s and len(s) > 5:
            edu.append(re.sub(r"^[•\-*]\s*", "", s).strip())
    return edu


_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _estimate_experience(text: str) -> int:
    ranges = re.findall(
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s*[–—-]\s*"
        r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|Present|Current)",
        text, re.IGNORECASE,
    )
    months = 0
    for dr in ranges:
        parts = re.split(r"\s*[–—-]\s*", dr)
        if len(parts) != 2:
            continue
        try:
            sm = re.match(r"([A-Za-z]+)\s+(\d{4})", parts[0])
            if not sm:
                continue
            s_m = _MONTH_MAP.get(sm.group(1)[:3].lower(), 1)
            s_y = int(sm.group(2))
            if re.match(r"present|current", parts[1], re.IGNORECASE):
                e_m, e_y = 4, 2026
            else:
                em = re.match(r"([A-Za-z]+)\s+(\d{4})", parts[1])
                if not em:
                    continue
                e_m = _MONTH_MAP.get(em.group(1)[:3].lower(), 1)
                e_y = int(em.group(2))
            months += max(0, (e_y - s_y) * 12 + (e_m - s_m))
        except (ValueError, AttributeError):
            continue
    return months


# ── Communication Scoring ─────────────────────────────────

ACTION_VERBS = frozenset({
    "built", "developed", "implemented", "created", "designed", "deployed",
    "led", "managed", "optimized", "reduced", "increased", "improved",
    "architected", "launched", "automated", "integrated", "migrated",
    "refactored", "scaled", "configured", "established", "maintained",
    "analyzed", "collaborated", "contributed", "delivered", "engineered",
    "executed", "generated", "mentored", "orchestrated", "resolved",
    "streamlined", "tested", "transformed", "wrote",
})

_PASSIVE_RE = [re.compile(p, re.IGNORECASE) for p in [
    r"\bwas\s+\w+ed\b", r"\bwere\s+\w+ed\b", r"\bresponsible\s+for\b",
    r"\btasked\s+with\b", r"\binvolved\s+in\b", r"\bworked\s+on\b",
    r"\bhelped\s+with\b", r"\bassisted\s+in\b",
]]


def score_communication(text: str) -> dict:
    """Score communication quality. Returns total (0-100) + subscores."""
    bullets = [l.strip() for l in text.split("\n") if l.strip().startswith(("•", "-", "*", "–"))]
    if not bullets:
        bullets = [l.strip() for l in text.split("\n") if len(l.strip()) > 10]
    n = max(len(bullets), 1)

    # Action verb density → 0-25
    act = sum(1 for b in bullets if any(w in ACTION_VERBS for w in b.lower().split()[:3]))
    act_score = min(25, round((act / n) * 30))

    # Active voice → 0-30
    pas = sum(1 for b in bullets if any(p.search(b) for p in _PASSIVE_RE))
    active_score = min(30, round((1 - pas / n) * 35))

    # Specificity (numbers) → 0-25
    met = sum(1 for b in bullets if re.search(r"\d+[%kKmM+]|\$\d|\d{2,}\s*(?:users|requests|ms)", b))
    spec_score = min(25, round((met / n) * 35))

    # Sentence quality → 0-20
    good = sum(1 for b in bullets if 8 <= len(b.split()) <= 25)
    qual_score = min(20, round((good / n) * 25))

    total = min(100, act_score + active_score + spec_score + qual_score)
    return {
        "total": total,
        "action_verb_pct": round(act / n * 100),
        "active_voice_pct": round((1 - pas / n) * 100),
        "specificity_pct": round(met / n * 100),
        "quality_pct": round(good / n * 100),
    }


def score_evidence_quality(signals: ProfileSignals | dict) -> dict:
    """Score evidence quality from profile signals (0-100)."""
    if isinstance(signals, dict):
        s = signals
    else:
        s = signals.to_dict()

    score = 0
    details: list[str] = []
    missing: list[str] = []

    # GitHub / deployed links (0-15 each)
    links = s.get("links", [])
    gh = any("github.com" in l for l in links)
    if gh:
        score += 15; details.append("GitHub profile linked")
    else:
        missing.append("No GitHub link found")
    deployed = any(d in " ".join(links) for d in [".io", ".app", ".dev", "vercel", "netlify", "heroku"])
    if deployed:
        score += 15; details.append("Deployed project link found")
    else:
        missing.append("No deployed project links")

    # Project metrics (0-20)
    projects = s.get("projects", [])
    with_metrics = sum(1 for p in projects if p.get("has_metrics"))
    if with_metrics >= 2:
        score += 20; details.append(f"{with_metrics} projects with quantified metrics")
    elif with_metrics == 1:
        score += 10; details.append("1 project with metrics")
    else:
        missing.append("No quantified metrics in projects")

    # Project depth (0-20)
    if len(projects) >= 3:
        score += 20; details.append(f"{len(projects)} projects described")
    elif len(projects) >= 2:
        score += 12; details.append(f"{len(projects)} projects described")
    elif len(projects) == 1:
        score += 6; details.append("1 project described")
    else:
        missing.append("No projects found")

    # Certifications (0-15)
    certs = s.get("certifications", [])
    if len(certs) >= 2:
        score += 15; details.append(f"{len(certs)} certifications")
    elif len(certs) == 1:
        score += 8; details.append("1 certification")
    else:
        missing.append("No certifications listed")

    # Internship / work experience (0-15)
    if s.get("has_internship") or s.get("experience_months", 0) > 0:
        score += 15; details.append("Work/internship experience present")
    else:
        missing.append("No work experience listed")

    return {"total": min(100, score), "details": details, "missing": missing}
