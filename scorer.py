"""
TRAXR — Scoring Engine
Deterministic 6-dimension scoring and gap classification.
Pure processing module: no Streamlit, no Gemini, no side effects.

Public API:
    compute_360_score(resume_text, role_dna, profile_signals) -> dict
    classify_gaps(resume_text, role_dna) -> list[dict]
"""

from __future__ import annotations

import re
from typing import Any


# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════

DIMENSION_NAMES = [
    "Technical Match",
    "Evidence Quality",
    "Communication Clarity",
    "Role Alignment",
    "Learning Momentum",
    "Interview Readiness",
]

_SKILL_CATEGORIES: dict[str, list[str]] = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c",
        "go", "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "dart",
    ],
    "Web & Frontend": [
        "react", "angular", "vue.js", "svelte", "next.js", "html", "css",
        "tailwind css", "bootstrap", "sass", "webpack", "vite", "redux",
    ],
    "Backend & Infrastructure": [
        "node.js", "express.js", "flask", "django", "fastapi", "spring boot",
        "rest apis", "graphql", "grpc", "microservices", "websockets",
    ],
    "Data & ML": [
        "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
        "machine learning", "deep learning", "data analysis",
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "ci/cd", "jenkins", "github actions", "gitlab ci", "linux",
        "prometheus", "grafana",
    ],
    "Tools & Practices": [
        "git", "github", "jira", "postman", "figma",
        "unit testing", "integration testing", "tdd",
        "agile", "scrum", "code review", "system design",
    ],
}


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _clamp(val: int | float, lo: int = 0, hi: int = 100) -> int:
    """Clamp a value to [lo, hi] and return as int."""
    return max(lo, min(hi, int(round(val))))


def _safe_list(d: dict, key: str) -> list[str]:
    """Safely extract a list of strings from a dict."""
    val = d.get(key, [])
    if isinstance(val, list):
        return [str(v).lower().strip() for v in val if v]
    return []


def _safe_str(d: dict, key: str) -> str:
    """Safely extract a string from a dict."""
    val = d.get(key, "")
    return str(val) if val else ""


def _find_category(skill: str) -> str:
    """Find the category for a skill."""
    s = skill.lower().strip()
    for cat, members in _SKILL_CATEGORIES.items():
        if s in members:
            return cat
    return "General"


def _find_related_skill(skill: str, resume_skills: set[str]) -> str | None:
    """Find a related skill in the resume from the same category."""
    cat = _find_category(skill)
    if cat == "General":
        return None
    for member in _SKILL_CATEGORIES.get(cat, []):
        if member != skill and member in resume_skills:
            return member
    return None


def _extract_skills_inline(text: str) -> list[str]:
    """Internal skill extraction (avoids circular import when needed)."""
    try:
        from nlppipeline import extract_skills
        return extract_skills(text)
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════
# PUBLIC: 6-DIMENSION SCORING
# ═══════════════════════════════════════════════════════════

def compute_360_score(
    resume_text: str,
    role_dna: dict[str, Any],
    profile_signals: dict[str, Any],
) -> dict[str, Any]:
    """Compute deterministic 6-dimension readiness score.

    Args:
        resume_text: Raw resume text.
        role_dna: Dict with required_skills, preferred_skills, tools, etc.
        profile_signals: Dict from nlppipeline.extract_profile_signals().

    Returns:
        Dict with dimension_scores, dimension_explanations, overall_score,
        confidence, strengths, concerns, plus backward-compatible dimensions list.
    """
    if not resume_text or not resume_text.strip():
        return _empty_score_result()

    # Extract skill sets
    resume_skills = set(s.lower() for s in profile_signals.get("skills", []))
    if not resume_skills:
        resume_skills = set(_extract_skills_inline(resume_text))

    required = set(_safe_list(role_dna, "required_skills"))
    preferred = set(_safe_list(role_dna, "preferred_skills"))
    tools = set(_safe_list(role_dna, "tools"))
    all_role_skills = required | preferred

    matched_req = resume_skills & required
    matched_pref = resume_skills & preferred
    matched_tools = resume_skills & tools
    matched_all = resume_skills & all_role_skills
    missing_req = required - resume_skills
    missing_pref = preferred - resume_skills

    req_ratio = len(matched_req) / max(1, len(required))
    all_ratio = len(matched_all) / max(1, len(all_role_skills))

    # Profile signals
    project_count = profile_signals.get("project_count", 0)
    metric_count = profile_signals.get("metric_bullets_count", 0)
    has_github = profile_signals.get("has_github", False)
    has_deployed = profile_signals.get("has_deployed_links", False) or profile_signals.get("has_deploy_proof", False)
    cert_count = profile_signals.get("certification_count", 0)
    action_density = profile_signals.get("action_verb_density", 0.0)
    leadership = profile_signals.get("leadership_signals", 0)
    word_count = profile_signals.get("word_count", len(resume_text.split()))
    bullet_count = profile_signals.get("bullet_count", 0)
    section_count = profile_signals.get("section_count", 0)
    has_summary = profile_signals.get("has_summary", False)
    has_collab = profile_signals.get("has_collab_language", False)
    has_learning = profile_signals.get("has_learning_signals", False)
    has_links = profile_signals.get("has_links", False) or has_github

    resume_low = resume_text.lower()

    scores: dict[str, int] = {}
    explanations: dict[str, str] = {}
    strengths: list[str] = []
    concerns: list[str] = []

    # ── 1. Technical Match ──────────────────────────
    tech = req_ratio * 60 + all_ratio * 20 + min(project_count, 3) * 5
    if matched_tools:
        tech += min(len(matched_tools), 3) * 2
    tech = _clamp(tech, 5, 95)
    scores["Technical Match"] = tech

    if len(matched_req) > 0:
        explanations["Technical Match"] = (
            f"Matched {len(matched_req)} of {len(required)} required skills "
            f"and {len(matched_pref)} of {len(preferred)} preferred skills."
        )
    else:
        explanations["Technical Match"] = (
            f"No direct overlap with the {len(required)} required skills detected."
        )

    if tech >= 70:
        strengths.append(f"Strong technical coverage: {len(matched_req)}/{len(required)} required skills matched")
    if missing_req and len(missing_req) >= 2:
        concerns.append(f"Missing {len(missing_req)} required skills: {', '.join(sorted(missing_req)[:3])}")

    # ── 2. Evidence Quality ─────────────────────────
    ev = 20
    ev += min(project_count, 5) * 8
    ev += min(metric_count, 5) * 5
    if has_github:
        ev += 8
    if has_deployed:
        ev += 8
    if has_links:
        ev += 4
    if action_density > 0.5:
        ev += 5
    ev = _clamp(ev, 10, 95)
    scores["Evidence Quality"] = ev

    evidence_parts = []
    if project_count:
        evidence_parts.append(f"{project_count} project(s)")
    if metric_count:
        evidence_parts.append(f"{metric_count} quantified metric(s)")
    if has_github:
        evidence_parts.append("GitHub")
    if has_deployed:
        evidence_parts.append("deployed links")
    explanations["Evidence Quality"] = (
        f"Found {', '.join(evidence_parts) or 'minimal evidence'}. "
        f"{'Strong proof of applied work.' if ev >= 60 else 'More quantified outcomes and public links would strengthen this.'}"
    )

    if ev >= 70:
        strengths.append("Rich evidence: projects with metrics and public links")
    if not metric_count:
        concerns.append("No quantified outcomes (percentages, user counts, etc.)")

    # ── 3. Communication Clarity ────────────────────
    comm = 35
    if 200 < word_count < 1200:
        comm += 12
    elif word_count >= 1200:
        comm -= 5
    if bullet_count >= 5:
        comm += 10
    if has_summary:
        comm += 8
    if section_count >= 3:
        comm += 8
    if action_density > 0.3:
        comm += 7
    if action_density > 0.6:
        comm += 5
    comm = _clamp(comm, 15, 92)
    scores["Communication Clarity"] = comm

    explanations["Communication Clarity"] = (
        f"{word_count} words, {bullet_count} bullet points, {section_count} sections. "
        f"Action verb density: {action_density:.0%}. "
        f"{'Well-structured.' if comm >= 60 else 'Consider stronger action verbs and more structured bullet points.'}"
    )

    if comm >= 70:
        strengths.append("Clear, well-structured resume with strong action verbs")
    if action_density < 0.3:
        concerns.append("Low action verb usage — consider starting bullets with Built, Deployed, Optimized, etc.")

    # ── 4. Role Alignment ──────────────────────────
    align = all_ratio * 50 + req_ratio * 30
    if has_collab:
        align += 8
    behavioral = _safe_list(role_dna, "behavioral_traits") or _safe_list(role_dna, "behavioral_requirements")
    behavioral_matches = [b for b in behavioral if b in resume_low]
    if behavioral_matches:
        align += min(len(behavioral_matches), 3) * 4

    # Check day-to-day alignment
    day_to_day = _safe_list(role_dna, "day_to_day") or _safe_list(role_dna, "day_to_day_activities")
    day_matches = sum(1 for d in day_to_day if any(w in resume_low for w in d.split() if len(w) > 4))
    align += min(day_matches, 4) * 3
    align = _clamp(align, 8, 92)
    scores["Role Alignment"] = align

    explanations["Role Alignment"] = (
        f"Overall skill overlap: {int(all_ratio * 100)}% with role requirements. "
        f"{'Collaboration and behavioral signals present.' if has_collab else 'Limited team or behavioral evidence.'}"
    )

    if align >= 65:
        strengths.append(f"Good role alignment: {int(all_ratio * 100)}% skill overlap")
    if align < 40:
        concerns.append("Low alignment with role requirements — tailor resume to emphasize relevant experience")

    # ── 5. Learning Momentum ───────────────────────
    learn = 25
    if has_learning:
        learn += 18
    if cert_count >= 1:
        learn += min(cert_count, 3) * 6
    if project_count >= 3:
        learn += 10
    if project_count >= 5:
        learn += 5
    if re.search(r"hackathon|winner|award|achievement|publication|open.?source", resume_low):
        learn += 10
    if re.search(r"20(?:2[4-6])", resume_text):
        learn += 5
    learn = _clamp(learn, 12, 95)
    scores["Learning Momentum"] = learn

    learn_parts: list[str] = []
    if cert_count:
        learn_parts.append(f"{cert_count} certification(s)")
    if project_count:
        learn_parts.append(f"{project_count} project(s)")
    if has_learning:
        learn_parts.append("courses/learning signals")
    explanations["Learning Momentum"] = (
        f"{', '.join(learn_parts) or 'Minimal learning signals'}. "
        f"{'Active growth trajectory.' if learn >= 60 else 'Add recent certifications or hackathon participation.'}"
    )

    if learn >= 70:
        strengths.append("Strong learning momentum: certifications, projects, and competitions")
    if not has_learning and cert_count == 0:
        concerns.append("No certifications, courses, or formal learning signals")

    # ── 6. Interview Readiness ─────────────────────
    interview = 20
    interview += int(req_ratio * 25)
    if has_deployed:
        interview += 10
    if metric_count >= 2:
        interview += 10
    if has_collab:
        interview += 8
    if project_count >= 2:
        interview += 5
    if has_summary:
        interview += 5
    if leadership >= 1:
        interview += 5
    interview = _clamp(interview, 10, 92)
    scores["Interview Readiness"] = interview

    ready_parts: list[str] = []
    if has_deployed:
        ready_parts.append("deployment experience")
    if metric_count >= 2:
        ready_parts.append(f"{metric_count} quantifiable outcomes")
    if has_collab:
        ready_parts.append("collaboration examples")
    explanations["Interview Readiness"] = (
        f"{'Strengths: ' + ', '.join(ready_parts) + '.' if ready_parts else 'Limited concrete talking points.'} "
        f"{'Prepare STAR stories around projects.' if interview < 60 else 'Strong portfolio for interview discussion.'}"
    )

    if interview >= 65:
        strengths.append("Interview-ready: metrics, deployments, and team examples to discuss")
    if interview < 40:
        concerns.append("Weak interview readiness — add measurable outcomes and deployment experience")

    # ── Aggregate ──────────────────────────────────
    dim_values = list(scores.values())
    overall = _clamp(sum(dim_values) / len(dim_values))

    if overall >= 65:
        confidence = "high"
    elif overall >= 40:
        confidence = "medium"
    else:
        confidence = "low"

    # Build backward-compatible dimensions list (for ScoreDimension.from_dict)
    dim_key_map = {
        "Technical Match": "technical_match",
        "Evidence Quality": "evidence_quality",
        "Communication Clarity": "communication_clarity",
        "Role Alignment": "role_alignment",
        "Learning Momentum": "learning_momentum",
        "Interview Readiness": "interview_readiness",
    }

    dimensions_list: list[dict[str, Any]] = []
    for label in DIMENSION_NAMES:
        key = dim_key_map.get(label, label.lower().replace(" ", "_"))
        sc = scores.get(label, 0)
        exp_text = explanations.get(label, "")
        dimensions_list.append({
            "key": key,
            "label": label,
            "score": sc,
            "why": exp_text,
            "evidence_found": _collect_evidence(label, profile_signals, matched_req, matched_pref),
            "missing_proof": _collect_missing(label, missing_req, missing_pref, profile_signals),
            "action": _generate_action(label, sc, missing_req, profile_signals),
        })

    return {
        # Primary spec keys
        "overall_score": overall,
        "dimension_scores": dict(scores),
        "dimension_explanations": dict(explanations),
        "strengths": strengths[:6],
        "concerns": concerns[:6],
        "confidence": confidence,
        # Backward-compatible keys for app.py integration
        "dimensions": dimensions_list,
        "matched_skills": sorted(matched_all),
        "missing_skills": sorted(missing_req | missing_pref),
    }


def _empty_score_result() -> dict[str, Any]:
    """Return a zeroed-out score result for empty input."""
    empty_scores = {d: 0 for d in DIMENSION_NAMES}
    empty_expl = {d: "No resume text provided." for d in DIMENSION_NAMES}
    empty_dims = [
        {"key": d.lower().replace(" ", "_"), "label": d, "score": 0,
         "why": "No resume text provided.", "evidence_found": [],
         "missing_proof": [], "action": "Provide resume text to begin analysis."}
        for d in DIMENSION_NAMES
    ]
    return {
        "overall_score": 0,
        "dimension_scores": empty_scores,
        "dimension_explanations": empty_expl,
        "strengths": [],
        "concerns": ["No resume content available for analysis."],
        "confidence": "low",
        "dimensions": empty_dims,
        "matched_skills": [],
        "missing_skills": [],
    }


def _collect_evidence(
    dim: str,
    signals: dict,
    matched_req: set[str],
    matched_pref: set[str],
) -> list[str]:
    """Generate evidence-found list for a dimension."""
    ev: list[str] = []
    if dim == "Technical Match":
        ev.extend(f"{s} ✓" for s in sorted(matched_req)[:6])
        ev.extend(f"{s} ✓ (preferred)" for s in sorted(matched_pref)[:3])
    elif dim == "Evidence Quality":
        if signals.get("project_count"):
            ev.append(f"{signals['project_count']} project(s)")
        if signals.get("metric_bullets_count"):
            ev.append(f"{signals['metric_bullets_count']} quantified metric(s)")
        if signals.get("has_github"):
            ev.append("GitHub profile linked")
        if signals.get("has_deployed_links") or signals.get("has_deploy_proof"):
            ev.append("Deployed/production evidence")
    elif dim == "Communication Clarity":
        ev.append(f"{signals.get('word_count', 0)} words")
        ev.append(f"{signals.get('bullet_count', 0)} bullet points")
        if signals.get("has_summary"):
            ev.append("Professional summary present")
    elif dim == "Role Alignment":
        ev.extend(f"{s} ✓" for s in sorted(matched_req | matched_pref)[:6])
    elif dim == "Learning Momentum":
        if signals.get("certification_count"):
            ev.append(f"{signals['certification_count']} certification(s)")
        if signals.get("project_count"):
            ev.append(f"{signals['project_count']} project(s)")
        if signals.get("has_learning_signals"):
            ev.append("Courses/learning signals")
    elif dim == "Interview Readiness":
        if signals.get("has_deployed_links") or signals.get("has_deploy_proof"):
            ev.append("Deployment experience available")
        if signals.get("metric_bullets_count", 0) >= 2:
            ev.append("Quantifiable outcomes for discussion")
        if signals.get("has_collab_language"):
            ev.append("Team collaboration examples")
    return ev


def _collect_missing(
    dim: str,
    missing_req: set[str],
    missing_pref: set[str],
    signals: dict,
) -> list[str]:
    """Generate missing-proof list for a dimension."""
    mi: list[str] = []
    if dim == "Technical Match":
        mi.extend(sorted(missing_req)[:5])
    elif dim == "Evidence Quality":
        if not signals.get("metric_bullets_count"):
            mi.append("No quantified outcomes")
        if not signals.get("has_deployed_links") and not signals.get("has_deploy_proof"):
            mi.append("No deployment evidence")
        if not signals.get("has_github"):
            mi.append("No GitHub profile")
    elif dim == "Communication Clarity":
        if not signals.get("has_summary"):
            mi.append("No professional summary")
        if signals.get("action_verb_density", 0) < 0.3:
            mi.append("Low action verb usage")
    elif dim == "Role Alignment":
        mi.extend(f"Missing: {s}" for s in sorted(missing_req | missing_pref)[:4])
    elif dim == "Learning Momentum":
        if not signals.get("certification_count"):
            mi.append("No certifications")
        if not signals.get("has_learning_signals"):
            mi.append("No formal learning signals")
    elif dim == "Interview Readiness":
        if not signals.get("has_deployed_links") and not signals.get("has_deploy_proof"):
            mi.append("No deployment experience")
        if signals.get("metric_bullets_count", 0) < 2:
            mi.append("Few quantifiable results")
    return mi


def _generate_action(dim: str, score: int, missing_req: set[str], signals: dict) -> str:
    """Generate a concise recommended action for a dimension."""
    if dim == "Technical Match":
        if missing_req:
            top = sorted(missing_req)[:3]
            return f"Focus on missing required skills: {', '.join(top)}."
        return "Strengthen evidence for existing skills with project outcomes."
    if dim == "Evidence Quality":
        return "Add measurable outcomes, deploy projects publicly, and link to GitHub."
    if dim == "Communication Clarity":
        return "Use strong action verbs and quantify every achievement with STAR format."
    if dim == "Role Alignment":
        return "Tailor resume to emphasize skills and experiences most relevant to this role."
    if dim == "Learning Momentum":
        return "Add recent certifications, courses, or hackathon participation."
    if dim == "Interview Readiness":
        return "Prepare STAR stories around your strongest projects with measurable outcomes."
    return "Continue building evidence and refining your resume."


# ═══════════════════════════════════════════════════════════
# PUBLIC: GAP CLASSIFICATION
# ═══════════════════════════════════════════════════════════

def classify_gaps(
    resume_text: str,
    role_dna_or_skills: Any = None,
    role_dna_legacy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Classify gaps between resume evidence and role requirements.

    Supports two call signatures:
        classify_gaps(resume_text, role_dna)              — new spec (2-arg)
        classify_gaps(resume_text, resume_skills, role_dna)  — legacy (3-arg)

    Returns sorted list of gap dicts with: skill, gap_type, severity,
    evidence_missing, why_it_matters, suggested_fix.
    """
    # Resolve role_dna from either call signature
    if role_dna_legacy is not None and isinstance(role_dna_legacy, dict):
        # Legacy 3-arg: classify_gaps(text, skills_list, dna_dict)
        role_dna = role_dna_legacy
    elif isinstance(role_dna_or_skills, dict):
        # New 2-arg: classify_gaps(text, dna_dict)
        role_dna = role_dna_or_skills
    else:
        role_dna = {}

    if not resume_text or not resume_text.strip():
        return []

    resume_skills = set(s.lower() for s in _extract_skills_inline(resume_text))
    resume_low = resume_text.lower()

    required = _safe_list(role_dna, "required_skills")
    preferred = _safe_list(role_dna, "preferred_skills")

    gaps: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Analyze required skills
    for skill in required:
        if skill in seen:
            continue
        seen.add(skill)
        gap = _classify_one_gap(skill, resume_low, resume_skills, is_required=True)
        if gap:
            gaps.append(gap)

    # Analyze preferred skills
    for skill in preferred:
        if skill in seen:
            continue
        seen.add(skill)
        gap = _classify_one_gap(skill, resume_low, resume_skills, is_required=False)
        if gap:
            gaps.append(gap)

    # Sort: high severity first, then by gap type priority
    sev_order = {"high": 0, "medium": 1, "low": 2}
    type_order = {"hard_gap": 0, "soft_gap": 1, "context_gap": 2}
    gaps.sort(key=lambda g: (sev_order.get(g["severity"], 9), type_order.get(g["gap_type"], 9)))

    return gaps


def _classify_one_gap(
    skill: str,
    resume_low: str,
    resume_skills: set[str],
    is_required: bool,
) -> dict[str, Any] | None:
    """Classify a single skill gap. Returns None if no gap detected."""
    base_severity = "high" if is_required else "medium"
    cat = _find_category(skill)
    pattern = re.escape(skill)

    if skill in resume_skills:
        # Skill IS present — check if it has sufficient proof
        occurrences = len(re.findall(r"(?<![a-z])" + pattern + r"(?![a-z])", resume_low))

        has_project_ctx = bool(re.search(
            rf"(?:built|developed|implemented|deployed|created|used|integrated).*{pattern}"
            rf"|{pattern}.*(?:project|app|system|api|service|pipeline|tool|platform)",
            resume_low,
        ))

        has_metric = bool(re.search(
            rf"{pattern}.*\d+[%+x]|\d+[%+x].*{pattern}",
            resume_low,
        ))

        if occurrences <= 1 and not has_project_ctx:
            return {
                "skill": skill,
                "gap_type": "soft_gap",
                "severity": base_severity,
                "evidence_missing": f"'{skill}' is listed but not tied to any project, outcome, or experience.",
                "why_it_matters": f"'{skill}' appears in resume but lacks supporting evidence or project context.",
                "suggested_fix": f"Add a project or work entry demonstrating practical {skill} usage with measurable outcomes.",
                # Backward-compatible keys for GapItem.from_dict()
                "name": skill,
                "category": cat,
                "action": f"Add a project demonstrating practical {skill} usage with measurable outcomes.",
            }

        if has_project_ctx and not has_metric:
            return {
                "skill": skill,
                "gap_type": "context_gap",
                "severity": "low",
                "evidence_missing": f"'{skill}' used in project context but results are not quantified.",
                "why_it_matters": f"'{skill}' has project evidence but lacks metrics (users, performance gains, coverage).",
                "suggested_fix": f"Quantify the outcome of your {skill} work with specific numbers or percentages.",
                "name": skill,
                "category": cat,
                "action": f"Quantify the outcome of your {skill} work with specific numbers or percentages.",
            }

        # Strong evidence — no gap
        return None

    # Skill NOT present — check for related skills (context gap) vs hard gap
    related = _find_related_skill(skill, resume_skills)
    if related:
        return {
            "skill": skill,
            "gap_type": "context_gap",
            "severity": base_severity,
            "evidence_missing": f"Resume mentions '{related}' but not '{skill}' directly.",
            "why_it_matters": f"'{skill}' is {'required' if is_required else 'preferred'} but only '{related}' is present — related but not identical.",
            "suggested_fix": f"Learn {skill} specifically; your {related} background will accelerate this.",
            "name": skill,
            "category": cat,
            "action": f"Learn {skill} specifically; your {related} background will accelerate this.",
        }

    # Completely absent — hard gap
    role_label = "a required skill" if is_required else "a preferred skill"
    return {
        "skill": skill,
        "gap_type": "hard_gap",
        "severity": base_severity,
        "evidence_missing": f"'{skill}' is not mentioned anywhere in the resume.",
        "why_it_matters": f"'{skill}' is {role_label} and is completely absent from the resume.",
        "suggested_fix": f"Take a focused crash course on {skill} and build a small project to demonstrate competence.",
        "name": skill,
        "category": cat,
        "action": f"Take a focused crash course on {skill} and build a small project to demonstrate competence.",
    }
