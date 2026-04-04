"""
TRAXR — Deterministic Scoring Engine
6-dimension readiness model with gap classification. No LLM dependency.
"""

from __future__ import annotations

from skilltaxonomy import resolve_alias, find_category, ROLE_WEIGHTS
from nlppipeline import score_communication, score_evidence_quality, ProfileSignals


# ── Gap Classification ───────────────────────────────────

def classify_gaps(
    role_dna: dict,
    claimed_skills: list[str],
    supported_skills: list[str],
) -> list[dict]:
    """
    Classify skill gaps as hard / soft / context.
    - Hard Gap:    required skill absent from resume entirely
    - Soft Gap:    skill claimed but not supported with evidence
    - Context Gap: skill present but via alias / adjacent tech (e.g. MySQL vs PostgreSQL)
    """
    required = [resolve_alias(s) for s in role_dna.get("required_skills", [])]
    preferred = [resolve_alias(s) for s in role_dna.get("preferred_skills", [])]
    claimed_set = {resolve_alias(s) for s in claimed_skills}
    supported_set = {resolve_alias(s) for s in supported_skills}

    gaps: list[dict] = []

    # Check required skills first (higher severity)
    for skill in required:
        if skill in supported_set:
            continue
        cat = find_category(skill) or "General"
        if skill in claimed_set:
            gaps.append({
                "skill": skill, "type": "soft", "severity": "high",
                "category": cat,
                "explanation": f"'{skill}' appears in your resume but lacks supporting evidence (no project, metric, or outcome).",
                "action": f"Add a project or measurable outcome demonstrating {skill}.",
            })
        else:
            # Check for context gap (adjacent skill present)
            adjacent = _find_adjacent(skill, claimed_set)
            if adjacent:
                gaps.append({
                    "skill": skill, "type": "context", "severity": "high",
                    "category": cat,
                    "explanation": f"'{skill}' is required but you have '{adjacent}' instead — related but not the same.",
                    "action": f"Learn {skill} specifically; your {adjacent} background will accelerate this.",
                })
            else:
                gaps.append({
                    "skill": skill, "type": "hard", "severity": "high",
                    "category": cat,
                    "explanation": f"'{skill}' is a required skill and is completely absent from your resume.",
                    "action": f"Take a focused crash course on {skill} and build a small project.",
                })

    # Check preferred skills (lower severity)
    for skill in preferred:
        if skill in supported_set or skill in claimed_set:
            continue
        cat = find_category(skill) or "General"
        gaps.append({
            "skill": skill, "type": "hard", "severity": "medium",
            "category": cat,
            "explanation": f"'{skill}' is a preferred skill that would strengthen your application.",
            "action": f"Explore {skill} through a tutorial or side project.",
        })

    # Sort: high severity first, then by type priority
    severity_order = {"high": 0, "medium": 1, "low": 2}
    type_order = {"hard": 0, "context": 1, "soft": 2}
    gaps.sort(key=lambda g: (severity_order.get(g["severity"], 9), type_order.get(g["type"], 9)))
    return gaps


_ADJACENT_MAP: dict[str, list[str]] = {
    "postgresql": ["mysql", "sql", "mongodb"],
    "mysql": ["postgresql", "sql"],
    "java": ["kotlin", "c#", "c++"],
    "typescript": ["javascript"],
    "go": ["python", "rust"],
    "kubernetes": ["docker"],
    "aws": ["gcp", "azure"],
    "gcp": ["aws", "azure"],
    "azure": ["aws", "gcp"],
    "react": ["angular", "vue.js", "svelte"],
    "angular": ["react", "vue.js"],
    "vue.js": ["react", "angular"],
    "node.js": ["express.js", "flask", "django"],
    "flask": ["django", "fastapi"],
    "django": ["flask", "fastapi"],
    "rabbitmq": ["kafka"],
    "kafka": ["rabbitmq"],
    "prometheus": ["grafana"],
    "grafana": ["prometheus"],
}


def _find_adjacent(skill: str, claimed: set[str]) -> str | None:
    """Check if user has an adjacent skill that partially covers the gap."""
    for adj in _ADJACENT_MAP.get(skill, []):
        if adj in claimed:
            return adj
    return None


# ── Technical Skill Match ─────────────────────────────────

def _score_technical(role_dna: dict, claimed: list[str]) -> dict:
    required = [resolve_alias(s) for s in role_dna.get("required_skills", [])]
    preferred = [resolve_alias(s) for s in role_dna.get("preferred_skills", [])]
    claimed_set = {resolve_alias(s) for s in claimed}

    req_matched = [s for s in required if s in claimed_set]
    pref_matched = [s for s in preferred if s in claimed_set]
    req_missing = [s for s in required if s not in claimed_set]

    req_total = max(len(required), 1)
    pref_total = max(len(preferred), 1)
    score = round((len(req_matched) / req_total) * 70 + (len(pref_matched) / pref_total) * 30)
    score = min(100, score)

    return {
        "score": score,
        "why": f"Matched {len(req_matched)} of {len(required)} required and {len(pref_matched)} of {len(preferred)} preferred skills.",
        "evidence_found": [f"{s} ✓" for s in req_matched] + [f"{s} ✓ (preferred)" for s in pref_matched],
        "missing": req_missing + [s for s in preferred if s not in claimed_set],
        "action": f"Focus on the {len(req_missing)} missing required skill(s): {', '.join(req_missing[:3]) or 'none'}.",
    }


# ── Role Alignment ────────────────────────────────────────

def _score_alignment(role_dna: dict, resume_text: str, claimed: list[str]) -> dict:
    context_words = set()
    for field in ("role_context", "day_to_day_activities"):
        val = role_dna.get(field, "")
        if isinstance(val, list):
            val = " ".join(val)
        context_words.update(w.lower() for w in val.split() if len(w) > 3)

    resume_words = set(w.lower() for w in resume_text.split() if len(w) > 3)
    if not context_words:
        return {"score": 50, "why": "Insufficient role context to score.", "evidence_found": [],
                "missing": [], "action": "Provide a more detailed job description."}

    overlap = context_words & resume_words
    ratio = len(overlap) / max(len(context_words), 1)
    score = min(100, round(ratio * 120))  # slight boost so matching profiles hit 80+

    return {
        "score": score,
        "why": f"Resume shares {len(overlap)} context keywords with the role description ({round(ratio*100)}% overlap).",
        "evidence_found": sorted(list(overlap)[:8]),
        "missing": sorted(list(context_words - resume_words)[:5]),
        "action": "Tailor your resume language to match the role's context and responsibilities.",
    }


# ── Learning Momentum ────────────────────────────────────

def _score_momentum(signals: dict) -> dict:
    score = 0
    evidence: list[str] = []
    missing: list[str] = []

    certs = signals.get("certifications", [])
    if len(certs) >= 2:
        score += 30; evidence.append(f"{len(certs)} certifications")
    elif len(certs) == 1:
        score += 15; evidence.append("1 certification")
    else:
        missing.append("No certifications")

    projects = signals.get("projects", [])
    if len(projects) >= 3:
        score += 30; evidence.append(f"{len(projects)} projects show active building")
    elif len(projects) >= 2:
        score += 18; evidence.append(f"{len(projects)} projects")
    else:
        missing.append("Build more projects to show momentum")

    exp = signals.get("experience_months", 0)
    if exp >= 6:
        score += 20; evidence.append(f"{exp} months of experience")
    elif exp > 0:
        score += 12; evidence.append(f"{exp} months of experience")
    else:
        missing.append("No work experience yet")

    # Diversity bonus: check number of unique tech categories
    claimed = signals.get("claimed_skills", [])
    categories_used = {find_category(s) for s in claimed if find_category(s)}
    if len(categories_used) >= 4:
        score += 20; evidence.append(f"Skills across {len(categories_used)} categories")
    elif len(categories_used) >= 2:
        score += 10; evidence.append(f"Skills in {len(categories_used)} categories")

    return {
        "score": min(100, score),
        "why": f"Learning momentum scored based on certifications, project cadence, and skill diversity.",
        "evidence_found": evidence,
        "missing": missing,
        "action": "Keep building projects consistently and pursue one more certification.",
    }


# ── Interview Readiness ──────────────────────────────────

def _score_interview(tech_score: int, evidence_score: int, comm_score: int, signals: dict) -> dict:
    # Weighted composite
    behavioral = 0
    evidence_list: list[str] = []
    missing: list[str] = []

    if signals.get("has_internship"):
        behavioral += 25; evidence_list.append("Internship experience for STAR stories")
    else:
        missing.append("No internship for behavioral examples")

    projects = signals.get("projects", [])
    deep_projects = sum(1 for p in projects if p.get("has_metrics") and p.get("has_link"))
    if deep_projects >= 1:
        behavioral += 15; evidence_list.append(f"{deep_projects} deep project(s) for detailed discussion")

    if comm_score >= 60:
        behavioral += 10; evidence_list.append("Good communication clarity")
    else:
        missing.append("Improve bullet point specificity for interviews")

    composite = round(0.30 * evidence_score + 0.25 * comm_score + 0.25 * tech_score + 0.20 * behavioral)
    composite = min(100, composite)

    return {
        "score": composite,
        "why": f"Composite score: evidence {evidence_score}, communication {comm_score}, technical {tech_score}, behavioral readiness {behavioral}.",
        "evidence_found": evidence_list,
        "missing": missing,
        "action": "Prepare 3 STAR stories and practice mock interviews.",
    }


# ── Main: 360° Score ──────────────────────────────────────

def compute_360_score(
    resume_text: str,
    role_dna: dict,
    signals: ProfileSignals | dict,
    role_type: str = "General",
) -> dict:
    """
    Compute the full 6-dimension TRAXR readiness score.
    Returns dimensions, overall_score, ranked_gaps, explanations, confidence.
    """
    if isinstance(signals, ProfileSignals):
        sig = signals.to_dict()
    else:
        sig = signals

    claimed = sig.get("claimed_skills", [])
    supported = sig.get("supported_skills", [])

    # 1. Technical Skill Match
    tech = _score_technical(role_dna, claimed)

    # 2. Evidence Quality
    ev = score_evidence_quality(sig)
    ev_expl = {
        "score": ev["total"], "why": "Evidence scored on GitHub links, deployed URLs, project metrics, certifications, and work experience.",
        "evidence_found": ev["details"], "missing": ev["missing"],
        "action": "Add GitHub links and deploy your projects publicly.",
    }

    # 3. Communication Clarity
    comm = score_communication(resume_text)
    comm_expl = {
        "score": comm["total"],
        "why": f"Action verb density: {comm['action_verb_pct']}%, Active voice: {comm['active_voice_pct']}%, Specificity: {comm['specificity_pct']}%.",
        "evidence_found": [f"Action verbs: {comm['action_verb_pct']}%", f"Active voice: {comm['active_voice_pct']}%",
                           f"Specificity: {comm['specificity_pct']}%"],
        "missing": (["Increase use of action verbs"] if comm["action_verb_pct"] < 60 else []) +
                   (["Add more quantified metrics"] if comm["specificity_pct"] < 40 else []),
        "action": "Rewrite weak bullets using STAR format with quantified outcomes.",
    }

    # 4. Role Alignment
    align = _score_alignment(role_dna, resume_text, claimed)

    # 5. Learning Momentum
    momentum = _score_momentum(sig)

    # 6. Interview Readiness
    interview = _score_interview(tech["score"], ev["total"], comm["total"], sig)

    dimensions = {
        "Technical Skill Match": tech["score"],
        "Evidence Quality": ev["total"],
        "Communication Clarity": comm["total"],
        "Role Alignment": align["score"],
        "Learning Momentum": momentum["score"],
        "Interview Readiness": interview["score"],
    }

    explanations = {
        "Technical Skill Match": tech,
        "Evidence Quality": ev_expl,
        "Communication Clarity": comm_expl,
        "Role Alignment": align,
        "Learning Momentum": momentum,
        "Interview Readiness": interview,
    }

    # Weighted overall
    weights = ROLE_WEIGHTS.get(role_type, ROLE_WEIGHTS["General"])
    overall = round(sum(dimensions[d] * weights[d] for d in dimensions))

    # Gaps
    gaps = classify_gaps(role_dna, claimed, supported)

    # Confidence
    text_len = len(resume_text.strip())
    if text_len > 1500 and len(sig.get("projects", [])) >= 2:
        confidence = "high"
    elif text_len > 500:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "dimensions": dimensions,
        "overall_score": overall,
        "ranked_gaps": gaps,
        "dimension_explanations": explanations,
        "confidence": confidence,
    }
