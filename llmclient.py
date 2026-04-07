"""
TRAXR — LLM Client (Google Gemini)
Primary reasoning layer with deterministic fallbacks.
No Streamlit dependency. No UI logic.

Public API:
    build_role_dna(resume_text, job_description_text) -> dict
    build_score_breakdown(resume_text, job_description_text, role_dna) -> dict
    build_gaps(resume_text, job_description_text, role_dna) -> list[dict]
    build_skillprint(resume_text, job_description_text, role_dna, gaps) -> dict
    build_roadmap(resume_text, job_description_text, role_dna, gaps) -> list[dict]
    evaluate_skillprint_submission(code, challenge) -> dict
    is_llm_available() -> bool
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════

DIMENSION_DEFS = [
    ("technical_match", "Technical Match"),
    ("evidence_quality", "Evidence Quality"),
    ("communication_clarity", "Communication Clarity"),
    ("role_alignment", "Role Alignment"),
    ("learning_momentum", "Learning Momentum"),
    ("interview_readiness", "Interview Readiness"),
]

_KNOWN_SKILLS: list[str] = [
    "python", "java", "javascript", "typescript", "c++", "c#", "c", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "dart", "shell scripting",
    "react", "angular", "vue.js", "svelte", "next.js", "html", "css",
    "tailwind css", "bootstrap", "sass", "webpack", "vite", "redux",
    "node.js", "express.js", "flask", "django", "fastapi", "spring boot",
    "rest apis", "graphql", "grpc", "microservices", "websockets",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
    "elasticsearch", "pandas", "numpy", "scikit-learn", "tensorflow",
    "pytorch", "keras", "machine learning", "deep learning",
    "natural language processing", "computer vision", "data analysis",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "ci/cd", "jenkins", "github actions", "gitlab ci", "linux",
    "ansible", "prometheus", "grafana",
    "git", "github", "jira", "postman", "swagger", "figma",
    "unit testing", "integration testing", "tdd",
    "agile", "scrum", "kanban", "code review", "system design",
]

_SKILL_ALIASES: dict[str, str] = {
    "js": "javascript", "ts": "typescript", "py": "python",
    "cpp": "c++", "golang": "go", "node": "node.js", "nodejs": "node.js",
    "react.js": "react", "reactjs": "react", "vue": "vue.js",
    "nextjs": "next.js", "expressjs": "express.js", "express": "express.js",
    "spring": "spring boot", "fast api": "fastapi",
    "tailwind": "tailwind css", "postgres": "postgresql", "mongo": "mongodb",
    "ml": "machine learning", "dl": "deep learning",
    "nlp": "natural language processing", "cv": "computer vision",
    "sklearn": "scikit-learn", "k8s": "kubernetes", "tf": "terraform",
    "ci cd": "ci/cd", "cicd": "ci/cd", "rest": "rest apis",
    "rest api": "rest apis", "restful": "rest apis", "restful apis": "rest apis",
    "gh actions": "github actions",
}

_TOOL_PATTERNS: list[str] = [
    "git", "github", "jira", "postman", "swagger", "figma", "vs code",
    "docker", "kubernetes", "jenkins", "terraform", "aws", "gcp", "azure",
    "linux", "nginx", "redis", "elasticsearch", "grafana", "prometheus",
    "datadog", "sentry", "circleci", "vercel", "netlify", "railway",
]

_BEHAVIORAL_KEYWORDS: list[str] = [
    "communication", "teamwork", "leadership", "problem solving",
    "collaboration", "self-motivated", "detail-oriented", "fast-paced",
    "ownership", "initiative", "curiosity", "mentoring", "presentation",
    "critical thinking", "time management", "project management",
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
    ],
    "Tools & Practices": [
        "git", "github", "jira", "postman", "figma",
        "unit testing", "integration testing", "tdd",
        "agile", "scrum", "code review", "system design",
    ],
}


# ═══════════════════════════════════════════════════════════
# INTERNAL: Gemini Call Result
# ═══════════════════════════════════════════════════════════

@dataclass
class _LLMCallResult:
    """Internal tracker for a single Gemini call attempt."""
    data: Any = None
    gemini_used: bool = False
    fallback_used: bool = False
    fallback_reason: str = ""


# ═══════════════════════════════════════════════════════════
# GEMINI CLIENT
# ═══════════════════════════════════════════════════════════

def _get_api_key() -> str | None:
    """Resolve the next available Gemini API key from the pool."""
    from key_pool import get_next_api_key
    return get_next_api_key()


def is_llm_available() -> bool:
    """Check if at least one Gemini API key is configured."""
    from key_pool import get_all_api_keys
    return len(get_all_api_keys()) > 0


def _get_model(api_key: str | None = None):
    try:
        import google.generativeai as genai
        key = api_key or _get_api_key()
        if not key:
            return None
        genai.configure(api_key=key)
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        return None


def _call_gemini(prompt: str, json_mode: bool = True) -> str | None:
    """Call Gemini with automatic key rotation on 429 errors.

    Tries each available API key. If a key hits rate limits, marks it
    as exhausted and moves to the next key. Returns raw text or None.
    """
    from key_pool import get_all_api_keys, mark_key_exhausted
    import time

    keys = get_all_api_keys()
    if not keys:
        return None

    for key in keys:
        model = _get_model(api_key=key)
        if model is None:
            continue

        # Try this key with a single retry (short backoff)
        for attempt in range(2):
            try:
                import google.generativeai as genai
                config_kwargs: dict[str, Any] = {"temperature": 0.3}
                if json_mode:
                    config_kwargs["response_mime_type"] = "application/json"
                config = genai.GenerationConfig(**config_kwargs)
                response = model.generate_content(prompt, generation_config=config)
                return response.text
            except Exception as exc:
                exc_str = str(exc)
                if "429" in exc_str:
                    if attempt == 0:
                        time.sleep(5)  # Brief retry before rotating
                        continue
                    # This key is exhausted — mark it and try the next
                    mark_key_exhausted(key)
                    break
                return None  # Non-quota error — don't retry

    return None  # All keys exhausted


def _parse_json(text: str | None) -> dict | list | None:
    """Parse JSON from Gemini response, tolerating fenced code blocks."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


# ═══════════════════════════════════════════════════════════
# TEXT UTILITIES
# ═══════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _resolve_alias(skill: str) -> str:
    s = skill.strip().lower()
    return _SKILL_ALIASES.get(s, s)


def _extract_skills_from_text(text: str) -> list[str]:
    """Extract known skills from text using multi-word matching."""
    norm = _normalize(text)
    found: list[str] = []
    seen: set[str] = set()

    sorted_skills = sorted(_KNOWN_SKILLS, key=len, reverse=True)
    for skill in sorted_skills:
        pattern = r"(?<![a-z])" + re.escape(skill) + r"(?![a-z])"
        if re.search(pattern, norm):
            canonical = _resolve_alias(skill)
            if canonical not in seen:
                found.append(canonical)
                seen.add(canonical)

    for alias, canonical in _SKILL_ALIASES.items():
        pattern = r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])"
        if re.search(pattern, norm) and canonical not in seen:
            found.append(canonical)
            seen.add(canonical)

    return found


def _find_category(skill: str) -> str:
    s = _resolve_alias(skill)
    for cat, members in _SKILL_CATEGORIES.items():
        if s in members:
            return cat
    return "General"


def _extract_bullets(text: str) -> list[str]:
    """Extract bullet-pointed lines from text."""
    bullets = re.findall(r"[•\-\*]\s*(.+)", text)
    return [b.strip() for b in bullets if len(b.strip()) > 10][:8]


def _detect_projects(text: str) -> list[str]:
    """Detect likely project names from resume text."""
    patterns = [
        r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\s*[\-–—]\s*",
        r"(?:project|built|developed|created|designed)\s+(?:a\s+)?(.+?)(?:\.|,|\n)",
    ]
    projects: list[str] = []
    for p in patterns:
        for m in re.finditer(p, text):
            name = m.group(1).strip()
            if 3 < len(name) < 60 and name not in projects:
                projects.append(name)
    return projects[:6]


def _count_metrics(text: str) -> int:
    """Count quantified achievements (numbers with context)."""
    return len(re.findall(r"\d+[%+x]|\d+\s*(?:users|requests|ms|seconds|projects|stars|views|downloads|contributions)", text.lower()))


def _has_links(text: str) -> bool:
    return bool(re.search(r"https?://|github\.com|linkedin\.com|gitlab\.com|heroku|vercel|netlify|railway", text.lower()))


def _has_deploy_proof(text: str) -> bool:
    return bool(re.search(r"deploy|hosted|live|production|shipped|released|launched", text.lower()))


def _has_collab_language(text: str) -> bool:
    return bool(re.search(r"team|collaborat|standup|sprint|pair|code review|mentored|led|managed|cross-functional", text.lower()))


def _has_learning_signals(text: str) -> bool:
    return bool(re.search(r"certif|coursera|udemy|course|bootcamp|hackathon|contest|competition|self.taught|learning|mooc", text.lower()))


def _infer_seniority(text: str) -> str:
    low = text.lower()
    if re.search(r"senior|lead|principal|staff|architect", low):
        return "senior"
    if re.search(r"mid.?level|3\+\s*years|4\+\s*years|5\+\s*years", low):
        return "mid"
    if re.search(r"junior|entry|1\+\s*year|0.?2\s*years|new\s*grad", low):
        return "entry"
    return "intern"


def _infer_job_title(jd_text: str) -> str:
    """Extract a likely job title from the first few lines."""
    lines = jd_text.strip().split("\n")
    for line in lines[:5]:
        line = line.strip()
        if 5 < len(line) < 80 and not line.startswith(("About", "We ", "Our ", "The ", "Join")):
            cleaned = re.sub(r"[–—\-]\s*(Summer|Fall|Spring|Winter)\s*\d{4}", "", line).strip()
            if cleaned:
                return cleaned
    return "Software Role"


def _dedupe(items: list[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        low = item.lower().strip()
        if low and low not in seen:
            seen.add(low)
            result.append(item.strip())
    return result


# ═══════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════

_ROLE_DNA_PROMPT = """Analyze this Job Description and extract structured role requirements.

Return ONLY a JSON object with exactly these keys:
- "required_skills": list of technical skills explicitly required (lowercase, concise)
- "preferred_skills": list of skills marked preferred or nice-to-have (lowercase)
- "tools": list of specific tools, platforms, or technologies
- "behavioral_traits": list of soft skills or behavioral traits expected
- "role_context": 2-3 sentence summary of the role and team context
- "day_to_day": list of 4-6 daily responsibilities
- "seniority_hint": one of "intern", "entry", "mid", "senior"

Do not add explanatory text. Return valid JSON only.

JOB DESCRIPTION:
{jd_text}"""


_SCORE_PROMPT = """You are evaluating a student resume against a Job Description.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

ROLE REQUIREMENTS (extracted):
Required skills: {required_skills}
Preferred skills: {preferred_skills}

Score the candidate on these 6 dimensions (0-100 each). For EACH dimension, cite specific resume content as evidence.

Return ONLY a JSON object with exactly these keys, each mapping to an object:
- "technical_match": {{"score": int, "why": str, "evidence_found": [str], "missing_proof": [str], "action": str}}
- "evidence_quality": {{"score": int, "why": str, "evidence_found": [str], "missing_proof": [str], "action": str}}
- "communication_clarity": {{"score": int, "why": str, "evidence_found": [str], "missing_proof": [str], "action": str}}
- "role_alignment": {{"score": int, "why": str, "evidence_found": [str], "missing_proof": [str], "action": str}}
- "learning_momentum": {{"score": int, "why": str, "evidence_found": [str], "missing_proof": [str], "action": str}}
- "interview_readiness": {{"score": int, "why": str, "evidence_found": [str], "missing_proof": [str], "action": str}}

For "why": cite specific resume items (project names, skills, metrics).
For "evidence_found": list specific items FROM the resume.
For "missing_proof": list what the Job Description needs but the resume lacks.
For "action": one specific, actionable recommendation.

Return valid JSON only. No markdown. No preamble."""


_GAPS_PROMPT = """Identify skill and evidence gaps between this resume and Job Description.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

REQUIRED SKILLS: {required_skills}
PREFERRED SKILLS: {preferred_skills}

For each gap, classify as:
- "hard_gap": skill completely absent from resume
- "soft_gap": skill mentioned but without evidence (no project, no metric, no outcome)
- "proof_gap": skill used in a project but results are not quantified
- "context_gap": related skill exists but not the exact one needed

Return ONLY a JSON array of objects, each with:
- "name": skill name (lowercase)
- "gap_type": one of hard_gap, soft_gap, proof_gap, context_gap
- "severity": "high" for required skills, "medium" for preferred
- "category": skill category
- "why_it_matters": one sentence on why this gap matters for this role
- "evidence_missing": what resume proof is absent
- "action": specific, concrete next step

Sort by severity (high first), then by gap_type (hard_gap first).
Return valid JSON only."""


_SKILLPRINT_PROMPT = """Create a 5-minute proof-of-skill micro-challenge.

TARGET SKILL: {skill}
ROLE CONTEXT: {role_context}
GAP TYPE: {gap_type}

The challenge must:
- Be solvable in one code block (Python preferred)
- Test what this specific ROLE needs, not generic knowledge
- Be achievable in 5 minutes
- Produce output that demonstrates practical competence

Return ONLY a JSON object with:
- "title": short challenge title
- "challenge_brief": clear problem statement (3-5 sentences, role-specific)
- "deliverables": list of 2-4 concrete things the solution must include
- "rubric": list of 4-6 evaluation criteria
- "starter_prompt": Python starter code with TODOs
- "proof_signal": one sentence on how completion strengthens the resume for this role

Return valid JSON only."""


_ROADMAP_PROMPT = """Create a personalized 30-day learning roadmap for a student.

STUDENT PROFILE:
- Matched skills: {matched_skills}
- Experience level: {seniority}

SKILL GAPS TO CLOSE:
{gaps_text}

TARGET ROLE: {role_context}

Create a 4-week roadmap where each week builds on the previous. Return ONLY a JSON array of 4 objects:
- "week": week number (1-4)
- "goal": one-sentence weekly goal
- "focus_skills": list of 2-3 skills to focus on this week
- "tasks": list of 4-6 concrete tasks (specific, not generic advice)
- "proof_milestone": what artifact proves this week's work (must be resume-addable)
- "resources": list of objects with "name" and "url" (free resources only)

Each week must target different gaps.
Tasks must produce artifacts that strengthen the resume.
Use only free resources (official docs, YouTube, freeCodeCamp, etc).
Return valid JSON only."""


_EVAL_PROMPT = """Evaluate this micro-challenge submission.

CHALLENGE: {title}
SKILL: {skill}

RUBRIC:
{rubric}

SUBMITTED CODE:
```
{code}
```

Return ONLY a JSON object with:
- "score": integer 0-100
- "passed": boolean (true if score >= 60)
- "feedback": 2-3 sentence assessment
- "rubric_results": list of {{"criterion": str, "met": bool}}
- "improvements": list of 1-3 specific suggestions

Be fair but rigorous. Return valid JSON only."""


# ═══════════════════════════════════════════════════════════
# GEMINI-BACKED BUILDERS
# ═══════════════════════════════════════════════════════════

def build_role_dna(resume_text: str, job_description_text: str) -> dict[str, Any]:
    """Extract Role DNA from a Job Description.

    Returns a dict with keys matching RoleDNA fields plus:
        _gemini_used, _fallback_used, _fallback_reason
    """
    result = _try_gemini_role_dna(job_description_text)
    if result.gemini_used:
        out = result.data
    else:
        out = _fallback_role_dna(job_description_text)
    out["_gemini_used"] = result.gemini_used
    out["_fallback_used"] = result.fallback_used
    out["_fallback_reason"] = result.fallback_reason
    return out


def build_score_breakdown(
    resume_text: str,
    job_description_text: str,
    role_dna: dict[str, Any],
) -> dict[str, Any]:
    """Generate evidence-backed score breakdown.

    Returns a dict with:
        dimensions: list of dicts (ScoreDimension-compatible),
        overall_score: int,
        confidence: str,
        _gemini_used, _fallback_used, _fallback_reason
    """
    result = _try_gemini_scores(resume_text, job_description_text, role_dna)
    if result.gemini_used:
        out = result.data
    else:
        out = _fallback_scores(resume_text, job_description_text, role_dna)
    out["_gemini_used"] = result.gemini_used
    out["_fallback_used"] = result.fallback_used
    out["_fallback_reason"] = result.fallback_reason
    return out


def build_gaps(
    resume_text: str,
    job_description_text: str,
    role_dna: dict[str, Any],
) -> dict[str, Any]:
    """Identify skill and evidence gaps.

    Returns a list of dicts (GapItem-compatible) with provenance metadata
    stored in a wrapper dict under key 'items' alongside '_gemini_used', etc.
    For convenience, this function returns the full wrapper dict.
    """
    result = _try_gemini_gaps(resume_text, job_description_text, role_dna)
    if result.gemini_used:
        items = result.data
    else:
        items = _fallback_gaps(resume_text, job_description_text, role_dna)
    return {
        "items": items,
        "_gemini_used": result.gemini_used,
        "_fallback_used": result.fallback_used,
        "_fallback_reason": result.fallback_reason,
    }


def build_skillprint(
    resume_text: str,
    job_description_text: str,
    role_dna: dict[str, Any],
    gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate a SkillPrint micro-challenge for the top gap.

    Returns a dict matching SkillPrint fields plus provenance metadata.
    """
    top_hard = next((g for g in gaps if g.get("gap_type") == "hard_gap"), None)
    if not top_hard and gaps:
        top_hard = gaps[0]

    if not top_hard:
        return {
            "title": "", "challenge_brief": "", "deliverables": [],
            "rubric": [], "starter_prompt": "", "proof_signal": "",
            "_gemini_used": False, "_fallback_used": False, "_fallback_reason": "no_gaps",
        }

    skill = top_hard.get("name", "")
    gap_type = top_hard.get("gap_type", "hard_gap")
    role_context = role_dna.get("role_context", "software engineering role")

    result = _try_gemini_skillprint(skill, role_context, gap_type)
    if result.gemini_used:
        out = result.data
    else:
        out = _fallback_skillprint(skill, role_context)
    out["_gemini_used"] = result.gemini_used
    out["_fallback_used"] = result.fallback_used
    out["_fallback_reason"] = result.fallback_reason
    return out


def build_roadmap(
    resume_text: str,
    job_description_text: str,
    role_dna: dict[str, Any],
    gaps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate a personalized 4-week roadmap.

    Returns a wrapper dict with 'weeks' list and provenance metadata.
    """
    resume_skills = _extract_skills_from_text(resume_text)
    result = _try_gemini_roadmap(resume_skills, role_dna, gaps)
    if result.gemini_used:
        weeks = result.data
    else:
        weeks = _fallback_roadmap(gaps, role_dna)
    return {
        "weeks": weeks,
        "_gemini_used": result.gemini_used,
        "_fallback_used": result.fallback_used,
        "_fallback_reason": result.fallback_reason,
    }


def evaluate_skillprint_submission(code: str, challenge: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a SkillPrint code submission against the challenge rubric."""
    if not code.strip():
        return {
            "score": 0, "passed": False,
            "feedback": "No code submitted.",
            "rubric_results": [], "improvements": ["Submit your solution."],
        }

    rubric_text = "\n".join(f"- {r}" for r in challenge.get("rubric", []))
    raw = _call_gemini(_EVAL_PROMPT.format(
        title=challenge.get("title", ""),
        skill=challenge.get("name", challenge.get("title", "")),
        rubric=rubric_text,
        code=code,
    ))
    parsed = _parse_json(raw)
    if parsed and isinstance(parsed, dict) and "score" in parsed:
        return parsed

    return {
        "score": 30, "passed": False,
        "feedback": "Automatic evaluation unavailable. Review the rubric criteria manually.",
        "rubric_results": [],
        "improvements": ["Ensure your code runs without errors", "Address each rubric criterion"],
    }


# ═══════════════════════════════════════════════════════════
# GEMINI CALL WRAPPERS
# ═══════════════════════════════════════════════════════════

def _try_gemini_role_dna(jd_text: str) -> _LLMCallResult:
    raw = _call_gemini(_ROLE_DNA_PROMPT.format(jd_text=jd_text[:4000]))
    parsed = _parse_json(raw)
    if parsed and isinstance(parsed, dict):
        required_keys = {"required_skills", "role_context"}
        if required_keys.issubset(parsed.keys()):
            return _LLMCallResult(data=parsed, gemini_used=True)
        return _LLMCallResult(
            fallback_used=True,
            fallback_reason="gemini_response_missing_keys",
        )
    reason = "gemini_no_response" if raw is None else "gemini_parse_failed"
    return _LLMCallResult(fallback_used=True, fallback_reason=reason)


def _try_gemini_scores(
    resume_text: str,
    jd_text: str,
    role_dna: dict[str, Any],
) -> _LLMCallResult:
    required = ", ".join(role_dna.get("required_skills", [])[:15])
    preferred = ", ".join(role_dna.get("preferred_skills", [])[:10])
    raw = _call_gemini(_SCORE_PROMPT.format(
        resume_text=resume_text[:3000],
        jd_text=jd_text[:2000],
        required_skills=required,
        preferred_skills=preferred,
    ))
    parsed = _parse_json(raw)
    if parsed and isinstance(parsed, dict) and len(parsed) >= 4:
        dimensions = []
        for key, label in DIMENSION_DEFS:
            entry = parsed.get(key, {})
            if isinstance(entry, dict) and "score" in entry:
                dimensions.append({
                    "key": key,
                    "label": label,
                    "score": max(0, min(100, int(entry["score"]))),
                    "why": entry.get("why", ""),
                    "evidence_found": entry.get("evidence_found", []),
                    "missing_proof": entry.get("missing_proof", []),
                    "action": entry.get("action", ""),
                })
        if len(dimensions) >= 4:
            scores = [d["score"] for d in dimensions]
            overall = round(sum(scores) / len(scores))
            confidence = "high" if overall >= 60 else "medium" if overall >= 35 else "low"
            return _LLMCallResult(
                data={"dimensions": dimensions, "overall_score": overall, "confidence": confidence},
                gemini_used=True,
            )
    reason = "gemini_no_response" if raw is None else "gemini_score_parse_failed"
    return _LLMCallResult(fallback_used=True, fallback_reason=reason)


def _try_gemini_gaps(
    resume_text: str,
    jd_text: str,
    role_dna: dict[str, Any],
) -> _LLMCallResult:
    required = ", ".join(role_dna.get("required_skills", [])[:15])
    preferred = ", ".join(role_dna.get("preferred_skills", [])[:10])
    raw = _call_gemini(_GAPS_PROMPT.format(
        resume_text=resume_text[:3000],
        jd_text=jd_text[:2000],
        required_skills=required,
        preferred_skills=preferred,
    ))
    parsed = _parse_json(raw)
    if parsed and isinstance(parsed, list) and len(parsed) >= 1:
        valid = []
        for item in parsed:
            if isinstance(item, dict) and "name" in item:
                valid.append({
                    "name": item.get("name", ""),
                    "gap_type": item.get("gap_type", "hard_gap"),
                    "severity": item.get("severity", "medium"),
                    "category": item.get("category", ""),
                    "why_it_matters": item.get("why_it_matters", ""),
                    "evidence_missing": item.get("evidence_missing", ""),
                    "action": item.get("action", ""),
                })
        if valid:
            return _LLMCallResult(data=valid, gemini_used=True)
    reason = "gemini_no_response" if raw is None else "gemini_gaps_parse_failed"
    return _LLMCallResult(fallback_used=True, fallback_reason=reason)


def _try_gemini_skillprint(skill: str, role_context: str, gap_type: str) -> _LLMCallResult:
    raw = _call_gemini(_SKILLPRINT_PROMPT.format(
        skill=skill,
        role_context=role_context,
        gap_type=gap_type,
    ))
    parsed = _parse_json(raw)
    if parsed and isinstance(parsed, dict) and "title" in parsed:
        return _LLMCallResult(data=parsed, gemini_used=True)
    reason = "gemini_no_response" if raw is None else "gemini_skillprint_parse_failed"
    return _LLMCallResult(fallback_used=True, fallback_reason=reason)


def _try_gemini_roadmap(
    resume_skills: list[str],
    role_dna: dict[str, Any],
    gaps: list[dict[str, Any]],
) -> _LLMCallResult:
    gaps_text = "\n".join(
        f"- {g['name']} ({g.get('gap_type','gap')}, {g.get('severity','medium')}): {g.get('why_it_matters','')}"
        for g in gaps[:8]
    )
    matched = ", ".join(resume_skills[:10]) or "none identified"
    raw = _call_gemini(_ROADMAP_PROMPT.format(
        matched_skills=matched,
        seniority=role_dna.get("seniority_hint", "intern"),
        gaps_text=gaps_text or "no specific gaps identified",
        role_context=role_dna.get("role_context", "software engineering role"),
    ))
    parsed = _parse_json(raw)
    if parsed and isinstance(parsed, list) and len(parsed) >= 2:
        weeks = []
        for item in parsed:
            if isinstance(item, dict) and "week" in item:
                weeks.append({
                    "week": item.get("week", 1),
                    "goal": item.get("goal", ""),
                    "focus_skills": item.get("focus_skills", []),
                    "tasks": item.get("tasks", []),
                    "proof_milestone": item.get("proof_milestone", ""),
                    "resources": item.get("resources", []),
                })
        if weeks:
            return _LLMCallResult(data=weeks, gemini_used=True)
    reason = "gemini_no_response" if raw is None else "gemini_roadmap_parse_failed"
    return _LLMCallResult(fallback_used=True, fallback_reason=reason)


# ═══════════════════════════════════════════════════════════
# DETERMINISTIC FALLBACKS
# ═══════════════════════════════════════════════════════════

def _fallback_role_dna(jd_text: str) -> dict[str, Any]:
    """Extract Role DNA from Job Description using keyword matching."""
    all_skills = _extract_skills_from_text(jd_text)
    split = max(1, int(len(all_skills) * 0.6))
    required = all_skills[:split]
    preferred = all_skills[split:]

    jd_lower = jd_text.lower()
    tools = [t for t in _TOOL_PATTERNS if t in jd_lower]
    behavioral = [b for b in _BEHAVIORAL_KEYWORDS if b in jd_lower]

    sentences = re.split(r"[.!?]+", jd_text.strip())
    context_parts = [s.strip() for s in sentences[:3] if len(s.strip()) > 15]
    role_context = ". ".join(context_parts) + "." if context_parts else "Details extracted from the Job Description."

    day_to_day = _extract_bullets(jd_text)
    seniority = _infer_seniority(jd_text)

    return {
        "required_skills": _dedupe(required),
        "preferred_skills": _dedupe(preferred),
        "tools": _dedupe(tools),
        "behavioral_traits": _dedupe(behavioral),
        "role_context": role_context,
        "day_to_day": day_to_day,
        "seniority_hint": seniority,
    }


def _fallback_scores(
    resume_text: str,
    jd_text: str,
    role_dna: dict[str, Any],
) -> dict[str, Any]:
    """Generate deterministic score breakdown using heuristics."""
    resume_skills = set(_extract_skills_from_text(resume_text))
    required = set(role_dna.get("required_skills", []))
    preferred = set(role_dna.get("preferred_skills", []))
    all_role = required | preferred

    matched_required = resume_skills & required
    matched_preferred = resume_skills & preferred
    matched_all = resume_skills & all_role

    req_ratio = len(matched_required) / max(1, len(required))
    all_ratio = len(matched_all) / max(1, len(all_role))

    projects = _detect_projects(resume_text)
    metric_count = _count_metrics(resume_text)
    has_links = _has_links(resume_text)
    has_deploy = _has_deploy_proof(resume_text)
    has_collab = _has_collab_language(resume_text)
    has_learning = _has_learning_signals(resume_text)

    # Technical Match
    tech_score = int(req_ratio * 60 + all_ratio * 20 + min(len(projects), 3) * 5)
    tech_score = max(10, min(95, tech_score))
    tech_evidence = [f"{s} ✓" for s in sorted(matched_required)]
    tech_missing = [s for s in sorted(required - resume_skills)]

    # Evidence Quality
    ev_score = 30
    if projects:
        ev_score += min(len(projects), 4) * 8
    if metric_count:
        ev_score += min(metric_count, 5) * 5
    if has_links:
        ev_score += 10
    if has_deploy:
        ev_score += 8
    ev_score = max(10, min(95, ev_score))
    ev_evidence = []
    if projects:
        ev_evidence.append(f"{len(projects)} project(s) detected")
    if metric_count:
        ev_evidence.append(f"{metric_count} quantified metric(s)")
    if has_links:
        ev_evidence.append("External links present")
    ev_missing = []
    if not metric_count:
        ev_missing.append("No quantified metrics or outcomes")
    if not has_deploy:
        ev_missing.append("No deployment or production evidence")

    # Communication Clarity
    word_count = len(resume_text.split())
    bullet_count = len(re.findall(r"[•\-\*]\s", resume_text))
    comm_score = 50
    if 200 < word_count < 1200:
        comm_score += 15
    if bullet_count >= 5:
        comm_score += 10
    if re.search(r"(summary|objective|about)", resume_text.lower()):
        comm_score += 5
    comm_score = max(20, min(90, comm_score))

    # Role Alignment
    align_score = int(all_ratio * 50 + req_ratio * 30)
    if has_collab:
        align_score += 10
    align_score = max(10, min(90, align_score))

    # Learning Momentum
    learn_score = 35
    if has_learning:
        learn_score += 25
    if re.search(r"hackathon|winner|award|achievement|publication", resume_text.lower()):
        learn_score += 15
    if len(projects) >= 3:
        learn_score += 10
    learn_score = max(15, min(95, learn_score))

    # Interview Readiness
    interview_score = 30
    interview_score += int(req_ratio * 25)
    if has_deploy:
        interview_score += 10
    if metric_count >= 2:
        interview_score += 10
    if has_collab:
        interview_score += 8
    interview_score = max(15, min(90, interview_score))

    dimensions = [
        {
            "key": "technical_match", "label": "Technical Match", "score": tech_score,
            "why": f"Matched {len(matched_required)} of {len(required)} required and {len(matched_preferred)} of {len(preferred)} preferred skills.",
            "evidence_found": tech_evidence, "missing_proof": tech_missing,
            "action": f"Focus on the {len(tech_missing)} missing required skill(s): {', '.join(tech_missing[:3])}." if tech_missing else "Strengthen evidence for existing skills.",
        },
        {
            "key": "evidence_quality", "label": "Evidence Quality", "score": ev_score,
            "why": f"Found {len(projects)} project(s) and {metric_count} quantified metric(s).",
            "evidence_found": ev_evidence, "missing_proof": ev_missing,
            "action": "Add measurable outcomes (percentage improvements, user counts, performance gains) to your project descriptions.",
        },
        {
            "key": "communication_clarity", "label": "Communication Clarity", "score": comm_score,
            "why": f"Resume has {word_count} words and {bullet_count} bullet points.",
            "evidence_found": [f"{word_count} words", f"{bullet_count} bullet points"],
            "missing_proof": ["Add a concise professional summary"] if not re.search(r"(summary|objective)", resume_text.lower()) else [],
            "action": "Use strong action verbs at the start of each bullet point and keep descriptions concise.",
        },
        {
            "key": "role_alignment", "label": "Role Alignment", "score": align_score,
            "why": f"Overall skill overlap is {int(all_ratio*100)}% with the role requirements.",
            "evidence_found": [f"{s} ✓" for s in sorted(matched_all)[:6]],
            "missing_proof": [f"Missing: {s}" for s in sorted((required | preferred) - resume_skills)[:4]],
            "action": "Tailor your resume to emphasize the skills and experiences most relevant to this specific role.",
        },
        {
            "key": "learning_momentum", "label": "Learning Momentum", "score": learn_score,
            "why": "Evaluated based on certifications, competitions, project count, and learning signals.",
            "evidence_found": ([f"{len(projects)} projects"] if projects else []) + (["Learning signals detected"] if has_learning else []),
            "missing_proof": (["No certifications or courses mentioned"] if not has_learning else []),
            "action": "Add recent certifications, courses, or hackathon participation to show continuous growth.",
        },
        {
            "key": "interview_readiness", "label": "Interview Readiness", "score": interview_score,
            "why": "Assessed based on demonstrable skills, deployment experience, and collaboration evidence.",
            "evidence_found": (["Deployment proof found"] if has_deploy else []) + (["Collaboration language present"] if has_collab else []),
            "missing_proof": (["No deployment evidence"] if not has_deploy else []) + (["No collaboration examples"] if not has_collab else []),
            "action": "Prepare to discuss your strongest project in detail: architecture decisions, challenges faced, and outcomes achieved.",
        },
    ]

    scores = [d["score"] for d in dimensions]
    overall = round(sum(scores) / len(scores))
    confidence = "high" if overall >= 60 else "medium" if overall >= 35 else "low"

    return {
        "dimensions": dimensions,
        "overall_score": overall,
        "confidence": confidence,
    }


def _fallback_gaps(
    resume_text: str,
    jd_text: str,
    role_dna: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate gaps by comparing role requirements to resume evidence."""
    resume_skills = set(_extract_skills_from_text(resume_text))
    resume_lower = resume_text.lower()
    required = role_dna.get("required_skills", [])
    preferred = role_dna.get("preferred_skills", [])

    claimed_but_unproven = set()
    for skill in resume_skills:
        pattern = re.escape(skill)
        occurrences = len(re.findall(pattern, resume_lower))
        has_project_context = bool(re.search(
            rf"(?:built|developed|implemented|used|deployed|created).*{pattern}|{pattern}.*(?:project|app|system|api|service)",
            resume_lower,
        ))
        if occurrences <= 1 and not has_project_context:
            claimed_but_unproven.add(skill)

    gaps: list[dict[str, Any]] = []

    for skill in required:
        canonical = _resolve_alias(skill)
        cat = _find_category(canonical)
        if canonical in resume_skills and canonical not in claimed_but_unproven:
            continue
        if canonical in claimed_but_unproven:
            gaps.append({
                "name": canonical, "gap_type": "soft_gap", "severity": "high",
                "category": cat,
                "why_it_matters": f"'{canonical}' appears in your resume but lacks supporting evidence.",
                "evidence_missing": f"No project, metric, or outcome demonstrates '{canonical}'. It is listed but unproven.",
                "action": f"Add a project or measurable outcome demonstrating {canonical}.",
            })
        elif any(
            _resolve_alias(rs) in resume_skills
            for rs in _KNOWN_SKILLS
            if _find_category(_resolve_alias(rs)) == cat and _resolve_alias(rs) != canonical
        ):
            related = next(
                (rs for rs in resume_skills if _find_category(rs) == cat and rs != canonical),
                "",
            )
            gaps.append({
                "name": canonical, "gap_type": "context_gap", "severity": "high",
                "category": cat,
                "why_it_matters": f"'{canonical}' is required but you have '{related}' instead — related but not identical.",
                "evidence_missing": f"Resume mentions '{related}' but not '{canonical}'. No direct project or experience using '{canonical}'.",
                "action": f"Learn {canonical} specifically; your {related} background will accelerate this.",
            })
        else:
            gaps.append({
                "name": canonical, "gap_type": "hard_gap", "severity": "high",
                "category": cat,
                "why_it_matters": f"'{canonical}' is a required skill and is completely absent from your resume.",
                "evidence_missing": f"'{canonical}' is not mentioned anywhere in your resume.",
                "action": f"Take a focused crash course on {canonical} and build a small project to demonstrate it.",
            })

    for skill in preferred:
        canonical = _resolve_alias(skill)
        if canonical in resume_skills:
            continue
        cat = _find_category(canonical)
        gaps.append({
            "name": canonical, "gap_type": "hard_gap", "severity": "medium",
            "category": cat,
            "why_it_matters": f"'{canonical}' is a preferred skill that would strengthen your application.",
            "evidence_missing": f"'{canonical}' is not mentioned in your resume.",
            "action": f"Explore {canonical} through a tutorial or side project to differentiate your application.",
        })

    severity_order = {"high": 0, "medium": 1, "low": 2}
    type_order = {"hard_gap": 0, "context_gap": 1, "soft_gap": 2, "proof_gap": 3}
    gaps.sort(key=lambda g: (severity_order.get(g["severity"], 9), type_order.get(g["gap_type"], 9)))
    return gaps


def _fallback_skillprint(skill: str, role_context: str) -> dict[str, Any]:
    """Generate a deterministic skill-specific challenge without Gemini."""
    return {
        "title": f"{skill.title()} Proof Challenge",
        "challenge_brief": (
            f"Write a small program that demonstrates practical competence in {skill}. "
            f"Implement a function that solves a real-world problem relevant to {role_context}. "
            f"Focus on clean code, proper error handling, and correct output."
        ),
        "deliverables": [
            f"A working implementation that uses {skill}",
            "Proper error handling for edge cases",
            "Clean, readable code with comments",
            "At least one test or demonstration of output",
        ],
        "rubric": [
            f"Correct use of {skill} concepts and patterns",
            "Code runs without errors",
            "Handles edge cases gracefully",
            "Clean, readable code structure",
            "Demonstrates practical understanding, not just syntax",
        ],
        "starter_prompt": f"# {skill.title()} Proof Challenge\n# Demonstrate your {skill} competence\n\ndef solve():\n    # TODO: Implement your solution\n    pass\n\n# Test\nsolve()",
        "proof_signal": f"Demonstrates practical {skill} competence through a working implementation relevant to the target role.",
    }


def _fallback_roadmap(
    gaps: list[dict[str, Any]],
    role_dna: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate a 4-week roadmap from actual gaps."""
    if not gaps:
        return [{
            "week": 1, "goal": "Strengthen your profile",
            "focus_skills": [],
            "tasks": [
                "Add README files to all GitHub projects",
                "Deploy at least one project to a public URL",
                "Add GitHub and project links to your resume",
                "Write measurable outcomes for each project bullet point",
            ],
            "proof_milestone": "Updated resume with quantified project descriptions and live links",
            "resources": [
                {"name": "GitHub Pages docs", "url": "https://pages.github.com/"},
                {"name": "Resume action verbs", "url": "https://hbr.org/2023/01/how-to-write-a-great-resume"},
            ],
        }]

    chunk_size = max(1, (len(gaps) + 3) // 4)
    weeks: list[dict[str, Any]] = []

    for i in range(0, min(len(gaps), chunk_size * 4), chunk_size):
        chunk = gaps[i:i + chunk_size]
        week_num = (i // chunk_size) + 1
        skill_names = [g["name"] for g in chunk]
        primary = skill_names[0] if skill_names else "general skills"

        tasks = []
        for g in chunk[:3]:
            s = g["name"]
            gt = g.get("gap_type", "hard_gap")
            if gt == "hard_gap":
                tasks.extend([
                    f"Study {s.title()} fundamentals through official documentation",
                    f"Build a small project demonstrating {s.title()}",
                ])
            elif gt == "soft_gap" or gt == "proof_gap":
                tasks.extend([
                    f"Add measurable outcomes for {s.title()} in your resume",
                    f"Create a focused project that proves {s.title()} competence",
                ])
            elif gt == "context_gap":
                tasks.extend([
                    f"Learn {s.title()} specifically (your related experience will help)",
                    f"Convert an existing project to use {s.title()}",
                ])
            else:
                tasks.append(g.get("action", f"Work on {s.title()}"))

        tasks = _dedupe(tasks)[:6]

        weeks.append({
            "week": week_num,
            "goal": f"Build demonstrable competence in {', '.join(s.title() for s in skill_names[:2])}",
            "focus_skills": skill_names[:3],
            "tasks": tasks,
            "proof_milestone": f"Working project or contribution demonstrating {primary.title()}",
            "resources": [],
        })

    return weeks or [{
        "week": 1, "goal": "Getting started", "focus_skills": [],
        "tasks": ["Review your resume", "Identify areas for improvement"],
        "proof_milestone": "Updated resume draft",
        "resources": [],
    }]
