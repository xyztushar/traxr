"""
TRAXR — LLM Client (Google Gemini)
Handles Role DNA extraction, roadmap generation, and SkillPrint evaluation.
Falls back to demodata.py on any failure.
"""

from __future__ import annotations

import json
import os
import re
import streamlit as st
from pydantic import BaseModel, Field, ValidationError

import demodata

# ── Pydantic Schema ──────────────────────────────────────

class RoleDNA(BaseModel):
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    behavioral_requirements: list[str] = Field(default_factory=list)
    role_context: str = ""
    experience_level: str = "intern"
    day_to_day_activities: list[str] = Field(default_factory=list)


# ── Gemini Client Setup ──────────────────────────────────

def _get_api_key() -> str | None:
    """Retrieve Gemini API key from st.secrets or environment."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


def _get_model():
    """Lazy-load and cache the Gemini generative model."""
    try:
        import google.generativeai as genai
        key = _get_api_key()
        if not key:
            return None
        genai.configure(api_key=key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None


def _call_gemini(prompt: str, json_mode: bool = True) -> str | None:
    """Call Gemini and return the text response. Returns None on failure."""
    model = _get_model()
    if model is None:
        return None
    try:
        import google.generativeai as genai
        config = {}
        if json_mode:
            config = genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.3,
            )
        else:
            config = genai.GenerationConfig(temperature=0.4)
        response = model.generate_content(prompt, generation_config=config)
        return response.text
    except Exception:
        return None


def _extract_json(text: str) -> dict | list | None:
    """Extract JSON from a response that may contain markdown fences."""
    if not text:
        return None
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from code fences
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


# ── Role DNA Extraction ──────────────────────────────────

_ROLE_DNA_PROMPT = """You are an expert career analyst. Analyze the following job description and extract structured information.

Return a JSON object with exactly these keys:
- "required_skills": list of technical skills explicitly required
- "preferred_skills": list of skills marked as preferred/nice-to-have
- "tools": list of specific tools, platforms, or technologies mentioned
- "behavioral_requirements": list of soft skills or behavioral traits expected
- "role_context": a 2-3 sentence summary of what the role involves and team context
- "experience_level": one of "intern", "entry", "mid", "senior"
- "day_to_day_activities": list of daily responsibilities

Keep skill names lowercase and concise (e.g., "python" not "Python programming language").

JOB DESCRIPTION:
{jd_text}
"""


@st.cache_data(ttl=600, show_spinner=False)
def extract_role_dna(jd_text: str) -> dict:
    """Extract Role DNA from a job description using Gemini. Falls back to demo."""
    raw = _call_gemini(_ROLE_DNA_PROMPT.format(jd_text=jd_text))
    parsed = _extract_json(raw)
    if parsed and isinstance(parsed, dict):
        try:
            dna = RoleDNA(**parsed)
            return dna.model_dump()
        except ValidationError:
            # Try to salvage what we can
            for key in list(parsed.keys()):
                if key not in RoleDNA.model_fields:
                    del parsed[key]
            try:
                dna = RoleDNA(**parsed)
                return dna.model_dump()
            except ValidationError:
                pass
    return dict(demodata.DEMO_ROLE_DNA)


# ── Roadmap Generation ────────────────────────────────────

_ROADMAP_PROMPT = """You are a career coach creating a 30-day learning roadmap for a student.

The student has these skill gaps for a {role_context} role:
{gaps_text}

Create a 4-week roadmap. Return a JSON array of 4 objects, each with:
- "week": week number (1-4)
- "title": short title for the week
- "days": array of day blocks, each with:
  - "day": day range (e.g., "1-2")
  - "goal": what to achieve
  - "tasks": list of concrete tasks
  - "resources": list of objects with "name" and "url" (free resources only)
  - "proof": what artifact proves completion

Make it specific to the gaps, not generic advice. Use only free resources.
"""


@st.cache_data(ttl=600, show_spinner=False)
def generate_roadmap(gaps: list[dict], role_context: str = "") -> list[dict]:
    """Generate a 30-day roadmap from gaps. Falls back to demo."""
    gaps_text = "\n".join(
        f"- {g['skill']} ({g['type']} gap, {g['severity']} severity): {g['explanation']}"
        for g in gaps[:7]
    )
    raw = _call_gemini(_ROADMAP_PROMPT.format(role_context=role_context, gaps_text=gaps_text))
    parsed = _extract_json(raw)
    if parsed and isinstance(parsed, list) and len(parsed) >= 2:
        return parsed
    return list(demodata.DEMO_ROADMAP)


# ── SkillPrint Evaluation ─────────────────────────────────

_SKILLPRINT_PROMPT = """You are a code reviewer evaluating a micro-challenge submission.

CHALLENGE: {title}
SKILL BEING TESTED: {skill}

RUBRIC:
{rubric}

SUBMITTED CODE:
```
{code}
```

Evaluate the submission against each rubric item. Return a JSON object with:
- "score": integer 0-100
- "passed": boolean (true if score >= 60)
- "feedback": 2-3 sentence overall assessment
- "rubric_results": list of objects with "criterion" and "met" (boolean)
- "improvements": list of 1-3 specific improvement suggestions

Be fair but rigorous. Give partial credit for reasonable attempts.
"""


def evaluate_skillprint(code: str, challenge: dict) -> dict:
    """Evaluate a SkillPrint micro-challenge submission."""
    if not code.strip():
        return {"score": 0, "passed": False, "feedback": "No code submitted.",
                "rubric_results": [], "improvements": ["Submit your solution to get evaluated."]}

    rubric_text = "\n".join(f"- {r}" for r in challenge.get("rubric", []))
    raw = _call_gemini(_SKILLPRINT_PROMPT.format(
        title=challenge.get("title", ""),
        skill=challenge.get("skill", ""),
        rubric=rubric_text,
        code=code,
    ))
    parsed = _extract_json(raw)
    if parsed and isinstance(parsed, dict) and "score" in parsed:
        return parsed

    # Conservative fallback
    return {
        "score": 35,
        "passed": False,
        "feedback": "Could not evaluate automatically. Consider reviewing the rubric yourself.",
        "rubric_results": [],
        "improvements": ["Ensure your code compiles/runs", "Address each rubric criterion"],
    }


# ── SkillPrint Challenge Generation ───────────────────────

_CHALLENGE_PROMPT = """Create a 5-minute coding micro-challenge for the skill: {skill}

The challenge should be:
- Solvable in a single code block (Python preferred)
- Testable without running infrastructure
- Focused on demonstrating practical knowledge

Return a JSON object with:
- "skill": the skill being tested
- "title": short challenge title
- "description": clear problem statement (3-5 sentences)
- "starter_code": Python starter code with TODOs
- "rubric": list of 4-6 evaluation criteria
- "time_estimate": "5 minutes"
- "max_score_boost": 8
"""


def generate_skillprint_challenge(skill: str) -> dict:
    """Generate a SkillPrint micro-challenge for a given skill."""
    raw = _call_gemini(_CHALLENGE_PROMPT.format(skill=skill))
    parsed = _extract_json(raw)
    if parsed and isinstance(parsed, dict) and "title" in parsed:
        parsed["skill"] = skill
        return parsed
    return dict(demodata.DEMO_SKILLPRINT_CHALLENGE)


def is_llm_available() -> bool:
    """Quick check if Gemini API is configured."""
    return _get_api_key() is not None
