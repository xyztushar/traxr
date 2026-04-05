"""
TRAXR — Data Models
Structured dataclasses for the entire analysis pipeline.
Single source of truth: every dashboard tab reads from AnalysisResult.
No Streamlit dependency. No UI logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ═══════════════════════════════════════════════════════════
# Role DNA
# ═══════════════════════════════════════════════════════════

@dataclass
class RoleDNA:
    """Structured extraction of a Job Description's requirements."""

    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    behavioral_traits: list[str] = field(default_factory=list)
    role_context: str = ""
    day_to_day: list[str] = field(default_factory=list)
    seniority_hint: str = "intern"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RoleDNA:
        return cls(
            required_skills=d.get("required_skills", []),
            preferred_skills=d.get("preferred_skills", []),
            tools=d.get("tools", []),
            behavioral_traits=d.get("behavioral_traits", []),
            role_context=d.get("role_context", ""),
            day_to_day=d.get("day_to_day", []),
            seniority_hint=d.get("seniority_hint", "intern"),
        )


# ═══════════════════════════════════════════════════════════
# Score Dimension
# ═══════════════════════════════════════════════════════════

@dataclass
class ScoreDimension:
    """One of the six readiness dimensions with evidence-backed reasoning."""

    key: str = ""
    label: str = ""
    score: int = 0
    why: str = ""
    evidence_found: list[str] = field(default_factory=list)
    missing_proof: list[str] = field(default_factory=list)
    action: str = ""

    def __post_init__(self) -> None:
        self.score = max(0, min(100, self.score))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScoreDimension:
        return cls(
            key=d.get("key", ""),
            label=d.get("label", ""),
            score=d.get("score", 0),
            why=d.get("why", ""),
            evidence_found=d.get("evidence_found", []),
            missing_proof=d.get("missing_proof", []),
            action=d.get("action", ""),
        )


# ═══════════════════════════════════════════════════════════
# Gap Item
# ═══════════════════════════════════════════════════════════

@dataclass
class GapItem:
    """A specific skill or evidence gap between resume and role requirements."""

    name: str = ""
    gap_type: str = "hard_gap"      # hard_gap | soft_gap | proof_gap | context_gap
    severity: str = "medium"        # high | medium | low
    category: str = ""
    why_it_matters: str = ""
    evidence_missing: str = ""
    action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GapItem:
        return cls(
            name=d.get("name", ""),
            gap_type=d.get("gap_type", "hard_gap"),
            severity=d.get("severity", "medium"),
            category=d.get("category", ""),
            why_it_matters=d.get("why_it_matters", ""),
            evidence_missing=d.get("evidence_missing", ""),
            action=d.get("action", ""),
        )


# ═══════════════════════════════════════════════════════════
# SkillPrint Challenge
# ═══════════════════════════════════════════════════════════

@dataclass
class SkillPrint:
    """A micro-challenge designed to prove a claimed or missing skill."""

    title: str = ""
    challenge_brief: str = ""
    deliverables: list[str] = field(default_factory=list)
    rubric: list[str] = field(default_factory=list)
    starter_prompt: str = ""
    proof_signal: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SkillPrint:
        return cls(
            title=d.get("title", ""),
            challenge_brief=d.get("challenge_brief", ""),
            deliverables=d.get("deliverables", []),
            rubric=d.get("rubric", []),
            starter_prompt=d.get("starter_prompt", ""),
            proof_signal=d.get("proof_signal", ""),
        )


# ═══════════════════════════════════════════════════════════
# Roadmap Week
# ═══════════════════════════════════════════════════════════

@dataclass
class RoadmapWeek:
    """One week of the personalized 30-day readiness roadmap."""

    week: int = 1
    goal: str = ""
    focus_skills: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    proof_milestone: str = ""
    resources: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RoadmapWeek:
        return cls(
            week=d.get("week", 1),
            goal=d.get("goal", ""),
            focus_skills=d.get("focus_skills", []),
            tasks=d.get("tasks", []),
            proof_milestone=d.get("proof_milestone", ""),
            resources=d.get("resources", []),
        )


# ═══════════════════════════════════════════════════════════
# Analysis Result — Single Source of Truth
# ═══════════════════════════════════════════════════════════

DIMENSION_KEYS = [
    "technical_match",
    "evidence_quality",
    "communication_clarity",
    "role_alignment",
    "learning_momentum",
    "interview_readiness",
]

DIMENSION_LABELS = {
    "technical_match": "Technical Match",
    "evidence_quality": "Evidence Quality",
    "communication_clarity": "Communication Clarity",
    "role_alignment": "Role Alignment",
    "learning_momentum": "Learning Momentum",
    "interview_readiness": "Interview Readiness",
}


@dataclass
class AnalysisResult:
    """Complete analysis output. Every dashboard tab reads from this object.

    Session state contract: app.py stores this as st.session_state.result.
    No other session keys are needed for analysis data.
    """

    # ── Identity ──────────────────────────────────────────
    user_name: str = ""

    # ── Inputs ────────────────────────────────────────────
    resume_text: str = ""
    job_description_text: str = ""
    job_title: str = ""
    role_summary: str = ""
    is_demo: bool = False
    input_hash: str = ""

    # ── Role DNA ──────────────────────────────────────────
    role_dna: RoleDNA = field(default_factory=RoleDNA)

    # ── Scores ────────────────────────────────────────────
    score_breakdown: list[ScoreDimension] = field(default_factory=list)
    overall_score: int = 0
    confidence: str = "low"             # low | medium | high

    # ── Skills ────────────────────────────────────────────
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)

    # ── Gaps ──────────────────────────────────────────────
    gaps: list[GapItem] = field(default_factory=list)

    # ── SkillPrint ────────────────────────────────────────
    skillprint: SkillPrint = field(default_factory=SkillPrint)

    # ── Roadmap ───────────────────────────────────────────
    roadmap: list[RoadmapWeek] = field(default_factory=list)

    # ── Provenance ────────────────────────────────────────
    gemini_used: bool = False
    fallback_used: bool = False
    fallback_reasons: list[str] = field(default_factory=list)
    analysis_version: str = "2.0"

    def __post_init__(self) -> None:
        self.overall_score = max(0, min(100, self.overall_score))

    @property
    def dimensions_dict(self) -> dict[str, int]:
        """Shortcut: {label: score} for radar charts and progress bars."""
        return {d.label: d.score for d in self.score_breakdown}

    @property
    def top_gaps(self) -> list[GapItem]:
        """Gaps sorted by severity: high first."""
        order = {"high": 0, "medium": 1, "low": 2}
        return sorted(self.gaps, key=lambda g: order.get(g.severity, 9))

    def to_dict(self) -> dict[str, Any]:
        """Full serialization for debugging and export."""
        return {
            "user_name": self.user_name,
            "resume_text": self.resume_text[:200] + "…" if len(self.resume_text) > 200 else self.resume_text,
            "job_description_text": self.job_description_text[:200] + "…" if len(self.job_description_text) > 200 else self.job_description_text,
            "job_title": self.job_title,
            "role_summary": self.role_summary,
            "is_demo": self.is_demo,
            "input_hash": self.input_hash,
            "role_dna": self.role_dna.to_dict(),
            "score_breakdown": [d.to_dict() for d in self.score_breakdown],
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "gaps": [g.to_dict() for g in self.gaps],
            "skillprint": self.skillprint.to_dict(),
            "roadmap": [w.to_dict() for w in self.roadmap],
            "gemini_used": self.gemini_used,
            "fallback_used": self.fallback_used,
            "fallback_reasons": self.fallback_reasons,
            "analysis_version": self.analysis_version,
        }
