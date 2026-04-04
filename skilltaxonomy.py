"""
TRAXR — Curated Skill Taxonomy & Ontology
Maps skill categories, canonical names, aliases, and role-specific weight presets.
"""

from __future__ import annotations

# ── Canonical Skill Taxonomy ─────────────────────────────

SKILL_TAXONOMY: dict[str, list[str]] = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c",
        "go", "rust", "ruby", "php", "swift", "kotlin", "scala", "r",
        "matlab", "dart", "shell scripting",
    ],
    "Web & Frontend": [
        "react", "angular", "vue.js", "svelte", "next.js", "html", "css",
        "tailwind css", "bootstrap", "sass", "webpack", "vite", "redux",
        "jquery", "responsive design",
    ],
    "Backend & Infrastructure": [
        "node.js", "express.js", "flask", "django", "fastapi", "spring boot",
        "rest apis", "graphql", "grpc", "microservices", "websockets",
        "rabbitmq", "kafka", "nginx",
    ],
    "Data & ML": [
        "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
        "elasticsearch", "pandas", "numpy", "scikit-learn", "tensorflow",
        "pytorch", "keras", "machine learning", "deep learning",
        "natural language processing", "computer vision", "data analysis",
        "apache spark", "data pipelines",
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "ci/cd", "jenkins", "github actions", "gitlab ci", "linux",
        "ansible", "prometheus", "grafana",
    ],
    "Tools & Practices": [
        "git", "github", "jira", "postman", "swagger", "figma",
        "unit testing", "integration testing", "tdd",
        "agile", "scrum", "kanban", "code review", "system design",
        "design patterns",
    ],
    "Soft Skills": [
        "leadership", "communication", "teamwork", "problem solving",
        "critical thinking", "time management", "project management",
        "mentoring", "presentation skills",
    ],
}

# ── Alias → Canonical ────────────────────────────────────

SKILL_ALIASES: dict[str, str] = {
    "js": "javascript", "ts": "typescript", "py": "python",
    "cpp": "c++", "csharp": "c#", "golang": "go",
    "node": "node.js", "nodejs": "node.js",
    "react.js": "react", "reactjs": "react",
    "vue": "vue.js", "vuejs": "vue.js",
    "nextjs": "next.js",
    "expressjs": "express.js", "express": "express.js",
    "spring": "spring boot",
    "fast api": "fastapi",
    "tailwind": "tailwind css",
    "postgres": "postgresql", "mongo": "mongodb",
    "ml": "machine learning", "dl": "deep learning",
    "nlp": "natural language processing", "cv": "computer vision",
    "sklearn": "scikit-learn", "sk-learn": "scikit-learn",
    "k8s": "kubernetes", "tf": "terraform",
    "ci cd": "ci/cd", "cicd": "ci/cd",
    "gh actions": "github actions",
    "rest": "rest apis", "rest api": "rest apis", "restful": "rest apis",
    "restful apis": "rest apis",
}

# ── Role Weight Presets ──────────────────────────────────

DIMENSION_NAMES = [
    "Technical Skill Match",
    "Evidence Quality",
    "Communication Clarity",
    "Role Alignment",
    "Learning Momentum",
    "Interview Readiness",
]

ROLE_WEIGHTS: dict[str, dict[str, float]] = {
    "SDE / Backend": {
        "Technical Skill Match": 0.30, "Evidence Quality": 0.25,
        "Communication Clarity": 0.10, "Role Alignment": 0.15,
        "Learning Momentum": 0.10, "Interview Readiness": 0.10,
    },
    "Frontend / UI": {
        "Technical Skill Match": 0.25, "Evidence Quality": 0.25,
        "Communication Clarity": 0.15, "Role Alignment": 0.15,
        "Learning Momentum": 0.10, "Interview Readiness": 0.10,
    },
    "Data / ML": {
        "Technical Skill Match": 0.30, "Evidence Quality": 0.25,
        "Communication Clarity": 0.10, "Role Alignment": 0.15,
        "Learning Momentum": 0.10, "Interview Readiness": 0.10,
    },
    "DevOps / Cloud": {
        "Technical Skill Match": 0.30, "Evidence Quality": 0.20,
        "Communication Clarity": 0.10, "Role Alignment": 0.15,
        "Learning Momentum": 0.15, "Interview Readiness": 0.10,
    },
    "General": {
        "Technical Skill Match": 0.25, "Evidence Quality": 0.20,
        "Communication Clarity": 0.15, "Role Alignment": 0.15,
        "Learning Momentum": 0.10, "Interview Readiness": 0.15,
    },
}

# ── Helpers ───────────────────────────────────────────────

def resolve_alias(skill: str) -> str:
    """Resolve a skill name to its canonical form."""
    return SKILL_ALIASES.get(skill.strip().lower(), skill.strip().lower())


def get_all_skills_flat() -> set[str]:
    """Return set of all canonical skill names."""
    skills: set[str] = set()
    for cat_skills in SKILL_TAXONOMY.values():
        skills.update(cat_skills)
    return skills


def find_category(skill: str) -> str | None:
    """Find which category a canonical skill belongs to."""
    s = resolve_alias(skill)
    for cat, skills in SKILL_TAXONOMY.items():
        if s in skills:
            return cat
    return None
