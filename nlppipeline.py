"""
TRAXR — NLP Pipeline
Text extraction, skill detection, semantic matching, and profile signal analysis.
Pure processing module: no Streamlit, no UI logic.

Public API (PDF extraction — called directly by app.py for per-stage spinner control):
    extract_pdf_text_basic(uploaded_file) -> str
    is_text_low_quality(text: str) -> bool
    render_pdf_pages_for_ocr(uploaded_file) -> list[bytes]
    extract_text_with_gemini_from_pages(page_images, api_key) -> str
    normalize_resume_text(text: str) -> str
    extract_resume_text(uploaded_file, api_key) -> tuple[str, dict]  (non-UI orchestrator)

Public API (skills & signals — unchanged):
    extract_skills(text: str) -> list[str]
    semantic_match_score(resume_skills: list[str], jd_skills: list[str]) -> float
    extract_profile_signals(resume_text: str) -> dict
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from typing import Any, BinaryIO

import gemini_client

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════

_MIN_USEFUL_TEXT_LENGTH = 100

# Canonical skill taxonomy — longest-match-first extraction
_SKILL_TAXONOMY: list[str] = sorted([
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "c", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "dart", "lua", "perl",
    "haskell", "elixir", "clojure", "shell scripting", "bash", "powershell",
    # Web & Frontend
    "react", "angular", "vue.js", "svelte", "next.js", "nuxt.js", "gatsby",
    "html", "css", "tailwind css", "bootstrap", "sass", "less",
    "webpack", "vite", "rollup", "redux", "zustand", "mobx",
    "jquery", "three.js", "d3.js",
    # Backend & APIs
    "node.js", "express.js", "flask", "django", "fastapi", "spring boot",
    "spring", "asp.net", "ruby on rails", "laravel", "gin", "fiber",
    "rest apis", "graphql", "grpc", "websockets", "microservices",
    "serverless", "lambda",
    # Databases
    "sql", "nosql", "mongodb", "postgresql", "mysql", "sqlite",
    "redis", "cassandra", "dynamodb", "elasticsearch", "neo4j",
    "firebase", "supabase",
    # Data & ML
    "pandas", "numpy", "scipy", "scikit-learn", "tensorflow", "pytorch",
    "keras", "opencv", "hugging face", "langchain", "openai api",
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "data analysis", "data engineering",
    "apache spark", "apache kafka", "airflow", "dbt",
    "tableau", "power bi", "matplotlib", "seaborn", "plotly",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "ansible", "pulumi", "cloudformation",
    "ci/cd", "jenkins", "github actions", "gitlab ci", "circleci",
    "linux", "nginx", "apache",
    "prometheus", "grafana", "datadog", "new relic",
    "aws lambda", "aws ec2", "aws s3", "aws rds",
    # Tools & Practices
    "git", "github", "gitlab", "bitbucket", "jira", "confluence",
    "postman", "swagger", "figma", "sketch", "adobe xd",
    "vs code", "intellij", "vim",
    "unit testing", "integration testing", "e2e testing", "tdd", "bdd",
    "jest", "pytest", "mocha", "cypress", "selenium", "playwright",
    "agile", "scrum", "kanban", "code review", "system design",
    "design patterns", "clean architecture", "solid principles",
    # Data Formats & Messaging
    "json", "xml", "yaml", "protobuf",
    "rabbitmq", "kafka", "celery", "bull",
    # Mobile
    "react native", "flutter", "android", "ios", "xcode",
    "swiftui", "jetpack compose",
], key=len, reverse=True)

_SKILL_ALIASES: dict[str, str] = {
    "js": "javascript", "ts": "typescript", "py": "python",
    "cpp": "c++", "golang": "go", "node": "node.js", "nodejs": "node.js",
    "react.js": "react", "reactjs": "react", "vue": "vue.js", "vuejs": "vue.js",
    "nextjs": "next.js", "nuxtjs": "nuxt.js",
    "expressjs": "express.js", "express": "express.js",
    "spring": "spring boot", "fast api": "fastapi", "fastapi": "fastapi",
    "rails": "ruby on rails", "ror": "ruby on rails",
    "tailwind": "tailwind css", "tw": "tailwind css",
    "postgres": "postgresql", "psql": "postgresql",
    "mongo": "mongodb", "elastic": "elasticsearch",
    "ml": "machine learning", "dl": "deep learning",
    "nlp": "natural language processing", "cv": "computer vision",
    "sklearn": "scikit-learn", "sk-learn": "scikit-learn",
    "k8s": "kubernetes", "tf": "terraform",
    "ci cd": "ci/cd", "cicd": "ci/cd",
    "ci/cd pipelines": "ci/cd", "ci/cd pipeline": "ci/cd",
    "rest": "rest apis", "rest api": "rest apis",
    "restful": "rest apis", "restful apis": "rest apis",
    "gh actions": "github actions", "gha": "github actions",
    "rn": "react native",
}

_ACTION_VERBS: set[str] = {
    "built", "developed", "designed", "implemented", "created", "deployed",
    "automated", "integrated", "led", "managed", "optimized", "reduced",
    "improved", "achieved", "delivered", "launched", "scaled", "architected",
    "collaborated", "mentored", "analyzed", "engineered", "configured",
    "established", "maintained", "migrated", "refactored", "streamlined",
    "orchestrated", "spearheaded", "authored", "contributed", "resolved",
    "debugged", "tested", "monitored", "provisioned", "containerized",
}


# ═══════════════════════════════════════════════════════════
# PRIVATE: LAZY LOADERS
# ═══════════════════════════════════════════════════════════

_nlp_cache = None
_sbert_cache = None


def _get_nlp():
    """Lazy-load spaCy pipeline. Returns None if unavailable."""
    global _nlp_cache
    if _nlp_cache is not None:
        return _nlp_cache
    try:
        import spacy
        _nlp_cache = spacy.load("en_core_web_sm")
        return _nlp_cache
    except (ImportError, OSError):
        return None


def _get_sbert():
    """Lazy-load sentence-transformers model. Returns None if unavailable."""
    global _sbert_cache
    if _sbert_cache is not None:
        return _sbert_cache
    try:
        from sentence_transformers import SentenceTransformer
        _sbert_cache = SentenceTransformer("all-MiniLM-L6-v2")
        return _sbert_cache
    except (ImportError, OSError):
        return None


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


# ═══════════════════════════════════════════════════════════
# PUBLIC: PDF EXTRACTION PIPELINE
# These stage functions are called directly by app.py for
# per-stage st.spinner() control.
# ═══════════════════════════════════════════════════════════

_MIN_OCR_TEXT = 100         # minimum OCR chars to consider usable
_LOW_QUALITY_LENGTH = 300   # below this, trigger OCR
_LOW_QUALITY_ALPHA = 0.45   # alpha ratio below this, trigger OCR
_LOW_QUALITY_TOKENS = 40    # fewer meaningful tokens, trigger OCR
_JUNK_RATIO = 0.05          # junk char ratio above this, trigger OCR
_MAX_OCR_PAGES = 3          # default pages to render for OCR
_MAX_OCR_PAGES_ABS = 4      # absolute max pages for OCR
_RENDER_DPI = 200           # OCR rendering DPI

_RESUME_SECTION_HEADERS = {
    "education", "experience", "skills", "projects", "summary",
    "work", "certifications", "achievements", "objective", "awards",
    "technical", "publications", "research", "volunteer",
    "extracurricular", "profile", "about",
}


def extract_pdf_text_basic(uploaded_file: BinaryIO) -> str:
    """Stage A: Extract text from a PDF using pypdf.

    Resets file pointer before reading. Returns raw extracted text
    (not yet normalized). Returns empty string on any failure.
    """
    try:
        import pypdf
    except ImportError:
        logger.warning("pypdf is not installed.")
        return ""

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        raw_bytes = uploaded_file.getvalue()
    except AttributeError:
        try:
            uploaded_file.seek(0)
            raw_bytes = uploaded_file.read()
        except Exception as exc:
            logger.warning("Could not read uploaded file: %s", exc)
            return ""

    if not raw_bytes:
        return ""

    try:
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
    except Exception as exc:
        logger.warning("pypdf could not read PDF: %s", exc)
        return ""

    if len(reader.pages) == 0:
        return ""

    page_texts: list[str] = []
    for page in reader.pages:
        try:
            raw = page.extract_text() or ""
            page_texts.append(raw)
        except Exception:
            page_texts.append("")

    return "\n\n".join(page_texts)


def is_text_low_quality(text: str) -> bool:
    """Quality gate: returns True if text should trigger OCR fallback.

    Triggers if ANY of these are true:
    - stripped text is shorter than 300 chars
    - alpha ratio is below 0.45
    - fewer than 40 meaningful tokens (len > 1)
    - high junk/replacement character ratio (> 5%)
    - no recognizable resume section headers found
    """
    stripped = text.strip()

    # Empty or very short
    if len(stripped) < _LOW_QUALITY_LENGTH:
        return True

    # Alpha ratio too low (metadata noise, binary junk)
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if alpha_count / max(1, len(stripped)) < _LOW_QUALITY_ALPHA:
        return True

    # Too few meaningful tokens
    tokens = [w for w in stripped.split() if len(w) > 1]
    if len(tokens) < _LOW_QUALITY_TOKENS:
        return True

    # High junk character ratio
    junk_count = stripped.count("\ufffd") + stripped.count("\uFFFD")
    if junk_count > len(stripped) * _JUNK_RATIO:
        return True

    # No resume-like structure
    lower = stripped.lower()
    has_section = any(header in lower for header in _RESUME_SECTION_HEADERS)
    if not has_section:
        return True

    return False


def render_pdf_pages_for_ocr(uploaded_file: BinaryIO) -> list[bytes]:
    """Render PDF pages to PNG images using pypdfium2.

    Returns list of PNG byte arrays (first 3 pages, max 4).
    Returns empty list on failure.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.warning("pypdfium2 is not installed for PDF rendering.")
        return []

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        raw_bytes = uploaded_file.getvalue()
    except AttributeError:
        try:
            uploaded_file.seek(0)
            raw_bytes = uploaded_file.read()
        except Exception as exc:
            logger.warning("Could not read uploaded file for rendering: %s", exc)
            return []

    if not raw_bytes:
        return []

    try:
        pdf = pdfium.PdfDocument(raw_bytes)
    except Exception as exc:
        logger.warning("pypdfium2 could not open PDF: %s", exc)
        return []

    images: list[bytes] = []
    pages_to_render = min(len(pdf), _MAX_OCR_PAGES)
    scale = _RENDER_DPI / 72  # 200 DPI → ~2.78x

    for i in range(pages_to_render):
        try:
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            images.append(buf.getvalue())
        except Exception as exc:
            logger.warning("Failed to render page %d: %s", i + 1, exc)
            continue

    try:
        pdf.close()
    except Exception:
        pass

    return images


def extract_text_with_gemini_from_pages(
    page_images: list[bytes], api_key: str
) -> str:
    """OCR each page image via Gemini, rotating API keys on 429 errors.

    Returns merged text. Returns empty string if all keys/pages fail.
    """
    if not page_images:
        return ""

    from key_pool import get_all_api_keys, mark_key_exhausted

    # Build ordered key list: provided key first, then others from pool
    all_keys = get_all_api_keys()
    ordered_keys = [api_key] if api_key else []
    for k in all_keys:
        if k not in ordered_keys:
            ordered_keys.append(k)

    if not ordered_keys:
        return ""

    # Track current key/model — rotate on 429
    current_key_idx = 0
    model = gemini_client.get_gemini_model(ordered_keys[current_key_idx])

    page_texts: list[str] = []
    for i, img_bytes in enumerate(page_images):
        extracted = ""
        if model is not None:
            extracted = gemini_client.ocr_page_image(model, img_bytes)

        # If OCR returned empty and we have more keys, try rotating
        if not extracted or not extracted.strip():
            rotated = False
            while current_key_idx < len(ordered_keys) - 1:
                mark_key_exhausted(ordered_keys[current_key_idx])
                current_key_idx += 1
                logger.info("Rotating to API key ...%s for OCR",
                            ordered_keys[current_key_idx][-6:])
                model = gemini_client.get_gemini_model(ordered_keys[current_key_idx])
                if model is not None:
                    extracted = gemini_client.ocr_page_image(model, img_bytes)
                    if extracted and extracted.strip():
                        rotated = True
                        break
            if not rotated:
                logger.info("Gemini OCR page %d: all keys exhausted.", i + 1)

        if extracted and extracted.strip() and extracted.strip() != "[BLANK PAGE]":
            page_texts.append(extracted.strip())
            logger.info("Gemini OCR page %d: %d chars.", i + 1, len(extracted))
        else:
            logger.info("Gemini OCR page %d: blank or failed.", i + 1)

    return "\n\n".join(page_texts)


def normalize_resume_text(text: str) -> str:
    """Normalize extracted resume text for the analysis pipeline.

    - Unicode NFKC normalization
    - Collapse 3+ consecutive blank lines to 2
    - Preserve bullet points, section headings, URLs, emails, metrics
    - Strip leading/trailing whitespace
    - Does NOT collapse all whitespace (preserves resume structure)
    """
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces/tabs on a single line (but preserve newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip trailing whitespace from each line
    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]
    text = "\n".join(lines)

    return text.strip()


def _compute_quality_score(text: str) -> float:
    """Compute a simple 0.0–1.0 quality score for extracted text."""
    if not text.strip():
        return 0.0

    # Length score (40% weight)
    length_score = min(1.0, len(text.strip()) / 2000)

    # Alpha ratio (30% weight)
    alpha_count = sum(1 for c in text if c.isalpha())
    alpha_score = alpha_count / max(1, len(text))

    # Section headers (30% weight)
    lower = text.lower()
    header_count = sum(1 for h in _RESUME_SECTION_HEADERS if h in lower)
    header_score = min(1.0, header_count / 4)

    return round(length_score * 0.4 + alpha_score * 0.3 + header_score * 0.3, 3)


def _count_pdf_pages(uploaded_file: BinaryIO) -> int:
    """Count pages in a PDF using pypdf. Returns 0 on failure."""
    try:
        import pypdf
        uploaded_file.seek(0)
        raw_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else uploaded_file.read()
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
        return len(reader.pages)
    except Exception:
        return 0


def extract_resume_text(
    uploaded_file: BinaryIO, api_key: str | None = None
) -> tuple[str, dict]:
    """Non-UI orchestrator: basic extraction → quality gate → OCR fallback → normalize.

    NOT called by app.py's upload flow (app.py calls stage functions directly
    for per-stage st.spinner() control). Exists for programmatic callers,
    tests, scripts, and non-Streamlit contexts.

    Returns:
        (extracted_text, metadata_dict)
    """
    # Stage A: basic extraction
    raw_text = extract_pdf_text_basic(uploaded_file)
    low_quality = is_text_low_quality(raw_text)
    page_count = _count_pdf_pages(uploaded_file)

    # Good quality — return directly
    if not low_quality and raw_text.strip():
        text = normalize_resume_text(raw_text)
        return (text, {
            "method": "pypdf",
            "used_ocr": False,
            "pages_processed": page_count,
            "chars_extracted": len(text),
            "quality_score": _compute_quality_score(text),
            "error": None,
        })

    # Stage B: OCR fallback
    if low_quality and api_key:
        uploaded_file.seek(0)
        page_images = render_pdf_pages_for_ocr(uploaded_file)
        ocr_text = extract_text_with_gemini_from_pages(page_images, api_key)

        if ocr_text and len(ocr_text.strip()) >= _MIN_OCR_TEXT:
            text = normalize_resume_text(ocr_text)
            return (text, {
                "method": "gemini_ocr",
                "used_ocr": True,
                "pages_processed": len(page_images),
                "chars_extracted": len(text),
                "quality_score": _compute_quality_score(text),
                "error": None,
            })

        # OCR failed but we have partial text
        if raw_text.strip() and len(raw_text.strip()) >= 50:
            text = normalize_resume_text(raw_text)
            return (text, {
                "method": "pypdf",
                "used_ocr": True,
                "pages_processed": page_count,
                "chars_extracted": len(text),
                "quality_score": _compute_quality_score(text),
                "error": "OCR fallback failed. Using partial text extraction.",
            })

    # Total failure
    return ("", {
        "method": "none",
        "used_ocr": bool(api_key and low_quality),
        "pages_processed": 0,
        "chars_extracted": 0,
        "quality_score": 0.0,
        "error": "Could not extract readable text from this PDF.",
    })



# ═══════════════════════════════════════════════════════════
# PUBLIC: SKILL EXTRACTION
# ═══════════════════════════════════════════════════════════

def extract_skills(text: str) -> list[str]:
    """Extract known skills from text using spaCy PhraseMatcher + regex fallback.

    Returns a deduplicated list of canonical skill names in stable order.
    """
    if not text or not text.strip():
        return []

    # Try spaCy-based extraction first
    skills = _spacy_extract_skills(text)
    if skills is not None:
        return skills

    # Regex-based fallback
    return _regex_extract_skills(text)


def _spacy_extract_skills(text: str) -> list[str] | None:
    """Extract skills using spaCy PhraseMatcher. Returns None if spaCy unavailable."""
    nlp = _get_nlp()
    if nlp is None:
        return None

    try:
        from spacy.matcher import PhraseMatcher
    except ImportError:
        return None

    # Truncate to avoid spaCy memory issues on very long docs
    doc = nlp(text.lower()[:100_000])

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in _SKILL_TAXONOMY]
    matcher.add("SKILLS", patterns)

    found: list[str] = []
    seen: set[str] = set()

    for _, start, end in matcher(doc):
        span_text = doc[start:end].text.strip()
        canonical = _resolve_alias(span_text)
        if canonical not in seen:
            found.append(canonical)
            seen.add(canonical)

    # Also check aliases that might not be in taxonomy
    norm = _normalize(text)
    for alias, canonical in _SKILL_ALIASES.items():
        if canonical not in seen:
            pattern = r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])"
            if re.search(pattern, norm):
                found.append(canonical)
                seen.add(canonical)

    return found


def _regex_extract_skills(text: str) -> list[str]:
    """Extract skills using pure regex matching (no spaCy)."""
    norm = _normalize(text)
    found: list[str] = []
    seen: set[str] = set()

    # Taxonomy skills — longest first to avoid partial matches
    for skill in _SKILL_TAXONOMY:
        pattern = r"(?<![a-z])" + re.escape(skill) + r"(?![a-z])"
        if re.search(pattern, norm):
            canonical = _resolve_alias(skill)
            if canonical not in seen:
                found.append(canonical)
                seen.add(canonical)

    # Check aliases
    for alias, canonical in _SKILL_ALIASES.items():
        if canonical not in seen:
            pattern = r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])"
            if re.search(pattern, norm):
                found.append(canonical)
                seen.add(canonical)

    return found


def _resolve_alias(skill: str) -> str:
    """Resolve a skill token to its canonical form."""
    s = skill.strip().lower()
    return _SKILL_ALIASES.get(s, s)


# ═══════════════════════════════════════════════════════════
# PUBLIC: SEMANTIC MATCHING
# ═══════════════════════════════════════════════════════════

def semantic_match_score(resume_skills: list[str], jd_skills: list[str]) -> float:
    """Compute semantic similarity between resume and JD skill sets.

    Returns a float on a 0–100 scale. Higher = better alignment.
    Uses sentence-transformers if available, else TF-IDF cosine fallback.
    """
    if not resume_skills or not jd_skills:
        return 0.0

    # Try sentence-transformers
    score = _sbert_similarity(resume_skills, jd_skills)
    if score is not None:
        return score

    # Pure-Python TF-IDF fallback
    return _tfidf_cosine(resume_skills, jd_skills)


def _sbert_similarity(resume_skills: list[str], jd_skills: list[str]) -> float | None:
    """Compute similarity using sentence-transformers. Returns None if unavailable."""
    model = _get_sbert()
    if model is None:
        return None

    try:
        import numpy as np
        resume_text = ", ".join(resume_skills)
        jd_text = ", ".join(jd_skills)
        embeddings = model.encode([resume_text, jd_text])
        cos = float(np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
        ))
        return max(0.0, min(100.0, cos * 100.0))
    except Exception:
        return None


def _tfidf_cosine(resume_skills: list[str], jd_skills: list[str]) -> float:
    """Pure-Python TF-IDF cosine similarity, scaled to 0–100."""
    import math
    from collections import Counter

    r_tokens = _tokenize_skills(resume_skills)
    j_tokens = _tokenize_skills(jd_skills)

    if not r_tokens or not j_tokens:
        return 0.0

    vocab = set(r_tokens) | set(j_tokens)
    r_tf = Counter(r_tokens)
    j_tf = Counter(j_tokens)

    dot = sum(r_tf.get(t, 0) * j_tf.get(t, 0) for t in vocab)
    mag_r = math.sqrt(sum(v * v for v in r_tf.values()))
    mag_j = math.sqrt(sum(v * v for v in j_tf.values()))

    if mag_r < 1e-8 or mag_j < 1e-8:
        return 0.0

    cos = dot / (mag_r * mag_j)
    return max(0.0, min(100.0, cos * 100.0))


def _tokenize_skills(skills: list[str]) -> list[str]:
    """Tokenize skill names into words for TF-IDF."""
    tokens: list[str] = []
    for skill in skills:
        parts = re.split(r"[\s/\-_.]+", skill.lower().strip())
        tokens.extend(p for p in parts if len(p) > 1)
    return tokens


# ═══════════════════════════════════════════════════════════
# PUBLIC: PROFILE SIGNAL EXTRACTION
# ═══════════════════════════════════════════════════════════

def extract_profile_signals(resume_text: str) -> dict[str, Any]:
    """Extract deterministic structural and evidence signals from resume text.

    Returns a dict with stable, predictable keys for use by the scoring engine.
    Pure regex heuristics — no network calls, no LLM.
    """
    if not resume_text or not resume_text.strip():
        return _empty_signals()

    low = resume_text.lower()
    lines = resume_text.strip().splitlines()

    # ── Project detection ──
    project_count = _count_projects(resume_text)

    # ── Link detection ──
    has_github = bool(re.search(r"github\.com/\w", low))
    has_linkedin = bool(re.search(r"linkedin\.com/in/\w", low))
    has_portfolio = bool(re.search(
        r"portfolio|personal\s*site|personal\s*website|\.dev/|\.me/|\.io/\w",
        low,
    ))
    has_deployed_links = bool(re.search(
        r"deploy|hosted|live\s*(url|link|site|demo)|production|vercel\.app|netlify\.app|"
        r"herokuapp\.com|railway\.app|render\.com|\.onrender\.com|"
        r"aws\.amazon|cloudfront|pages\.dev",
        low,
    ))

    # ── Certification detection ──
    cert_patterns = re.findall(
        r"certif|certified|aws\s+(?:cloud\s+)?practitioner|aws\s+solutions?\s+architect|"
        r"az-\d{2,3}|google\s+cloud\s+(?:certified|professional)|"
        r"coursera|udemy|edx|udacity|pluralsight|"
        r"cka\b|ckad\b|ckss?\b|pmp\b|csm\b|"
        r"hackerrank|leetcode\s+(?:rating|badge)",
        low,
    )
    certification_count = min(len(cert_patterns), 10)

    # ── Metric/quantified impact bullets ──
    metric_patterns = [
        r"\d+[%+x]",
        r"\d+\s*(?:users?|requests?|ms|seconds?|projects?|stars?|views?|"
        r"downloads?|contributions?|clients?|customers?|transactions?|"
        r"members?|endpoints?|repositories|repos)",
        r"\$\d+",
        r"(?:reduced|improved|increased|boosted|grew|cut|saved|decreased|accelerated)"
        r"\s+.*?\d+",
    ]
    metric_bullets_count = 0
    for p in metric_patterns:
        metric_bullets_count += len(re.findall(p, low))
    metric_bullets_count = min(metric_bullets_count, 30)

    # ── Education ──
    education_mentions = len(re.findall(
        r"b\.?tech|m\.?tech|b\.?sc|m\.?sc|b\.?e\b|m\.?e\b|bachelor|master|"
        r"ph\.?d|diploma|mba|cgpa|gpa|university|college|institute|"
        r"degree|engineering|computer\s+science",
        low,
    ))
    education_mentions = min(education_mentions, 8)

    # ── Experience ──
    experience_mentions = len(re.findall(
        r"intern|internship|engineer|developer|analyst|consultant|associate|"
        r"work(?:ed|ing)\s+(?:at|on|with|for)|"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+20\d{2}\s*[-–—]\s*"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|present|current)",
        low,
    ))
    experience_mentions = min(experience_mentions, 10)

    # ── Leadership ──
    leadership_signals = len(re.findall(
        r"led\s|lead\s|team\s*lead|managed|mentored|spearheaded|orchestrated|"
        r"captain|president|vice.?president|founder|co.?founder|"
        r"organized|coordinated|supervised|directed",
        low,
    ))
    leadership_signals = min(leadership_signals, 8)

    # ── Action verb density ──
    bullet_lines = re.findall(r"[•\-\*]\s*(\w+)", resume_text)
    if bullet_lines:
        action_count = sum(1 for w in bullet_lines if w.lower() in _ACTION_VERBS)
        action_verb_density = round(action_count / len(bullet_lines), 2)
    else:
        # fallback: check first words of lines
        first_words = [l.strip().split()[0].lower() for l in lines
                       if l.strip() and l.strip()[0].isalpha()]
        if first_words:
            action_count = sum(1 for w in first_words if w in _ACTION_VERBS)
            action_verb_density = round(action_count / len(first_words), 2)
        else:
            action_verb_density = 0.0

    # ── Contact info ──
    has_email = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", resume_text))
    has_phone = bool(re.search(r"\+?\d[\d\s\-()]{7,}", resume_text))
    contact_info_present = has_email or has_phone

    # ── Additional stable signals ──
    skills = extract_skills(resume_text)
    word_count = len(resume_text.split())
    bullet_count = len(re.findall(r"[•\-\*]\s", resume_text))
    section_headers = re.findall(
        r"^[\s]*(education|experience|projects?|skills?|certif|awards?|"
        r"summary|objective|work|technical|achievements?|publications?|"
        r"research|extracurricular|volunteer)",
        low, re.MULTILINE,
    )
    section_count = len(set(section_headers))
    has_summary = bool(re.search(r"(summary|objective|about\s+me|profile)", low))

    has_collab = bool(re.search(
        r"team|collaborat|standup|sprint|pair|code\s+review|cross.?functional",
        low,
    ))
    has_learning = bool(re.search(
        r"certif|coursera|udemy|course|bootcamp|hackathon|contest|competition|"
        r"self.?taught|mooc|nanodegree",
        low,
    ))

    return {
        # Canonical keys (required by spec)
        "project_count": project_count,
        "has_github": has_github,
        "has_linkedin": has_linkedin,
        "has_portfolio": has_portfolio,
        "has_deployed_links": has_deployed_links,
        "certification_count": certification_count,
        "metric_bullets_count": metric_bullets_count,
        "education_mentions": education_mentions,
        "experience_mentions": experience_mentions,
        "leadership_signals": leadership_signals,
        "action_verb_density": action_verb_density,
        "contact_info_present": contact_info_present,
        # Additional stable keys for scorer integration
        "skills": skills,
        "word_count": word_count,
        "bullet_count": bullet_count,
        "section_count": section_count,
        "has_summary": has_summary,
        "has_collab_language": has_collab,
        "has_learning_signals": has_learning,
        "has_links": has_github or has_linkedin or has_portfolio,
        "has_deploy_proof": has_deployed_links,
    }


def _empty_signals() -> dict[str, Any]:
    """Return zeroed-out signal dict for empty input."""
    return {
        "project_count": 0,
        "has_github": False,
        "has_linkedin": False,
        "has_portfolio": False,
        "has_deployed_links": False,
        "certification_count": 0,
        "metric_bullets_count": 0,
        "education_mentions": 0,
        "experience_mentions": 0,
        "leadership_signals": 0,
        "action_verb_density": 0.0,
        "contact_info_present": False,
        "skills": [],
        "word_count": 0,
        "bullet_count": 0,
        "section_count": 0,
        "has_summary": False,
        "has_collab_language": False,
        "has_learning_signals": False,
        "has_links": False,
        "has_deploy_proof": False,
    }


def _count_projects(text: str) -> int:
    """Count likely distinct projects in resume text."""
    count = 0
    seen: set[str] = set()

    # Pattern: "ProjectName — description" or "ProjectName - description"
    for m in re.finditer(
        r"([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3})\s*[–—\-:]\s*",
        text,
    ):
        name = m.group(1).strip().lower()
        skip = {"education", "experience", "skills", "summary", "projects",
                "work", "technical", "achievements", "awards", "certifications"}
        if 3 < len(name) < 60 and name not in seen and name not in skip:
            seen.add(name)
            count += 1

    # Pattern: "Built/Developed/Created X"
    for m in re.finditer(
        r"(?:built|developed|created|designed|implemented|launched)\s+(?:a\s+)?(.+?)(?:\.|,|\n|;|using)",
        text, re.IGNORECASE,
    ):
        name = m.group(1).strip().lower()[:60]
        if 3 < len(name) < 80 and name not in seen:
            seen.add(name)
            count += 1

    return min(count, 12)
