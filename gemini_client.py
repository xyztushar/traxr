"""
TRAXR — Gemini OCR Client
Thin, Streamlit-independent wrapper around the Google Generative AI SDK.
Used exclusively for OCR fallback on scanned/image-based PDF pages.

No Streamlit dependency. No UI logic. API key is received as a parameter.

Public API:
    get_gemini_model(api_key, model_name="gemini-1.5-flash") -> GenerativeModel | None
    ocr_page_image(model, image_bytes) -> str
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# OCR PROMPT — spec-mandated, do not modify
# ═══════════════════════════════════════════════════════════

OCR_PROMPT = (
    "You are performing OCR on a resume.\n"
    "Extract only the readable resume text from this image.\n"
    "Preserve section headers, bullet points, line breaks, dates, project titles, "
    "technologies, links, certifications, metrics, and role descriptions where visible.\n"
    "Do not summarize.\n"
    "Do not rewrite for style.\n"
    "Do not invent missing text.\n"
    "If a region is unreadable, skip it rather than guessing."
)


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════

def get_gemini_model(api_key: str, model_name: str = "gemini-2.5-flash") -> Any | None:
    """Configure the Gemini SDK and return a GenerativeModel instance.

    Args:
        api_key: Gemini API key (must not be empty).
        model_name: Model identifier. Default: gemini-2.5-flash for OCR.

    Returns:
        A GenerativeModel instance, or None on failure.
    """
    if not api_key or not api_key.strip():
        logger.warning("No API key provided for Gemini OCR.")
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel(model_name)
        return model
    except ImportError:
        logger.warning("google-generativeai package is not installed.")
        return None
    except Exception as exc:
        logger.warning("Failed to initialize Gemini model: %s", exc)
        return None


def ocr_page_image(model: Any, image_bytes: bytes) -> str:
    """Send a single page image to Gemini for OCR text extraction.

    Retries up to 3 times on 429 quota errors with exponential backoff,
    since free-tier limits typically reset within 15-60 seconds.

    Args:
        model: A GenerativeModel instance from get_gemini_model().
        image_bytes: PNG image bytes of a single PDF page.

    Returns:
        Extracted text string, or empty string on failure.
    """
    if model is None or not image_bytes:
        return ""

    import time

    max_retries = 3
    base_delay = 20  # seconds — free-tier resets are typically 15-40s

    for attempt in range(max_retries):
        try:
            import google.generativeai as genai

            image_part = {
                "mime_type": "image/png",
                "data": image_bytes,
            }
            response = model.generate_content(
                [OCR_PROMPT, image_part],
                generation_config=genai.GenerationConfig(temperature=0.1),
            )
            extracted = (response.text or "").strip()
            return extracted
        except Exception as exc:
            exc_str = str(exc)
            # Retry on 429 quota errors — they reset quickly on free tier
            if "429" in exc_str and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(
                    "Gemini OCR rate-limited (attempt %d/%d). "
                    "Retrying in %ds...", attempt + 1, max_retries, delay
                )
                time.sleep(delay)
                continue
            logger.warning("Gemini OCR failed on page image: %s", exc)
            return ""

