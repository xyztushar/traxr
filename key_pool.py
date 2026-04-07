"""
TRAXR — Gemini API Key Pool

Manages multiple API keys with automatic rotation when free-tier quotas
are exhausted (429 errors). Keys are loaded from:

  1. st.secrets["GEMINI_API_KEYS"] — comma-separated string or TOML list
  2. st.secrets["GEMINI_API_KEY"]  — single key (backward compat)
  3. os.environ["GEMINI_API_KEYS"] — comma-separated string
  4. os.environ["GEMINI_API_KEY"]  — single key (backward compat)

Usage:
    from key_pool import get_next_api_key, mark_key_exhausted, get_all_api_keys

    keys = get_all_api_keys()
    for key in keys:
        try:
            result = call_api(key)
            break
        except RateLimitError:
            mark_key_exhausted(key)
            continue
"""

from __future__ import annotations

import logging
import os
import time
import threading

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# KEY STATE
# ═══════════════════════════════════════════════════════════

_lock = threading.Lock()
_exhausted_keys: dict[str, float] = {}  # key -> timestamp when exhausted
_COOLDOWN_SECONDS = 65  # Free-tier quotas typically reset within 60s


def _load_keys_from_source() -> list[str]:
    """Load API keys from Streamlit secrets and/or environment variables."""
    keys: list[str] = []

    # 1. Try Streamlit secrets (list form)
    try:
        import streamlit as st
        raw = st.secrets.get("GEMINI_API_KEYS", None)
        if raw:
            if isinstance(raw, (list, tuple)):
                keys.extend(k.strip() for k in raw if k.strip())
            elif isinstance(raw, str):
                keys.extend(k.strip() for k in raw.split(",") if k.strip())
    except Exception:
        pass

    # 2. Try Streamlit secrets (single key — backward compat)
    if not keys:
        try:
            import streamlit as st
            single = st.secrets.get("GEMINI_API_KEY", "").strip()
            if single:
                # Support comma-separated in single key field too
                keys.extend(k.strip() for k in single.split(",") if k.strip())
        except Exception:
            pass

    # 3. Try env var (list form)
    if not keys:
        env_keys = os.environ.get("GEMINI_API_KEYS", "")
        if env_keys:
            keys.extend(k.strip() for k in env_keys.split(",") if k.strip())

    # 4. Try env var (single key — backward compat)
    if not keys:
        env_single = os.environ.get("GEMINI_API_KEY", "").strip()
        if env_single:
            keys.extend(k.strip() for k in env_single.split(",") if k.strip())

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)

    return unique


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════

def get_all_api_keys() -> list[str]:
    """Return all configured API keys (order preserved)."""
    return _load_keys_from_source()


def get_available_api_keys() -> list[str]:
    """Return keys that aren't currently in cooldown."""
    now = time.time()
    all_keys = _load_keys_from_source()
    with _lock:
        return [
            k for k in all_keys
            if k not in _exhausted_keys
            or (now - _exhausted_keys[k]) > _COOLDOWN_SECONDS
        ]


def mark_key_exhausted(key: str) -> None:
    """Mark a key as rate-limited. It will be skipped for COOLDOWN_SECONDS."""
    with _lock:
        _exhausted_keys[key] = time.time()
        logger.info("API key ...%s marked as exhausted (cooldown %ds)",
                     key[-6:], _COOLDOWN_SECONDS)


def get_next_api_key() -> str | None:
    """Get the first available (non-exhausted) key, or None if all exhausted."""
    available = get_available_api_keys()
    if available:
        return available[0]

    # All keys exhausted — return the first one anyway (retry might work)
    all_keys = get_all_api_keys()
    return all_keys[0] if all_keys else None


def key_count() -> int:
    """Return the total number of configured keys."""
    return len(get_all_api_keys())
