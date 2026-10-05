"""Gemini-specific settings shared by every Gemini call: location, thinking level, today's date.

Timeouts and retries apply to every Google call, Vision included, so they live
in google_clients.

Each Gemini call names its own model, location and thinking level in its
settings file, because they differ by model: every Gemini 3.x model is served
only from the `global` location (404 in regional ones), and thinking is set by
level on Gemini 3 but by token budget on 2.5, which rejects a level.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# LOW is the lowest every Gemini 3 model accepts: 3.8 Flash rejects MINIMAL.
THINKING_LEVELS = ("MINIMAL", "LOW", "MEDIUM", "HIGH")


def thinking_level(value: Any, where: str) -> str | None:
    """Validate a `thinking_level` setting. None (or absent) keeps the model's default."""
    if value is None:
        return None
    if not isinstance(value, str) or value.strip().upper() not in THINKING_LEVELS:
        raise ValueError(f"{where}: thinking_level must be one of {', '.join(THINKING_LEVELS)}, got {value!r}")
    return value.strip().upper()


def location(value: Any, where: str) -> str | None:
    """Validate a `location` setting. None (or absent) falls back to $VERTEX_LOCATION."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: location must be a Vertex location such as global or us-east1")
    return value.strip()


def thinking_config(gentypes: Any, level: str | None) -> Any:
    """The GenerateContentConfig.thinking_config for `level`, or None for the model default."""
    return gentypes.ThinkingConfig(thinking_level=level) if level else None


def today() -> str:
    """Today's date (UTC) for prompts. Without it Gemini judges dates against its
    training cutoff, so every certificate issued since looked future-dated."""
    return datetime.now(timezone.utc).date().isoformat()
