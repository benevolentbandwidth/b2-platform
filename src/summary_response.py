from __future__ import annotations

import functools
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from tools.fake_image_detector import gemini_settings

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(__file__).parent / "config" / "whatsapp.yaml"


@dataclass(frozen=True)
class SummarySettings:
    model: str
    location: str | None  # None: $VERTEX_LOCATION
    thinking_level: str | None  # None: the model's default thinking


@functools.lru_cache(maxsize=1)
def summary_settings() -> SummarySettings:
    """The summary model's settings, from src/config/whatsapp.yaml. Refuses a
    missing model rather than silently falling back to some other one."""
    raw = yaml.safe_load(_SETTINGS_PATH.read_text()) or {}
    summary = raw.get("summary") or {}
    model = summary.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{_SETTINGS_PATH}: summary.model must be a model name")
    where = f"{_SETTINGS_PATH}: summary"
    return SummarySettings(
        model=model.strip(),
        location=gemini_settings.location(summary.get("location"), where),
        thinking_level=gemini_settings.thinking_level(summary.get("thinking_level"), where),
    )


@functools.lru_cache(maxsize=1)
def _summary_model() -> GoogleModel:
    """Vertex through the runtime's own credentials, built the way the agent
    builds its model, once per process like the agents. The "google-vertex:..."
    shortcut picked up GEMINI_API_KEY whenever one was set (e.g. from a local
    .env), and Vertex rejects API keys, so every summary failed silently."""
    settings = summary_settings()
    return GoogleModel(
        settings.model,
        provider=GoogleProvider(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=settings.location or os.getenv("VERTEX_LOCATION", "us-central1"),
        ),
    )


def _summary_model_settings() -> dict[str, Any]:
    level = summary_settings().thinking_level
    return {"google_thinking_config": {"thinking_level": level}} if level else {}


async def generate_summary_tool_response(
    *,
    final_response: str,
    tool_events: list[dict[str, Any]] | None = None,
) -> str | None:
    """A short WhatsApp message telling the claimant the outcome, or None on failure."""
    try:
        return await call_summary_model(summary_prompt(final_response, tool_events or []))
    except Exception as exc:
        logger.warning("summary_response.generate skipped error=%s", exc)
        return None


def summary_prompt(final_response: str, tool_events: list[dict[str, Any]]) -> str:
    reference = case_reference_from(tool_events)
    reference_rule = (
        f"Include the case reference {reference} exactly as written, so they can quote it. "
        if reference
        else ""
    )
    return (
        "Write a WhatsApp-ready message to someone who has just sent their family "
        "member's death certificate, telling them the outcome below. Use plain, warm, "
        "compassionate language; at most two sentences. "
        f"{reference_rule}"
        "Do not mention internal tools, debug output, e2e tests, JSON, prompts, scores, "
        "checks, AI, or implementation details. Reply with the message text only.\n\n"
        f"{json.dumps(summary_payload(final_response, tool_events), indent=2, ensure_ascii=False, default=str)}"
    )


def summary_payload(final_response: str, tool_events: list[dict[str, Any]]) -> dict[str, Any]:
    verification = death_certificate_verification(tool_events)
    payload: dict[str, Any] = {"assistant_response": final_response}
    if verification is not None:
        payload["death_certificate_verification"] = verification
    return payload


def death_certificate_verification(tool_events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(tool_events):
        if event.get("tool") != "death_certificate_verification":
            continue
        return {
            "status": event.get("status"),
            "accepted": event.get("accepted"),
            "handed_off": event.get("handed_off"),
            "summary": event.get("summary"),
            "flags": list(event.get("flags", [])),
        }
    return None


def case_reference_from(tool_events: list[dict[str, Any]]) -> str | None:
    """The case reference from the latest verification event, if it made one."""
    for event in reversed(tool_events):
        if event.get("tool") == "death_certificate_verification" and event.get("case_reference"):
            return str(event["case_reference"])
    return None


async def call_summary_model(prompt: str) -> str:
    """Ask Gemini for the message directly.

    This used to wrap the prompt as a conversation and hand it to the conversation
    summariser, whose own instruction ("summarise this conversation") won: claimants
    were sent a summary of these instructions instead of a message.
    """
    result = await Agent(_summary_model(), model_settings=_summary_model_settings()).run(prompt)
    return str(result.output).strip()
