from __future__ import annotations

import json
import logging
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from tools.summary.summary import ConversationSummaryTool

logger = logging.getLogger(__name__)

SUMMARY_MODEL = "google-vertex:gemini-2.5-flash"


async def generate_summary_tool_response(
    *,
    final_response: str,
    tool_events: list[dict[str, Any]] | None = None,
    session_id: str
) -> str | None:
    try:
        messages = build_summary_messages(final_response=final_response, tool_events=tool_events or [])
        return await call_summary_tool(messages, session_id)
    except Exception as exc:
        logger.warning("summary_response.generate skipped error=%s", exc)
        return None


def build_summary_messages(
    *,
    final_response: str,
    tool_events: list[dict[str, Any]],
) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content=summary_prompt(final_response, tool_events))]),
        ModelResponse(parts=[TextPart(content=final_response)]),
    ]


def summary_prompt(final_response: str, tool_events: list[dict[str, Any]]) -> str:
    return (
        "Write a concise WhatsApp-ready message for the user based on this result. "
        "Do not mention internal tools, debug output, e2e tests, JSON, prompts, or implementation details. "
        "Use plain, compassionate language. Maximum two sentences.\n\n"
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


async def call_summary_tool(messages: list[ModelMessage]) -> str:
    tool = ConversationSummaryTool(
        agent=Agent(SUMMARY_MODEL),
        scrubber=SummaryResponseScrubber(),
        audit_store=SummaryResponseAuditStore(),
    )
    return await tool.run(summary_tool_messages(messages))


def summary_tool_messages(messages: list[ModelMessage]) -> list[SummaryToolMessage]:
    return [SummaryToolMessage(message) for message in messages]


class SummaryToolMessage:
    def __init__(self, message: ModelMessage) -> None:
        self.message = message

    def model_dump_json(self, *, exclude_none: bool = False, **kwargs: Any) -> str:
        dumped = ModelMessagesTypeAdapter.dump_python([self.message], exclude_none=exclude_none)[0]
        return json.dumps(dumped, ensure_ascii=False, default=str)


class SummaryResponseScrubber:
    def scrub(self, transcript: str, *, audit_store: Any) -> str:
        return transcript


class SummaryResponseAuditStore:
    pass
