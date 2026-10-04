from collections.abc import Sequence

from summary.summary import ConversationSummaryTool
from pydantic_ai import Agent, ModelMessage
from app.core.pii import PiiScrubber, PiiAuditStore


async def handle(value: Sequence[ModelMessage], uuid: str) -> str:
    """Return a summary of the user's conversation"""
    """Parameters : ModelMessages for complete conversation"""
    agent = Agent("google-vertex:gemini-2.5-flash")

    tool = ConversationSummaryTool(
        agent=agent,
        scrubber=PiiScrubber(),
        audit_store=PiiAuditStore(),
    )

    summary = await tool.run(value, uuid)

    return summary

