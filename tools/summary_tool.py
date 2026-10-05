from collections.abc import Sequence

from summary.summary import ConversationSummaryTool
from pydantic_ai import Agent, ModelMessage
from app.core.pii import PiiScrubber, PiiAuditStore


async def handle(value: Sequence[ModelMessage], case_reference: str | None = None) -> str:
    """Return a summary of the user's conversation"""
    """Parameters : ModelMessages for complete conversation"""
    agent = Agent("google-vertex:gemini-2.5-flash")

    tool = ConversationSummaryTool(
        agent=agent,
        scrubber=PiiScrubber(),
        audit_store=PiiAuditStore(),
    )

    # Keyword, so a caller can never bind something else (such as the session
    # id, which is the claimant's phone number) as the case reference.
    summary = await tool.run(value, case_reference=case_reference)

    return summary

