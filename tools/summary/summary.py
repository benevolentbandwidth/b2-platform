from collections.abc import Sequence

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from tools.privacy.pii_scrubber import PiiScrubber, PiiAuditStore


class ConversationSummaryTool:
    def __init__(
        self,
        agent: Agent,
        scrubber: PiiScrubber,
        audit_store: PiiAuditStore,
    ):
        self._agent = agent
        self._scrubber = scrubber
        self._audit_store = audit_store

    async def run(
        self,
        messages: Sequence[ModelMessage],
    ) -> str:
        transcript = self._build_transcript(messages)

        scrubbed = self._scrubber.scrub(
            transcript,
            audit_store=self._audit_store,
        )

        return await self._summarize(scrubbed)

    def _build_transcript(
        self,
        messages: Sequence[ModelMessage],
    ) -> str:
        return "\n".join(
            message.model_dump_json(exclude_none=True)
            for message in messages
        )

    async def _summarize(
        self,
        scrubbed_conversation: str,
    ) -> str:
        prompt = f"""
            Summarize the following conversation history.

            Requirements:
            - The conversation has already been scrubbed of PII.
            - Include:
                • User goals
                • Important decisions
                • Completed work
                • Remaining work
                • Important tool calls, if relevant
            - Maximum 200 words.
            - Do not speculate.
            - Return plain text.

            Conversation:

            {scrubbed_conversation}
        """

        result = await self._agent.run(prompt)

        return result.output