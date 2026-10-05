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
        case_reference: str | None = None,
    ) -> str:
        transcript = self._build_transcript(messages)

        scrubbed = self._scrubber.scrub(
            transcript,
            audit_store=self._audit_store,
        )

        return await self._summarize(scrubbed, case_reference)

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
        case_reference: str | None = None,
    ) -> str:
        # Added after scrubbing so the scrubber cannot mangle it, and only when
        # a case exists. Never the session id: that is the claimant's phone
        # number, which the scrubbing step exists to keep out of this prompt.
        reference_rule = (
            f"- Include this case reference exactly as written, so the user can "
            f"quote it: {case_reference}\n"
            if case_reference
            else ""
        )
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
            {reference_rule}
            Conversation:

            {scrubbed_conversation}
        """

        result = await self._agent.run(prompt)

        return result.output


    async def _summarize_user_facing(
    self,
    scrubbed_conversation: str,
    uuid: str
    ) -> str:
        prompt = f"""
            Create a plain-language summary of the following conversation for the user.

            Requirements:
            - The conversation has already been scrubbed of PII.
            - Summarize what happened from the user's perspective.
            - Include:
                • What the user was trying to accomplish
                • Important information the user provided
                • Important decisions or conclusions
                • What was successfully completed
                • What is still incomplete or needs follow-up
                • The current status of the conversation
                • The UUID of the conversation
            - Do NOT mention:
                • Tool calls
                • Agents, models, or AI
                • APIs, functions, code, or technical implementation details
                • Internal processing or system behavior
                • PII or details that were removed during scrubbing
            - Preserve important context needed for the user to understand the outcome.
            - Do not speculate or invent information.
            - Maximum 200 words.
            - Return plain text.
            - Write for a non-technical user.
            - UUID is absolutely essential for this summary - without it the user will not be able to retreive their history

            Conversation:

            {scrubbed_conversation}

            UUID:
            {uuid}

        """