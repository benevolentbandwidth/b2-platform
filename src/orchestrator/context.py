"""Per-run session context passed to context-aware tools.

Threaded through pydantic-ai as the agent's ``deps`` so a tool can pull
out-of-band state — most importantly the user's most recent uploaded image —
without that data ever passing through the model. This keeps the mechanism
generic: any tool that declares ``needs_context`` receives this object and can
reach the transient store or the rendered conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionContext:
    """Runtime context for a single agent turn."""

    session_id: str | None = None
    store: Any | None = None          # FirestoreSessionStore (history + media)
    # The conversation so far, including the message being answered, rendered
    # as "user:"/"assistant:" lines; the verification tool's narrative.
    history_text: str = ""
    debug_events: list[dict[str, Any]] | None = None
    # The claimant's own messages, including the one being answered, verbatim and
    # one entry per message, so they reach GiveLight exactly as written rather
    # than only as an AI summary.
    claimant_messages: list[str] = field(default_factory=list)
