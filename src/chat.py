from __future__ import annotations

from collections.abc import Sequence

from dotenv import load_dotenv
from pydantic_ai.messages import (
    BinaryImage,
    ImageUrl,
    ModelMessage,
    TextContent,
    TextPart,
    UserContent,
    UserPromptPart,
)

from .orchestrator.context import SessionContext
from .router import AgentRouter
from .session import Session
from .session_store import FirestoreSessionStore

load_dotenv()

DEFAULT_IMAGE_MEDIA_TYPE = "image/jpeg"
IMAGE_PROMPT_TEXT = "The user sent this image on WhatsApp. Analyze it and provide a helpful response."
IMAGE_ROUTING_TEXT = "A WhatsApp user uploaded a document image such as a death certificate for verification."
# Text notice given to the model when an image arrives. The image bytes go to the
# transient store (for tools to pull), never into the model's prompt.
IMAGE_ARRIVED_PROMPT = "The user has just uploaded a document image."
# Tail of prior conversation kept alongside IMAGE_ROUTING_TEXT when routing an image turn.
IMAGE_ROUTING_HISTORY_CHARS = 400

_router: AgentRouter | None = None


def _render_history_text(history: Sequence[ModelMessage] | None) -> str:
    """Flatten prior user/assistant turns into plain text for use as a narrative."""
    lines: list[str] = []
    for message in history or []:
        for part in getattr(message, "parts", []):
            content = getattr(part, "content", None)
            if not isinstance(content, str) or not content.strip():
                continue
            if isinstance(part, UserPromptPart):
                lines.append(f"user: {content.strip()}")
            elif isinstance(part, TextPart):
                lines.append(f"assistant: {content.strip()}")
    return "\n".join(lines)


def _get_router() -> AgentRouter:
    global _router
    if _router is None:
        _router = AgentRouter()
    return _router


def chat(
    *,
    text: str | None = None,
    image_bytes: bytes | None = None,
    image_url: str | None = None,
    image_media_type: str = DEFAULT_IMAGE_MEDIA_TYPE,
    session_id: str | None = None,
    channel: str = "whatsapp",
    debug_events: list[dict[str, object]] | None = None,
) -> str:
    """Route one WhatsApp text or image message through the agent loop and return the response."""

    store = FirestoreSessionStore() if session_id is not None else None
    history = store.load_history(session_id) if store is not None else None
    history_text = _render_history_text(history)

    prompt, route_query = _prepare_turn(
        text=text,
        image_bytes=image_bytes,
        image_url=image_url,
        image_media_type=image_media_type,
        history_text=history_text,
        store=store,
        session_id=session_id,
    )

    router = _get_router()
    agent, _metadata = router.route_with_metadata(route_query)
    deps = SessionContext(
        session_id=session_id,
        store=store,
        history_text=history_text,
        debug_events=debug_events,
    )
    session = Session(agent, history=history, deps=deps)

    response = "".join(session.send_stream(prompt))
    if store is not None:
        store.save_history(
            session_id,
            session.history,
            agent_name=getattr(agent, "name", None),
            channel=channel,
        )
    return response


def _prepare_turn(
    *,
    text: str | None,
    image_bytes: bytes | None,
    image_url: str | None,
    image_media_type: str,
    history_text: str,
    store: FirestoreSessionStore | None,
    session_id: str | None,
) -> tuple[str | Sequence[UserContent], str]:
    """Return (model_prompt, routing_query) and persist any uploaded image.

    An uploaded image (image_bytes) is saved to the transient store and the model
    is handed only a text notice — the bytes are pulled later by a context-aware
    tool and never enter the model's prompt.
    """
    supplied = sum(value is not None for value in (text, image_bytes, image_url))
    if supplied != 1:
        raise ValueError("Provide exactly one of text, image_bytes, or image_url")

    if image_bytes is not None:
        if not image_bytes:
            raise ValueError("image_bytes must not be empty")
        if store is not None and session_id is not None:
            store.save_media(session_id, image_bytes, mime_type=image_media_type)
        return IMAGE_ARRIVED_PROMPT, _image_route_query(history_text)

    if text is not None:
        prompt = _build_prompt(text=text, image_bytes=None, image_url=None, image_media_type=image_media_type)
        return prompt, text.strip()

    # image_url — legacy path: the referenced image is passed to the model directly.
    prompt = _build_prompt(text=None, image_bytes=None, image_url=image_url, image_media_type=image_media_type)
    return prompt, IMAGE_ROUTING_TEXT


def _image_route_query(history_text: str) -> str:
    """Build the routing query for an image turn.

    Always leads with the document-verification signal so an image reaches the
    verification agent even when prior small talk dominates the history. Only
    the tail of the history is kept, to stop a long chat diluting that signal.
    """
    if not history_text:
        return IMAGE_ROUTING_TEXT
    return f"{IMAGE_ROUTING_TEXT}\n{history_text[-IMAGE_ROUTING_HISTORY_CHARS:]}"


def _build_prompt(
    *,
    text: str | None,
    image_bytes: bytes | None,
    image_url: str | None,
    image_media_type: str,
) -> str | Sequence[UserContent]:
    supplied = sum(value is not None for value in (text, image_bytes, image_url))
    if supplied != 1:
        raise ValueError("Provide exactly one of text, image_bytes, or image_url")

    if text is not None:
        if not text.strip():
            raise ValueError("text must not be blank")
        return text.strip()

    if image_bytes is not None:
        if not image_bytes:
            raise ValueError("image_bytes must not be empty")
        return [
            TextContent(content=IMAGE_PROMPT_TEXT),
            BinaryImage(data=image_bytes, media_type=image_media_type),
        ]

    if image_url is None or not image_url.strip():
        raise ValueError("image_url must not be blank")

    return [
        TextContent(content=IMAGE_PROMPT_TEXT),
        ImageUrl(url=image_url.strip(), media_type=image_media_type),
    ]
