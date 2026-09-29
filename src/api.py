"""FastAPI entry point for the b2 WhatsApp Channel Adapter.

Exposes POST /message — called by the central benevolent-bandwidth webhook
for every inbound WhatsApp message routed to b2-platform.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .chat import chat
from .summary_response import generate_summary_tool_response
from .whatsapp_outbound import can_send_text_message, send_text_message
from .whatsapp_media import download_media

logger = logging.getLogger(__name__)

app = FastAPI(title="b2 WhatsApp Adapter")

_WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
_ENABLE_E2E_DEBUG = os.getenv("ENABLE_E2E_DEBUG", "")

# WhatsApp media message types that carry a downloadable media ID.
_MEDIA_TYPES = ("image", "document")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class InboundMessage:
    """A normalized inbound WhatsApp message."""

    wa_id: str | None = None
    text: str | None = None
    media_id: str | None = None
    mime_type: str | None = None
    phone_number_id: str | None = None


def _debug_response(
    response: str | None,
    message: InboundMessage,
    debug_events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "response": response,
        "debug": {
            "session_id": message.wa_id,
            "tool_results": debug_events,
        },
    }


def _extract_message(payload: dict[str, Any]) -> InboundMessage:
    """Normalize a WhatsApp webhook payload into an InboundMessage.

    Handles text and media (image/document) messages; returns an empty
    InboundMessage for anything else (e.g. status updates).
    """
    try:
        entry = (payload.get("entry") or [])[0]
        value = (entry.get("changes") or [])[0].get("value", {})
        metadata = value.get("metadata") or {}
        contacts = value.get("contacts") or []
        messages = value.get("messages") or []
        phone_number_id = metadata.get("phone_number_id")

        if not messages:
            return InboundMessage(phone_number_id=phone_number_id)

        msg = messages[0]
        msg_type = msg.get("type")
        wa_id = (contacts[0].get("wa_id") if contacts else None) or msg.get("from")

        if msg_type == "text":
            text = (msg.get("text") or {}).get("body", "").strip()
            if not text:
                return InboundMessage(wa_id=wa_id, phone_number_id=phone_number_id)
            return InboundMessage(wa_id=wa_id, text=text, phone_number_id=phone_number_id)

        if msg_type in _MEDIA_TYPES:
            media = msg.get(msg_type) or {}
            media_id = media.get("id")
            if not media_id:
                return InboundMessage(wa_id=wa_id, phone_number_id=phone_number_id)
            return InboundMessage(
                wa_id=wa_id,
                media_id=media_id,
                mime_type=media.get("mime_type"),
                phone_number_id=phone_number_id,
            )

        return InboundMessage(wa_id=wa_id, phone_number_id=phone_number_id)
    except (IndexError, AttributeError, TypeError, KeyError):
        return InboundMessage()


@app.post("/message")
async def message_endpoint(
    request: Request,
    x_webhook_secret: str | None = Header(default=None),
    x_e2e_debug: str | None = Header(default=None),
) -> dict[str, Any]:
    if _WEBHOOK_SECRET and x_webhook_secret != _WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    payload = await request.json()
    logger.info("api.message received object=%s", payload.get("object"))

    message = _extract_message(payload)
    debug_enabled = _truthy(_ENABLE_E2E_DEBUG) and _truthy(x_e2e_debug)
    debug_events: list[dict[str, Any]] = []

    if message.text is not None:
        logger.info("api.message routing text wa_id=%s chars=%d", message.wa_id, len(message.text))
        response = await run_in_threadpool(
            chat,
            text=message.text,
            session_id=message.wa_id,
            debug_events=debug_events,
        )
        logger.info("api.message done wa_id=%s response_chars=%d", message.wa_id, len(response))
        await _send_summary_tool_response(message, message.wa_id, response, debug_events)
        if debug_enabled:
            return _debug_response(response, message, debug_events or [])
        return {"response": response}

    if message.media_id is not None:
        logger.info("api.message media wa_id=%s media=%s", message.wa_id, message.media_id)
        media = await download_media(message.media_id)
        if media is None:
            logger.info("api.message skipped — media download unavailable")
            if debug_enabled:
                return _debug_response(None, message, debug_events or [])
            return {"response": None}
        image_bytes, mime_type = media
        response = await run_in_threadpool(
            chat,
            image_bytes=image_bytes,
            image_media_type=mime_type,
            session_id=message.wa_id,
            debug_events=debug_events,
        )
        logger.info("api.message done wa_id=%s response_chars=%d", message.wa_id, len(response))
        await _send_summary_tool_response(message, message.wa_id, response, debug_events)
        if debug_enabled:
            return _debug_response(response, message, debug_events or [])
        return {"response": response}

    logger.info("api.message skipped — no actionable payload")
    if debug_enabled:
        return _debug_response(None, message, debug_events or [])
    return {"response": None}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def _send_summary_tool_response(
    message: InboundMessage,
    session_id: str,
    response: str,
    debug_events: list[dict[str, Any]],
) -> None:
    if not can_send_text_message(wa_id=message.wa_id, phone_number_id=message.phone_number_id):
        return

    summary = await generate_summary_tool_response(
        final_response=response,
        tool_events=debug_events,
        session_id=session_id
    )
    if not summary:
        return

    await send_text_message(
        wa_id=message.wa_id,
        text=summary,
        phone_number_id=message.phone_number_id,
    )
