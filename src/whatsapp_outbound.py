from __future__ import annotations

import logging
import os

import httpx

from .gcp_secrets import resolve as resolve_secret

logger = logging.getLogger(__name__)

_GRAPH_BASE = "https://graph.facebook.com"
_DEFAULT_GRAPH_VERSION = "v21.0"
_TIMEOUT = 30.0


def outbound_disabled() -> bool:
    return _truthy(os.getenv("E2E_DISABLE_WHATSAPP_OUTBOUND"))


def can_send_text_message(*, wa_id: str | None, phone_number_id: str | None = None) -> bool:
    if outbound_disabled():
        logger.info("whatsapp_outbound.send skipped disabled=true")
        return False
    if not resolve_secret("WHATSAPP_TOKEN", "WHATSAPP_ACCESS_TOKEN"):
        logger.warning("whatsapp_outbound.send skipped — no WhatsApp access token available")
        return False
    if not (phone_number_id or resolve_secret("WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_PHONE_NUMBER_ID")):
        logger.warning("whatsapp_outbound.send skipped missing phone_number_id")
        return False
    if not wa_id:
        logger.warning("whatsapp_outbound.send skipped missing recipient")
        return False
    return True


async def send_text_message(
    *,
    wa_id: str | None,
    text: str | None,
    phone_number_id: str | None = None,
) -> bool:
    body = (text or "").strip()

    if not can_send_text_message(wa_id=wa_id, phone_number_id=phone_number_id):
        return False
    if not body:
        logger.warning("whatsapp_outbound.send skipped empty text")
        return False

    token = resolve_secret("WHATSAPP_TOKEN", "WHATSAPP_ACCESS_TOKEN")
    sender_id = phone_number_id or resolve_secret("WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_PHONE_NUMBER_ID")
    version = os.getenv("WHATSAPP_GRAPH_VERSION", _DEFAULT_GRAPH_VERSION)
    url = f"{_GRAPH_BASE}/{version}/{sender_id}/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": wa_id,
        "type": "text",
        "text": {"body": body},
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("whatsapp_outbound.send failed recipient=%.8s error=%s", wa_id, exc)
        return False

    logger.info("whatsapp_outbound.send ok recipient=%.8s chars=%d", wa_id, len(body))
    return True


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
