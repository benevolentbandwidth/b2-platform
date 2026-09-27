"""Resolve a WhatsApp media ID to raw bytes via the Meta Graph API.

Inbound WhatsApp image/document messages carry only a media *ID*. Turning that
into bytes is a two-step authenticated call:

  1. GET /{media-id}                 → JSON with a short-lived, authenticated URL
  2. GET <that url>                  → the media bytes

Both require a WhatsApp access token: ``WHATSAPP_TOKEN`` env var if set, else the
``WHATSAPP_ACCESS_TOKEN`` secret in the benevolent-bandwidth project (see
gcp_secrets.resolve). If neither resolves, the function logs and returns None
so the caller can degrade gracefully rather than crash.

This module is intentionally the single seam for media retrieval: if the upstream
webhook is later changed to pre-resolve media to bytes/URL, only this file and the
payload extractor in api.py need to change.
"""

from __future__ import annotations

import logging
import os

import httpx

from .gcp_secrets import resolve as resolve_secret

logger = logging.getLogger(__name__)

_GRAPH_BASE = "https://graph.facebook.com"
_DEFAULT_GRAPH_VERSION = "v21.0"
_TIMEOUT = 30.0


async def download_media(media_id: str) -> tuple[bytes, str] | None:
    """Return (bytes, mime_type) for a WhatsApp media ID, or None if unavailable."""
    token = resolve_secret("WHATSAPP_TOKEN", "WHATSAPP_ACCESS_TOKEN")
    if not token:
        logger.warning("whatsapp_media.download skipped — no WhatsApp access token available")
        return None
    if not media_id:
        return None

    version = os.getenv("WHATSAPP_GRAPH_VERSION", _DEFAULT_GRAPH_VERSION)
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            meta_resp = await client.get(f"{_GRAPH_BASE}/{version}/{media_id}", headers=headers)
            meta_resp.raise_for_status()
            meta = meta_resp.json()

            media_url = meta.get("url")
            mime_type = meta.get("mime_type") or "image/jpeg"
            if not media_url:
                logger.warning("whatsapp_media.download no url in metadata media=%s", media_id)
                return None

            # The media URL must be fetched with the same bearer token.
            bytes_resp = await client.get(media_url, headers=headers)
            bytes_resp.raise_for_status()
            data = bytes_resp.content
    except httpx.HTTPError as exc:
        logger.error("whatsapp_media.download failed media=%s error=%s", media_id, exc)
        return None

    logger.info("whatsapp_media.download ok media=%s bytes=%d mime=%s", media_id, len(data), mime_type)
    return data, mime_type
