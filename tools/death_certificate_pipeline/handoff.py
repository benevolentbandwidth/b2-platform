"""Package scored death-certificate results and upload them to GiveLight."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx

from tools.death_certificate_pipeline.debug import authenticity_summary
from tools.death_certificate_pipeline.models import ReliabilityResult
from tools.fake_image_detector.models import ToolResult
from tools.fake_image_detector.file_formats import UNKNOWN_MIME_TYPE, by_mime, sniff
from tools.fake_image_detector.google_clients import access_token

logger = logging.getLogger(__name__)

_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"
_DRIVE_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"

# Waits between attempts after a temporary failure; one more attempt than
# there are entries. Drive's documented retryable responses are 429, 5xx and
# 403 rateLimitExceeded / userRateLimitExceeded.
_RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.0, 3.0)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


DECISION_ACCEPTED = "accepted"
DECISION_NEEDS_REVIEW = "needs_review"


def new_case_reference() -> str:
    """A short reference a claimant can read out or type, e.g. DC-7F3A-9C21-B4E8.

    48 random bits: collisions are negligible at this volume. It also names the
    case's files in Drive, so GiveLight can find a case by searching for it.
    """
    raw = secrets.token_hex(6).upper()
    return f"DC-{raw[:4]}-{raw[4:8]}-{raw[8:]}"


def build_handoff_payload(
    result: ReliabilityResult,
    contact_identifier: str | None,
    case_fields: dict[str, Any],
    *,
    contact_phone: str | None = None,
    case_reference: str | None = None,
    decision: str = DECISION_ACCEPTED,
    review_reasons: list[str] | None = None,
    authenticity: ToolResult | None = None,
    claimant_messages: list[str] | None = None,
) -> dict[str, Any]:
    """Build the JSON-serialisable payload sent to GiveLight.

    Every case with a document is sent: `decision` says whether it cleared
    automatic acceptance or needs a person to review it, and `review_reasons`
    says why in plain terms.

    contact_phone is the claimant's WhatsApp ID as received — their phone
    number in international format without a leading "+". It is sent in the
    clear deliberately: GiveLight needs it to contact the family.
    """
    return {
        "schema_version":      "1.0",
        "submitted_at":        datetime.now(timezone.utc).isoformat(),
        "case_reference":      case_reference,
        "decision":            decision,
        "review_reasons":      list(review_reasons or []),
        # For a caseworker: who is asking, how they are related, any dependants,
        # what they said happened and whether it matches the certificate.
        "case_note":           result.case_note,
        "claimant":            result.claimant,
        # Exactly as the claimant wrote them; the case note is an AI summary.
        "claimant_messages":   list(claimant_messages or []),
        "contact_phone":       contact_phone,
        "contact_identifier":  contact_identifier,
        "config_versions":     result.config_versions,
        "score":               result.score,
        "band":                result.band.value,
        "sub_scores":          result.sub_scores,
        "flags":               result.flags,
        "justification":       result.justification,
        "extracted_fields":    result.extracted_fields,
        "story_check": {
            "summary":          result.story_summary,
            "matches":          result.matches,
            "mismatches":       result.mismatches,
            "uncertain_points": result.uncertain_points,
        },
        "authenticity":        authenticity_summary(authenticity) if authenticity else None,
        "case_fields":         case_fields,
    }


def _multipart_body(metadata: dict[str, Any], content: bytes, mime_type: str) -> tuple[str, bytes]:
    # Random per request: a fixed boundary could also occur inside the image
    # bytes and silently corrupt the body.
    boundary = f"b2-{secrets.token_hex(16)}"
    body = (
        f"--{boundary}\r\n"
        "Content-Type: application/json; charset=UTF-8\r\n\r\n"
    ).encode() + json.dumps(metadata).encode("utf-8") + (
        f"\r\n--{boundary}\r\n"
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return boundary, body


def _get_access_token() -> str:
    from google.auth.exceptions import GoogleAuthError

    try:
        # Shared credentials, refreshed only on expiry (see google_clients).
        token = access_token(_DRIVE_SCOPE)
    except GoogleAuthError as exc:
        raise ValueError("Google Drive authentication failed") from exc
    if not token:
        raise ValueError("Google Drive authentication returned no access token")
    return token


def _retryable(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        if response.status_code in _RETRYABLE_STATUS:
            return True
        return response.status_code == 403 and "ratelimitexceeded" in response.text.lower()
    return isinstance(exc, httpx.TransportError)


def _describe(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"status={exc.response.status_code}"
    return type(exc).__name__


async def _upload_file(
    client: httpx.AsyncClient,
    token: str,
    folder_id: str,
    name: str,
    content: bytes,
    mime_type: str,
) -> None:
    """Upload one file, retrying temporary failures.

    A retry after a network error can, rarely, duplicate a file whose first
    attempt did reach Drive. Both copies share the same name, so they are
    recognisable as one case.
    """
    boundary, body = _multipart_body(
        {"name": name, "parents": [folder_id], "mimeType": mime_type},
        content,
        mime_type,
    )
    for attempt, delay in enumerate((*_RETRY_DELAYS_SECONDS, None), start=1):
        try:
            response = await client.post(
                _DRIVE_UPLOAD_URL,
                params={"uploadType": "multipart", "supportsAllDrives": "true"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": f"multipart/related; boundary={boundary}",
                },
                content=body,
            )
            response.raise_for_status()
            return
        except httpx.HTTPError as exc:
            if delay is None or not _retryable(exc):
                raise
            logger.warning(
                "handoff.retry file=%s attempt=%d %s wait=%.0fs",
                name, attempt, _describe(exc), delay,
            )
            await asyncio.sleep(delay)


async def deliver_to_gl(
    payload: dict[str, Any], image_bytes: bytes, image_mime_type: str
) -> bool:
    """Upload a case to GiveLight's Drive folder: the document, then its JSON.

    The JSON goes last and names its document in `image_file`, so a JSON file
    in the folder always means a complete case. If the JSON upload fails after
    the document succeeded, the stray document has no JSON and can be ignored.
    (Deleting it instead would need more than Contributor access on a Shared
    Drive.)

    Returns False on every failure and logs a distinct reason for each.
    """
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if not folder_id:
        logger.warning("handoff.skipped reason=GOOGLE_DRIVE_FOLDER_ID_unset")
        return False

    try:
        token = await asyncio.to_thread(_get_access_token)
    except ValueError as exc:
        logger.error("handoff.failed stage=auth error=%s", exc)
        return False

    stem = _stem(payload)
    # Trust the bytes over the declared type; fall back to the declaration.
    fmt = sniff(image_bytes) or by_mime(image_mime_type)
    image_name = f"{stem}{fmt.extension if fmt else '.bin'}"
    json_name = f"{stem}.json"
    upload_mime_type = fmt.mime_type if fmt else UNKNOWN_MIME_TYPE
    document = {**payload, "image_file": image_name}

    current = image_name
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await _upload_file(client, token, folder_id, image_name, image_bytes, upload_mime_type)
            current = json_name
            await _upload_file(
                client,
                token,
                folder_id,
                json_name,
                json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8"),
                "application/json",
            )
    except httpx.HTTPStatusError as exc:
        # Drive error bodies describe the request, not the payload, so they are
        # safe to log; truncated regardless.
        logger.error(
            "handoff.failed stage=upload file=%s status=%d detail=%.200s",
            current,
            exc.response.status_code,
            exc.response.text,
        )
        _log_incomplete(current, image_name)
        return False
    except (httpx.HTTPError, OSError) as exc:
        logger.error("handoff.failed stage=upload file=%s error=%s", current, exc)
        _log_incomplete(current, image_name)
        return False

    logger.info("handoff.ok stem=%s mime=%s bytes=%d", stem, upload_mime_type, len(image_bytes))
    return True


def _stem(payload: dict[str, Any]) -> str:
    """File name stem, e.g. death-certificate-needs-review-DC-7F3A-9C21-B4E8.

    The decision makes review cases easy to filter in Drive; the reference
    makes a case findable by the number the claimant was given.
    """
    reference = payload.get("case_reference")
    if not reference:
        return f"death-certificate-{uuid4()}"
    decision = str(payload.get("decision") or DECISION_ACCEPTED).replace("_", "-")
    return f"death-certificate-{decision}-{reference}"


def _log_incomplete(failed: str, image_name: str) -> None:
    if failed != image_name:
        logger.warning(
            "handoff.incomplete image_uploaded=%s json_missing — GiveLight can ignore it",
            image_name,
        )
