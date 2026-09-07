"""Package scored death-certificate results and upload them to GiveLight."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx

from tools.death_certificate_pipeline.models import ReliabilityResult

_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"
_DRIVE_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"
_IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "application/pdf": ".pdf",
}


def build_handoff_payload(
    result: ReliabilityResult,
    contact_identifier: str | None,
    case_fields: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON-serialisable payload sent to GiveLight."""
    return {
        "schema_version":      "1.0",
        "submitted_at":        datetime.now(timezone.utc).isoformat(),
        "contact_identifier":  contact_identifier,
        "score":               result.score,
        "band":                result.band.value,
        "sub_scores":          result.sub_scores,
        "flags":               result.flags,
        "justification":       result.justification,
        "extracted_fields":    result.extracted_fields,
        "case_fields":         case_fields,
    }


def _multipart_body(metadata: dict[str, Any], content: bytes, mime_type: str) -> tuple[str, bytes]:
    boundary = "b2-drive-handoff"
    body = (
        f"--{boundary}\r\n"
        "Content-Type: application/json; charset=UTF-8\r\n\r\n"
    ).encode() + json.dumps(metadata).encode("utf-8") + (
        f"\r\n--{boundary}\r\n"
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return boundary, body


def _get_access_token() -> str:
    import google.auth
    from google.auth.exceptions import GoogleAuthError
    from google.auth.transport.requests import Request

    try:
        credentials, _ = google.auth.default(scopes=[_DRIVE_SCOPE])
        credentials.refresh(Request())
    except GoogleAuthError as exc:
        raise ValueError("Google Drive authentication failed") from exc
    if not credentials.token:
        raise ValueError("Google Drive authentication returned no access token")
    return credentials.token


async def _upload_file(
    client: httpx.AsyncClient,
    token: str,
    folder_id: str,
    name: str,
    content: bytes,
    mime_type: str,
) -> None:
    boundary, body = _multipart_body(
        {"name": name, "parents": [folder_id], "mimeType": mime_type},
        content,
        mime_type,
    )
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


async def deliver_to_gl(
    payload: dict[str, Any], image_bytes: bytes, image_mime_type: str
) -> bool:
    """Upload a JSON payload and its source image to GiveLight's Drive folder."""
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if not folder_id:
        return False

    try:
        token = await asyncio.to_thread(_get_access_token)

        stem = f"death-certificate-{uuid4()}"
        extension = _IMAGE_EXTENSIONS.get(image_mime_type, ".bin")
        upload_mime_type = (
            image_mime_type if image_mime_type in _IMAGE_EXTENSIONS else "application/octet-stream"
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            await _upload_file(
                client,
                token,
                folder_id,
                f"{stem}.json",
                json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                "application/json",
            )
            await _upload_file(
                client,
                token,
                folder_id,
                f"{stem}{extension}",
                image_bytes,
                upload_mime_type,
            )
        return True
    except (httpx.HTTPError, OSError, ValueError):
        return False
