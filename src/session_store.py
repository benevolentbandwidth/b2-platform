"""Transient Firestore store for chat history and the latest uploaded image.

Expiry: every record (and every image piece) carries `expires_at`. The code
deletes an expired record only when the same session reads it again, so
deletion of abandoned sessions depends on Firestore TTL policies on
`expires_at` for the `sessions` and `session_media` collection groups. Image
pieces deliberately live in a `session_media` sub-collection so that one
policy covers them too. Without these policies nothing is ever deleted for a
session that does not come back.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TTL = timedelta(hours=72)

# Firestore caps a document at ~1 MB and base64 inflates bytes by ~33%, so an
# image up to this size fits in one document. Larger images are split into
# pieces of this size.
MEDIA_CHUNK_BYTES = 700_000
# Largest image accepted. Matches WhatsApp's photo limit and the 5 MB ceiling of
# Drive's multipart upload used for the GiveLight handoff: anything bigger would
# pass verification and then fail to reach GiveLight.
MAX_MEDIA_BYTES = 5_000_000


class FirestoreSessionStore:
    def __init__(
        self,
        *,
        client: Any | None = None,
        project: str | None = None,
        collection: str = "sessions",
        media_collection: str = "session_media",
        server_timestamp: Any | None = None,
        ttl: timedelta = DEFAULT_SESSION_TTL,
        now: Callable[[], datetime] | None = None,
    ):
        firestore = None if client is not None else _firestore_module()
        self._client = client or firestore.Client(project=project or os.getenv("GOOGLE_CLOUD_PROJECT"))
        self._collection = collection
        self._media_collection = media_collection
        self._server_timestamp = (
            server_timestamp
            if server_timestamp is not None
            else (firestore or _firestore_module()).SERVER_TIMESTAMP
        )
        self._ttl = ttl
        self._now = now or _utc_now

    def load_history(self, session_id: str) -> list[ModelMessage]:
        document = self._document(session_id)
        snapshot = document.get()
        if not snapshot.exists:
            return []

        data = snapshot.to_dict() or {}
        expires_at = data.get("expires_at")
        if isinstance(expires_at, datetime) and expires_at <= self._now():
            document.delete()
            return []

        history = data.get("history") or []
        return list(ModelMessagesTypeAdapter.validate_python(history))

    def save_history(
        self,
        session_id: str,
        history: Sequence[ModelMessage],
        *,
        agent_name: str | None = None,
        channel: str | None = None,
    ) -> None:
        document = self._document(session_id)
        snapshot = document.get()
        data: dict[str, Any] = {
            "session_id": session_id,
            "history": ModelMessagesTypeAdapter.dump_python(history, mode="json"),
            "updated_at": self._server_timestamp,
            "expires_at": self._now() + self._ttl,
        }

        if not snapshot.exists:
            data["created_at"] = self._server_timestamp
        if agent_name is not None:
            data["agent_name"] = agent_name
        if channel is not None:
            data["channel"] = channel

        document.set(data, merge=True)

    def save_media(self, session_id: str, image_bytes: bytes, *, mime_type: str) -> bool:
        """Persist the most recent inbound image for a session.

        Stores one record per session (overwritten on each upload) so tools can
        later pull the latest media by session_id without the image ever passing
        through the model. Images over MEDIA_CHUNK_BYTES are split into pieces in
        a sub-collection; the pieces are written before the record that points
        at them, so a reader never follows a half-written upload.

        Returns False (and writes nothing) if the image is over MAX_MEDIA_BYTES.
        """
        if not image_bytes:
            return False
        if len(image_bytes) > MAX_MEDIA_BYTES:
            logger.warning(
                "session_store.save_media skipped — image too large bytes=%d limit=%d session=%.8s",
                len(image_bytes),
                MAX_MEDIA_BYTES,
                _session_tag(session_id),
            )
            return False

        document = self._media_document(session_id)
        replaced = self._chunk_refs(session_id, _snapshot_data(document.get()))

        expires_at = self._now() + self._ttl
        record: dict[str, Any] = {
            "session_id": session_id,
            "mime_type": mime_type,
            "created_at": self._server_timestamp,
            "expires_at": expires_at,
        }
        if len(image_bytes) <= MEDIA_CHUNK_BYTES:
            record["image_b64"] = base64.b64encode(image_bytes).decode("ascii")
        else:
            upload_id = uuid4().hex[:12]
            pieces = [
                image_bytes[i : i + MEDIA_CHUNK_BYTES]
                for i in range(0, len(image_bytes), MEDIA_CHUNK_BYTES)
            ]
            for index, piece in enumerate(pieces):
                self._chunk(session_id, upload_id, index).set(
                    {
                        "data_b64": base64.b64encode(piece).decode("ascii"),
                        # Same expiry as the record, so a TTL policy can reap them.
                        "expires_at": expires_at,
                    }
                )
            record.update(
                upload_id=upload_id,
                chunk_count=len(pieces),
                total_bytes=len(image_bytes),
            )

        document.set(record)
        _delete_all(replaced)
        return True

    def load_latest_media(self, session_id: str) -> tuple[bytes, str] | None:
        """Return (image_bytes, mime_type) for the most recent image, or None."""
        document = self._media_document(session_id)
        snapshot = document.get()
        if not snapshot.exists:
            return None

        data = snapshot.to_dict() or {}
        expires_at = data.get("expires_at")
        if isinstance(expires_at, datetime) and expires_at <= self._now():
            _delete_all(self._chunk_refs(session_id, data))
            document.delete()
            return None

        mime_type = data.get("mime_type") or "image/jpeg"
        encoded = data.get("image_b64")
        if encoded:
            return base64.b64decode(encoded), mime_type

        refs = self._chunk_refs(session_id, data)
        if not refs:
            return None
        pieces: list[bytes] = []
        for ref in refs:
            piece = _snapshot_data(ref.get()).get("data_b64")
            if not piece:
                logger.warning(
                    "session_store.load_media missing piece session=%.8s", _session_tag(session_id)
                )
                return None
            pieces.append(base64.b64decode(piece))

        image = b"".join(pieces)
        if data.get("total_bytes") not in (None, len(image)):
            logger.warning(
                "session_store.load_media size mismatch session=%.8s", _session_tag(session_id)
            )
            return None
        return image, mime_type

    def _chunk(self, session_id: str, upload_id: str, index: int) -> Any:
        # The pieces' sub-collection deliberately shares the media collection's
        # name. Firestore TTL policies apply per collection group (every
        # collection with the same name), so the policy that expires media
        # records also expires their pieces, including any orphaned by two
        # uploads racing. A differently named sub-collection would need its own
        # policy, and without one its pieces would never be deleted.
        return (
            self._media_document(session_id)
            .collection(self._media_collection)
            .document(f"{upload_id}-{index:03d}")
        )

    def _chunk_refs(self, session_id: str, data: dict[str, Any]) -> list[Any]:
        upload_id, count = data.get("upload_id"), data.get("chunk_count")
        if not upload_id or not isinstance(count, int):
            return []
        return [self._chunk(session_id, upload_id, i) for i in range(count)]

    def _document(self, session_id: str) -> Any:
        return self._client.collection(self._collection).document(session_id)

    def _media_document(self, session_id: str) -> Any:
        return self._client.collection(self._media_collection).document(session_id)


def _snapshot_data(snapshot: Any) -> dict[str, Any]:
    return (snapshot.to_dict() or {}) if snapshot.exists else {}


def _delete_all(refs: list[Any]) -> None:
    for ref in refs:
        ref.delete()


def _session_tag(session_id: str) -> str:
    """Log-safe stand-in for the session id (a phone number)."""
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _firestore_module() -> Any:
    from google.cloud import firestore

    return firestore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
