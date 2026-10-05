import base64
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse, TextPart, UserPromptPart

from src.session_store import MAX_MEDIA_BYTES, MEDIA_CHUNK_BYTES, FirestoreSessionStore


class FakeSnapshot:
    def __init__(self, exists: bool, data=None):
        self.exists = exists
        self._data = data

    def to_dict(self):
        return self._data


class FakeDocument:
    def __init__(self, snapshot: FakeSnapshot):
        self.snapshot = snapshot
        self.writes = []
        self.deletes = 0

    def get(self):
        return self.snapshot

    def set(self, data, *, merge: bool):
        self.writes.append((data, merge))
        self.snapshot = FakeSnapshot(True, data)

    def delete(self):
        self.deletes += 1
        self.snapshot = FakeSnapshot(False)


class FakeCollection:
    def __init__(self, document: FakeDocument):
        self.document_ref = document

    def document(self, session_id: str):
        assert session_id == "session-1"
        return self.document_ref


class FakeClient:
    def __init__(self, document: FakeDocument):
        self.document_ref = document
        self.collection_name = None

    def collection(self, name: str):
        self.collection_name = name
        return FakeCollection(self.document_ref)


def test_load_history_deserializes_stored_messages() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="hi")]),
    ]
    store = FirestoreSessionStore(
        client=FakeClient(
            FakeDocument(
                FakeSnapshot(
                    True,
                    {
                        "history": ModelMessagesTypeAdapter.dump_python(history, mode="json"),
                        "expires_at": now + timedelta(hours=1),
                    },
                )
            )
        ),
        server_timestamp="SERVER_TIME",
        now=lambda: now,
    )

    assert store.load_history("session-1") == history


def test_load_history_returns_empty_list_and_deletes_expired_document() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    document = FakeDocument(
        FakeSnapshot(
            True,
            {
                "history": ModelMessagesTypeAdapter.dump_python(
                    [ModelRequest(parts=[UserPromptPart(content="hello")])],
                    mode="json",
                ),
                "expires_at": now,
            },
        )
    )
    store = FirestoreSessionStore(
        client=FakeClient(document),
        server_timestamp="SERVER_TIME",
        now=lambda: now,
    )

    assert store.load_history("session-1") == []
    assert document.deletes == 1


def test_load_history_returns_empty_list_for_missing_document() -> None:
    store = FirestoreSessionStore(
        client=FakeClient(FakeDocument(FakeSnapshot(False))),
        server_timestamp="SERVER_TIME",
    )

    assert store.load_history("session-1") == []


def test_save_history_serializes_messages_and_sets_metadata_on_create() -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    document = FakeDocument(FakeSnapshot(False))
    client = FakeClient(document)
    store = FirestoreSessionStore(client=client, server_timestamp="SERVER_TIME", now=lambda: now)
    history = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="hi")]),
    ]

    store.save_history("session-1", history, agent_name="support", channel="whatsapp")

    assert client.collection_name == "sessions"
    assert len(document.writes) == 1
    data, merge = document.writes[0]
    assert merge is True
    assert data["session_id"] == "session-1"
    assert data["updated_at"] == "SERVER_TIME"
    assert data["created_at"] == "SERVER_TIME"
    assert data["expires_at"] == now + timedelta(hours=72)
    assert data["agent_name"] == "support"
    assert data["channel"] == "whatsapp"
    assert data["history"][0]["kind"] == "request"
    assert data["history"][1]["kind"] == "response"


# ---------------------------------------------------------------------------
# transient media store
# ---------------------------------------------------------------------------


class FakeMediaDocument:
    def __init__(self, snapshot: FakeSnapshot):
        self.snapshot = snapshot
        self.writes = []
        self.deletes = 0

    def get(self):
        return self.snapshot

    def set(self, data, merge: bool = False):
        self.writes.append((data, merge))
        self.snapshot = FakeSnapshot(True, data)

    def delete(self):
        self.deletes += 1
        self.snapshot = FakeSnapshot(False)


class FakeMediaClient:
    def __init__(self, document: FakeMediaDocument):
        self.document_ref = document
        self.collection_names: list[str] = []

    def collection(self, name: str):
        self.collection_names.append(name)
        return FakeCollection(self.document_ref)


def _media_store(document: FakeMediaDocument, now: datetime):
    return FirestoreSessionStore(
        client=FakeMediaClient(document),
        server_timestamp="SERVER_TIME",
        now=lambda: now,
    )


def test_save_media_stores_base64_and_metadata() -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    document = FakeMediaDocument(FakeSnapshot(False))
    client = FakeMediaClient(document)
    store = FirestoreSessionStore(client=client, server_timestamp="SERVER_TIME", now=lambda: now)

    ok = store.save_media("session-1", b"\xff\xd8abc", mime_type="image/jpeg")

    assert ok is True
    assert "session_media" in client.collection_names
    data, _merge = document.writes[0]
    assert data["session_id"] == "session-1"
    assert data["image_b64"] == base64.b64encode(b"\xff\xd8abc").decode("ascii")
    assert data["mime_type"] == "image/jpeg"
    assert data["expires_at"] == now + timedelta(hours=72)


def test_load_latest_media_round_trips_bytes() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    document = FakeMediaDocument(
        FakeSnapshot(
            True,
            {
                "image_b64": base64.b64encode(b"\xff\xd8abc").decode("ascii"),
                "mime_type": "image/png",
                "expires_at": now + timedelta(hours=1),
            },
        )
    )
    store = _media_store(document, now)

    assert store.load_latest_media("session-1") == (b"\xff\xd8abc", "image/png")


def test_load_latest_media_deletes_expired_and_returns_none() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    document = FakeMediaDocument(
        FakeSnapshot(
            True,
            {
                "image_b64": base64.b64encode(b"abc").decode("ascii"),
                "expires_at": now,
            },
        )
    )
    store = _media_store(document, now)

    assert store.load_latest_media("session-1") is None
    assert document.deletes == 1


def test_load_latest_media_returns_none_when_missing() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = _media_store(FakeMediaDocument(FakeSnapshot(False)), now)
    assert store.load_latest_media("session-1") is None


def test_save_media_skips_oversized_image() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    document = FakeMediaDocument(FakeSnapshot(False))
    store = _media_store(document, now)

    ok = store.save_media("session-1", b"x" * (MAX_MEDIA_BYTES + 1), mime_type="image/jpeg")

    assert ok is False
    assert document.writes == []


# ---------------------------------------------------------------------------
# large images, split across several records
# ---------------------------------------------------------------------------

class _MemoryFirestore:
    """Path-keyed stand-in for Firestore that supports sub-collections."""

    def __init__(self):
        self.docs: dict[str, dict] = {}

    def collection(self, name):
        return _MemoryCollection(self, name)


class _MemoryCollection:
    def __init__(self, db, path):
        self.db, self.path = db, path

    def document(self, doc_id):
        return _MemoryDocument(self.db, f"{self.path}/{doc_id}")


class _MemoryDocument:
    def __init__(self, db, path):
        self.db, self.path = db, path

    def get(self):
        data = self.db.docs.get(self.path)
        return SimpleNamespace(exists=data is not None, to_dict=lambda: dict(data) if data else None)

    def set(self, data, merge=False):
        self.db.docs[self.path] = dict(data)

    def delete(self):
        self.db.docs.pop(self.path, None)

    def collection(self, name):
        return _MemoryCollection(self.db, f"{self.path}/{name}")


def _pieces(db):
    return {path: doc for path, doc in db.docs.items() if "data_b64" in doc}


def _store(db, now=datetime(2026, 1, 1, tzinfo=timezone.utc)):
    return FirestoreSessionStore(client=db, server_timestamp="SERVER_TIME", now=lambda: now)


def _image(size: int, seed: int = 7) -> bytes:
    return bytes((i * seed) % 251 for i in range(size))


def test_large_image_round_trips_across_several_records() -> None:
    """Regression: anything over 700 KB used to be dropped, and the claimant was
    told no document had been received. Most PDFs and screenshots are bigger."""
    db = _MemoryFirestore()
    store = _store(db)
    image = _image(int(MEDIA_CHUNK_BYTES * 3.5))

    assert store.save_media("s1", image, mime_type="application/pdf") is True
    assert store.load_latest_media("s1") == (image, "application/pdf")
    assert len(_pieces(db)) == 4


def test_every_record_stays_under_the_firestore_document_limit() -> None:
    db = _MemoryFirestore()
    _store(db).save_media("s1", _image(MAX_MEDIA_BYTES), mime_type="image/png")

    for doc in db.docs.values():
        assert sum(len(str(v)) for v in doc.values()) < 1_000_000


def test_pieces_carry_the_record_expiry() -> None:
    db = _MemoryFirestore()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _store(db, now).save_media("s1", _image(MEDIA_CHUNK_BYTES * 2), mime_type="image/png")

    assert all(doc["expires_at"] == now + timedelta(hours=72) for doc in _pieces(db).values())


@pytest.mark.parametrize("second_size", [1_000, MEDIA_CHUNK_BYTES * 2], ids=["small", "large"])
def test_a_new_upload_clears_the_previous_pieces(second_size) -> None:
    db = _MemoryFirestore()
    store = _store(db)
    store.save_media("s1", _image(MEDIA_CHUNK_BYTES * 3), mime_type="image/png")
    second = _image(second_size, seed=13)

    store.save_media("s1", second, mime_type="image/jpeg")

    assert store.load_latest_media("s1") == (second, "image/jpeg")
    expected = 0 if second_size <= MEDIA_CHUNK_BYTES else 2
    assert len(_pieces(db)) == expected


def test_a_missing_piece_reads_as_no_image_rather_than_a_corrupt_one() -> None:
    db = _MemoryFirestore()
    store = _store(db)
    store.save_media("s1", _image(MEDIA_CHUNK_BYTES * 2), mime_type="image/png")
    db.docs.pop(sorted(_pieces(db))[0])

    assert store.load_latest_media("s1") is None


def test_expired_large_image_is_deleted_with_its_pieces() -> None:
    db = _MemoryFirestore()
    saved_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _store(db, saved_at).save_media("s1", _image(MEDIA_CHUNK_BYTES * 2), mime_type="image/png")

    later = _store(db, saved_at + timedelta(hours=73))
    assert later.load_latest_media("s1") is None
    assert db.docs == {}


def test_image_over_the_limit_is_refused_and_nothing_is_written() -> None:
    db = _MemoryFirestore()
    assert _store(db).save_media("s1", _image(MAX_MEDIA_BYTES + 1), mime_type="image/png") is False
    assert db.docs == {}



def test_pieces_share_the_media_collection_group_for_ttl() -> None:
    """Firestore TTL policies are per collection group. Pieces in a differently
    named sub-collection would need a policy of their own; without one they
    would never be deleted."""
    db = _MemoryFirestore()
    _store(db).save_media("s1", _image(MEDIA_CHUNK_BYTES * 2), mime_type="image/png")

    for path in _pieces(db):
        collection_id = path.split("/")[-2]
        assert collection_id == "session_media"
