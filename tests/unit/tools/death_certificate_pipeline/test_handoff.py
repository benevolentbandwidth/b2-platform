"""Tests for the GiveLight handoff upload to Cloud Storage."""

import json

import httpx
import pytest

from tools.death_certificate_pipeline import handoff


async def test_missing_bucket_configuration_returns_false(monkeypatch):
    monkeypatch.delenv("GIVELIGHT_HANDOFF_BUCKET", raising=False)

    assert await handoff.deliver_to_gl({}, b"image", "image/jpeg") is False


async def test_uploads_json_and_original_image(monkeypatch):
    monkeypatch.setenv("GIVELIGHT_HANDOFF_BUCKET", "give-light-bucket")

    monkeypatch.setattr(handoff, "_get_access_token", lambda: "storage-token")
    monkeypatch.setattr(
        handoff,
        "uuid4",
        lambda: "12345678-1234-5678-1234-567812345678",
    )

    requests = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, timeout):
            assert timeout == 30.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            requests.append((url, kwargs))
            return Response()

    monkeypatch.setattr(handoff.httpx, "AsyncClient", Client)
    payload = {
        "submitted_at": "2026-08-07T12:34:56+00:00",
        "contact_identifier": "a" * 64,
        "score": 92,
    }

    assert await handoff.deliver_to_gl(payload, b"jpeg-bytes", "image/jpeg") is True
    assert len(requests) == 2

    for url, kwargs in requests:
        assert url == "https://storage.googleapis.com/upload/storage/v1/b/give-light-bucket/o"
        assert kwargs["headers"]["Authorization"] == "Bearer storage-token"
        # Create-only: B2 can never overwrite an object already in the bucket.
        assert kwargs["params"]["uploadType"] == "media"
        assert kwargs["params"]["ifGenerationMatch"] == "0"

    # Document first, JSON last: a JSON in the bucket means a complete case.
    image_name, json_name = (kwargs["params"]["name"] for _, kwargs in requests)
    assert json_name == "death-certificate-12345678-1234-5678-1234-567812345678.json"
    assert image_name == "death-certificate-12345678-1234-5678-1234-567812345678.jpg"
    assert "aaaaaaaaaaaaaaaa" not in json_name
    assert "aaaaaaaaaaaaaaaa" not in image_name
    assert "2026-08-07" not in json_name
    assert "2026-08-07" not in image_name
    assert json_name.removesuffix(".json") == image_name.removesuffix(".jpg")
    assert requests[0][1]["headers"]["Content-Type"] == "image/jpeg"
    assert requests[1][1]["headers"]["Content-Type"] == "application/json"
    assert requests[0][1]["content"] == b"jpeg-bytes"
    assert json.loads(requests[1][1]["content"]) == {**payload, "image_file": image_name}


async def test_file_extension_comes_from_the_bytes(monkeypatch):
    """TIFF used to upload as .bin; and the bytes outrank a wrong declared type."""
    monkeypatch.setenv("GIVELIGHT_HANDOFF_BUCKET", "give-light-bucket")
    monkeypatch.setattr(handoff, "_get_access_token", lambda: "storage-token")
    uploads = []

    async def capture(client, token, bucket, name, content, mime_type):
        uploads.append((name, mime_type))

    monkeypatch.setattr(handoff, "_upload_file", capture)

    assert await handoff.deliver_to_gl({}, b"II*\x00 tiff body", "image/jpeg") is True

    image_name, image_mime = uploads[0]
    assert image_name.endswith(".tiff")
    assert image_mime == "image/tiff"


# --- retries and ordering ----------------------------------------------------



def _response(status: int, text: str = "") -> httpx.Response:
    return httpx.Response(
        status, text=text, request=httpx.Request("POST", handoff._STORAGE_UPLOAD_URL.format(bucket="b"))
    )


class _ScriptedClient:
    """Answers each POST from a script of statuses, (status, body) pairs or exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        status, text = item if isinstance(item, tuple) else (item, "")
        return _response(status, text)


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr(handoff, "_RETRY_DELAYS_SECONDS", (0.0, 0.0))


async def _upload(client):
    await handoff._upload_file(client, "tok", "bucket", "x.jpg", b"data", "image/jpeg")


@pytest.mark.parametrize(
    "script",
    [
        [503, 429, 200],
        [408, 200],
        [httpx.ConnectError("connection reset"), 200],
    ],
    ids=["server-and-quota", "request-timeout", "network"],
)
async def test_temporary_failures_are_retried(no_wait, script):
    client = _ScriptedClient(script)
    await _upload(client)
    assert len(client.calls) == len(script)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 412])
async def test_permanent_failures_are_not_retried(no_wait, status):
    """A 404 (no such bucket), 403 (no write access) or a name already taken on
    the first attempt will not fix itself; fail fast."""
    client = _ScriptedClient([status])
    with pytest.raises(httpx.HTTPStatusError):
        await _upload(client)
    assert len(client.calls) == 1


async def test_gives_up_after_the_last_attempt(no_wait):
    client = _ScriptedClient([503, 503, 503])
    with pytest.raises(httpx.HTTPStatusError):
        await _upload(client)
    assert len(client.calls) == 3


async def test_a_retry_that_finds_its_object_already_uploaded_succeeds(no_wait):
    """The first attempt landed but its response was lost; the retry is refused
    as a duplicate name, which means the file is already there in full."""
    client = _ScriptedClient([httpx.ReadTimeout("response lost"), 412])
    await _upload(client)
    assert len(client.calls) == 2


async def test_json_failure_after_document_leaves_no_complete_case(monkeypatch, no_wait, caplog):
    """Document lands, JSON fails: report failure, and flag the stray document."""
    monkeypatch.setenv("GIVELIGHT_HANDOFF_BUCKET", "give-light-bucket")
    monkeypatch.setattr(handoff, "_get_access_token", lambda: "storage-token")
    scripted = _ScriptedClient([200, 400])

    class Client:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return scripted

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr(handoff.httpx, "AsyncClient", Client)
    caplog.set_level("WARNING", logger=handoff.__name__)

    assert await handoff.deliver_to_gl({"score": 1}, b"\xff\xd8 jpeg", "image/jpeg") is False
    assert "handoff.incomplete" in caplog.text
    assert ".jpg" in caplog.text  # names the stray document so it can be found


def test_case_reference_is_short_and_readable():
    import re

    reference = handoff.new_case_reference()
    assert re.fullmatch(r"DC-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}", reference)
    assert handoff.new_case_reference() != reference


def test_file_names_carry_the_decision_and_reference():
    assert handoff._stem({"case_reference": "DC-1111-2222-3333", "decision": "needs_review"}) == (
        "death-certificate-needs-review-DC-1111-2222-3333"
    )
    assert handoff._stem({"case_reference": "DC-1111-2222-3333", "decision": "accepted"}) == (
        "death-certificate-accepted-DC-1111-2222-3333"
    )
