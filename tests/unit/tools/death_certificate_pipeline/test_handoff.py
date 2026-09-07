"""Tests for the Google Drive GiveLight handoff."""

import json

from tools.death_certificate_pipeline import handoff


async def test_missing_folder_configuration_returns_false(monkeypatch):
    monkeypatch.delenv("GOOGLE_DRIVE_FOLDER_ID", raising=False)

    assert await handoff.deliver_to_gl({}, b"image", "image/jpeg") is False


async def test_uploads_json_and_original_image(monkeypatch):
    monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "give-light-folder")

    monkeypatch.setattr(handoff, "_get_access_token", lambda: "drive-token")
    monkeypatch.setattr(
        handoff,
        "uuid4",
        lambda: "12345678-1234-5678-1234-567812345678",
    )

    requests = []

    class Response:
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

    metadata = []
    contents = []
    for url, kwargs in requests:
        assert url == handoff._DRIVE_UPLOAD_URL
        assert kwargs["params"] == {"uploadType": "multipart", "supportsAllDrives": "true"}
        assert kwargs["headers"]["Authorization"] == "Bearer drive-token"
        assert kwargs["headers"]["Content-Type"].startswith("multipart/related; boundary=")
        _, remainder = kwargs["content"].split(b"\r\n\r\n", 1)
        raw_metadata, remainder = remainder.split(b"\r\n--b2-drive-handoff\r\n", 1)
        metadata.append(json.loads(raw_metadata))
        contents.append(remainder.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n--", 1)[0])

    json_name, image_name = metadata[0]["name"], metadata[1]["name"]
    assert json_name == "death-certificate-12345678-1234-5678-1234-567812345678.json"
    assert image_name == "death-certificate-12345678-1234-5678-1234-567812345678.jpg"
    assert "aaaaaaaaaaaaaaaa" not in json_name
    assert "aaaaaaaaaaaaaaaa" not in image_name
    assert "2026-08-07" not in json_name
    assert "2026-08-07" not in image_name
    assert json_name.removesuffix(".json") == image_name.removesuffix(".jpg")
    assert metadata[0]["parents"] == metadata[1]["parents"] == ["give-light-folder"]
    assert metadata[0]["mimeType"] == "application/json"
    assert metadata[1]["mimeType"] == "image/jpeg"
    assert json.loads(contents[0]) == payload
    assert contents[1] == b"jpeg-bytes"
