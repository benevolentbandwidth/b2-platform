"""SynthID via Google's Imagen watermark verification model."""

import asyncio
import base64

import httpx
import pytest

from tools.fake_image_detector.checks import synthid_check
from tools.fake_image_detector.checks.synthid_check import VertexSynthIDCheck

_PNG = b"\x89PNG\r\n\x1a\n rest of png"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def api(monkeypatch):
    """Record the outgoing request and answer with a scripted response."""
    calls = []
    reply = {"status": 200, "json": {"predictions": [{"decision": "REJECT"}]}}

    def fake_post(url, *, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers})
        if isinstance(reply.get("raise"), Exception):
            raise reply["raise"]
        return httpx.Response(
            reply["status"], json=reply.get("json"), request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(synthid_check, "_access_token", lambda: "tok")
    monkeypatch.setattr(synthid_check.httpx, "post", fake_post)
    return calls, reply


def _check():
    return VertexSynthIDCheck(project="b2-platform")


def test_calls_the_imagen_verification_model_in_us_central1(api):
    calls, _ = api
    run(_check().run(_PNG, {}))

    (call,) = calls
    assert call["url"] == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/b2-platform"
        "/locations/us-central1/publishers/google/models/imageverification@001:predict"
    )
    image = call["json"]["instances"][0]["image"]["bytesBase64Encoded"]
    assert base64.b64decode(image) == _PNG
    assert call["headers"]["Authorization"] == "Bearer tok"


def test_watermark_detected_is_flagged_for_review(api):
    _, reply = api
    reply["json"] = {"predictions": [{"decision": "ACCEPT"}]}

    result = run(_check().run(_PNG, {}))

    assert result.skipped is False
    assert result.passed is False
    assert "SYNTHID_WATERMARK_DETECTED" in result.flags
    assert "LIKELY_AI_GENERATED" in result.flags


def test_no_watermark_carries_no_weight(api):
    """No Imagen watermark is not evidence the document is genuine."""
    result = run(_check().run(_PNG, {}))

    assert result.skipped is True
    assert result.confidence == 0.0
    assert result.flags == []
    assert result.signals["decision"] == "REJECT"


@pytest.mark.parametrize(
    ("reply_update", "reason"),
    [
        ({"status": 403, "json": {"error": {"code": 403}}}, "Google restricts this model"),
        ({"status": 404, "json": {"error": {"code": 404}}}, "no access"),
        ({"status": 503, "json": {}}, "HTTP 503"),
        ({"json": {"predictions": []}}, "unexpected response shape"),
        ({"json": {"predictions": [{"decision": "MAYBE"}]}}, "unrecognised decision"),
        ({"raise": httpx.ConnectError("reset")}, "request failed"),
    ],
    ids=["restricted", "not-found", "server-error", "empty", "unknown-decision", "network"],
)
def test_anything_that_stops_it_running_skips_and_never_escalates(api, reply_update, reason):
    _, reply = api
    reply.update(reply_update)

    result = run(_check().run(_PNG, {}))

    assert result.skipped is True
    assert result.error is None  # an error would early-exit the detector to review
    assert result.human_escalate is False
    assert reason in result.signals["reason"]


def test_without_a_project_it_skips_without_calling(api, monkeypatch):
    calls, _ = api
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    result = run(VertexSynthIDCheck().run(_PNG, {}))

    assert result.skipped is True
    assert calls == []


def test_pdfs_are_skipped_and_tiffs_converted(api):
    import io

    from PIL import Image

    calls, _ = api
    assert run(_check().run(b"%PDF-1.7 ...", {})).skipped is True
    assert calls == []

    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="TIFF")
    run(_check().run(buf.getvalue(), {}))
    sent = base64.b64decode(calls[0]["json"]["instances"][0]["image"]["bytesBase64Encoded"])
    assert sent.startswith(b"\x89PNG")


def test_location_comes_from_settings():
    check = VertexSynthIDCheck(params={"location": "europe-west4"}, project="p")
    assert check._location == "europe-west4"
