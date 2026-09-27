"""Unit tests for src/api.py — WhatsApp Channel Adapter endpoint."""

# These tests call message_endpoint() directly with FakeRequest instead of using
# FastAPI TestClient, which currently hangs in this Python/dependency setup.

import importlib


SAMPLE_TEXT_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "id": "1745012400192435",
            "changes": [
                {
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {
                            "display_phone_number": "16179164660",
                            "phone_number_id": "1104821716055506",
                        },
                        "contacts": [
                            {"profile": {"name": "Test User"}, "wa_id": "16508106640"}
                        ],
                        "messages": [
                            {
                                "from": "16508106640",
                                "id": "wamid.test",
                                "timestamp": "1780358445",
                                "text": {"body": "Hello B2, what is Givelight?"},
                                "type": "text",
                            }
                        ],
                    },
                    "field": "messages",
                }
            ],
        }
    ],
}

STATUS_UPDATE_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "id": "1745012400192435",
            "changes": [
                {
                    "value": {
                        "messaging_product": "whatsapp",
                        "statuses": [
                            {
                                "id": "wamid.test",
                                "status": "delivered",
                                "timestamp": "1780358446",
                                "recipient_id": "16508106640",
                            }
                        ],
                    },
                    "field": "messages",
                }
            ],
        }
    ],
}

IMAGE_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "id": "1745012400192435",
            "changes": [
                {
                    "value": {
                        "messaging_product": "whatsapp",
                        "contacts": [
                            {"profile": {"name": "Test User"}, "wa_id": "16508106640"}
                        ],
                        "messages": [
                            {
                                "from": "16508106640",
                                "id": "wamid.test2",
                                "timestamp": "1780358447",
                                "type": "image",
                                "image": {"id": "media-123", "mime_type": "image/jpeg"},
                            }
                        ],
                    },
                    "field": "messages",
                }
            ],
        }
    ],
}


def _make_client(secret: str = "", monkeypatch=None):
    """Import api module fresh with WEBHOOK_SECRET env var set."""
    if monkeypatch is not None:
        monkeypatch.setenv("WEBHOOK_SECRET", secret)

    # Re-import to pick up env var at module level
    import src.api as api_module
    importlib.reload(api_module)
    return api_module.app, api_module


class FakeRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


async def _run_direct(func, **kwargs):
    return func(**kwargs)


async def _no_summary(**kwargs):
    return None


# ---------------------------------------------------------------------------
# _extract_message unit tests
# ---------------------------------------------------------------------------

def test_extract_message_text():
    from src.api import _extract_message
    msg = _extract_message(SAMPLE_TEXT_PAYLOAD)
    assert msg.wa_id == "16508106640"
    assert msg.text == "Hello B2, what is Givelight?"
    assert msg.media_id is None
    assert msg.phone_number_id == "1104821716055506"


def test_extract_message_status_update_returns_empty():
    from src.api import _extract_message
    msg = _extract_message(STATUS_UPDATE_PAYLOAD)
    assert msg.text is None
    assert msg.media_id is None


def test_extract_message_image_returns_media_id():
    from src.api import _extract_message
    msg = _extract_message(IMAGE_PAYLOAD)
    assert msg.wa_id == "16508106640"
    assert msg.text is None
    assert msg.media_id == "media-123"
    assert msg.mime_type == "image/jpeg"
    assert msg.phone_number_id is None


def test_extract_message_empty_payload():
    from src.api import _extract_message
    msg = _extract_message({})
    assert msg.wa_id is None
    assert msg.text is None
    assert msg.media_id is None


def test_extract_message_blank_text():
    from src.api import _extract_message
    payload = {
        "entry": [{"changes": [{"value": {"messages": [{"type": "text", "text": {"body": "  "}, "from": "123"}]}}]}]
    }
    msg = _extract_message(payload)
    assert msg.text is None
    assert msg.media_id is None


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------

def test_health():
    from src.api import health
    assert health() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /message — no secret configured (open mode)
# ---------------------------------------------------------------------------

async def test_message_status_update_returns_null(monkeypatch):
    import src.api as api_module
    monkeypatch.setenv("WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_ALLOW_OPEN_WEBHOOK", "true")

    def fake_chat(**kwargs):
        return "should not be called"

    monkeypatch.setattr(api_module, "chat", fake_chat)

    response = await api_module.message_endpoint(FakeRequest(STATUS_UPDATE_PAYLOAD))
    assert response == {"response": None}


async def test_message_image_download_unavailable_returns_null(monkeypatch):
    """If media can't be downloaded (e.g. no WHATSAPP_TOKEN), respond with null."""
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_ALLOW_OPEN_WEBHOOK", "true")
    monkeypatch.setattr(api_module, "chat", lambda **kw: "should not be called")

    async def fake_download(media_id):
        return None

    monkeypatch.setattr(api_module, "download_media", fake_download)

    response = await api_module.message_endpoint(FakeRequest(IMAGE_PAYLOAD))
    assert response == {"response": None}


async def test_message_image_downloads_and_calls_chat(monkeypatch):
    """A downloadable image is fetched and routed to chat as image_bytes."""
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_ALLOW_OPEN_WEBHOOK", "true")

    captured = {}

    async def fake_download(media_id):
        captured["media_id"] = media_id
        return b"\xff\xd8jpeg-bytes", "image/jpeg"

    def fake_chat(*, image_bytes=None, image_media_type=None, session_id=None, **kwargs):
        captured["image_bytes"] = image_bytes
        captured["image_media_type"] = image_media_type
        captured["session_id"] = session_id
        return "Your request is being processed."

    monkeypatch.setattr(api_module, "download_media", fake_download)
    monkeypatch.setattr(api_module, "chat", fake_chat)
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_direct)
    monkeypatch.setattr(api_module, "generate_summary_tool_response", _no_summary)

    response = await api_module.message_endpoint(FakeRequest(IMAGE_PAYLOAD))
    assert response == {"response": "Your request is being processed."}
    assert captured["media_id"] == "media-123"
    assert captured["image_bytes"] == b"\xff\xd8jpeg-bytes"
    assert captured["image_media_type"] == "image/jpeg"
    assert captured["session_id"] == "16508106640"


async def test_message_text_calls_chat(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_ALLOW_OPEN_WEBHOOK", "true")

    captured = {}

    def fake_chat(*, text, session_id, **kwargs):
        captured["text"] = text
        captured["session_id"] = session_id
        return "Givelight is an orphan aid programme."

    monkeypatch.setattr(api_module, "chat", fake_chat)
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_direct)
    monkeypatch.setattr(api_module, "generate_summary_tool_response", _no_summary)

    response = await api_module.message_endpoint(FakeRequest(SAMPLE_TEXT_PAYLOAD))
    assert response == {"response": "Givelight is an orphan aid programme."}
    assert captured["text"] == "Hello B2, what is Givelight?"
    assert captured["session_id"] == "16508106640"


async def test_message_text_debug_requires_env_and_header(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_ALLOW_OPEN_WEBHOOK", "true")
    monkeypatch.setattr(api_module, "_ENABLE_E2E_DEBUG", "true")

    captured = {}

    def fake_chat(*, text, session_id, debug_events=None, **kwargs):
        captured["debug_events"] = debug_events
        return "ok"

    monkeypatch.setattr(api_module, "chat", fake_chat)
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_direct)
    monkeypatch.setattr(api_module, "generate_summary_tool_response", _no_summary)

    response = await api_module.message_endpoint(FakeRequest(SAMPLE_TEXT_PAYLOAD))
    assert response == {"response": "ok"}
    assert captured["debug_events"] == []


async def test_message_text_debug_header_without_env_returns_normal_response(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_ALLOW_OPEN_WEBHOOK", "true")
    monkeypatch.setattr(api_module, "_ENABLE_E2E_DEBUG", "")

    def fake_chat(*, text, session_id, debug_events=None, **kwargs):
        debug_events.append({"tool": "death_certificate_verification", "accepted": False})
        return "ok"

    monkeypatch.setattr(api_module, "chat", fake_chat)
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_direct)
    monkeypatch.setattr(api_module, "generate_summary_tool_response", _no_summary)

    response = await api_module.message_endpoint(
        FakeRequest(SAMPLE_TEXT_PAYLOAD), x_e2e_debug="true"
    )

    assert response == {"response": "ok"}


async def test_message_text_sends_summary_tool_response_as_second_message(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_ALLOW_OPEN_WEBHOOK", "true")
    monkeypatch.setattr(api_module, "_SEND_SUMMARY_MESSAGE", "true")

    captured = {}

    def fake_chat(*, text, session_id, debug_events=None, **kwargs):
        debug_events.append(
            {
                "tool": "death_certificate_verification",
                "accepted": True,
                "handed_off": True,
                "summary": "Verification passed.",
                "flags": [],
            }
        )
        return "The request was accepted."

    async def fake_generate_summary(*, final_response, tool_events):
        captured["summary_input"] = {
            "final_response": final_response,
            "tool_events": list(tool_events),
        }
        return "Your certificate was verified and forwarded to GiveLight."

    async def fake_send_text_message(*, wa_id, text, phone_number_id=None):
        captured["send"] = {
            "wa_id": wa_id,
            "text": text,
            "phone_number_id": phone_number_id,
        }
        return True

    monkeypatch.setattr(api_module, "chat", fake_chat)
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_direct)
    monkeypatch.setattr(api_module, "can_send_text_message", lambda *, wa_id, phone_number_id=None: True)
    monkeypatch.setattr(api_module, "generate_summary_tool_response", fake_generate_summary)
    monkeypatch.setattr(api_module, "send_text_message", fake_send_text_message)

    response = await api_module.message_endpoint(FakeRequest(SAMPLE_TEXT_PAYLOAD))

    assert response == {"response": "The request was accepted."}
    assert captured["summary_input"]["final_response"] == "The request was accepted."
    assert captured["summary_input"]["tool_events"][0]["summary"] == "Verification passed."
    assert captured["send"] == {
        "wa_id": "16508106640",
        "text": "Your certificate was verified and forwarded to GiveLight.",
        "phone_number_id": "1104821716055506",
    }


async def test_message_text_skips_summary_when_outbound_is_not_configured(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_ALLOW_OPEN_WEBHOOK", "true")

    called = {"summary": False}

    def fake_chat(*, text, session_id, debug_events=None, **kwargs):
        return "ok"

    async def fake_generate_summary(*, final_response, tool_events):
        called["summary"] = True
        return "summary"

    monkeypatch.setattr(api_module, "chat", fake_chat)
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_direct)
    monkeypatch.setattr(api_module, "can_send_text_message", lambda *, wa_id, phone_number_id=None: False)
    monkeypatch.setattr(api_module, "generate_summary_tool_response", fake_generate_summary)

    response = await api_module.message_endpoint(FakeRequest(SAMPLE_TEXT_PAYLOAD))

    assert response == {"response": "ok"}
    assert called["summary"] is False


async def test_message_text_keeps_original_response_when_summary_generation_fails(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_ALLOW_OPEN_WEBHOOK", "true")

    sent = {"called": False}

    def fake_chat(*, text, session_id, debug_events=None, **kwargs):
        return "ok"

    async def fake_generate_summary(*, final_response, tool_events):
        return None

    async def fake_send_text_message(*, wa_id, text, phone_number_id=None):
        sent["called"] = True
        return True

    monkeypatch.setattr(api_module, "chat", fake_chat)
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_direct)
    monkeypatch.setattr(api_module, "can_send_text_message", lambda *, wa_id, phone_number_id=None: True)
    monkeypatch.setattr(api_module, "generate_summary_tool_response", fake_generate_summary)
    monkeypatch.setattr(api_module, "send_text_message", fake_send_text_message)

    response = await api_module.message_endpoint(FakeRequest(SAMPLE_TEXT_PAYLOAD))

    assert response == {"response": "ok"}
    assert sent["called"] is False


async def test_message_text_debug_returns_tool_results(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_ALLOW_OPEN_WEBHOOK", "true")
    monkeypatch.setattr(api_module, "_ENABLE_E2E_DEBUG", "yes")

    def fake_chat(*, text, session_id, debug_events=None, **kwargs):
        assert debug_events == []
        debug_events.append(
            {
                "tool": "death_certificate_verification",
                "status": "verified",
                "score": 91,
                "band": "high",
                "accepted": True,
                "handed_off": True,
                "flags": [],
                "extracted_fields": {"full_name": "Jane Doe"},
                "summary": "Verification passed.",
            }
        )
        return "ok"

    monkeypatch.setattr(api_module, "chat", fake_chat)
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_direct)
    monkeypatch.setattr(api_module, "generate_summary_tool_response", _no_summary)

    response = await api_module.message_endpoint(FakeRequest(SAMPLE_TEXT_PAYLOAD), x_e2e_debug="true")
    assert response == {
        "response": "ok",
        "debug": {
            "session_id": "16508106640",
            "tool_results": [
                {
                    "tool": "death_certificate_verification",
                    "status": "verified",
                    "score": 91,
                    "band": "high",
                    "accepted": True,
                    "handed_off": True,
                    "flags": [],
                    "extracted_fields": {"full_name": "Jane Doe"},
                    "summary": "Verification passed.",
                }
            ],
        },
    }


# ---------------------------------------------------------------------------
# /message — secret header enforcement
# ---------------------------------------------------------------------------

async def test_message_missing_secret_returns_401(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "mysecret")

    try:
        await api_module.message_endpoint(FakeRequest(SAMPLE_TEXT_PAYLOAD))
    except api_module.HTTPException as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("expected HTTPException")


async def test_message_wrong_secret_returns_401(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "mysecret")

    try:
        await api_module.message_endpoint(
            FakeRequest(SAMPLE_TEXT_PAYLOAD),
            x_webhook_secret="wrongsecret",
        )
    except api_module.HTTPException as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("expected HTTPException")


async def test_message_correct_secret_passes(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "mysecret")
    monkeypatch.setattr(api_module, "chat", lambda *, text, **kw: "ok")
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_direct)
    monkeypatch.setattr(api_module, "generate_summary_tool_response", _no_summary)

    response = await api_module.message_endpoint(
        FakeRequest(SAMPLE_TEXT_PAYLOAD),
        x_webhook_secret="mysecret",
    )
    assert response["response"] == "ok"


# ---------------------------------------------------------------------------
# /message — fail-closed auth, turn failure fallback, summary send gate
# ---------------------------------------------------------------------------

async def test_message_unset_secret_refuses_with_503(monkeypatch):
    """An unconfigured secret must refuse the request, not accept it."""
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_ALLOW_OPEN_WEBHOOK", "")

    try:
        await api_module.message_endpoint(FakeRequest(SAMPLE_TEXT_PAYLOAD))
    except api_module.HTTPException as exc:
        assert exc.status_code == 503
    else:
        raise AssertionError("expected HTTPException")


async def test_message_chat_failure_returns_fallback_not_500(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_ALLOW_OPEN_WEBHOOK", "true")

    def boom(**kwargs):
        raise RuntimeError("firestore unavailable")

    monkeypatch.setattr(api_module, "chat", boom)
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_direct)
    monkeypatch.setattr(api_module, "generate_summary_tool_response", _no_summary)

    response = await api_module.message_endpoint(FakeRequest(SAMPLE_TEXT_PAYLOAD))

    assert response["response"] == api_module.FALLBACK_RESPONSE


async def test_message_unroutable_query_returns_guidance(monkeypatch):
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_ALLOW_OPEN_WEBHOOK", "true")

    def below_threshold(**kwargs):
        raise api_module.BelowThresholdError(0.1, 0.35)

    monkeypatch.setattr(api_module, "chat", below_threshold)
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_direct)
    monkeypatch.setattr(api_module, "generate_summary_tool_response", _no_summary)

    response = await api_module.message_endpoint(FakeRequest(SAMPLE_TEXT_PAYLOAD))

    assert response["response"] == api_module.UNROUTABLE_RESPONSE


async def test_message_does_not_send_summary_when_gate_is_off(monkeypatch):
    """The caller delivers the response; b2 stays silent unless explicitly enabled."""
    import src.api as api_module
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "_ALLOW_OPEN_WEBHOOK", "true")
    monkeypatch.setattr(api_module, "_SEND_SUMMARY_MESSAGE", "")

    sent = []
    monkeypatch.setattr(api_module, "chat", lambda *, text, **kw: "ok")
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_direct)
    monkeypatch.setattr(api_module, "can_send_text_message", lambda **kw: sent.append("checked") or True)

    response = await api_module.message_endpoint(FakeRequest(SAMPLE_TEXT_PAYLOAD))

    assert response["response"] == "ok"
    assert sent == []
