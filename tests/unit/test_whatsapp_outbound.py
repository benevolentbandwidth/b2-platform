from __future__ import annotations

import httpx

from src import whatsapp_outbound


async def test_send_text_message_posts_to_graph_api(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, *, json, headers):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.delenv("E2E_DISABLE_WHATSAPP_OUTBOUND", raising=False)
    monkeypatch.setenv("WHATSAPP_TOKEN", "token-123")
    monkeypatch.setenv("WHATSAPP_GRAPH_VERSION", "v99.0")
    monkeypatch.setattr(whatsapp_outbound.httpx, "AsyncClient", FakeClient)

    sent = await whatsapp_outbound.send_text_message(
        wa_id="16508106640",
        phone_number_id="1104821716055506",
        text="Verification summary.",
    )

    assert sent is True
    assert captured["url"] == "https://graph.facebook.com/v99.0/1104821716055506/messages"
    assert captured["headers"]["Authorization"] == "Bearer token-123"
    assert captured["json"] == {
        "messaging_product": "whatsapp",
        "to": "16508106640",
        "type": "text",
        "text": {"body": "Verification summary."},
    }


async def test_send_text_message_skips_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("E2E_DISABLE_WHATSAPP_OUTBOUND", "1")
    monkeypatch.setenv("WHATSAPP_TOKEN", "token-123")

    sent = await whatsapp_outbound.send_text_message(
        wa_id="16508106640",
        phone_number_id="1104821716055506",
        text="Verification summary.",
    )

    assert sent is False


async def test_send_text_message_skips_missing_configuration(monkeypatch) -> None:
    monkeypatch.delenv("E2E_DISABLE_WHATSAPP_OUTBOUND", raising=False)
    monkeypatch.delenv("WHATSAPP_TOKEN", raising=False)
    monkeypatch.delenv("WHATSAPP_PHONE_NUMBER_ID", raising=False)
    # No Secret Manager fallback available either — must not attempt a live call.
    monkeypatch.setattr(whatsapp_outbound, "resolve_secret", lambda *a, **k: "")

    assert await whatsapp_outbound.send_text_message(wa_id="16508106640", text="x") is False


async def test_send_text_message_falls_back_to_secret_manager(monkeypatch) -> None:
    """When the env vars are unset, the token and sender id come from Secret Manager."""
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, *, json, headers):
            captured["url"] = url
            captured["headers"] = headers
            return FakeResponse()

    def fake_resolve(env_var: str, secret_name: str) -> str:
        return {"WHATSAPP_TOKEN": "secret-token", "WHATSAPP_PHONE_NUMBER_ID": "secret-sender"}[env_var]

    monkeypatch.delenv("E2E_DISABLE_WHATSAPP_OUTBOUND", raising=False)
    monkeypatch.delenv("WHATSAPP_TOKEN", raising=False)
    monkeypatch.delenv("WHATSAPP_PHONE_NUMBER_ID", raising=False)
    monkeypatch.setattr(whatsapp_outbound, "resolve_secret", fake_resolve)
    monkeypatch.setattr(whatsapp_outbound.httpx, "AsyncClient", FakeClient)

    sent = await whatsapp_outbound.send_text_message(wa_id="16508106640", text="x")

    assert sent is True
    assert captured["url"] == "https://graph.facebook.com/v21.0/secret-sender/messages"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"


async def test_send_text_message_returns_false_on_http_error(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "bad",
                request=httpx.Request("POST", "http://test"),
                response=httpx.Response(500),
            )

    class FakeClient:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, *, json, headers):
            return FakeResponse()

    monkeypatch.delenv("E2E_DISABLE_WHATSAPP_OUTBOUND", raising=False)
    monkeypatch.setenv("WHATSAPP_TOKEN", "token-123")
    monkeypatch.setattr(whatsapp_outbound.httpx, "AsyncClient", FakeClient)

    sent = await whatsapp_outbound.send_text_message(
        wa_id="16508106640",
        phone_number_id="1104821716055506",
        text="Verification summary.",
    )

    assert sent is False
