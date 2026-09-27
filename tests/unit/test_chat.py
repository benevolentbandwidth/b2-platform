from pydantic_ai.messages import (
    BinaryImage,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    UserPromptPart,
)

from src import chat as chat_module
from src.chat import (
    IMAGE_ARRIVED_PROMPT,
    IMAGE_PROMPT_TEXT,
    _build_prompt,
    _render_history_text,
    chat,
)


def test_build_prompt_accepts_text_only() -> None:
    assert _build_prompt(
        text="  hello  ",
        image_bytes=None,
        image_url=None,
        image_media_type="image/jpeg",
    ) == "hello"


def test_build_prompt_accepts_image_bytes_only() -> None:
    prompt = _build_prompt(
        text=None,
        image_bytes=b"image",
        image_url=None,
        image_media_type="image/png",
    )

    assert isinstance(prompt, list)
    assert prompt == [
        TextContent(content=IMAGE_PROMPT_TEXT),
        BinaryImage(data=b"image", media_type="image/png"),
    ]


def test_build_prompt_accepts_image_url_only() -> None:
    prompt = _build_prompt(
        text=None,
        image_bytes=None,
        image_url=" https://example.test/image.jpg ",
        image_media_type="image/jpeg",
    )

    assert isinstance(prompt, list)
    assert prompt == [
        TextContent(content=IMAGE_PROMPT_TEXT),
        ImageUrl(url="https://example.test/image.jpg", media_type="image/jpeg"),
    ]


def test_build_prompt_rejects_multiple_inputs() -> None:
    try:
        _build_prompt(
            text="hello",
            image_bytes=b"image",
            image_url=None,
            image_media_type="image/jpeg",
        )
    except ValueError as exc:
        assert str(exc) == "Provide exactly one of text, image_bytes, or image_url"
    else:
        raise AssertionError("expected ValueError")


def test_chat_routes_and_returns_streamed_text(monkeypatch) -> None:
    created_sessions = []

    class FakeRouter:
        def route_with_metadata(self, query: str):
            assert query == "hello"
            return object(), {"score": 1.0}

    class FakeSession:
        def __init__(self, agent: object, history=None, deps=None):
            self.agent = agent
            self.history = list(history or [])
            self.deps = deps
            created_sessions.append(self)

        def send_stream(self, prompt: str):
            assert prompt == "hello"
            yield "hi"
            yield " there"

    monkeypatch.setattr(chat_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(chat_module, "_router", None)
    monkeypatch.setattr(chat_module, "AgentRouter", FakeRouter)
    monkeypatch.setattr(chat_module, "Session", FakeSession)
    monkeypatch.setattr(
        chat_module,
        "FirestoreSessionStore",
        lambda: (_ for _ in ()).throw(AssertionError("store should not be used")),
    )

    assert chat(text="hello") == "hi there"
    assert created_sessions[0].history == []


def test_chat_loads_and_saves_stateful_history(monkeypatch) -> None:
    loaded_history = [object()]
    saved = {}

    class FakeAgent:
        name = "support"

    class FakeRouter:
        def route_with_metadata(self, query: str):
            assert query == "hello"
            return FakeAgent(), {"score": 1.0}

    class FakeStore:
        def load_history(self, session_id: str):
            assert session_id == "session-1"
            return loaded_history

        def save_history(self, session_id: str, history, *, agent_name=None, channel=None) -> None:
            saved["session_id"] = session_id
            saved["history"] = history
            saved["agent_name"] = agent_name
            saved["channel"] = channel

    class FakeSession:
        def __init__(self, agent: object, history=None, deps=None):
            assert isinstance(agent, FakeAgent)
            assert history == loaded_history
            self.deps = deps
            self.history = ["updated"]

        def send_stream(self, prompt: str):
            assert prompt == "hello"
            yield "hi"

    monkeypatch.setattr(chat_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(chat_module, "_router", None)
    monkeypatch.setattr(chat_module, "AgentRouter", FakeRouter)
    monkeypatch.setattr(chat_module, "Session", FakeSession)
    monkeypatch.setattr(chat_module, "FirestoreSessionStore", FakeStore)

    assert chat(text="hello", session_id="session-1", channel="sms") == "hi"
    assert saved == {
        "session_id": "session-1",
        "history": ["updated"],
        "agent_name": "support",
        "channel": "sms",
    }


def test_chat_threads_debug_events_into_session_context(monkeypatch) -> None:
    debug_events = []
    captured = {}

    class FakeRouter:
        def route_with_metadata(self, query: str):
            return object(), {"score": 1.0}

    class FakeSession:
        def __init__(self, agent: object, history=None, deps=None):
            captured["deps"] = deps
            self.history = []

        def send_stream(self, prompt: str):
            yield "ok"

    monkeypatch.setattr(chat_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(chat_module, "_router", None)
    monkeypatch.setattr(chat_module, "AgentRouter", FakeRouter)
    monkeypatch.setattr(chat_module, "Session", FakeSession)
    monkeypatch.setattr(
        chat_module,
        "FirestoreSessionStore",
        lambda: (_ for _ in ()).throw(AssertionError("store should not be used")),
    )

    assert chat(text="hello", debug_events=debug_events) == "ok"
    assert captured["deps"].debug_events is debug_events


def test_render_history_text_flattens_user_and_assistant_turns() -> None:
    history = [
        ModelRequest(parts=[UserPromptPart(content="my mother passed away")]),
        ModelResponse(parts=[TextPart(content="I'm so sorry for your loss")]),
    ]
    assert _render_history_text(history) == (
        "user: my mother passed away\nassistant: I'm so sorry for your loss"
    )


def test_chat_image_turn_stores_media_and_hides_image_from_model(monkeypatch) -> None:
    """An uploaded image is saved to the store; the model only sees a text notice."""
    saved_media = {}
    captured = {}

    class FakeAgent:
        name = "death_certificate_poc_agent"

    class FakeRouter:
        def route_with_metadata(self, query: str):
            captured["route_query"] = query
            return FakeAgent(), {"score": 1.0}

    class FakeStore:
        def load_history(self, session_id: str):
            # prior conversation establishes the death-certificate context
            return [ModelRequest(parts=[UserPromptPart(content="my mother passed away")])]

        def save_media(self, session_id: str, image_bytes: bytes, *, mime_type: str) -> bool:
            saved_media["session_id"] = session_id
            saved_media["bytes"] = image_bytes
            saved_media["mime_type"] = mime_type
            return True

        def save_history(self, session_id, history, *, agent_name=None, channel=None) -> None:
            pass

    class FakeSession:
        def __init__(self, agent: object, history=None, deps=None):
            captured["deps"] = deps
            self.history = list(history or [])

        def send_stream(self, prompt):
            captured["prompt"] = prompt
            yield "Your request is being processed."

    monkeypatch.setattr(chat_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(chat_module, "_router", None)
    monkeypatch.setattr(chat_module, "AgentRouter", FakeRouter)
    monkeypatch.setattr(chat_module, "Session", FakeSession)
    monkeypatch.setattr(chat_module, "FirestoreSessionStore", FakeStore)

    result = chat(image_bytes=b"\xff\xd8jpeg", image_media_type="image/jpeg", session_id="wa-1")

    assert result == "Your request is being processed."
    # image persisted to the transient store
    assert saved_media == {"session_id": "wa-1", "bytes": b"\xff\xd8jpeg", "mime_type": "image/jpeg"}
    # the model prompt is a plain text notice — never the image bytes
    assert captured["prompt"] == IMAGE_ARRIVED_PROMPT
    assert not isinstance(captured["prompt"], list)
    # routing used the prior conversation, biasing toward the death-cert agent
    assert "my mother passed away" in captured["route_query"]
    # deps carry the store + session so the tool can pull the image
    assert captured["deps"].session_id == "wa-1"
    assert captured["deps"].store is not None
    assert "my mother passed away" in captured["deps"].history_text


def test_image_route_query_keeps_document_signal_when_history_is_small_talk() -> None:
    """Prior chatter must not steer an image turn away from the verification agent."""
    from src.chat import IMAGE_ROUTING_TEXT, _image_route_query

    query = _image_route_query("user: hello\nassistant: Hi! How can I help you today?")

    assert query.startswith(IMAGE_ROUTING_TEXT)
    assert "hello" in query


def test_image_route_query_trims_long_history() -> None:
    from src.chat import IMAGE_ROUTING_HISTORY_CHARS, IMAGE_ROUTING_TEXT, _image_route_query

    query = _image_route_query("x" * 5000)

    assert query.startswith(IMAGE_ROUTING_TEXT)
    assert len(query) == len(IMAGE_ROUTING_TEXT) + 1 + IMAGE_ROUTING_HISTORY_CHARS


def test_image_route_query_without_history_uses_document_signal() -> None:
    from src.chat import IMAGE_ROUTING_TEXT, _image_route_query

    assert _image_route_query("") == IMAGE_ROUTING_TEXT
