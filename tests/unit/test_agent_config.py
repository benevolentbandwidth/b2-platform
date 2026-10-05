from pathlib import Path

from src.orchestrator import agent as agent_module


AGENTS_DIR = Path(__file__).resolve().parents[2] / "agents"
RETIRED_MODELS = {"gemini-2.0-flash", "gemini-3.1-flash-lite-preview"}


class FakeProvider:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)


class FakeModel:
    calls = []

    def __init__(self, model_name, *, provider):
        self.model_name = model_name
        self.provider = provider
        self.calls.append((model_name, provider))


class FakePydanticAgent:
    def __init__(self, *, model, system_prompt, name, deps_type):
        self.model = model
        self.system_prompt = system_prompt
        self.name = name
        self.deps_type = deps_type
        self.tools = []

    def tool(self, fn, *, name, description):
        self.tools.append((name, description, fn))
        return fn

    def tool_plain(self, fn, *, name, description):
        self.tools.append((name, description, fn))
        return fn


def _agent_yaml_files():
    return sorted(AGENTS_DIR.glob("*.yaml"))


def test_agent_yaml_files_do_not_use_retired_gemini_models():
    for yaml_file in _agent_yaml_files():
        definition = agent_module._load_agent_definition_from_file(yaml_file)
        assert definition.provider["model"] not in RETIRED_MODELS


def test_all_agent_definitions_construct_google_vertex_models(monkeypatch):
    FakeProvider.calls = []
    FakeModel.calls = []
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setattr(agent_module, "GoogleProvider", FakeProvider)
    monkeypatch.setattr(agent_module, "GoogleModel", FakeModel)
    monkeypatch.setattr(agent_module, "PydanticAgent", FakePydanticAgent)

    for yaml_file in _agent_yaml_files():
        definition = agent_module._load_agent_definition_from_file(yaml_file)
        agent = agent_module.Agent(definition)
        assert agent.name == definition.name
        assert agent.pydantic_ai_agent.model.model_name == definition.provider["model"]

    assert FakeModel.calls
    # Gemini 3.x is only served from `global`; a regional location returns 404.
    for model_name, provider in FakeModel.calls:
        if model_name.startswith("gemini-3"):
            assert provider.kwargs["location"] == "global", model_name
    assert all(call["vertexai"] is True for call in FakeProvider.calls)
    assert all(call["project"] == "test-project" for call in FakeProvider.calls)


def test_thinking_level_setting_reaches_gemini():
    settings = agent_module.Agent._model_settings_from_provider(
        {"settings": {"thinking_level": "low"}}, "gemini-3.8-flash"
    )
    assert settings == {"google_thinking_config": {"thinking_level": "LOW"}}


def test_thinking_budget_zero_on_gemini_3_asks_for_low_not_minimal():
    """3.8 Flash rejects MINIMAL with a 400, which would fail every chat turn."""
    settings = agent_module.Agent._model_settings_from_provider(
        {"settings": {"thinking_budget": 0}}, "gemini-3.8-flash"
    )
    assert settings == {"google_thinking_config": {"thinking_level": "LOW"}}
    settings = agent_module.Agent._model_settings_from_provider(
        {"settings": {"thinking_budget": 0}}, "gemini-2.5-flash"
    )
    assert settings == {"google_thinking_config": {"thinking_budget": 0}}
