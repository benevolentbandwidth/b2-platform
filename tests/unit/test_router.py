from types import SimpleNamespace

import numpy as np

from src import router as router_module
from src.router import AgentRouter

# Unit vectors: a query embedding equal to an agent's routes to that agent.
VECTORS = {"intake": np.array([1.0, 0.0]), "general": np.array([0.0, 1.0])}


class FakeEmbedder:
    model = "fake-embedding"
    output_dimensionality = 2

    def embed_documents(self, texts):
        return np.stack([VECTORS[text.split("\n", 1)[0]] for text in texts]) if texts else np.zeros((0, 2))

    def embed_query(self, text):
        return VECTORS[text]


def _write_agent(directory, name, *, sticky):
    lines = [f"name: {name}", f"description: {name} agent", "system_prompt: be helpful"]
    if sticky:
        lines.append("sticky: true")
    lines += ["provider:", "  type: gemini", "  model: gemini-2.5-flash"]
    (directory / f"{name}.yaml").write_text("\n".join(lines) + "\n")


def _router(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    agents.mkdir()
    _write_agent(agents, "intake", sticky=True)
    _write_agent(agents, "general", sticky=False)
    # Building a real Agent needs Vertex; routing only needs the definition.
    monkeypatch.setattr(
        router_module.agent_module,
        "Agent",
        lambda definition: SimpleNamespace(definition=definition, name=definition.name),
    )
    return AgentRouter(agents_dir=str(agents), db_path=str(tmp_path / "agents.sqlite3"), embedder=FakeEmbedder())


def test_a_conversation_with_a_sticky_agent_stays_with_it(tmp_path, monkeypatch) -> None:
    router = _router(tmp_path, monkeypatch)

    agent, metadata = router.route_with_metadata("general", current_agent="intake")

    assert agent.name == "intake"
    assert metadata["sticky"] is True


def test_a_conversation_with_a_non_sticky_agent_is_routed_by_the_message(tmp_path, monkeypatch) -> None:
    router = _router(tmp_path, monkeypatch)

    agent, _ = router.route_with_metadata("intake", current_agent="general")

    assert agent.name == "intake"


def test_a_removed_current_agent_falls_back_to_routing(tmp_path, monkeypatch) -> None:
    router = _router(tmp_path, monkeypatch)

    agent, metadata = router.route_with_metadata("general", current_agent="retired_agent")

    assert agent.name == "general"
    assert metadata["score"] == 1.0
