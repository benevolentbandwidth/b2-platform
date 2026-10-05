"""Integration test: real orchestrator agent + verify tool, LLM faked with TestModel.

Drives the novel mechanism end to end — the model calls the context-aware tool,
which pulls the user's image from the transient store (via SessionContext deps)
and runs the pipeline + handoff. Only the external boundaries (the LLM, the
reliability pipeline, and GiveLight delivery) are faked; the tool registration,
deps threading, and image-through-context path are all exercised for real.
"""

import json
from src.chat import IMAGE_ARRIVED_PROMPT
from pathlib import Path

from pydantic_ai.models.test import TestModel

from src.orchestrator import agent as agent_module
from src.orchestrator.context import SessionContext
from src.session import Session
from tools.death_certificate_pipeline import verify as verify_module
from tools.death_certificate_pipeline.models import Band, ReliabilityResult
from tools.death_certificate_pipeline.pipeline import PipelineExecution
from tools.fake_image_detector.models import Escalation, ToolResult, Verdict

_AGENT_YAML = Path(__file__).resolve().parents[2] / "agents" / "poc_deathCertParserAgent.yaml"


class _FakeStore:
    def __init__(self, media):
        self._media = media
        self.pulled_for = None

    def load_latest_media(self, session_id):
        self.pulled_for = session_id
        return self._media


def test_model_call_pulls_image_and_hands_off(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-proj")

    seen = {}

    async def fake_pipeline(submission):
        seen["narrative"] = submission.narrative
        seen["image"] = submission.image
        return PipelineExecution(result=ReliabilityResult(
            score=82,
            band=Band.HIGH,
            sub_scores={},
            weights={},
            flags=[],
            justification="ok",
            extracted_fields={"full_name": "Jane Doe"},
        ), authenticity=ToolResult(
            verdict=Verdict.PASS,
            risk_score=0.1,
            escalation=Escalation.AUTO_ACCEPT,
            checks=[],
        ))

    async def fake_deliver(payload, image_bytes, mime_type):
        seen["handoff"] = payload
        seen["handoff_image"] = image_bytes
        seen["handoff_mime"] = mime_type
        return True

    monkeypatch.setattr(verify_module, "run_pipeline_with_diagnostics", fake_pipeline)
    monkeypatch.setattr(verify_module, "deliver_to_gl", fake_deliver)
    monkeypatch.setattr(
        agent_module.Agent,
        "_build_model",
        classmethod(lambda cls, provider: (TestModel(), {})),
    )

    definition = agent_module._load_agent_definition_from_file(_AGENT_YAML)
    agent = agent_module.Agent(definition)

    store = _FakeStore((b"\xff\xd8jpeg-image-bytes", "image/jpeg"))
    deps = SessionContext(
        session_id="wa-1",
        store=store,
        history_text="My mother Jane Doe passed away last week.",
    )

    with agent.pydantic_ai_agent.override(model=TestModel()):
        session = Session(agent, deps=deps)
        "".join(session.send_stream(IMAGE_ARRIVED_PROMPT))

    # the tool pulled the image for the right session, out of band from the model
    assert store.pulled_for == "wa-1"
    assert seen["image"] == b"\xff\xd8jpeg-image-bytes"
    assert seen["narrative"] == "My mother Jane Doe passed away last week."
    # and the passing result was forwarded to GiveLight
    assert seen["handoff"]["score"] == 82
    assert len(seen["handoff"]["contact_identifier"]) == 64
    assert seen["handoff_image"] == b"\xff\xd8jpeg-image-bytes"
    assert seen["handoff_mime"] == "image/jpeg"


def test_orphan_claim_document_upload_returns_tool_validation_output(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-proj")

    seen = {}

    async def fake_pipeline(submission):
        seen["submission"] = submission
        return PipelineExecution(result=ReliabilityResult(
            score=91,
            band=Band.HIGH,
            sub_scores={"document": 1.0, "authenticity": 0.95, "consistency": 0.9},
            weights={"document": 0.2, "authenticity": 0.4, "consistency": 0.4},
            flags=[],
            justification=(
                "document=legible (type=death_certificate), "
                "authenticity=PASS (risk=0.05), consistency=high (score=0.90)"
            ),
            extracted_fields={
                "full_name": "Jane Doe",
                "date_of_death": "2024-05-01",
                "place_of_death": "Seattle",
            },
        ), authenticity=ToolResult(
            verdict=Verdict.PASS,
            risk_score=0.05,
            escalation=Escalation.AUTO_ACCEPT,
            checks=[],
        ))

    async def fake_deliver(payload, image_bytes, mime_type):
        seen["handoff"] = payload
        seen["handoff_image"] = image_bytes
        seen["handoff_mime"] = mime_type
        return True

    monkeypatch.setattr(verify_module, "run_pipeline_with_diagnostics", fake_pipeline)
    monkeypatch.setattr(verify_module, "deliver_to_gl", fake_deliver)
    monkeypatch.setattr(
        agent_module.Agent,
        "_build_model",
        classmethod(lambda cls, provider: (TestModel(), {})),
    )

    definition = agent_module._load_agent_definition_from_file(_AGENT_YAML)
    agent = agent_module.Agent(definition)

    store = _FakeStore((b"\xff\xd8jpeg-death-certificate", "image/jpeg"))
    deps = SessionContext(
        session_id="wa-orphan-claim-1",
        store=store,
        history_text=(
            "user: Hello, I am an orphan and I need help applying for GiveLight support.\n"
            "assistant: I am sorry for your loss. Please send the death certificate so I can verify the request.\n"
            "user: My mother Jane Doe died on 2024-05-01 in Seattle. I am uploading the document now."
        ),
    )

    with agent.pydantic_ai_agent.override(model=TestModel(call_tools=["death_certificate_verification"])):
        session = Session(agent, deps=deps)
        response_text = "".join(session.send_stream(IMAGE_ARRIVED_PROMPT))

    assert store.pulled_for == "wa-orphan-claim-1"
    assert seen["submission"].image == b"\xff\xd8jpeg-death-certificate"
    assert "I am an orphan" in seen["submission"].narrative
    assert "Jane Doe died on 2024-05-01 in Seattle" in seen["submission"].narrative

    handoff = seen["handoff"]
    assert handoff["score"] == 91
    assert handoff["band"] == "high"
    assert handoff["sub_scores"] == {"document": 1.0, "authenticity": 0.95, "consistency": 0.9}
    assert handoff["extracted_fields"]["full_name"] == "Jane Doe"
    assert handoff["case_fields"]["channel"] == "whatsapp"
    assert len(handoff["contact_identifier"]) == 64
    assert seen["handoff_image"] == b"\xff\xd8jpeg-death-certificate"
    assert seen["handoff_mime"] == "image/jpeg"

    tool_output = json.loads(response_text)["death_certificate_verification"]
    assert tool_output["status"] == "verified"
    assert tool_output["score"] == 91
    assert tool_output["band"] == "high"
    assert tool_output["handed_off"] is True
    assert tool_output["extracted_fields"]["full_name"] == "Jane Doe"
