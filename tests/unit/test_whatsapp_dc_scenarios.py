import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit.test_api import FakeRequest
from tools.death_certificate_pipeline import verify as verify_module
from tools.death_certificate_pipeline import pipeline as pipeline_module
from tools.fake_image_detector.models import Escalation, ToolResult, Verdict


ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_DC_DIR = ROOT_DIR / "tests" / "test_dc"


class ScenarioStore:
    def __init__(self):
        self.history: dict[str, list[str]] = {}
        self.media: dict[str, tuple[bytes, str]] = {}

    def add_message(self, session_id: str, role: str, text: str) -> None:
        self.history.setdefault(session_id, []).append(f"{role}: {text}")

    def history_text(self, session_id: str) -> str:
        return "\n".join(self.history.get(session_id, []))

    def save_media(self, session_id: str, image_bytes: bytes, mime_type: str) -> None:
        self.media[session_id] = (image_bytes, mime_type)

    def load_latest_media(self, session_id: str):
        return self.media.get(session_id)


async def _run_inline(func, **kwargs):
    result = func(**kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _text_payload(text: str, *, wa_id: str, message_id: str) -> dict:
    return {
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
                            "contacts": [{"profile": {"name": "Test User"}, "wa_id": wa_id}],
                            "messages": [
                                {
                                    "from": wa_id,
                                    "id": message_id,
                                    "timestamp": "1780358445",
                                    "text": {"body": text},
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


def _image_payload(media_id: str, *, wa_id: str, message_id: str) -> dict:
    return {
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
                            "contacts": [{"profile": {"name": "Test User"}, "wa_id": wa_id}],
                            "messages": [
                                {
                                    "from": wa_id,
                                    "id": message_id,
                                    "timestamp": "1780358447",
                                    "type": "image",
                                    "image": {"id": media_id, "mime_type": "image/png"},
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }


async def test_whatsapp_death_certificate_scenarios_print_expected_vs_actual_scores(monkeypatch, capsys):
    import src.api as api_module

    image_files = sorted(
        path for path in TEST_DC_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    expected_consistency_scores = [1.0, 0.67, 0.33, 0.0, 1.0, 0.67, 0.33, 0.0, 1.0]
    expected_scores = [100, 87, 73, 60, 100, 87, 73, 60, 100]
    names = [
        "Jane Doe",
        "John Smith",
        "Amina Hassan",
        "Omar Benali",
        "Fatima El Idrissi",
        "Noah Williams",
        "Maya Johnson",
        "Sam Carter",
        "Leila Haddad",
    ]
    places = ["Seattle", "Nairobi", "Casablanca", "Rabat", "Marrakesh", "Austin", "Detroit", "Chicago", "Fez"]

    if not image_files:
        # tests/test_dc/* is gitignored: real certificates must not be committed,
        # so this only runs where someone has put fixtures there locally.
        pytest.skip(f"no certificate fixtures in {TEST_DC_DIR} (see its README)")
    assert len(expected_scores) >= len(image_files)

    scenarios = []
    for index, image_file in enumerate(image_files):
        full_name = names[index]
        date_of_death = f"2024-05-{index + 1:02d}"
        claim_name = full_name
        claim_date = date_of_death
        claim_place = places[index]
        if index % 4 >= 1:
            claim_place = "Tangier"
        if index % 4 >= 2:
            claim_date = "2020-01-01"
        if index % 4 >= 3:
            claim_name = "Unrelated Person"

        score = expected_scores[index]
        consistency_score = expected_consistency_scores[index]
        if score >= 75:
            band = "high"
        elif score >= 50:
            band = "medium"
        elif score >= 25:
            band = "low"
        else:
            band = "escalate"

        scenarios.append(
            {
                "name": image_file.stem,
                "wa_id": f"1650810664{index}",
                "messages": [
                    "Hello B2, I am an orphan and need help applying for GiveLight support.",
                    f"My parent {claim_name} died on {claim_date} in {claim_place}.",
                    "I am sending the death certificate image now.",
                ],
                "image": image_file,
                "expected_score": score,
                "expected_band": band,
                "expected_consistency_score": consistency_score,
                "fields": {
                    "full_name": full_name,
                    "date_of_death": date_of_death,
                    "place_of_death": places[index],
                },
            }
        )

    media_by_id = {}
    fields_by_image = {}
    expected_consistency_by_image = {}
    for index, scenario in enumerate(scenarios, start=1):
        image_bytes = scenario["image"].read_bytes()
        media_id = f"test-dc-{index}"
        scenario["media_id"] = media_id
        media_by_id[media_id] = (image_bytes, "image/png")
        fields_by_image[image_bytes] = scenario["fields"]
        expected_consistency_by_image[image_bytes] = scenario["expected_consistency_score"]

    store = ScenarioStore()
    downloaded_media_ids = []
    chat_text_calls = []
    chat_image_calls = []

    async def fake_download(media_id):
        downloaded_media_ids.append(media_id)
        return media_by_id[media_id]

    class FakeAuthenticityPipeline:
        async def run(self, image_bytes, context=None):
            assert context == {"input_type": "document"}
            return ToolResult(
                verdict=Verdict.PASS,
                risk_score=0.0,
                escalation=Escalation.AUTO_ACCEPT,
                checks=[],
            )

    def fake_build_authenticity_pipeline():
        return FakeAuthenticityPipeline()

    def fake_consistency(chat_history, image_bytes, **_kwargs):
        certificate = fields_by_image[image_bytes]
        transcript = chat_history.lower()
        checks = [
            ("name", certificate["full_name"].lower(), "name matches claimant narrative"),
            ("date", certificate["date_of_death"].lower(), "date of death matches claimant narrative"),
            ("place", certificate["place_of_death"].lower(), "place of death matches claimant narrative"),
        ]
        matches = []
        mismatches = []
        for label, expected_text, match_message in checks:
            if expected_text in transcript:
                matches.append(match_message)
            else:
                mismatches.append(f"{label} does not align with claimant narrative")

        consistency_score = round(len(matches) / len(checks), 2)
        expected_consistency = expected_consistency_by_image[image_bytes]
        assert consistency_score == expected_consistency

        if consistency_score >= 0.8:
            consistency_label = "high"
        elif consistency_score >= 0.34:
            consistency_label = "moderate"
        else:
            consistency_label = "low"

        return {
            "certificate": certificate,
            "consistency_score": consistency_score,
            "consistency_label": consistency_label,
            "confidence": 0.95,
            "matches": matches,
            "mismatches": mismatches,
            "uncertain_points": [],
            "summary": (
                "Claimant narrative aligns with certificate facts."
                if not mismatches
                else "Claimant narrative has certificate fact mismatches."
            ),
        }

    async def fake_deliver(payload):
        return True

    async def scenario_chat(*, text=None, image_bytes=None, image_media_type="image/jpeg", session_id=None, **_kwargs):
        assert session_id is not None
        if text is not None:
            chat_text_calls.append({"session_id": session_id, "text": text})
            store.add_message(session_id, "user", text)
            return "Please send the death certificate image when ready."

        assert image_bytes is not None
        chat_image_calls.append(
            {"session_id": session_id, "image_bytes": image_bytes, "image_media_type": image_media_type}
        )
        store.save_media(session_id, image_bytes, image_media_type)
        ctx = SimpleNamespace(
            deps=SimpleNamespace(
                session_id=session_id,
                store=store,
                history_text=store.history_text(session_id),
            )
        )
        tool_output = await verify_module.verify_death_certificate(ctx)
        return json.dumps({"death_certificate_verification": tool_output}, sort_keys=True)

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-proj")
    monkeypatch.setattr(api_module, "_WEBHOOK_SECRET", "")
    monkeypatch.setattr(api_module, "download_media", fake_download)
    monkeypatch.setattr(api_module, "chat", scenario_chat)
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(pipeline_module, "_build_authenticity_pipeline", fake_build_authenticity_pipeline)
    monkeypatch.setattr(pipeline_module, "analyze_death_certificate_consistency", fake_consistency)
    monkeypatch.setattr(verify_module, "deliver_to_gl", fake_deliver)

    rows = []
    for scenario in scenarios:
        for message_index, message in enumerate(scenario["messages"], start=1):
            await api_module.message_endpoint(
                FakeRequest(
                    _text_payload(
                        message,
                        wa_id=scenario["wa_id"],
                        message_id=f"{scenario['media_id']}.text.{message_index}",
                    )
                )
            )

        response = await api_module.message_endpoint(
            FakeRequest(
                _image_payload(
                    scenario["media_id"],
                    wa_id=scenario["wa_id"],
                    message_id=f"{scenario['media_id']}.image",
                )
            )
        )

        tool_output = json.loads(response["response"])["death_certificate_verification"]
        rows.append((scenario, tool_output))

    assert downloaded_media_ids == [scenario["media_id"] for scenario in scenarios]
    assert len(chat_text_calls) == sum(len(scenario["messages"]) for scenario in scenarios)
    assert len(chat_image_calls) == len(scenarios)
    for scenario, image_call in zip(scenarios, chat_image_calls):
        assert image_call["session_id"] == scenario["wa_id"]
        assert image_call["image_bytes"] == scenario["image"].read_bytes()
        assert image_call["image_media_type"] == "image/png"
        for message in scenario["messages"]:
            assert message in store.history_text(scenario["wa_id"])

    report_lines = [
        "",
        "WhatsApp death-certificate scenario score report",
        (
            "case | expected_score | actual_score | expected_band | actual_band | "
            "expected_consistency | actual_consistency | mismatches"
        ),
    ]
    for scenario, tool_output in rows:
        report_lines.append(
            f"{scenario['name']} | {scenario['expected_score']} | {tool_output['score']} | "
            f"{scenario['expected_band']} | {tool_output['band']} | "
            f"{scenario['expected_consistency_score']} | {tool_output['sub_scores']['consistency']} | "
            f"{len(tool_output['mismatches'])}"
        )

    report = "\n".join(report_lines)
    with capsys.disabled():
        print(report)
    assert "WhatsApp death-certificate scenario score report" in report

    for scenario, tool_output in rows:
        assert tool_output["status"] == "verified"
        assert tool_output["score"] == scenario["expected_score"]
        assert tool_output["band"] == scenario["expected_band"]
        assert tool_output["sub_scores"]["consistency"] == scenario["expected_consistency_score"]
        assert len(tool_output["mismatches"]) == 3 - round(scenario["expected_consistency_score"] * 3)
        assert tool_output["handed_off"] is (scenario["expected_band"] in {"high", "medium"})
