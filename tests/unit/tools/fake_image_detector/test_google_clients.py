"""Google clients and credentials are created once per process and shared.

Creating them per call ran credential discovery every time; on a machine signed
in with gcloud that starts a subprocess, and doing so while Vision gRPC calls
were in flight froze the whole process for about a minute.
"""

import asyncio
import json
import sys
from unittest.mock import MagicMock

from tools.fake_image_detector import google_clients
from tools.fake_image_detector.checks.gemini_vision_check import GeminiVisionCheck
from tools.fake_image_detector.checks.reverse_image_check import ReverseImageCheck


def _jpeg() -> bytes:
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="JPEG")
    return buf.getvalue()


def test_gemini_client_is_built_once_and_timeout_is_per_request():
    response = MagicMock()
    response.text = json.dumps({"is_deceptive": False, "fake_likelihood": 0.1, "confidence": 0.9})
    genai = MagicMock()
    genai.Client.return_value.models.generate_content.return_value = response
    types = MagicMock()
    sys.modules["google.genai"] = genai
    sys.modules["google.genai.types"] = types
    try:
        check = GeminiVisionCheck(project="p", location="global", timeout_seconds=30)
        for _ in range(3):
            asyncio.run(check.run(_jpeg(), {}))
    finally:
        sys.modules.pop("google.genai", None)
        sys.modules.pop("google.genai.types", None)

    genai.Client.assert_called_once_with(vertexai=True, project="p", location="global")
    assert genai.Client.return_value.models.generate_content.call_count == 3
    types.HttpOptions.assert_called_with(timeout=30_000)
    assert types.GenerateContentConfig.call_args.kwargs["http_options"] is types.HttpOptions.return_value


def test_vision_client_is_shared_and_each_call_has_its_own_timeout(monkeypatch):
    client = MagicMock()
    client.web_detection.return_value.error.message = ""
    created = []

    def make():
        created.append(1)
        return client

    vision = MagicMock()
    vision.ImageAnnotatorClient.side_effect = make
    monkeypatch.setitem(sys.modules, "google.cloud.vision", vision)
    import google.cloud

    monkeypatch.setattr(google.cloud, "vision", vision, raising=False)

    check = ReverseImageCheck(params={"timeout_seconds": 7, "attempts": 2})
    for _ in range(3):
        asyncio.run(check.run(_jpeg(), {}))

    assert len(created) == 1
    kwargs = client.web_detection.call_args.kwargs
    assert kwargs["timeout"] == 7
    assert kwargs["retry"] is None  # our retries only, not stacked on the library's


def test_access_token_is_cached_until_it_expires(monkeypatch):
    import google.auth

    credentials = MagicMock()
    credentials.valid = False
    credentials.token = "t1"

    def refresh(_request):
        credentials.valid = True

    credentials.refresh.side_effect = refresh
    default = MagicMock(return_value=(credentials, "p"))
    monkeypatch.setattr(google.auth, "default", default)

    assert google_clients.access_token("scope-a") == "t1"
    assert google_clients.access_token("scope-a") == "t1"
    default.assert_called_once()
    credentials.refresh.assert_called_once()

    credentials.valid = False  # expired
    google_clients.access_token("scope-a")
    assert credentials.refresh.call_count == 2
    default.assert_called_once()


def test_only_transient_failures_are_retried():
    import httpx
    from tools.fake_image_detector.google_clients import retryable

    def status(code):
        return type("E", (Exception,), {"code": code})()

    assert retryable(status(429)) and retryable(status(503)) and retryable(status(504))
    assert retryable(TimeoutError()) and retryable(httpx.ReadTimeout("slow"))
    # Fail the same way again: bad setting, API switched off, wrong model, a bug.
    assert not retryable(status(400)) and not retryable(status(403)) and not retryable(status(404))
    assert not retryable(ValueError("bug"))


def test_reverse_image_does_not_retry_a_switched_off_api(monkeypatch):
    from google.api_core import exceptions

    client = MagicMock()
    client.web_detection.side_effect = exceptions.PermissionDenied("Vision API has not been used")
    monkeypatch.setattr(google_clients, "vision_client", lambda: client)
    import tools.fake_image_detector.checks.reverse_image_check as module

    monkeypatch.setattr(module, "vision_client", lambda: client)
    result = asyncio.run(ReverseImageCheck(params={"attempts": 3}).run(_jpeg(), {}))

    assert client.web_detection.call_count == 1
    assert result.skipped is True


def test_warm_up_creates_each_client_once_and_never_raises(monkeypatch):
    """Done at startup so the first claimant's turn does not run credential
    discovery mid-request; a failure is logged, not raised, so the app still starts."""
    from tools.death_certificate_pipeline import pipeline

    from google.genai import types as real_types

    genai = MagicMock()
    genai.Client.return_value.models.count_tokens.side_effect = RuntimeError("no network")
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", real_types)
    vision = MagicMock()
    monkeypatch.setattr(google_clients, "vision_client", vision)
    import tools.fake_image_detector.google_clients as gc

    import dataclasses

    from tools.fake_image_detector.config_loader import GeminiConfig, PipelineConfig

    # Fixed settings, not the shipped files, so a model change there cannot
    # break this test: the story check and the fraud check on different
    # models and locations.
    scoring = dataclasses.replace(
        pipeline.default_scoring_config(),
        consistency_model="story-model",
        consistency_location="us-east1",
    )
    detector = PipelineConfig(
        clear_fail=0.8,
        clear_pass=0.2,
        gemini=GeminiConfig(enabled=True, model="fraud-model", location="global"),
    )
    monkeypatch.setattr(pipeline, "default_scoring_config", lambda: scoring)
    monkeypatch.setattr(
        "tools.fake_image_detector.config_loader.load_pipeline_config", lambda *a, **k: detector
    )

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    pipeline.warm_up_google_clients()  # must not raise, though every call fails

    built = {call.kwargs["location"] for call in genai.Client.call_args_list}
    assert built == {"us-east1", "global"}  # one client per location
    counted = {c.kwargs["model"] for c in genai.Client.return_value.models.count_tokens.call_args_list}
    assert counted == {"story-model", "fraud-model"}
    assert gc._objects  # the clients stay cached for the first real call
    # Startup waits on these calls, so each carries its own timeout.
    config = genai.Client.return_value.models.count_tokens.call_args.kwargs["config"]
    assert config.http_options.timeout == int(pipeline._WARM_UP_TIMEOUT_SECONDS * 1000)


def test_warm_up_does_nothing_without_a_project(monkeypatch):
    from tools.death_certificate_pipeline import pipeline

    genai = MagicMock()
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    pipeline.warm_up_google_clients()
    genai.Client.assert_not_called()
