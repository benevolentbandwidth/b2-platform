import asyncio
import io
import json
import os
import sys
from unittest.mock import MagicMock

from PIL import Image

from tools.fake_image_detector import google_clients
from tools.fake_image_detector.checks.gemini_vision_check import GeminiVisionCheck


def run(coro):
    return asyncio.run(coro)


def _jpeg_bytes(width=64, height=64) -> bytes:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _install_mock_genai(response_text: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.text = response_text

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client

    sys.modules["google.genai"] = mock_genai
    sys.modules["google.genai.types"] = MagicMock()
    return mock_client


def _remove_mock_genai() -> None:
    sys.modules.pop("google.genai", None)
    sys.modules.pop("google.genai.types", None)


class TestGeminiVisionCheck:
    def test_check_id(self):
        assert GeminiVisionCheck.check_id == "gemini_vision"

    def test_detects_synthetic_image(self):
        _install_mock_genai(json.dumps({
            "is_deceptive": True,
            "fake_likelihood": 0.87,
            "confidence": 0.87,
            "signals": ["unnatural skin texture", "bilateral symmetry"],
            "flags": ["GAN_ARTIFACTS", "UNNATURAL_TEXTURE"],
        }))
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert result.passed is False
        assert result.skipped is False
        assert result.fake_score == 0.87
        assert result.confidence == 0.87
        assert "GAN_ARTIFACTS" in result.flags
        assert "EDITING_ARTIFACTS" in result.flags
        assert result.human_escalate is False
        assert result.signals["is_deceptive"] is True
        assert result.normalized_signals is not None
        assert result.normalized_signals.category == "synthetic"
        assert result.normalized_signals.synthetic_score == 0.87

    def test_possible_stock_is_reported_as_a_flag(self):
        """The check reports the flag; whether it forces review is decided by
        the pipeline from hard_escalation_flags in pipeline.yaml."""
        _install_mock_genai(json.dumps({
            "is_deceptive": True,
            "fake_likelihood": 0.7,
            "confidence": 0.8,
            "signals": ["looks like stock imagery"],
            "flags": ["STOCK_PHOTO_INDICATORS"],
        }))
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert "POSSIBLE_STOCK" in result.flags
        assert result.human_escalate is False

    def test_passes_real_image(self):
        _install_mock_genai(json.dumps({
            "is_deceptive": False,
            "fake_likelihood": 0.05,
            "confidence": 0.9,
            "signals": [],
            "flags": ["CLEAN"],
        }))
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert result.passed is True
        assert result.skipped is False
        assert result.fake_score == 0.05
        assert result.confidence == 0.9
        assert result.normalized_signals is not None
        assert result.normalized_signals.category == "synthetic"
        assert result.normalized_signals.synthetic_score == 0.05

    def test_markdown_wrapped_json_is_parsed(self):
        payload = json.dumps({
            "is_deceptive": True,
            "fake_likelihood": 0.75,
            "confidence": 0.75,
            "signals": ["diffusion artifacts"],
            "flags": ["DIFFUSION_ARTIFACTS"],
        })
        _install_mock_genai(f"```json\n{payload}\n```")
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert result.passed is False
        assert result.confidence == 0.75

    def test_json_parse_failure_escalates_to_human_review(self):
        _install_mock_genai("sorry, I cannot analyze this image")
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert result.skipped is False
        assert result.passed is False
        assert result.fake_score == 0.0
        assert result.confidence == 0.0
        assert result.human_escalate is True
        assert "GEMINI_PARSE_ERROR" in result.flags
        assert "JSON parse error" in result.error

    def test_api_exception_escalates_to_human_review(self, monkeypatch):
        from tools.fake_image_detector.checks import gemini_vision_check as module

        monkeypatch.setattr(google_clients, "RETRY_BACKOFF_SECONDS", (0.0, 0.0))
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("quota exceeded")
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        sys.modules["google.genai"] = mock_genai
        sys.modules["google.genai.types"] = MagicMock()
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert result.skipped is False
        assert result.passed is False
        assert result.fake_score == 0.0
        assert result.confidence == 0.0
        assert result.human_escalate is True
        assert "CHECK_RUNTIME_ERROR" in result.flags
        assert "quota exceeded" in result.error

    def test_missing_project_escalates_to_human_review(self):
        env_backup = os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
        sys.modules["google.genai"] = MagicMock()
        sys.modules["google.genai.types"] = MagicMock()
        try:
            result = run(GeminiVisionCheck(project=None).run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()
            if env_backup is not None:
                os.environ["GOOGLE_CLOUD_PROJECT"] = env_backup

        assert result.skipped is False
        assert result.passed is False
        assert result.fake_score == 0.0
        assert result.confidence == 0.0
        assert result.human_escalate is True
        assert "CHECK_RUNTIME_ERROR" in result.flags
        assert "GOOGLE_CLOUD_PROJECT" in result.error

    def test_confidence_clamped_above_one(self):
        _install_mock_genai(json.dumps({
            "is_deceptive": True,
            "confidence": 1.5,
            "signals": [],
            "flags": [],
        }))
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert result.confidence == 1.0

    def test_confidence_clamped_below_zero(self):
        _install_mock_genai(json.dumps({
            "is_deceptive": False,
            "fake_likelihood": 0.0,
            "confidence": -0.3,
            "signals": [],
            "flags": ["CLEAN"],
        }))
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert result.confidence == 0.0  # clamp(-0.3) = 0.0
        assert result.fake_score == 0.0

    def test_document_context_is_sanitized_in_prompt_and_signals(self):
        mock_client = _install_mock_genai(json.dumps({
            "is_deceptive": False,
            "fake_likelihood": 0.1,
            "confidence": 0.8,
            "signals": [],
            "flags": ["CLEAN"],
        }))
        try:
            result = run(GeminiVisionCheck(project="test-project").run(
                _jpeg_bytes(),
                {"doc_type": "passport<script>alert(1)</script>", "country": "u s-1; DROP"},
            ))
        finally:
            _remove_mock_genai()

        assert result.skipped is False
        assert result.signals["doc_type"] == "passportscriptalert1script"
        assert result.signals["country"] == "US1"
        assert result.normalized_signals is not None
        assert result.normalized_signals.category == "document_authenticity"
        assert result.normalized_signals.document_type == "passportscriptalert1script"
        assert result.normalized_signals.country_code == "US1"
        prompt = mock_client.models.generate_content.call_args.kwargs["contents"][1]
        assert "passportscriptalert1script" in prompt
        assert "from US1" in prompt

    def test_document_prompt_excludes_capture_artifacts_without_concrete_evidence(self):
        mock_client = _install_mock_genai(json.dumps({
            "is_deceptive": False,
            "fake_likelihood": 0.1,
            "confidence": 0.8,
            "signals": [],
            "flags": ["CLEAN"],
        }))
        try:
            run(GeminiVisionCheck(project="test-project").run(
                _jpeg_bytes(), {"doc_type": "death certificate", "country": "ID"}
            ))
        finally:
            _remove_mock_genai()

        prompt = mock_client.models.generate_content.call_args.kwargs["contents"][1]
        assert "Ordinary capture and scan artifacts are NOT evidence of deception on their own" in prompt
        assert "photograph of a physical document" in prompt
        assert "perspective distortion" in prompt
        assert "glare or reflections" in prompt
        assert "scanner/CamScanner cleanup" in prompt
        assert "visible document-content or compositing inconsistency" in prompt
        assert "Do not use those findings for capture or scan quality alone" in prompt

    def test_document_prompt_asks_for_internal_consistency_with_calendar_allowance(self):
        """Contradictory dates on the certificate itself are checked by Gemini,
        without flagging dual-calendar documents (Hijri alongside Gregorian)."""
        mock_client = _install_mock_genai(json.dumps({
            "is_deceptive": True,
            "fake_likelihood": 0.7,
            "confidence": 0.8,
            "signals": ["registration date 2017-01-03 precedes date of death 2017-08-06"],
            "flags": ["INTERNAL_INCONSISTENCY"],
        }))
        try:
            result = run(GeminiVisionCheck(project="test-project").run(
                _jpeg_bytes(), {"doc_type": "death certificate", "country": "MA"}
            ))
        finally:
            _remove_mock_genai()

        prompt = mock_client.models.generate_content.call_args.kwargs["contents"][1]
        assert "INTERNAL INCONSISTENCY" in prompt
        assert "Hijri" in prompt
        assert "never report a difference that is only a calendar or format conversion" in prompt
        # Details decoded from ID numbers (an NIK's birth date, then its region
        # code once dates were ruled out) flagged genuine Indonesian certificates.
        assert "Treat every identity or registration number" in prompt
        assert "never decode one for any purpose" in prompt
        assert "INTERNAL_INCONSISTENCY" in result.flags

    def test_gemini_vision_keeps_default_temperature_and_names_the_country(self):
        """Google advises against lowering temperature on Gemini 3. And "ID" alone
        reads as "identity", so known countries are named."""
        mock_client = _install_mock_genai(json.dumps({
            "is_deceptive": False, "fake_likelihood": 0.1, "confidence": 0.9,
            "signals": [], "flags": ["CLEAN"],
        }))
        try:
            types = sys.modules["google.genai.types"]
            run(GeminiVisionCheck(project="test-project").run(
                _jpeg_bytes(), {"doc_type": "death_certificate", "country": "ID"}
            ))
        finally:
            _remove_mock_genai()

        assert "temperature" not in types.GenerateContentConfig.call_args.kwargs
        # JSON output, so a reply wrapped in prose cannot become a parse error.
        assert types.GenerateContentConfig.call_args.kwargs["response_mime_type"] == "application/json"
        assert "config" in mock_client.models.generate_content.call_args.kwargs
        prompt = mock_client.models.generate_content.call_args.kwargs["contents"][1]
        assert "from Indonesia" in prompt

    def test_thinking_level_and_location_reach_gemini(self):
        """Latency is almost all thinking; 3.8 Flash at its default level timed out.
        Gemini 3.x is only served from the global location."""
        _install_mock_genai(json.dumps({
            "is_deceptive": False, "fake_likelihood": 0.1, "confidence": 0.9,
            "signals": [], "flags": ["CLEAN"],
        }))
        try:
            genai = sys.modules["google.genai"]
            types = sys.modules["google.genai.types"]
            run(GeminiVisionCheck(
                project="test-project", location="global", thinking_level="LOW"
            ).run(_jpeg_bytes(), {"doc_type": "death_certificate", "country": "ID"}))
        finally:
            _remove_mock_genai()

        assert genai.Client.call_args.kwargs["location"] == "global"
        types.ThinkingConfig.assert_called_once_with(thinking_level="LOW")
        assert (
            types.GenerateContentConfig.call_args.kwargs["thinking_config"]
            is types.ThinkingConfig.return_value
        )

    def test_without_a_thinking_level_the_model_default_is_kept(self):
        _install_mock_genai(json.dumps({"is_deceptive": False, "fake_likelihood": 0.1}))
        try:
            types = sys.modules["google.genai.types"]
            run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert types.GenerateContentConfig.call_args.kwargs["thinking_config"] is None

    def test_a_stalled_attempt_is_retried_within_the_check(self, monkeypatch):
        """Regression: the whole check had one timeout equal to a single attempt's,
        so a stalled call ended the check before any retry, while the thread
        went on retrying unseen."""
        import tools.fake_image_detector.checks.gemini_vision_check as module

        monkeypatch.setattr(google_clients, "RETRY_BACKOFF_SECONDS", (0.0, 0.0))
        mock_client = _install_mock_genai("")
        ok = MagicMock()
        ok.text = json.dumps({"is_deceptive": False, "fake_likelihood": 0.1, "confidence": 0.9})
        mock_client.models.generate_content.side_effect = [TimeoutError("stalled"), ok]
        try:
            check = GeminiVisionCheck(project="p", timeout_seconds=30, attempts=2)
            result = run(check.run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert result.error is None
        assert mock_client.models.generate_content.call_count == 2

    def test_an_error_that_will_recur_is_not_retried(self, monkeypatch):
        """A 400 (a setting the model rejects) or 404 (wrong model name) fails the
        same way again; retrying only doubled the cost and the wait."""
        monkeypatch.setattr(google_clients, "RETRY_BACKOFF_SECONDS", (0.0, 0.0))

        class BadRequest(Exception):
            code = 400

        mock_client = _install_mock_genai("")
        mock_client.models.generate_content.side_effect = BadRequest("400 INVALID_ARGUMENT")
        try:
            result = run(GeminiVisionCheck(project="p", attempts=3).run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert mock_client.models.generate_content.call_count == 1
        assert result.human_escalate is True

    def test_prompt_states_todays_date(self, monkeypatch):
        """Without it Gemini judged dates against its training cutoff, so every
        genuine certificate issued since then looked future-dated."""
        from tools.fake_image_detector.checks import gemini_vision_check as module

        monkeypatch.setattr(module, "today", lambda: "2026-10-04")
        mock_client = _install_mock_genai(json.dumps({
            "is_deceptive": False, "fake_likelihood": 0.1, "confidence": 0.9,
            "signals": [], "flags": ["CLEAN"],
        }))
        try:
            run(GeminiVisionCheck(project="test-project").run(
                _jpeg_bytes(), {"doc_type": "death_certificate", "country": "ID"}
            ))
        finally:
            _remove_mock_genai()

        prompt = mock_client.models.generate_content.call_args.kwargs["contents"][1]
        assert "Today's date is 2026-10-04" in prompt

    def test_request_has_its_own_timeout_and_image_is_prepared_once(self, monkeypatch):
        """The HTTP timeout ends a hung request; the payload is prepared once,
        not re-decoded on every retry; retries back off."""
        from tools.fake_image_detector.checks import gemini_vision_check as module

        prepared, slept = [], []
        monkeypatch.setattr(module, "gemini_payload", lambda b: prepared.append(1) or (b, "image/jpeg"))
        monkeypatch.setattr(module.time, "sleep", slept.append)
        mock_client = MagicMock()
        class RateLimited(Exception):
            code = 429

        mock_client.models.generate_content.side_effect = RateLimited("429 RESOURCE_EXHAUSTED")
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        types = MagicMock()
        sys.modules["google.genai"] = mock_genai
        sys.modules["google.genai.types"] = types
        try:
            run(GeminiVisionCheck(project="p", timeout_seconds=60, attempts=3).run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert types.HttpOptions.call_args.kwargs["timeout"] == 60_000
        assert mock_client.models.generate_content.call_count == 3
        assert len(prepared) == 1
        assert slept == [1.0, 2.0]

    def test_extraction_readings_are_not_put_in_the_prompt(self):
        """With them, Gemini invented discrepancies on genuine certificates."""
        mock_client = _install_mock_genai(json.dumps({
            "is_deceptive": False, "fake_likelihood": 0.1, "confidence": 0.9,
            "signals": [], "flags": ["CLEAN"],
        }))
        try:
            run(GeminiVisionCheck(project="p").run(_jpeg_bytes(), {
                "doc_type": "death_certificate", "country": "ID",
                "extracted_fields": {"date_of_death": "DUA PULUH DESEMBER", "full_name": "SENTINEL-NAME"},
            }))
        finally:
            _remove_mock_genai()

        prompt = mock_client.models.generate_content.call_args.kwargs["contents"][1]
        assert "SENTINEL-NAME" not in prompt
        assert "Extracted fields" not in prompt

    def test_json_parse_failure_includes_bounded_raw_snippet(self):
        _install_mock_genai("not json at all")
        try:
            result = run(GeminiVisionCheck(project="test-project").run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_genai()

        assert "GEMINI_PARSE_ERROR" in result.flags
        assert "raw_snippet='not json at all'" in result.error
