import asyncio
import io
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from tools.fake_image_detector.checks.cnn_deepfake_check import CNNDeepfakeCheck


def _jpeg_bytes(width=64, height=64) -> bytes:
    img = Image.new("RGB", (width, height), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _mock_session(logits: list[float]):
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="pixel_values")]
    session.get_outputs.return_value = [MagicMock(name="logits")]
    session.run.return_value = [np.array([logits], dtype=np.float32)]
    return session


def run(coro):
    return asyncio.run(coro)


class TestCNNDeepfakeCheck:
    def test_skips_when_model_not_present(self, tmp_path):
        check = CNNDeepfakeCheck(model_path=tmp_path / "nonexistent.onnx")
        result = run(check.run(_jpeg_bytes(), {}))
        assert result.skipped is True
        assert result.passed is True
        assert result.confidence == 0.0

    def test_detects_fake_image(self, tmp_path):
        model_path = tmp_path / "model.onnx"
        model_path.touch()
        check = CNNDeepfakeCheck(model_path=model_path)
        # logits: [high_fake, low_real] → p_fake > 0.5 after softmax
        check._session = _mock_session([5.0, -5.0])
        result = run(check.run(_jpeg_bytes(), {}))
        assert result.passed is False
        assert "CNN_DEEPFAKE_DETECTED" in result.flags
        assert result.signals["p_fake"] > 0.5
        assert result.normalized_signals is not None
        assert result.normalized_signals.category == "synthetic"
        assert result.normalized_signals.synthetic_score == result.signals["p_fake"]

    def test_passes_real_image(self, tmp_path):
        model_path = tmp_path / "model.onnx"
        model_path.touch()
        check = CNNDeepfakeCheck(model_path=model_path)
        # logits: [low_fake, high_real] → p_fake < 0.5
        check._session = _mock_session([-5.0, 5.0])
        result = run(check.run(_jpeg_bytes(), {}))
        assert result.passed is True
        assert result.flags == []
        assert result.signals["p_real"] > 0.5

    def test_skips_on_corrupt_image(self, tmp_path):
        model_path = tmp_path / "model.onnx"
        model_path.touch()
        check = CNNDeepfakeCheck(model_path=model_path)
        check._session = _mock_session([-5.0, 5.0])
        result = run(check.run(b"not_an_image", {}))
        assert result.skipped is True
        assert result.error is not None

    def test_skips_on_inference_error(self, tmp_path):
        model_path = tmp_path / "model.onnx"
        model_path.touch()
        check = CNNDeepfakeCheck(model_path=model_path)
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="pixel_values")]
        session.get_outputs.return_value = [MagicMock(name="logits")]
        session.run.side_effect = RuntimeError("ONNX runtime error")
        check._session = session
        result = run(check.run(_jpeg_bytes(), {}))
        assert result.skipped is True
        assert "ONNX runtime error" in result.error

    def test_confidence_reflects_p_fake(self, tmp_path):
        model_path = tmp_path / "model.onnx"
        model_path.touch()
        check = CNNDeepfakeCheck(model_path=model_path)
        check._session = _mock_session([3.0, -3.0])
        result = run(check.run(_jpeg_bytes(), {}))
        assert result.passed is False
        assert result.confidence == result.signals["p_fake"]

    def test_check_id(self):
        assert CNNDeepfakeCheck.check_id == "cnn_deepfake"
