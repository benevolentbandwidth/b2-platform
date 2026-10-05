"""Scoring settings: the shipped file is valid, and bad files are refused."""

import textwrap

import pytest

from tools.death_certificate_pipeline.config_loader import (
    CONFIG_PATH_ENV,
    load_scoring_config,
)
from tools.death_certificate_pipeline.models import Band

_VALID = """
version: 7
stage_weights: {authenticity: 0.5, consistency: 0.5}
bands: {high: 75, medium: 50, low: 25}
accept_bands: [high, medium]
consistency: {min_score: 0.3, timeout_seconds: 30, attempts: 2, model: test-model}
"""


def _write(tmp_path, text):
    path = tmp_path / "scoring.yaml"
    path.write_text(textwrap.dedent(text))
    return path


def test_shipped_config_loads():
    config = load_scoring_config()
    assert sum(config.stage_weights.values()) == pytest.approx(1.0)
    assert Band.ESCALATE not in config.accept_bands


def test_valid_file_round_trips(tmp_path):
    config = load_scoring_config(_write(tmp_path, _VALID))
    assert config.version == 7
    assert config.stage_weights == {"authenticity": 0.5, "consistency": 0.5}
    assert config.accept_bands == frozenset({Band.HIGH, Band.MEDIUM})
    assert config.consistency_min_score == 0.3
    assert config.consistency_model == "test-model"
    # Unset: $VERTEX_LOCATION and the model's default thinking.
    assert config.consistency_location is None
    assert config.consistency_thinking_level is None


def test_location_and_thinking_level_are_read(tmp_path):
    text = _VALID.replace("model: test-model}", "model: test-model, location: global, thinking_level: low}")
    config = load_scoring_config(_write(tmp_path, text))
    assert config.consistency_location == "global"
    assert config.consistency_thinking_level == "LOW"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("consistency: 0.5}", "consistency: 0.6}", "add up to 1.0"),
        ("authenticity: 0.5, ", "", "exactly"),
        # a version-1 file still weighting the document stage is refused
        ("{authenticity: 0.5,", "{document: 0.0, authenticity: 0.5,", "exactly"),
        ("high: 75", "high: 40", "low < medium < high"),
        ("medium: 50", "medium: 50.5", "whole numbers"),
        ("[high, medium]", "[high, great]", "unknown accept_bands"),
        ("[high, medium]", "[high, escalate]", "cannot be auto-accepted"),
        ("min_score: 0.3", "min_score: 1.5", "between 0 and 1"),
        ("timeout_seconds: 30", "timeout_seconds: 0", "must be positive"),
        ("attempts: 2", "attempts: 0", "attempts"),
        ("attempts: 2, ", "", "consistency.attempts"),
        (", model: test-model", "", "consistency.model"),
        ("version: 7", "version: seven", "integer"),
        ("bands: {high: 75, medium: 50, low: 25}\n", "", "bands"),
        ("model: test-model}", "model: test-model, thinking_level: off}", "thinking_level"),
        ("model: test-model}", "model: test-model, location: ''}", "location"),
    ],
)
def test_bad_values_are_refused(tmp_path, old, new, message):
    assert old in _VALID
    with pytest.raises(ValueError, match=message):
        load_scoring_config(_write(tmp_path, _VALID.replace(old, new)))


def test_env_var_points_at_an_alternative_file(tmp_path, monkeypatch):
    monkeypatch.setenv(CONFIG_PATH_ENV, str(_write(tmp_path, _VALID)))
    assert load_scoring_config().version == 7


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scoring_config(tmp_path / "nope.yaml")
