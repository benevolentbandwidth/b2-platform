"""Load and validate death-certificate scoring settings (config/scoring.yaml).

Nothing is defaulted: a missing or malformed value raises, because a silently
wrong weight is worse than a crash.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.death_certificate_pipeline.models import Band
from tools.fake_image_detector import gemini_settings

CONFIG_DIR = Path(__file__).parent / "config"
CONFIG_PATH_ENV = "B2_SCORING_CONFIG"
# Scored stages. The document stage is a pass/fail gate, not a score.
STAGES = ("authenticity", "consistency")


@dataclass(frozen=True)
class ScoringConfig:
    version: int
    stage_weights: dict[str, float]
    band_high: int
    band_medium: int
    band_low: int
    accept_bands: frozenset[Band]
    consistency_min_score: float
    consistency_timeout_seconds: float  # per attempt
    consistency_attempts: int
    consistency_model: str
    # None: $VERTEX_LOCATION. None: the model's default thinking.
    consistency_location: str | None = None
    consistency_thinking_level: str | None = None


def _number(value: Any, where: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: {where} must be a number, got {value!r}")
    return float(value)


def _section(raw: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: missing section `{key}`")
    return value


def _get(section: dict[str, Any], key: str, where: str, path: Path) -> Any:
    if key not in section:
        raise ValueError(f"{path}: missing `{where}.{key}`")
    return section[key]


def _unit(value: float, where: str, path: Path) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{path}: {where} must be between 0 and 1, got {value}")
    return value


def load_scoring_config(path: Path | str | None = None) -> ScoringConfig:
    """Read and validate a scoring config.

    With no path, uses $B2_SCORING_CONFIG if set, else config/scoring.yaml.
    """
    if path is None:
        path = os.environ.get(CONFIG_PATH_ENV) or CONFIG_DIR / "scoring.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scoring config not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f"{path}: `version` must be an integer")

    weights_raw = _section(raw, "stage_weights", path)
    if set(weights_raw) != set(STAGES):
        raise ValueError(
            f"{path}: stage_weights must have exactly {', '.join(STAGES)}; "
            f"got {', '.join(sorted(weights_raw))}"
        )
    weights = {k: _number(weights_raw[k], f"stage_weights.{k}", path) for k in STAGES}
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"{path}: stage_weights cannot be negative")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"{path}: stage_weights must add up to 1.0, got {total:g}")

    bands = _section(raw, "bands", path)
    high = _number(_get(bands, "high", "bands", path), "bands.high", path)
    medium = _number(_get(bands, "medium", "bands", path), "bands.medium", path)
    low = _number(_get(bands, "low", "bands", path), "bands.low", path)
    if not all(v.is_integer() for v in (high, medium, low)):
        raise ValueError(f"{path}: bands must be whole numbers (scores are 1-100 integers)")
    if not 1 <= low < medium < high <= 100:
        raise ValueError(
            f"{path}: bands must satisfy 1 <= low < medium < high <= 100; "
            f"got low={low:g} medium={medium:g} high={high:g}"
        )

    accept_raw = raw.get("accept_bands")
    if not isinstance(accept_raw, list):
        raise ValueError(f"{path}: `accept_bands` must be a list")
    valid = {b.value for b in Band}
    unknown = [b for b in accept_raw if b not in valid]
    if unknown:
        raise ValueError(f"{path}: unknown accept_bands {unknown}; choose from {sorted(valid)}")
    if Band.ESCALATE.value in accept_raw:
        raise ValueError(f"{path}: `escalate` means human review and cannot be auto-accepted")

    consistency = _section(raw, "consistency", path)
    min_score = _unit(
        _number(_get(consistency, "min_score", "consistency", path), "consistency.min_score", path),
        "consistency.min_score",
        path,
    )
    timeout = _number(_get(consistency, "timeout_seconds", "consistency", path), "consistency.timeout_seconds", path)
    model = _get(consistency, "model", "consistency", path)
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{path}: consistency.model must be a model name")
    if timeout <= 0:
        raise ValueError(f"{path}: consistency.timeout_seconds must be positive")
    attempts = _get(consistency, "attempts", "consistency", path)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError(f"{path}: consistency.attempts must be a whole number of at least 1")

    return ScoringConfig(
        version=version,
        stage_weights=weights,
        band_high=int(high),
        band_medium=int(medium),
        band_low=int(low),
        accept_bands=frozenset(Band(b) for b in accept_raw),
        consistency_min_score=min_score,
        consistency_timeout_seconds=timeout,
        consistency_attempts=attempts,
        consistency_model=model.strip(),
        consistency_location=gemini_settings.location(
            consistency.get("location"), f"{path}: consistency"
        ),
        consistency_thinking_level=gemini_settings.thinking_level(
            consistency.get("thinking_level"), f"{path}: consistency"
        ),
    )


@functools.lru_cache(maxsize=1)
def default_scoring_config() -> ScoringConfig:
    """The process-wide config, loaded once. Pass a ScoringConfig explicitly to override."""
    return load_scoring_config()
