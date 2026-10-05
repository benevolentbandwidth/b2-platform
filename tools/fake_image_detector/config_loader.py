from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from tools.fake_image_detector import gemini_settings
from tools.fake_image_detector.models import GL9_HARD_ESCALATION_FLAGS

CONFIG_DIR = Path(__file__).parent / "config"


@dataclass
class CheckConfig:
    id: str
    enabled: bool
    early_exit_on_fail: bool = False
    params: dict = field(default_factory=dict)


@dataclass
class GeminiConfig:
    enabled: bool = False
    # Required in pipeline.yaml; the default only serves direct construction.
    model: str = "gemini-2.5-flash"
    # None: $VERTEX_LOCATION. None: the model's default thinking.
    location: str | None = None
    thinking_level: str | None = None
    timeout_seconds: float = 30.0
    attempts: int = 3  # tries per call, including the first
    max_concurrency: int = 5


@dataclass
class PipelineConfig:
    clear_fail: float
    clear_pass: float
    checks: list[CheckConfig] = field(default_factory=list)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    # Flags that send a case straight to human review, whatever the score.
    # Required in pipeline.yaml; the default only serves code that builds a
    # PipelineConfig directly.
    hard_escalation_flags: frozenset[str] = field(
        default_factory=lambda: frozenset(GL9_HARD_ESCALATION_FLAGS)
    )
    # Required in pipeline.yaml; None only for directly constructed configs.
    version: int | None = None


def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    path = path or CONFIG_DIR / "pipeline.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    thresholds = raw.get("thresholds", {})
    if "clear_fail" not in thresholds or "clear_pass" not in thresholds:
        raise ValueError(f"Missing required thresholds in pipeline config: {path}")

    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f"pipeline config `version` must be an integer: {path}")

    flags_raw = raw.get("hard_escalation_flags")
    if not isinstance(flags_raw, list) or not all(isinstance(f, str) for f in flags_raw):
        raise ValueError(
            f"pipeline config must list hard_escalation_flags as strings: {path}"
        )

    gemini_raw = raw.get("gemini", {})
    # Renamed: `max_retries: 2` meant two attempts (one retry), so `1` silently
    # gave no retry at all. An old copy of this file is refused, not misread.
    renamed = [where for where, section in [("gemini", gemini_raw)] + [
        (f"checks.{c.get('id')}.params", c.get("params") or {}) for c in raw.get("checks", [])
    ] if "max_retries" in section]
    if renamed:
        raise ValueError(
            f"{path}: `max_retries` is now `attempts` (the number of tries, including "
            f"the first); rename it in: {', '.join(renamed)}"
        )
    model = gemini_raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"pipeline config must set gemini.model: {path}")
    gemini = GeminiConfig(
        enabled=bool(gemini_raw.get("enabled", False)),
        model=model.strip(),
        location=gemini_settings.location(gemini_raw.get("location"), f"{path}: gemini"),
        thinking_level=gemini_settings.thinking_level(
            gemini_raw.get("thinking_level"), f"{path}: gemini"
        ),
        timeout_seconds=float(gemini_raw.get("timeout_seconds", 30.0)),
        attempts=int(gemini_raw.get("attempts", 3)),
        max_concurrency=int(gemini_raw.get("max_concurrency", 5)),
    )

    for check in raw.get("checks", []):
        if check.get("id") == "gemini_extract" and check.get("enabled"):
            _validate_gemini_params(check.get("params") or {}, f"{path}: gemini_extract params")

    return PipelineConfig(
        clear_fail=thresholds["clear_fail"],
        clear_pass=thresholds["clear_pass"],
        checks=[
            CheckConfig(
                id=c["id"],
                enabled=c["enabled"],
                early_exit_on_fail=c.get("early_exit_on_fail", False),
                params=c.get("params", {}),
            )
            for c in raw.get("checks", [])
        ],
        gemini=gemini,
        hard_escalation_flags=frozenset(flags_raw),
        version=version,
    )


def _validate_gemini_params(params: dict, where: str) -> None:
    """A Gemini check's model, location and thinking level, checked at load.

    The model is required, as for gemini.model: a missing one used to fall back
    to another model silently, which rejects a thinking level and so failed on
    every case without any error at startup.
    """
    model = params.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{where}: model must be a model name")
    gemini_settings.location(params.get("location"), where)
    gemini_settings.thinking_level(params.get("thinking_level"), where)


def load_document_schemas(path: Path | None = None) -> dict:
    path = path or CONFIG_DIR / "document_schemas.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
