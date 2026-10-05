from __future__ import annotations

import asyncio
import io
from typing import Literal

try:
    import pytesseract
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 20_000_000
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None
    Image = None

from tools.fake_image_detector.checks.base_check import BaseCheck
from tools.fake_image_detector.config_loader import PipelineConfig, load_pipeline_config
from tools.fake_image_detector.models import (
    CheckResult,
    Escalation,
    ToolResult,
    Verdict,
)

InputType = Literal["document", "face", "unknown"]

_DOCUMENT_CHECK_IDS = {
    "exif",
    "ocr_document",
    "mrz",
    "gemini_extract",
    "checksum",
    "reverse_image",
    "synthid",
}
_FACE_CHECK_IDS = {"exif", "ela", "cnn_deepfake", "reverse_image"}
_DOC_TEXT_MIN_ALNUM = 25


class FakeImageDetectorPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        checks: list[tuple[object, BaseCheck]],
        gemini_check: BaseCheck | None = None,
        gemini_max_concurrency: int = 5,
    ):
        # checks: list of (CheckConfig, BaseCheck instance)
        self._config = config
        self._checks = checks
        self._gemini_check = gemini_check
        self._vision_semaphore = asyncio.Semaphore(max(1, gemini_max_concurrency))

    @property
    def config(self) -> PipelineConfig:
        return self._config

    async def run(self, image_bytes: bytes, context: dict | None = None) -> ToolResult:
        if context is None:
            context = {}

        input_type = context.get("input_type")
        if input_type not in {"document", "face"}:
            input_type = await self._classify_input_type(image_bytes)
        context["input_type"] = input_type

        if input_type == "unknown":
            return ToolResult(
                verdict=Verdict.FLAG,
                risk_score=round(self._config.clear_pass, 4),
                escalation=Escalation.HUMAN_REVIEW,
                checks=[],
                early_exit=True,
                early_exit_reason="UNKNOWN_INPUT_TYPE",
            )

        check_ids = _DOCUMENT_CHECK_IDS if input_type == "document" else _FACE_CHECK_IDS
        active_checks = [(cfg, check) for cfg, check in self._checks if cfg.id in check_ids]

        if not active_checks:
            return ToolResult(
                verdict=Verdict.FLAG,
                risk_score=round(self._config.clear_pass, 4),
                escalation=Escalation.HUMAN_REVIEW,
                checks=[],
                early_exit=True,
                early_exit_reason=f"NO_ACTIVE_CHECKS_FOR_{input_type.upper()}",
            )

        results: list[CheckResult] = []

        for check_cfg, check in active_checks:
            try:
                result = await check.run(image_bytes, context)
            except Exception as e:
                result = CheckResult(
                    check=check_cfg.id,
                    passed=True,
                    confidence=0.0,
                    skipped=True,
                    error=str(e),
                )

            if result.error is not None and not result.skipped:
                results.append(
                    CheckResult(
                        check=check_cfg.id,
                        passed=False,
                        fake_score=round(self._config.clear_pass, 4),
                        confidence=1.0,
                        flags=["CHECK_RUNTIME_ERROR"],
                        signals={"error": result.error},
                        skipped=False,
                        error=result.error,
                    )
                )
                return ToolResult(
                    verdict=Verdict.FLAG,
                    risk_score=round(self._config.clear_pass, 4),
                    escalation=Escalation.HUMAN_REVIEW,
                    checks=results,
                    early_exit=True,
                    early_exit_reason=f"{check_cfg.id} runtime error",
                )

            results.append(result)

            if self._should_force_human_escalation(result):
                return ToolResult(
                    verdict=Verdict.FLAG,
                    risk_score=round(max(self._config.clear_pass, result.fake_score), 4),
                    escalation=Escalation.HUMAN_REVIEW,
                    checks=results,
                    early_exit=True,
                    early_exit_reason=self._escalation_reason(check_cfg.id, result),
                )

            if check_cfg.early_exit_on_fail and not result.passed and not result.skipped:
                risk_score = result.fake_score
                verdict, escalation = self._classify(risk_score)
                return ToolResult(
                    verdict=verdict,
                    risk_score=round(risk_score, 4),
                    escalation=escalation,
                    checks=results,
                    early_exit=True,
                    early_exit_reason=f"{check_cfg.id} failed with flags: {result.flags}",
                )

        risk_score = self._score(results)
        verdict, escalation = self._classify(risk_score)

        stage1 = ToolResult(
            verdict=verdict,
            risk_score=round(risk_score, 4),
            escalation=escalation,
            checks=results,
            early_exit=False,
            early_exit_reason=None,
        )

        # Documents always go to Gemini — OCR/MRZ are detection-only, Gemini validates content.
        # Photos go to Gemini when stage-1 is ambiguous OR has suspicious failed checks below clear_pass.
        has_failed_stage1_check = any(not r.skipped and not r.passed for r in results)
        should_run_gemini = (
            input_type == "document"
            or (self._config.clear_pass <= risk_score < self._config.clear_fail)
            or (input_type == "face" and has_failed_stage1_check and risk_score < self._config.clear_pass)
        )
        if should_run_gemini:
            return await self._run_gemini_stage(image_bytes, context, stage1)

        return stage1

    async def _run_gemini_stage(
        self, image_bytes: bytes, context: dict, stage1: ToolResult
    ) -> ToolResult:
        if self._gemini_check is None:
            return stage1

        async with self._vision_semaphore:
            result = await self._gemini_check.run(image_bytes, context)

        if result.skipped:
            return ToolResult(
                verdict=stage1.verdict,
                risk_score=stage1.risk_score,
                escalation=stage1.escalation,
                checks=stage1.checks + [result],
                early_exit=stage1.early_exit,
                early_exit_reason=stage1.early_exit_reason,
            )

        final_risk = result.fake_score
        if self._should_force_human_escalation(result):
            return ToolResult(
                verdict=Verdict.FLAG,
                risk_score=round(max(self._config.clear_pass, final_risk), 4),
                escalation=Escalation.HUMAN_REVIEW,
                checks=stage1.checks + [result],
                early_exit=True,
                early_exit_reason=self._escalation_reason("gemini_vision", result),
            )

        verdict, escalation = self._classify(final_risk)

        return ToolResult(
            verdict=verdict,
            risk_score=round(final_risk, 4),
            escalation=escalation,
            checks=stage1.checks + [result],
            early_exit=False,
            early_exit_reason=None,
        )

    def _score(self, results: list[CheckResult]) -> float:
        active = [r for r in results if not r.skipped]
        if not active:
            return self._config.clear_pass
        total_weight = sum(r.confidence for r in active)
        if total_weight <= 0.0:
            # Fail-safe: checks ran but produced no usable confidence → force human review band.
            return round(min(self._config.clear_fail, self._config.clear_pass + 0.01), 4)
        return round(sum(r.fake_score * r.confidence for r in active) / total_weight, 4)

    def _hard_escalation_flags(self, flags: list[str]) -> list[str]:
        return [flag for flag in flags if flag in self._config.hard_escalation_flags]

    def _escalation_reason(self, check_id: str, result: CheckResult) -> str:
        flags = self._hard_escalation_flags(result.flags)
        if flags:
            return f"{check_id} triggered hard escalation: {flags}"
        return f"{check_id} requested human review"

    def _should_force_human_escalation(self, result: CheckResult) -> bool:
        if result.human_escalate:
            return True
        return bool(self._hard_escalation_flags(result.flags))

    def _classify(self, risk_score: float) -> tuple[Verdict, Escalation]:
        if risk_score >= self._config.clear_fail:
            return Verdict.REJECT, Escalation.AUTO_REJECT
        if risk_score >= self._config.clear_pass:
            return Verdict.FLAG, Escalation.HUMAN_REVIEW
        return Verdict.PASS, Escalation.AUTO_ACCEPT

    async def _classify_input_type(self, image_bytes: bytes) -> InputType:
        return await asyncio.to_thread(self._classify_input_type_sync, image_bytes)

    def _classify_input_type_sync(self, image_bytes: bytes) -> InputType:
        # TIFF magic bytes (little-endian or big-endian) → scanned document, never a selfie
        if image_bytes[:4] in (b"\x49\x49\x2a\x00", b"\x4d\x4d\x00\x2a"):
            return "document"

        if Image is None or pytesseract is None:
            return "unknown"

        try:
            image = Image.open(io.BytesIO(image_bytes))
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except Exception:
            return "unknown"

        texts = [t for t in data["text"] if t.strip()]
        total_alnum = sum(sum(ch.isalnum() for ch in t) for t in texts)
        line_count = len({
            (b, l)
            for b, l, t in zip(data["block_num"], data["line_num"], data["text"])
            if t.strip()
        })

        if total_alnum >= _DOC_TEXT_MIN_ALNUM and line_count >= 3:
            return "document"
        if total_alnum < 10:
            return "face"
        return "unknown"


def build_pipeline(config_path=None) -> FakeImageDetectorPipeline:
    from importlib import import_module

    def _optional_check(module_name: str, class_name: str) -> type[BaseCheck] | None:
        try:
            module = import_module(module_name)
            cls = getattr(module, class_name, None)
            return cls if isinstance(cls, type) else None
        except Exception:
            return None

    registry: dict[str, type[BaseCheck]] = {}
    for check_id, module_name, class_name in (
        ("exif", "tools.fake_image_detector.checks.exif_check", "EXIFCheck"),
        ("ela", "tools.fake_image_detector.checks.ela_check", "ELACheck"),
        ("cnn_deepfake", "tools.fake_image_detector.checks.cnn_deepfake_check", "CNNDeepfakeCheck"),
        ("mrz", "tools.fake_image_detector.checks.mrz_check", "MRZCheck"),
        ("ocr_document", "tools.fake_image_detector.checks.ocr_document_check", "OCRDocumentCheck"),
        ("gemini_extract", "tools.fake_image_detector.checks.gemini_extract_check", "GeminiExtractCheck"),
        ("checksum", "tools.fake_image_detector.checks.checksum_check", "ChecksumCheck"),
        ("reverse_image", "tools.fake_image_detector.checks.reverse_image_check", "ReverseImageCheck"),
        ("synthid", "tools.fake_image_detector.checks.synthid_check", "VertexSynthIDCheck"),
    ):
        cls = _optional_check(module_name, class_name)
        if cls is not None:
            registry[check_id] = cls

    config = load_pipeline_config(config_path)

    checks = []
    for check_cfg in config.checks:
        if not check_cfg.enabled:
            continue
        cls = registry.get(check_cfg.id)
        if cls is None:
            continue
        checks.append((check_cfg, cls(params=check_cfg.params)))

    gemini_cls = _optional_check(
        "tools.fake_image_detector.checks.gemini_vision_check",
        "GeminiVisionCheck",
    )
    gemini_check = (
        gemini_cls(
            timeout_seconds=config.gemini.timeout_seconds,
            attempts=config.gemini.attempts,
            model=config.gemini.model,
            location=config.gemini.location,
            thinking_level=config.gemini.thinking_level,
        )
        if config.gemini.enabled and gemini_cls is not None
        else None
    )
    return FakeImageDetectorPipeline(
        config=config,
        checks=checks,
        gemini_check=gemini_check,
        gemini_max_concurrency=config.gemini.max_concurrency,
    )
