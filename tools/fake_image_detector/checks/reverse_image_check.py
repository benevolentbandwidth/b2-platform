from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from tools.fake_image_detector.checks.base_check import BaseCheck, CheckContext
from tools.fake_image_detector.google_clients import (
    attempts_budget_seconds,
    retry_wait,
    retryable,
    vision_client,
)
from tools.fake_image_detector.models import (
    CheckResult,
    GL9_FLAG_FOUND_ONLINE,
    GL9_FLAG_POSSIBLE_STOCK,
    NormalizedSignals,
)

_FLAG_ALIASES = {
    "FOUND_ONLINE_REUSE": GL9_FLAG_FOUND_ONLINE,
    "STOCK_PHOTO_REUSE": GL9_FLAG_POSSIBLE_STOCK,
    "STOCK_PHOTO_INDICATORS": GL9_FLAG_POSSIBLE_STOCK,
}


@dataclass
class _VisionResult:
    exact_count: int = 0
    similar_count: int = 0
    top_urls: list[str] | None = None


class ReverseImageCheck(BaseCheck):
    check_id = "reverse_image"

    def __init__(self, params: dict | None = None) -> None:
        p = params or {}
        self._timeout_seconds = float(p.get("timeout_seconds", 10.0))
        self._attempts = max(1, int(p.get("attempts", 2)))  # tries, including the first
        self._stock_domains = {d.lower() for d in p.get("stock_domains", [
            "shutterstock.com",
            "gettyimages.com",
            "istockphoto.com",
            "adobe.com",
            "alamy.com",
            "pexels.com",
            "unsplash.com",
            "dreamstime.com",
        ])}
        self._social_domains = {d.lower() for d in p.get("social_domains", [
            "facebook.com",
            "instagram.com",
            "x.com",
            "twitter.com",
            "tiktok.com",
            "linkedin.com",
            "reddit.com",
        ])}
        self._news_domains = {d.lower() for d in p.get("news_domains", [
            "cnn.com",
            "bbc.com",
            "reuters.com",
            "apnews.com",
            "nytimes.com",
            "theguardian.com",
        ])}

    async def run(self, image_bytes: bytes, context: CheckContext) -> CheckResult:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._run_sync, image_bytes),
                # Backstop only: timeout_seconds limits each attempt, so a
                # stalled call is retried rather than left running unseen.
                timeout=attempts_budget_seconds(self._attempts, self._timeout_seconds),
            )
        except TimeoutError:
            return CheckResult(
                check=self.check_id,
                passed=True,
                skipped=True,
                confidence=0.0,
                flags=["REVERSE_SEARCH_UNAVAILABLE", "CHECK_TIMEOUT"],
                signals={"error": "Google Vision reverse search timed out"},
                normalized_signals=NormalizedSignals(
                    category="staging",
                    confidence=0.0,
                    indicators=["REVERSE_SEARCH_UNAVAILABLE", "CHECK_TIMEOUT"],
                    staging_score=0.21,
                ),
            )

    def _run_sync(self, image_bytes: bytes) -> CheckResult:
        errors: list[str] = []
        for attempt in range(self._attempts):
            try:
                result = self._search_vision(image_bytes)
                return self._score_result(result)
            except Exception as e:
                errors.append(str(e))
                # A Vision API that is switched off (403) refuses every retry too.
                if attempt == self._attempts - 1 or not retryable(e):
                    break
                time.sleep(retry_wait(attempt))

        return CheckResult(
            check=self.check_id,
            passed=True,
            skipped=True,
            confidence=0.0,
            flags=["REVERSE_SEARCH_UNAVAILABLE"],
            signals={"errors": errors},
            normalized_signals=NormalizedSignals(
                category="staging",
                confidence=0.0,
                indicators=["REVERSE_SEARCH_UNAVAILABLE"],
                staging_score=0.21,
            ),
        )

    def _search_vision(self, image_bytes: bytes) -> _VisionResult:
        try:
            from google.cloud import vision
        except ImportError as e:
            raise RuntimeError(str(e)) from e

        # Shared per process: creating a client per call ran credential
        # discovery, which on a gcloud-signed-in machine started a subprocess
        # mid-gRPC and froze the whole process for about a minute.
        client = vision_client()
        image = vision.Image(content=image_bytes)
        # Our own timeout and retries; without a timeout the call could run on
        # after the check had given up, and the library's retries would stack
        # on top of ours.
        response = client.web_detection(image=image, timeout=self._timeout_seconds, retry=None)

        if response.error and response.error.message:
            raise RuntimeError(response.error.message)

        web = response.web_detection
        full = list(getattr(web, "full_matching_images", []) or [])
        partial = list(getattr(web, "partial_matching_images", []) or [])
        pages = list(getattr(web, "pages_with_matching_images", []) or [])

        urls: list[str] = []
        for item in full:
            url = getattr(item, "url", None)
            if url:
                urls.append(url)
        for item in partial:
            url = getattr(item, "url", None)
            if url:
                urls.append(url)
        for page in pages:
            page_url = getattr(page, "url", None) or getattr(page, "page_url", None)
            if page_url:
                urls.append(page_url)

        return _VisionResult(
            exact_count=len(full),
            similar_count=len(partial),
            top_urls=urls[:20],
        )

    def _score_result(self, result: _VisionResult) -> CheckResult:
        urls = result.top_urls or []
        domains = [self._domain_from_url(u) for u in urls if u]
        has_stock = any(self._is_domain_in(d, self._stock_domains) for d in domains)
        has_social = any(self._is_domain_in(d, self._social_domains) for d in domains)
        has_news = any(self._is_domain_in(d, self._news_domains) for d in domains)
        has_gov = any(d.endswith(".gov") for d in domains if d)

        total_matches = max(0, int(result.exact_count) + int(result.similar_count))
        confidence = round(min(1.0, 0.55 + 0.08 * min(total_matches, 5)), 3)

        if total_matches <= 0:
            return CheckResult(
                check=self.check_id,
                passed=True,
                fake_score=0.0,
                confidence=0.7,
                flags=["NO_MATCH"],
                signals={"provider": "google_vision_web_detection", "match_count": 0},
                normalized_signals=NormalizedSignals(
                    category="staging",
                    confidence=0.7,
                    indicators=["NO_MATCH"],
                    staging_score=0.0,
                ),
            )

        if has_stock:
            return self._fail_result(result, confidence, 1.0, [GL9_FLAG_POSSIBLE_STOCK], domains)
        if has_gov:
            return self._fail_result(result, confidence, 0.8, ["OFFICIAL_DOCUMENT_REUSE"], domains)
        if has_social or has_news:
            return self._fail_result(result, confidence, 0.9, [GL9_FLAG_FOUND_ONLINE], domains)
        if result.exact_count > 0:
            return self._fail_result(result, confidence, 0.75, [GL9_FLAG_FOUND_ONLINE], domains)

        return self._fail_result(result, confidence, 0.4, ["SIMILAR_IMAGE_FOUND"], domains)

    def _normalize_flags(self, flags: list[str]) -> list[str]:
        canonical: list[str] = []
        for flag in flags:
            normalized = _FLAG_ALIASES.get(flag, flag)
            if normalized not in canonical:
                canonical.append(normalized)
        return canonical

    def _fail_result(
        self,
        result: _VisionResult,
        confidence: float,
        fake_score: float,
        flags: list[str],
        domains: list[str],
    ) -> CheckResult:
        normalized_flags = self._normalize_flags(flags)
        return CheckResult(
            check=self.check_id,
            passed=False,
            fake_score=round(fake_score, 3),
            confidence=confidence,
            flags=normalized_flags,
            signals={
                "provider": "google_vision_web_detection",
                "exact_count": result.exact_count,
                "similar_count": result.similar_count,
                "domains": domains[:10],
            },
            normalized_signals=NormalizedSignals(
                category="staging",
                confidence=confidence,
                indicators=normalized_flags,
                staging_score=round(fake_score, 3),
            ),
            # Flags only: whether a flag forces human review is decided by the
            # pipeline from hard_escalation_flags in pipeline.yaml, so that list
            # is the single source of truth.
        )

    def _domain_from_url(self, value: str) -> str:
        try:
            netloc = urlparse(value).netloc.lower()
        except Exception:
            return ""
        if netloc.startswith("www."):
            return netloc[4:]
        return netloc

    def _is_domain_in(self, domain: str, domain_set: set[str]) -> bool:
        if not domain:
            return False
        return any(domain == d or domain.endswith(f".{d}") for d in domain_set)
