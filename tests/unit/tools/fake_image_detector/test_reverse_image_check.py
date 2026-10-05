import asyncio

from tools.fake_image_detector.checks.reverse_image_check import ReverseImageCheck, _VisionResult


def run(coro):
    return asyncio.run(coro)


class TestReverseImageCheck:
    def test_check_id(self):
        assert ReverseImageCheck.check_id == "reverse_image"

    def test_no_match_returns_pass_with_no_match_flag(self):
        check = ReverseImageCheck()
        check._search_vision = lambda _img: _VisionResult(exact_count=0, similar_count=0, top_urls=[])  # type: ignore[attr-defined]

        result = run(check.run(b"img", {}))

        assert result.passed is True
        assert result.skipped is False
        assert result.fake_score == 0.0
        assert "NO_MATCH" in result.flags
        assert result.signals["provider"] == "google_vision_web_detection"

    def test_stock_domain_reuse_returns_reject_signal(self):
        check = ReverseImageCheck()
        check._search_vision = lambda _img: _VisionResult(  # type: ignore[attr-defined]
            exact_count=1,
            similar_count=0,
            top_urls=["https://www.shutterstock.com/image-photo/example"],
        )

        result = run(check.run(b"img", {}))

        assert result.passed is False
        assert result.fake_score == 1.0
        assert "POSSIBLE_STOCK" in result.flags
        assert result.human_escalate is False  # the pipeline decides, from settings
        assert result.normalized_signals is not None
        assert result.normalized_signals.category == "staging"

    def test_exact_match_on_nonlisted_domains_returns_found_online(self):
        check = ReverseImageCheck()
        check._search_vision = lambda _img: _VisionResult(  # type: ignore[attr-defined]
            exact_count=2,
            similar_count=0,
            top_urls=["https://example.com/reused-photo"],
        )

        result = run(check.run(b"img", {}))

        assert result.passed is False
        assert "FOUND_ONLINE" in result.flags
        assert result.human_escalate is False  # the pipeline decides, from settings
        assert result.signals["provider"] == "google_vision_web_detection"

    def test_search_failure_returns_skipped_unavailable(self):
        check = ReverseImageCheck(params={"attempts": 2})

        def _boom(_img):
            raise RuntimeError("vision unavailable")

        check._search_vision = _boom  # type: ignore[attr-defined]

        result = run(check.run(b"img", {}))

        assert result.skipped is True
        assert result.passed is True
        assert "REVERSE_SEARCH_UNAVAILABLE" in result.flags
        assert "errors" in result.signals
