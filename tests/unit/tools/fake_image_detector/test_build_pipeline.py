from tools.fake_image_detector.pipeline import FakeImageDetectorPipeline, build_pipeline


def test_build_pipeline_returns_pipeline_instance():
    pipeline = build_pipeline()
    assert isinstance(pipeline, FakeImageDetectorPipeline)


def test_build_pipeline_loads_at_least_one_check():
    pipeline = build_pipeline()
    assert len(pipeline._checks) > 0, (
        "build_pipeline() produced no checks — likely a silent import failure in one or more check modules"
    )


def test_build_pipeline_check_ids_are_strings():
    pipeline = build_pipeline()
    for cfg, check in pipeline._checks:
        assert isinstance(cfg.id, str) and cfg.id, f"check config has empty id: {cfg!r}"


def test_shipped_gemini_settings_reach_both_gemini_checks():
    """Location and thinking level come from pipeline.yaml for each Gemini call."""
    from tools.fake_image_detector.config_loader import load_pipeline_config

    config = load_pipeline_config()
    pipeline = build_pipeline()
    vision = pipeline._gemini_check
    assert (vision._model, vision._location, vision._thinking_level) == (
        config.gemini.model, config.gemini.location, config.gemini.thinking_level,
    )
    params = next(c.params for c in config.checks if c.id == "gemini_extract")
    extract = next(check for _cfg, check in pipeline._checks if check.check_id == "gemini_extract")
    assert extract._location == params["location"]
    assert extract._thinking_level == params["thinking_level"].upper()


def test_bad_thinking_level_is_refused(tmp_path):
    import pytest
    from tools.fake_image_detector.config_loader import CONFIG_DIR, load_pipeline_config

    text = (CONFIG_DIR / "pipeline.yaml").read_text()
    path = tmp_path / "pipeline.yaml"
    path.write_text(text.replace("  thinking_level: LOW\n", "  thinking_level: OFF\n", 1))
    with pytest.raises(ValueError, match="thinking_level"):
        load_pipeline_config(path)


def test_extraction_without_a_model_is_refused_at_load(tmp_path):
    """It used to fall back to 2.5 Flash silently, which rejects a thinking
    level, so extraction failed on every case without any error at startup."""
    import pytest
    from tools.fake_image_detector.config_loader import CONFIG_DIR, load_pipeline_config

    text = (CONFIG_DIR / "pipeline.yaml").read_text()
    marker = "      model: gemini-3.8-flash\n      location: global\n      thinking_level: LOW\n      timeout_seconds: 60\n"
    assert marker in text
    path = tmp_path / "pipeline.yaml"
    path.write_text(text.replace(marker, marker.replace("      model: gemini-3.8-flash\n", "")))
    with pytest.raises(ValueError, match="gemini_extract params: model"):
        load_pipeline_config(path)


def test_the_old_max_retries_key_is_refused(tmp_path):
    """`max_retries: 2` meant two attempts, so `1` silently meant no retry."""
    import pytest
    from tools.fake_image_detector.config_loader import CONFIG_DIR, load_pipeline_config

    text = (CONFIG_DIR / "pipeline.yaml").read_text()
    path = tmp_path / "pipeline.yaml"
    path.write_text(text.replace("  attempts: 2\n", "  max_retries: 2\n", 1))
    with pytest.raises(ValueError, match="`max_retries` is now `attempts`"):
        load_pipeline_config(path)
