# PR 46 Unit Test Plan

## Rubric

This plan describes the current tests under `tests/unit`. It focuses on the runtime surfaces affected by the branch: agent configuration, request handling, routing, tool execution, death-certificate scoring, API behavior, and fake-image-detector checks.

| Area | Unit Test Files | What Is Being Tested | Pass Criteria |
| --- | --- | --- | --- |
| Dependency and agent configuration | `tests/unit/test_agent_config.py` | Agent YAML model IDs and Google Vertex provider construction. | No agent uses retired Gemini model IDs; every agent constructs with the model its YAML names and Vertex-style provider settings; Gemini 3.x agents use the `global` location; `thinking_level` reaches Gemini, and `thinking_budget: 0` on Gemini 3 becomes LOW (3.8 Flash rejects MINIMAL). |
| WhatsApp and API request handling | `tests/unit/test_api.py` | Webhook payload normalization, health endpoint behavior, media download flow, text/image chat calls, and webhook secret enforcement. | Text payloads call `chat(text=..., session_id=...)`; image payloads download media and call `chat(image_bytes=..., image_media_type=..., session_id=...)`; status/unusable payloads return `None`; invalid secrets raise 401. |
| Agent routing and chat state | `tests/unit/test_chat.py` | Prompt construction, text/image input validation, router selection, streaming response aggregation, session history, media persistence, and tool deps. | Text and image prompts are built correctly; multiple input types are rejected; route results stream into one response; session history/media are loaded and saved as expected; the message being answered reaches the verification tool's narrative and claimant messages. |
| Embeddings | `tests/unit/test_embeddings.py` | Vertex AI embedding client setup and document/query embedding behavior. | Client uses Vertex AI project/location config; document and query task types are correct; empty input avoids API calls; vectors are normalized. |
| Session store | `tests/unit/test_session_store.py` | Firestore-backed history and media storage using fake Firestore objects. | Stored histories deserialize; expired/missing histories return empty results; media round-trips as base64; expired/missing/oversized media are handled safely. |
| Context-aware tool calls | `tests/unit/test_verify_flow_integration.py`, `tests/unit/tools/death_certificate_pipeline/test_verify.py` | The `death_certificate_verification` tool path, session media lookup, narrative/history use, GiveLight handoff, and no-document handling. | The tool pulls latest media from session storage, runs the verification pipeline with history text, returns score/band/status details, hands off high-band results, and avoids handoff when escalation or missing media applies. |
| WhatsApp death-certificate scenarios | `tests/unit/test_whatsapp_dc_scenarios.py` | Offline full-pipeline scenarios using WhatsApp-style text and image payloads. | Messages and image bytes reach the chat/tool path; consistency checks compare claimant text to certificate facts; expected score, band, consistency sub-score, mismatch count, media delivery, message history, and handoff behavior match the scenario expectations. |
| Death-certificate scoring and consistency | `tests/unit/poc/test_pipeline.py`, `tests/unit/tools/death_certificate_pipeline/test_death_certificate_consistency.py`, `tests/unit/tools/death_certificate_pipeline/test_scoring.py`, `tests/unit/tools/death_certificate_pipeline/test_stage_consistency.py`, `tests/unit/tools/death_certificate_pipeline/test_config_loader.py` | Pipeline models, narrative validation, document staging, weighted score calculation, band thresholds, extracted field passthrough, API/CLI surfaces, and Gemini consistency wrapper behavior. | Scores use the expected document/authenticity/consistency weighting; bands map correctly; hard authenticity escalation forces `ESCALATE`; a file that cannot be read goes to review with that single reason and is never sent to Gemini; the consistency model, location, thinking level, per-attempt timeout and attempts come from `scoring.yaml`, and bad settings are refused at load; a stalled story call is retried within the stage's budget and a timeout marks the stage unavailable; invalid inputs and missing credentials raise expected errors. |
| Fake image detector build and routing | `tests/unit/tools/fake_image_detector/test_build_pipeline.py`, `tests/unit/tools/fake_image_detector/test_pipeline.py` | Detector construction, configured checks, route filtering, stage-one decisions, Gemini fallback, fail-closed behavior, and human review triggers. | Pipeline builds with check IDs; document routes run document checks; ambiguous results call Gemini; hard flags and runtime errors trigger human review; clear pass/fail cases avoid unnecessary Gemini calls. |
| Shared Google clients and retries | `tests/unit/tools/fake_image_detector/test_google_clients.py`, `tests/unit/tools/fake_image_detector/test_file_formats.py` | Per-process Gemini and Vision clients, cached credentials, per-request timeouts, which failures are retried, startup warm-up, and image preparation for Gemini. | Each client is built once and reused; credentials refresh only on expiry; each request carries its own timeout; only timeouts, 429 and 5xx are retried; warm-up has a timeout, never raises and does nothing without a project; images are resized and converted as documented. |
| Fake image detector checks | `tests/unit/tools/fake_image_detector/test_checksum_check.py`, `tests/unit/tools/fake_image_detector/test_checksums.py`, `tests/unit/tools/fake_image_detector/test_cnn_deepfake_check.py`, `tests/unit/tools/fake_image_detector/test_ela_check.py`, `tests/unit/tools/fake_image_detector/test_exif_check.py`, `tests/unit/tools/fake_image_detector/test_gemini_extract_check.py`, `tests/unit/tools/fake_image_detector/test_gemini_vision_check.py`, `tests/unit/tools/fake_image_detector/test_mrz_check.py`, `tests/unit/tools/fake_image_detector/test_ocr_document_check.py`, `tests/unit/tools/fake_image_detector/test_reverse_image_check.py`, `tests/unit/tools/fake_image_detector/test_synthid_check.py` | Individual authenticity and document-analysis checks, including OCR, EXIF, checksum, MRZ, ELA, CNN, Gemini extraction/vision, reverse image search, and SynthID (off in `pipeline.yaml` until Google grants access). | Each check skips safely when dependencies/config/context are missing, passes valid inputs, fails suspicious inputs with expected flags, and records normalized signals for downstream scoring; extraction skips when no check-digit field uses its readings; the fraud check keeps Gemini's default temperature and retries a stalled attempt but not an error that will recur. |

## Test Commands

Run the focused scripted plan with:

```bash
cd /b2-platform
tests/scripts/run_test_plan.sh
```

Run every current unit test directly with:

```bash
cd /b2-platform
python -m pytest -q tests/unit
```

If the `uv` test-plan environment has already been created, run the unit suite through that environment with:

```bash
cd /b2-platform
.venv-testplan/bin/python -m pytest -q tests/unit
```

## Live Gemini OCR/Vision Evaluation

The live Gemini runner is separate from the unit test plan because it makes real Gemini calls. It is useful when you want to verify the local WhatsApp webhook intake path plus live OCR/vision quality against the death-certificate fixture image.

The script runs `tests/evals/death_certificate_consistency/test_live_whatsapp_dc_scenarios.py`. That test sends two scenarios for `tests/test_dc/Draft_2_Morocco1_00001_.png`:

1. An aligned claim where full WhatsApp text webhook payloads provide a narrative matching the certificate name, date, and place.
2. A misaligned claim where full WhatsApp text webhook payloads provide unrelated name, date, and place values.

Each scenario also sends a full WhatsApp image webhook payload. The test does not call Meta/WhatsApp servers; it locally submits production-shaped webhook JSON into `message_endpoint()`. The image webhook carries a media ID, and the test maps that media ID to the local PNG fixture before the verification tool calls live Gemini.

Run it with an API key:

```bash
cd /b2-platform
export GEMINI_API_KEY=...
tests/scripts/run_live_gemini_test_plan.sh
```

Or run it with Vertex project auth:

```bash
cd /b2-platform
export GOOGLE_CLOUD_PROJECT=...
tests/scripts/run_live_gemini_test_plan.sh
```

When using `GOOGLE_CLOUD_PROJECT`, Vertex AI also needs Application Default Credentials. Use one of:

```bash
gcloud auth application-default login
```

or:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

The script loads `.env` before checking credentials because the app imports can load `.env` during pytest. If `GOOGLE_APPLICATION_CREDENTIALS` is set there or in your shell and points to a missing file, the script fails before running pytest and prints the missing path. To use `GEMINI_API_KEY` mode instead, unset `GOOGLE_CLOUD_PROJECT` because the consistency code uses Vertex mode whenever `GOOGLE_CLOUD_PROJECT` is present.

The script does the following:

1. Verifies `uv` is installed.
2. Creates or reuses `.venv-testplan`.
3. Installs `requirements-api.txt`, `pytest`, and `pytest-asyncio`.
4. Loads `.env` if present.
5. Requires either `GEMINI_API_KEY` or `GOOGLE_CLOUD_PROJECT`.
6. Validates Vertex ADC when `GOOGLE_CLOUD_PROJECT` is set.
7. Runs the live eval with `RUN_LIVE_GEMINI_TESTS=1`.

Expected output includes one printed report per live webhook scenario, headed by:

```text
Live Gemini WhatsApp webhook death-certificate comparison
```

Each report includes:

1. `image`
2. `aligned_claim`
3. `webhook_text_messages`
4. `webhook_media_id`
5. `score`
6. `band`
7. `consistency_score`
8. `certificate`
9. `matches`
10. `mismatches`
11. `uncertain_points`
12. `flags`
13. `summary`

The expected pytest result is:

```text
2 passed
```

The aligned case passes when the full webhook flow succeeds and Gemini returns a `consistency_score >= 0.5`. The misaligned case passes when the full webhook flow succeeds and Gemini returns a `consistency_score <= 0.8`. Both cases must also download the expected media ID, send image bytes through the chat/tool path, return a verified tool response, return a consistency score between `0.0` and `1.0`, return a certificate object, and include at least one match, mismatch, or uncertain point.

## Unit Test File Walkthrough

### `tests/unit/test_agent_config.py`

This file validates agent YAML configuration.

1. Fake Google/Pydantic AI classes let agent construction run without live provider calls.
2. Agent YAML files are discovered from the repo's `agents/` directory.
3. Retired Gemini model IDs are rejected.
4. Each agent definition is constructed and checked for the model its YAML names plus the expected Vertex project config; Gemini 3.x models must use the `global` location.

### `tests/unit/test_api.py`

This tests the WhatsApp/FastAPI adapter.

1. Representative text, status, image, empty, and blank-message payloads are parsed.
2. The health endpoint returns `{"status": "ok"}`.
3. Status updates return `{"response": None}` and do not call chat.
4. Image messages download media before calling chat with image bytes and MIME type.
5. Text messages call chat with the message body and WhatsApp sender ID.
6. Missing or wrong webhook secrets raise 401; the correct secret allows processing.

### `tests/unit/test_chat.py`

This tests prompt construction and chat session behavior.

1. Text-only, image-bytes, and image-URL prompts are built correctly.
2. Multiple simultaneous input types raise `ValueError`.
3. Routing and streamed response chunks are combined into the returned text.
4. Existing session history is loaded and saved with metadata.
5. Rendered history text flattens user and assistant turns.
6. Image uploads are stored out of band, hidden from the model prompt, and exposed to tools through session deps.
7. The message being answered is added to the tool's narrative and claimant messages, so a claimant who sends the certificate first and their account second is not checked without that account.

### `tests/unit/test_embeddings.py`

This tests the Vertex AI embedding wrapper.

1. Initialization uses Vertex AI with explicit or environment-derived project/location.
2. No Gemini API key is required for Vertex mode.
3. Document embeddings use `RETRIEVAL_DOCUMENT`.
4. Query embeddings use `RETRIEVAL_QUERY`.
5. Empty document input returns an empty array without calling the API.
6. Returned vectors are L2-normalized.

### `tests/unit/test_session_store.py`

This tests Firestore-backed session and media storage using fake Firestore objects.

1. Stored Pydantic AI message JSON deserializes into message objects.
2. Expired histories are deleted and return `[]`.
3. Missing histories return `[]`.
4. Saved histories include serialized messages and metadata.
5. Image media is stored as base64 with MIME type and expiration.
6. Latest media round-trips back into bytes.
7. Expired, missing, or oversized media is handled without unsafe writes.

### `tests/unit/test_verify_flow_integration.py`

This is an integration-style unit test for the real orchestrator agent plus the verification tool.

1. Pydantic AI's `TestModel` replaces a real LLM.
2. A fake store simulates transient media storage.
3. The pipeline and GiveLight delivery functions are patched.
4. The model/tool path pulls image bytes from session context.
5. History text becomes the verification narrative.
6. High-score results include score, band, extracted fields, WhatsApp metadata, and a handoff payload.
7. The orphan-claim document-upload scenario returns verified tool output.

### `tests/unit/test_whatsapp_dc_scenarios.py`

This runs deterministic WhatsApp death-certificate scenarios through the API boundary.

1. WhatsApp text and image payloads are generated for multiple users and cases.
2. A scenario store keeps conversation history and latest media per session.
3. Media download, chat, external authenticity, Gemini consistency, and GiveLight delivery are patched for offline determinism.
4. Text turns are sent before the image upload.
5. The fake chat handler records text calls, saves media, builds tool context, and calls the real verification tool handler.
6. Pipeline orchestration runs document, authenticity, consistency, and weighted-score stages.
7. Claimant history is compared against mapped certificate facts.
8. Expected and actual score, band, consistency sub-score, mismatch count, media delivery, history, and handoff behavior are asserted.

### `tests/unit/poc/test_pipeline.py`

This is the broadest death-certificate pipeline test file.

1. It defines fake PNG/JPEG/TIFF bytes and a sample death narrative.
2. It provides clean authenticity and mocked consistency fixtures.
3. Model construction tests verify `Submission`, `DocumentSignal`, `AuthenticitySignal`, `ConsistencySignal`, `ReliabilityResult`, and `Band`.
4. Narrative validation rejects blank input.
5. Document staging detects PNG/JPEG/TIFF and rejects garbage or empty bytes.
6. Pipeline tests confirm score range, valid band, expected weights, justification text, extracted fields, and high-band clean results.
7. Missing credentials skip consistency while keeping the pipeline functional.
8. Scoring tests cover hard escalation, low/escalate thresholds, and extracted-field passthrough.
9. API endpoint tests cover multipart image/narrative success and validation failures.
10. CLI tests cover parser setup, score arguments, local image reads, missing files, GCS URI handling, and invalid GCS URI exits.

### `tests/unit/tools/death_certificate_pipeline/test_death_certificate_consistency.py`

This tests Gemini-based certificate consistency analysis.

1. Fake Gemini modules are injected into `sys.modules`.
2. Result shape includes certificate fields, score, confidence, matches, mismatches, summary, and model.
3. Score and confidence values are clamped between `0.0` and `1.0`.
4. The model is the one set in `scoring.yaml` (`consistency.model`).
5. Empty chat history, empty image bytes, and missing credentials raise `ValueError`.
6. A stalled call is retried up to `attempts`; the last failure is raised.

### `tests/unit/tools/death_certificate_pipeline/test_scoring.py`

This focuses on weighted score calculation.

1. Strong document, authenticity, and consistency signals produce score `94`, high band, and expected subscores.
2. Medium and low weighted scores map to `Band.MEDIUM` and `Band.LOW`.
3. Hard authenticity escalation overrides a high numeric score and produces `Band.ESCALATE`.

### `tests/unit/tools/death_certificate_pipeline/test_verify.py`

This tests the context-aware verification tool directly.

1. A fake store returns latest media for a session.
2. Fake tool context supplies session deps.
3. High-band results run the pipeline, send GiveLight handoff, and return status `verified`.
4. Hard escalation still returns `verified` but does not call delivery.
5. Missing media returns `status == "no_document"` and never runs the pipeline.

### `tests/unit/tools/fake_image_detector/test_build_pipeline.py`

1. `build_pipeline()` returns a `FakeImageDetectorPipeline`.
2. The built pipeline contains at least one check.
3. Every configured check ID is a non-empty string.

### `tests/unit/tools/fake_image_detector/test_checksum_check.py`

1. The check skips when there is no `doc_type`, no matching schema, or no checksum fields.
2. Valid IBAN text passes.
3. Invalid or missing IBAN text fails and records failed fields.

### `tests/unit/tools/fake_image_detector/test_checksums.py`

1. Luhn validation covers valid, invalid, empty, and formatted values.
2. IBAN validation covers valid, invalid, too-short, and spaced values.
3. MRZ check digit validation covers valid, invalid, filler-character, and bad-character values.

### `tests/unit/tools/fake_image_detector/test_cnn_deepfake_check.py`

1. Missing model paths skip/pass.
2. Fake-biased logits fail.
3. Real-biased logits pass.
4. Corrupt images and inference errors skip with errors.
5. Confidence mirrors the fake probability.
6. The check ID is `cnn_deepfake`.

### `tests/unit/tools/fake_image_detector/test_ela_check.py`

1. Transparent PNG input is processed without skip.
2. ELA signal fields are present.
3. Manipulation score is normalized.

### `tests/unit/tools/fake_image_detector/test_exif_check.py`

1. Non-JPEG images are skipped.
2. JPEGs without EXIF pass with a low-confidence `NO_EXIF_DATA` signal.
3. Photoshop/software EXIF fails as edited.
4. Camera make/model EXIF passes and records make/model signals.

### `tests/unit/tools/fake_image_detector/test_gemini_extract_check.py`

1. The check skips without doc type, schema fields, project config, or valid Gemini output.
2. Valid JSON extraction stores fields in context.
3. `None` values are excluded.
4. Passport, bank statement, birth certificate, and death certificate extraction are covered.
5. Markdown-wrapped JSON is parsed.
6. Extraction is treated as enrichment rather than authenticity rejection.
7. Death certificates are not extracted: no check-digit field uses the readings, so the call is skipped.

### `tests/unit/tools/fake_image_detector/test_gemini_vision_check.py`

1. Synthetic/deceptive responses fail with fake score, confidence, flags, and normalized synthetic signals.
2. Stock-photo indicators trigger human escalation.
3. Real-image responses pass.
4. Markdown-wrapped JSON is accepted.
5. JSON parse failures, API exceptions, and missing project config fail closed.
6. Confidence is clamped.
7. Document context is sanitized before being used in prompts/signals.
8. Parse errors include a bounded raw response snippet.
9. Temperature is left at Gemini's default; thinking level and location come from settings.
10. Each attempt has its own timeout; a stalled attempt is retried, an error that will recur (e.g. 400) is not.

### `tests/unit/tools/fake_image_detector/test_google_clients.py`

Google clients are shared per process. Creating them per call ran credential discovery every time; on a machine signed in with gcloud that starts a subprocess, and doing so while Vision gRPC calls were in flight froze the whole process for about a minute.

1. The Gemini client is built once per configuration and reused; each request carries its own timeout.
2. The Vision client is built once; each `web_detection` call has its own timeout and no library retries.
3. An access token is fetched once and refreshed only when it expires.
4. Only timeouts, 429 and 5xx are retried; 400, 403, 404 and programming errors are not.
5. Startup warm-up builds each configured client once, gives its call a timeout so startup cannot hang, never raises, and does nothing without `GOOGLE_CLOUD_PROJECT`.

### `tests/unit/tools/fake_image_detector/test_mrz_check.py`

1. Missing `passporteye` skips/pass with an error.
2. No MRZ detected skips/pass.
3. A full valid MRZ score passes with document/country signals.
4. Partial MRZ reads skip/pass but record partial-read indicators.
5. Exceptions skip/pass with an error.

### `tests/unit/tools/fake_image_detector/test_ocr_document_check.py`

1. Missing Tesseract, no document keywords, and OCR errors skip/pass.
2. Passport, national ID, bank statement, and birth certificate keywords set `context["doc_type"]`.
3. Germany and Kenya text set country codes.
4. Existing country context is preserved.
5. Document-authenticity signals are emitted when a document is detected.

### `tests/unit/tools/fake_image_detector/test_pipeline.py`

This tests the detector pipeline decision engine.

1. Unknown input type immediately flags human review.
2. Document routing only runs document checks.
3. No active checks for a route flags review.
4. Runtime-error check results cause immediate human review.
5. Human-escalation flags override score-based classification.
6. `early_exit_on_fail` can auto-reject based on fake score.
7. Ambiguous stage-one scores call Gemini.
8. Clear pass/fail face checks do not call Gemini.
9. Skipped Gemini falls back to the stage-one flag result.
10. Gemini hard escalation overrides reject.
11. Gemini real verdict can override ambiguous stage-one.
12. Documents always call Gemini.
13. Zero-confidence stage-one results are treated as review-worthy.

### `tests/unit/tools/fake_image_detector/test_reverse_image_check.py`

1. No online matches pass with `NO_MATCH`.
2. Stock-domain matches fail, set `POSSIBLE_STOCK`, and require human escalation.
3. Exact matches on non-stock domains fail with `FOUND_ONLINE`.
4. Search failures skip/pass with `REVERSE_SEARCH_UNAVAILABLE`.
5. A switched-off Vision API (403) is not retried (see `test_google_clients.py`).

### `tests/unit/tools/fake_image_detector/test_synthid_check.py`

Calls Google's Imagen watermark verification model (`imageverification@001`, us-central1 only, access restricted by Google). The check is disabled in `pipeline.yaml` until access is granted; these tests cover it for when it is turned on.

1. An `ACCEPT` decision fails with `SYNTHID_WATERMARK_DETECTED` and `LIKELY_AI_GENERATED`, which force human review.
2. A `REJECT` decision carries no weight: no watermark is not evidence of authenticity.
3. No access (403), timeouts, unexpected responses and unrecognised decisions all skip with a reason, never escalate.
4. PDFs are skipped; TIFFs are converted to PNG before sending.
