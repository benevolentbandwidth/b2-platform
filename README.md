# b2 — AI Agents for Nonprofits

> Agents that take on a nonprofit's day-to-day work, on any LLM, with the guardrails and human oversight the people they serve deserve.

**b2** is an open-source platform from the [Benevolent Bandwidth Foundation](https://benevolentbandwidth.org) for building and running AI agents for nonprofits. Nonprofits do vital work with small teams, and much of their time goes to work an agent can share: taking in requests, checking documents, answering the same questions, coordinating people. b2 gives them agents that do this work where the people they serve already are, such as WhatsApp, in their own languages, and hands every decision that needs judgment to a person.

### Our agents

| Agent | What it does | Status |
|---|---|---|
| **GiveLight intake** | Families apply to GiveLight for orphan aid over WhatsApp. The agent takes their account, verifies the death certificate, and hands GiveLight's reviewers a ready-to-review case. | Running: our first agent |
| **Volunteer management** | Helps nonprofits recruit, coordinate and keep in touch with their volunteers. | Next |

### Where we're going

Any nonprofit should be able to pick up an agent for its work and trust it.
b2 brings the parts every nonprofit would otherwise have to build alone:

- **Any LLM.** Agents aren't tied to one model provider. Each agent names its
  model, so a nonprofit can choose the one that fits its cost, privacy and
  language needs.
- **Guardrails.** Personal data is kept out of model prompts, conversations
  expire, and anything uncertain goes to a person instead of being decided by
  the model.
- **Orchestration.** One number, many agents. A router hands each
  conversation to the agent built for it, and the agents call the tools they
  need.
- **Pretrained agents.** Ready-made agents for common nonprofit work, such as
  intake and volunteer management, that a nonprofit can adapt instead of
  building from scratch.
- **Channels people already use.** WhatsApp first. No app to download and no
  account to create.

> **Status:** b2 is early. What runs today is the WhatsApp channel and its first
> agent, GiveLight intake, on Gemini. Much of this document describes the
> platform b2 is building towards; [What's Running Today](#whats-running-today)
> describes what is built.

---

## The Problem

Nonprofits could gain a great deal from AI agents, but today's tools aren't
built for them:

- **Small teams carry manual work.** Staff spend their time on intake, document
  checks and coordination instead of on the people they serve.
- **The people they serve aren't on portals.** They're on WhatsApp, on basic
  phones, writing in many languages.
- **Generic AI doesn't know the work.** A general chatbot doesn't know a
  nonprofit's process, and shouldn't be trusted to make decisions about people
  on its own.
- **Sensitive data, tight budgets.** Beneficiaries share IDs, certificates and
  family details. Nonprofits need privacy and oversight they can explain, and
  can't afford to be locked into one vendor.

b2 is built for these.

---

## How It Works

```
User sends "My maize leaves are turning yellow" on WhatsApp from Kisumu, Kenya
           ↓
    Channel Adapter (WhatsApp)
    Normalizes to canonical message envelope
    Extracts location context (device GPS or prompt inference)
           ↓
    Agentic OS Core Platform
    ├── NeMo Guardrails        (safety + PII scrub)
    ├── Location Detector      (device signal · prompt inference · phone prefix)
    ├── Intent Classifier      (language detection + intent embedding)
    ├── Skill Router           (intent × location × language → best skill match)
    ├── Agent Registry         (resolves: kenya-farming-agent v1.2)
    ├── Tool Registry          (resolves: web_search, weather, rag:kenya-farming-kb)
    ├── Skill Agent Runtime    (builds self-contained, isolated prompt)
    └── LLM API Gateway        (b2 identity · no user signal · isolated context)
           ↓
    LLM sees only: "b2 system" + isolated conversation + skill context
    LLM never sees: user identity, session ID, location, channel, or any
                    signal linking this request to any other request
           ↓
    Gemini responds in Swahili with Kenya-specific farming advice
           ↓
User receives hyperlocal expert advice. Nothing stored. LLM learned nothing.
When session ends, b2 sends a plain-language summary. User holds their own memory.
If they reply to that summary later, b2 picks up where it left off.
```

---

## What's Running Today

```
WhatsApp user
     ↓
Meta Cloud API → central webhook (whatsapp-webhook, benevolent-bandwidth project)
     Acknowledges Meta at once, then POSTs each message to b2
     ↓
POST /message — b2 WhatsApp adapter (src/api.py, on Cloud Run)
     ↓
Router (src/router.py)
     Picks an agent from agents/*.yaml by embedding similarity
     (text-embedding-004). A conversation with a sticky agent stays with it.
     ↓
Agent (pydantic-ai on Gemini, via Vertex AI) and its tools
     ↓
Reply returned to the webhook, which sends it on WhatsApp
```

### Agents

| Agent | Handles |
|---|---|
| `whatsapp_agent` | Greetings, small talk, and general questions about GiveLight and b2. The fallback for anything no other agent covers. |
| `death_certificate_poc_agent` | Orphan-aid intake for GiveLight. It takes the claimant's account of the death and the death certificate, verifies the certificate, and passes the case to GiveLight. Sticky: once a claim starts here, the rest of the conversation stays here. |

`agents/examples/` holds templates. The router reads only `agents/*.yaml`, so
they never receive messages.

### Death-certificate verification

When the certificate arrives, the intake agent calls the
`death_certificate_verification` tool
(`tools/death_certificate_pipeline/verify.py`). The model never sees the image;
the tool loads it from the session store and:

1. **Reads the file.** Images and PDFs are accepted. A file that can't be read
   goes to review.
2. **Checks authenticity.** The checks in `tools/fake_image_detector` look for
   forgery and editing. Most of the weight is on a Gemini vision check.
3. **Checks consistency.** Gemini reads the certificate's details and compares
   them with what the claimant said. If the claimant hasn't described the
   death, this check is neutral and doesn't count against them.
4. **Scores the case from 1 to 100**, using the versioned settings in
   `tools/death_certificate_pipeline/config/scoring.yaml`. The high and medium
   bands are accepted. A fraud flag always sends the case to review.
5. **Hands every case to GiveLight** with a case reference, marked either
   `accepted` or `needs_review` with the reasons. GiveLight's team does the
   human review.

Accuracy on the current test set is under
[Latest results](#latest-results).

### GiveLight handoff

b2 uploads each case (the document and a JSON payload) to a Cloud Storage
bucket, named by `GIVELIGHT_HANDOFF_BUCKET`. It can only add files there, never
read or delete them. GiveLight's own Apps Script
([`integrations/givelight-drive-automation`](integrations/givelight-drive-automation/README.md))
moves each case into GiveLight's private Drive folder, emails their team, and
deletes it from the bucket. b2 never has access to GiveLight's Drive. Details:
[docs/integrations/google-drive-handoff.md](docs/integrations/google-drive-handoff.md).

### State

For each conversation, Firestore holds the message history, the agent handling
it, and the latest uploaded image. All of it expires 72 hours after the last
message (`src/session_store.py`).

### Deployment and configuration

Two Cloud Run services run the adapter: `b2-whatsapp-adapter` and
`b2-whatsapp-adapter-staging`. Both are built by `cloudbuild.yaml`. Their
environment variables and secrets are listed in
[docs/cloud-run-config.md](docs/cloud-run-config.md).

### Running it locally

```bash
pip install -r requirements.txt
pytest tests/unit               # unit tests
python -m src.main              # chat with the agents in the terminal
python scripts/e2e_cases.py     # e2e cases, see End-to-End Test Automation
```

The agents and the verification checks call Gemini on Vertex AI, so set
`GOOGLE_CLOUD_PROJECT` and sign in with
`gcloud auth application-default login` first.

---

## The LLM API Gateway — b2 as the Identity Layer

One of b2's most important architectural properties is invisible to users and contributors: **the LLM never knows who it is talking to.**

Every call to every LLM — Gemini, GPT-4o, Claude, LLaMA — flows through b2's LLM API Gateway. From the model's perspective, every single request originates from b2. There are no users. There are no sessions. There are no locations. There are no languages. There is only b2, asking a well-formed question on behalf of a skill.

This solves two critical problems.

### Problem 1: User identity isolation

Without a gateway, LLM providers can in principle link requests to users via IP addresses, API keys, or usage patterns. With the gateway, all requests originate from a single b2 service identity. A user in Kenya and a user in Indonesia are indistinguishable to the model provider — they are both b2.

```
WITHOUT b2 gateway:
  User A in Kenya → [Kenya IP, session token A] → Gemini
  User B in Kenya → [Kenya IP, session token B] → Gemini
  Gemini provider can potentially correlate A and B by geography

WITH b2 gateway:
  User A in Kenya → b2 API Gateway → [b2 identity only] → Gemini
  User B in Kenya → b2 API Gateway → [b2 identity only] → Gemini
  Gemini provider sees only: b2, b2. No correlation possible.
```

### Problem 2: Conversation context contamination

LLMs are stateless between API calls — they have no memory. But if prompt construction is sloppy, one user's context can accidentally bleed into another user's response. Consider:

```
BAD — implicit shared context:
  Prompt: "Continue helping the user with their farming question."
  History: [turn from user A in Kenya about maize]
  New user B's message: "What crops should I plant?"
  Result: LLM responds about maize in Kenya — wrong for user B

GOOD — explicit isolated context (b2's approach):
  Every prompt is fully self-contained.
  Every prompt begins with an isolation declaration.
  History is only the current user's current conversation.
  Location, language, and skill context are explicitly declared.
  The LLM cannot be confused by a previous conversation because
  the previous conversation is not present.
```

### The isolation protocol

Every LLM call from b2 follows this structure — without exception:

```
┌─────────────────────────────────────────────────────────────────┐
│ SYSTEM PROMPT — generated fresh for every single call          │
│                                                                 │
│ [ISOLATION DECLARATION]                                         │
│ This is a new, independent conversation. You have no memory     │
│ of any previous conversation. Each request is self-contained.  │
│                                                                 │
│ [SKILL CONTEXT]                                                 │
│ You are: {skill name and description}                           │
│ Your role: {what this skill does}                               │
│ Tone: {from agent.yaml persona}                                 │
│ Response length: {from agent.yaml}                              │
│                                                                 │
│ [LANGUAGE CONTEXT]                                              │
│ Respond in: {detected language — ISO 639-1}                     │
│                                                                 │
│ [LOCATION CONTEXT — only if detected, never precise coords]     │
│ Geographic context: {country/region name only}                  │
│ Adapt advice to this context when relevant.                     │
│                                                                 │
│ [TOOL RESULTS — injected before conversation]                   │
│ Relevant information retrieved:                                 │
│ {tool output 1}                                                 │
│ {tool output 2}                                                 │
│                                                                 │
│ [SAFETY RULES]                                                  │
│ {from agent.yaml safety block}                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CONVERSATION — current session only, in-memory, never stored   │
│                                                                 │
│ [Turn 1]  user: {text}                                          │
│ [Turn 1]  assistant: {text}                                     │
│ [Turn 2]  user: {text}                                          │
│ [Turn 2]  assistant: {text}                                     │
│ [Current] user: {current message}                               │
└─────────────────────────────────────────────────────────────────┘
```

**What is never in the prompt:**
- Session ID or any user identifier
- Phone number (even hashed)
- Channel name (whatsapp, sms, telegram)
- Precise coordinates
- Any reference to other users or other conversations
- Any implicit context carried over from previous sessions

The system prompt is generated from scratch on every single call. It contains everything the LLM needs and nothing it should not have.

---

## What Makes b2 Different — Three Registries, Location-Aware Routing, LLM Isolation

### Agent Registry
Every skill is registered with a semantic description embedding, language tags, a location scope, and tool dependencies. The skill router queries the agent registry on every message using intent × location × language simultaneously.

### Tool / MCP Registry
Every tool is a reusable capability with a standard [MCP](https://modelcontextprotocol.io) interface. Built once, shared across all skills. Location-aware tools receive `LocationContext` automatically from the runtime.

### Location Registry
Location context is detected per-session and used for routing and tool enrichment. Never stored. The location registry maintains country codes, regional groupings, and skill scope boundaries.

```
TOOL / MCP REGISTRY — built once, reused by everyone
────────────────────────────────────────────────────────
web_search     ← farming · health · legal · education · ...
weather        ← farming · health · disaster-response · ...
rag            ← every skill with a knowledge base
nearby_clinics ← health (location-aware)
ocr            ← legal-aid · medical-bill · lending-lens · ...
maps           ← farming (markets) · health (clinics) · ...
calculator     ← lending-lens · delivery-optimizer · ...
[your tool]    ← immediately available to every future skill
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        CHANNEL LAYER                             │
│     WhatsApp · SMS · Telegram · Messenger · Viber · LINE         │
│                ChannelAdapter interface (base.py)                │
│          Extracts device location when available                 │
└───────────────────────────────┬──────────────────────────────────┘
                                │ Canonical Message Envelope
┌───────────────────────────────▼──────────────────────────────────┐
│                  AGENTIC OS CORE PLATFORM                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  NeMo Guardrails Layer                     │  │
│  │   PII scrub · Content safety · Jailbreak detection         │  │
│  │   Topic control · Fraud signals · Location scrub           │  │
│  └───────────────────────────┬────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼────────────────────────────────┐  │
│  │                  Location Detector                         │  │
│  │   Device → Prompt inference → Phone prefix → None          │  │
│  └───────────────────────────┬────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼────────────────────────────────┐  │
│  │                  Intent Classifier                         │  │
│  │   Language detection · Intent embedding · Confidence       │  │
│  └───────────────────────────┬────────────────────────────────┘  │
│                              │                                   │
│  ┌──────────┬────────────────▼──────────┬─────────────────────┐  │
│  │  AGENT   │     SKILL ROUTER          │   TOOL / MCP        │  │
│  │ REGISTRY │  Intent × Location ×      │   REGISTRY          │  │
│  │          │  Language routing         │                     │  │
│  └──────────┴────────────────┬──────────┴─────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼────────────────────────────────┐  │
│  │                  Skill Agent Runtime                       │  │
│  │   Builds self-contained, isolated prompt                   │  │
│  │   Injects tool results · In-session memory only            │  │
│  └───────────────────────────┬────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼────────────────────────────────┐  │
│  │                  LLM API GATEWAY                           │  │
│  │   Single b2 identity for all LLM calls                     │  │
│  │   Isolation enforcement — strips all user signals          │  │
│  │   Prompt validation — rejects non-compliant prompts        │  │
│  │   Model routing — Gemini · GPT-4o · Claude · LLaMA         │  │
│  │   Retry logic · Rate limiting · Cost tracking              │  │
│  └───────────────────────────┬────────────────────────────────┘  │
└──────────────────────────────┼───────────────────────────────────┘
                               │ b2 identity only — no user signal
┌──────────────────────────────▼───────────────────────────────────┐
│                        MODEL LAYER                               │
│         LLMAdapter interface — model-agnostic, swappable         │
│           Gemini (MVP) · GPT-4o · Claude · LLaMA                 │
│   Every model sees: b2 · Every model sees: isolated context      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Defining a Skill Agent

An agent is one YAML file in `agents/`. These are the fields the loader reads
today (`src/orchestrator/agent.py`):

```yaml
# agents/my_agent.yaml

name: my_agent
description: >
  The requests this agent should handle. The router compares each message with
  the agent's name, description and system prompt, so describe the requests,
  not the agent.

sticky: false        # true: once a conversation reaches this agent, it stays

system_prompt: |
  How the agent should behave.

tools:               # names from tools/*.yaml
  - death_certificate_verification
skills: []           # names from skills/*.yaml

provider:
  type: gemini
  model: gemini-2.5-flash
```

### Planned format

The target design adds languages, location scope, session settings, persona and
safety rules. The loader doesn't read these fields yet:

```yaml
# agents/farming-kenya/agent.yaml

name: Kenya Farming Agent
version: "1.0"
description: >
  Crop advice, pest identification, and market prices for
  smallholder farmers in Kenya.

languages: [sw, en]

location_scope:
  type: country
  countries: [KE]

session:
  timeout_minutes: 10       # per-skill — platform maximum is 4320 (72 hours)
  max_turns: 8
  auto_summary: true        # send plain-language summary to user at session end
  summary_instructions: >   # optional — override default summary prompt
    Summarize key findings, recommendations, and next steps.
    5 bullets maximum. Plain language.

persona:
  tone: warm, practical, uses Kenyan farming examples
  response_length: concise
  voice_style: conversational, no jargon
  location_aware: true

tools:
  - web_search
  - weather
  - rag:knowledge/

safety:
  never_advise: [pesticide_dosage_medical, financial_loans]
  always_refer: local_agricultural_extension_officer

channels: all
owner: contributor:your-github-handle
```

---

## Defining a Tool

A tool is a YAML file in `tools/` that points at a Python function. The model
sees the name, description and input schema:

```yaml
# tools/deathCertificatePipeline.yaml

name: death_certificate_verification
description: >
  What the tool does and when the model should call it.
input_schema:
  type: object
  properties: {}
implementation:
  type: function
  needs_context: true     # the handler receives the conversation's context
  handler: tools.death_certificate_pipeline.verify.verify_death_certificate
```

### Planned format

The target design serves tools over MCP and injects location automatically:

```yaml
# tools/weather/tool.yaml

name: Weather Tool
id: weather
version: "1.0"
description: Current conditions, forecasts, and agricultural alerts.

input_schema:
  location:
    type: object
    auto_inject: true           # runtime injects LocationContext automatically
    required: false
  days_ahead:
    type: integer
    default: 1

output_schema:
  temperature_c: number
  humidity_percent: number
  rainfall_mm: number
  conditions: string
  agricultural_alert: string | null

mcp_endpoint: tools/weather/server.py
rate_limit: 100/minute
auth: none
owner: contributor:your-github-handle
```

---

## Language Support

Gemini understands and responds natively in 100+ languages. The intent classifier detects language from the message itself — no translation layer, no extra latency, no cost.

| Region | Languages |
|--------|-----------|
| East Africa | Swahili (sw), Amharic (am), Oromo (om), Somali (so) |
| West Africa | Hausa (ha), Yoruba (yo), Igbo (ig), French (fr) |
| North Africa + Middle East | Arabic (ar), Berber (ber), Persian (fa) |
| South Asia | Hindi (hi), Bengali (bn), Urdu (ur), Gujarati (gu), Punjabi (pa) |
| Southeast Asia | Indonesian (id), Filipino (fil), Vietnamese (vi), Khmer (km) |
| Latin America | Spanish (es), Portuguese (pt), Haitian Creole (ht), Quechua (qu) |
| + 80 more | All languages natively supported by Gemini |

---

## Privacy Principles

These are not aspirations. They are architectural constraints enforced in code:

- **No conversation content is ever stored beyond the active session.** Session memory is in-process only. Platform maximum is 72 hours — set per skill in `agent.yaml`. When a session ends, it is deleted with no archive and no recovery path.
- **Users hold their own memory.** When a session closes, b2 automatically sends a plain-language summary to the user. The summary lives in their own WhatsApp chat — b2 does not keep it. If a user replies to that summary later, b2 reads it as context and continues the conversation. The user decides what to share. b2 never stores it.
- **No PII is ever written.** Phone numbers are one-way hashed to session IDs on receipt. The raw identifier never leaves the channel adapter.
- **No location is ever stored.** Location context lives in session memory only. Never written to logs, databases, or analytics.
- **No user signal ever reaches the LLM.** All LLM calls flow through the b2 API Gateway. The model provider sees only b2 — never a user identity, session, or location.
- **Every LLM call is fully isolated.** Prompts are constructed fresh for every call. No conversation from any other user, session, or channel is present. Cross-contamination is architecturally impossible.
- **No user profiles.** b2 has no concept of a user across sessions.
- **No tracking, no analytics, no behavioral data.** Usage counted anonymously — request counts only.
- **NeMo Guardrails scrubs PII** — including precise location — from every message before it reaches any LLM or tool.

### Where the current implementation differs

- **Conversations are kept for up to 72 hours, in Firestore, not in memory.**
  Cloud Run runs several instances, so a WhatsApp conversation has to survive
  between them. Firestore holds each conversation's history, the agent handling
  it, and the latest uploaded image. Records are keyed by the user's WhatsApp
  number and expire 72 hours after the last message.
- **Death certificates are passed to GiveLight by design.** Each case goes
  through a staging bucket. It is deleted once GiveLight's script collects it,
  or after 7 days.
- **Not built yet:** NeMo Guardrails, the LLM API Gateway, location detection,
  and the end-of-session summary. Model calls go from the adapter straight to
  Gemini on Vertex AI, under the adapter's own service account.

---

## Tech Stack

| Layer | Technology | Status |
|-------|-----------|--------|
| Channels | Meta WhatsApp Cloud API | Live |
| | Twilio (SMS), Telegram Bot API | Planned |
| Async pipeline | Central webhook acknowledges Meta at once and processes in the background | Live |
| | GCP Pub/Sub | Planned |
| Agent framework | pydantic-ai | Live |
| Intent + routing | Embedding similarity (Vertex AI `text-embedding-004`), sticky agents | Live |
| | Intent × location × language routing | Planned |
| LLM | Gemini 3.8 Flash and 2.5 Flash via Vertex AI | Live |
| | GPT-4o, Claude, LLaMA adapters | Planned |
| Tools | YAML-defined Python functions | Live |
| | Model Context Protocol (MCP) | Planned |
| Session state | Firestore, 72-hour TTL | Live |
| Partner handoff | Cloud Storage bucket + GiveLight's Apps Script | Live |
| Compute | GCP Cloud Run, built by Cloud Build | Live |
| Language | Python 3.12, FastAPI | Live |
| Guardrails | NVIDIA NeMo Guardrails | Planned |
| Location detection | Gemini inference + geocoding + phone prefix | Planned |
| LLM API Gateway | b2 custom (single identity, isolation enforcement) | Planned |
| Knowledge / RAG | Vertex AI Vector Search | Planned |

---

## Roadmap

### MVP (Weeks 1–6) — One channel. Three registries. Location-aware. LLM-isolated. Real users.

- [ ] `#1` Canonical message envelope (with LocationContext)
- [ ] `#2` Agent registry + agent.yaml spec (with location_scope)
- [ ] `#3` Tool / MCP registry + tool.yaml spec (with location parameter)
- [ ] `#4` LLM adapter interface + Gemini 1.5 Pro implementation
- [ ] `#5` LLM API Gateway (b2 identity, isolation enforcement, prompt validation)
- [ ] `#6` Location detector (device · prompt inference · phone prefix)
- [ ] `#7` Intent classifier + language detection
- [ ] `#8` Skill router (intent × location × language)
- [ ] `#9` WhatsApp channel adapter
- [ ] `#10` Skill agent runtime (isolation-compliant prompt builder)
- [ ] `#11` Core tool: RAG
- [ ] `#12` Core tool: web_search
- [ ] `#13` Farming skill agents (global + Kenya reference)
- [ ] `#14` Pub/Sub async worker
- [ ] `#15` Session manager (per-skill TTL up to 72hrs, auto-summary trigger on close)
- [ ] `#16` NeMo Guardrails integration
- [ ] `#17` Cloud Run deploy + GitHub Actions CI/CD
- [ ] `#18` CONTRIBUTING.md + local dev guide

### Phase 1 (Weeks 7–14)
SMS · Telegram · Messenger · Health triage · Legal aid · Education · Location tools · GPT-4o adapter · Contributor SDK v1

### Phase 2 (Weeks 15–22)
Token optimizer · Claude + LLaMA adapters · PII pipeline · Fraud signals · Tool marketplace

### Phase 3 (Weeks 23–34)
PSTN voice · Streaming STT/TTS · b2 CLI · Contributor docs · Viber · LINE

### Phase 4 (Month 9+)
Multi-region · 500+ skills · 100+ tools · 1M+ users · Open source core

---

## Project Structure

```
b2-platform/
├── src/
│   ├── api.py                     # FastAPI app: POST /message, GET /health
│   ├── chat.py                    # One conversation turn: route, run, save
│   ├── router.py                  # Picks the agent for a message
│   ├── embeddings.py              # Vertex AI embeddings for routing
│   ├── store.py                   # SQLite index of agent embeddings
│   ├── session_store.py           # Firestore history + latest image, 72-hour TTL
│   ├── orchestrator/              # Agent, tool and skill loading
│   ├── whatsapp_media.py          # Downloads inbound media from Meta
│   ├── whatsapp_outbound.py       # Sends extra WhatsApp messages
│   ├── summary_response.py        # Optional summary message after a verification
│   ├── envelope.py                # Canonical message types
│   ├── config/whatsapp.yaml       # WhatsApp channel settings
│   └── main.py                    # Terminal chat for local testing
├── agents/
│   ├── whatsapp_agent.yaml
│   ├── poc_deathCertParserAgent.yaml
│   └── examples/                  # Templates, not routed
├── tools/
│   ├── deathCertificatePipeline.yaml
│   ├── death_certificate_pipeline/  # Verification, scoring and GiveLight handoff
│   ├── fake_image_detector/         # Authenticity checks
│   ├── privacy/                     # PII scrubber
│   └── summary/
├── skills/                        # Prompt snippets agents can include
├── integrations/
│   └── givelight-drive-automation/  # GiveLight's Apps Script
├── e2e/                           # e2e harness
├── e2e_cases/                     # e2e cases (real cases are not checked in)
├── scripts/                       # e2e runner, Cloud Run config check
├── poc/                           # Earlier standalone scoring API
├── app/core/                      # Re-exports kept for older imports
├── tests/
├── docs/
├── deploy/cloud_run_config.json   # Required Cloud Run env and secrets
├── cloudbuild.yaml
├── Dockerfile
└── requirements.txt
```

---

## End-to-End Test Automation

The e2e harness runs filesystem-driven WhatsApp `/message` cases from
`e2e_cases/` and writes machine-readable reports under `e2e_runs/`.
When `--kind` is omitted, the harness discovers both `real/` and `fake/` case
folders.

The automation scripts added under `tests/scripts/` wrap that harness for local
and remote runs:

```bash
tests/scripts/run_e2e_suite.sh
```

This command:

1. Creates or reuses `.venv-testplan`.
2. Installs `requirements-api.txt`, `pytest`, and `pytest-asyncio`.
3. Enables `ENABLE_E2E_DEBUG=true` for the local harness process.
4. Runs `scripts/e2e_cases.py --debug-tools` locally.
5. Finds the newest `e2e_runs/local/*/results.json`.
6. Gates the report with `tests/scripts/check_e2e_report.py`, including
   separate `real` and `fake` summaries.

The local runner restores the previous `ENABLE_E2E_DEBUG` value when it exits.
If the variable was not set before the run, it unsets it.

If `.env` exists, the e2e scripts load it automatically before running. To run
without loading `.env`, set `E2E_LOAD_ENV=0`:

```bash
E2E_LOAD_ENV=0 tests/scripts/run_e2e_suite.sh
```

Useful local filters are passed through to the underlying harness:

```bash
tests/scripts/run_e2e_suite.sh --country example --kind real
```

Run all real and fake cases for a country:

```bash
tests/scripts/run_e2e_suite.sh --country example
```

Run only the local harness wrapper:

```bash
tests/scripts/run_e2e_local.sh
```

Gate an existing report:

```bash
python tests/scripts/check_e2e_report.py e2e_runs/local/<timestamp>/results.json
```

By default, the report gate fails when:

- Any e2e case fails.
- Any classified case is missing a structured tool verdict.
- Any case is classified as `UNKNOWN`.
- Any false positive is reported.

Optional thresholds:

```bash
python tests/scripts/check_e2e_report.py e2e_runs/local/<timestamp>/results.json \
  --max-unknown 0 \
  --max-fp 0 \
  --max-fn 1 \
  --min-accuracy 0.95
```

Remote non-production smoke tests require a deployed service URL:

```bash
BASE_URL=https://example.run.app WEBHOOK_SECRET=... tests/scripts/run_e2e_remote.sh
```

Remote structured verdicts require `ENABLE_E2E_DEBUG=true` on the deployed
service. Remote media cases also need a pre-existing Meta media ID in each
case's `expected.json`; local certificate files are not uploaded in remote mode.

Do not commit real certificates or real-person narratives as e2e fixtures.

### Latest results

Run on 2026-10-31, locally with `--debug-tools`, on the code in
[#60](https://github.com/benevolentbandwidth/b2-platform/pull/60) (scoring
config v8, detector config v9, Gemini 3.8 Flash). The case set itself is not
checked in.

| Group | Correct | |
|---|---|---|
| Indonesia real | 7 / 7 | accepted |
| Bangladesh real | 7 / 8 | accepted |
| Pakistan real | 6 / 8 | accepted |
| Indonesia fake | 29 / 29 | caught |
| Bangladesh fake | 8 / 8 | caught |
| Pakistan fake | 8 / 8 | caught |
| Moroccan fake | 4 / 9 | caught |
| **Real** | **20 / 23 (87%)** | accepted |
| **Fake** | **49 / 54 (91%)** | caught |
| **Overall** | **69 / 77 (90%)** | |

How to read these numbers:

- **The fakes vary a lot in difficulty.** The Pakistan fakes are synthetic and
  print "SYNTHETIC TEST DATA — NOT A VALID CERTIFICATE" on the page, so they
  measure little; without them, 41 of 46 fakes (89%) were caught. The
  Indonesian fakes combine headers and stamps from different villages. The
  Bangladesh fakes are real certificates with the handwriting replaced by
  typed text.
- **Realistic edits are the weak spot.** The Moroccan fakes are one real scan
  with only the handwritten fields changed. Five of the nine (002, 005, 006,
  009, 010) were accepted, rated clean at a fraud risk of 0.10–0.15, as in
  every earlier run.
- **Real certificates sent to review:** Pakistan 002 and 006, flagged as forged
  over handwritten annotations on every run so far, and Bangladesh 001, which
  passed on the previous run (the fraud check varies between runs).
- A case counts as caught when it is not accepted, whether it is sent to review
  or rejected.

Each group was run with:

```bash
python scripts/e2e_cases.py --debug-tools --country <country> --kind <real|fake>
```

---

## Contributing

We are actively looking for contributors to own MVP components.

1. Read [CONTRIBUTING.md](CONTRIBUTING.md)
2. Browse [open issues](https://github.com/benevolentbandwidth/b2-platform/issues)
3. Comment to claim an issue
4. Build, test, PR

**The deal:** You build the component. The foundation provides infrastructure, code review, and deployment. Your work serves millions of people — in their language, in their location, on their device, with their privacy intact.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Links

- [Benevolent Bandwidth Foundation](https://benevolentbandwidth.org)
- [Contributing Guide](CONTRIBUTING.md)
- [Open Issues](https://github.com/benevolentbandwidth/b2-platform/issues)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)

---

*Built with purpose by the Benevolent Bandwidth Foundation.*
*Every tool built serves every skill. Every skill serves everyone.*
*Every location matters. Every conversation is private. Every person is just b2.*
