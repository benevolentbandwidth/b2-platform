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

The full design, using a nonprofit's intake agent as the example. Parts of
this are built and parts are planned; [What's Running Today](#whats-running-today)
says which.

```
A family messages a nonprofit on WhatsApp, in Indonesian:
"My brother has died and I'm caring for his children. Can you help?"
           ↓
    Channel Adapter (WhatsApp)
    Normalizes to a canonical message envelope
           ↓
    b2 Core
    ├── Guardrails             (safety · PII scrub · prompt-injection defense)
    ├── Router                 (picks the agent built for this request)
    ├── Agent Registry         (resolves: the nonprofit's intake agent)
    ├── Tool Registry          (resolves: document_verification, partner_handoff)
    ├── Agent Runtime          (builds a self-contained, isolated prompt)
    └── LLM Gateway            (any LLM · b2 identity · no beneficiary signal)
           ↓
    The model sees only the agent's instructions and this conversation.
    It never sees who the family is or any other conversation.
           ↓
    The agent asks for the documents it needs, checks them with its tools,
    and replies in Indonesian
           ↓
Anything that needs judgment goes to the nonprofit's own staff, with the
reasons. The family gets a summary and a case reference. Conversations
expire; nothing is kept that the work doesn't need.
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

## The LLM Gateway — Any Model, No Beneficiary Identity

*Planned. Today model calls go from the adapter straight to Gemini on Vertex
AI.*

Nonprofits shouldn't have to pick one AI vendor forever, and the people they
serve shouldn't become a profile at a model provider. The LLM Gateway handles
both: **every agent can run on any LLM, and the LLM never knows who it is
talking to.**

Every call to every model — Gemini, GPT, Claude, Llama or an open model a
nonprofit hosts itself — flows through b2's gateway. The agent's YAML names the
model, so moving an agent to another provider is a configuration change, not a
rewrite. From the model's perspective, every request comes from b2. There are
no beneficiaries, no sessions and no phone numbers, only b2 asking a
well-formed question on behalf of an agent.

This solves two problems.

### Problem 1: Beneficiary identity isolation

Without a gateway, model providers can in principle link requests to people
through IP addresses, API keys or usage patterns. With the gateway, all
requests come from a single b2 service identity. Two families applying for aid
are indistinguishable to the model provider: both are b2.

```
WITHOUT b2 gateway:
  Family A → [IP, session token A] → model provider
  Family B → [IP, session token B] → model provider
  The provider can potentially correlate A and B

WITH b2 gateway:
  Family A → b2 LLM Gateway → [b2 identity only] → any model
  Family B → b2 LLM Gateway → [b2 identity only] → any model
  The provider sees only: b2, b2. No correlation possible.
```

### Problem 2: Conversation context contamination

LLMs are stateless between API calls — they have no memory. But if prompt construction is sloppy, one person's context can bleed into another person's response. Consider:

```
BAD — implicit shared context:
  Prompt: "Continue helping the applicant with their case."
  History: [turns from family A about their certificate]
  Family B's message: "What documents do you need?"
  Result: the model answers using family A's case — wrong for family B

GOOD — explicit isolated context (b2's approach):
  Every prompt is fully self-contained.
  Every prompt begins with an isolation declaration.
  History is only this person's current conversation.
  Language, agent and (if known) regional context are explicitly declared.
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
│ You are: {agent name and description}                           │
│ Your role: {what this agent does}                               │
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
│ CONVERSATION — this person's current conversation only         │
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

## What b2 Brings to Nonprofits

Every nonprofit that wants an agent would otherwise have to build the same
pieces: a WhatsApp integration, routing, guardrails, document checks, a way to
hand work to staff. b2 builds them once and shares them.

### Pretrained agents
Agents for common nonprofit work, built once and adapted per organization:
intake and eligibility checks, volunteer management, and more over time. A
nonprofit starts from a working agent and changes its instructions, tools and
model in YAML.

### Orchestration
One number, many agents. The router matches each conversation to the agent
built for it, and a sticky agent keeps a case with the same agent from start to
finish. Agents call shared tools and hand finished work to the nonprofit's own
systems.

### Guardrails and human oversight
Agents should pass anything uncertain to a person, with the reasons, instead of
deciding on their own. Today the GiveLight intake agent sends every case to
GiveLight's reviewers marked accepted or needs-review, conversations expire
after 72 hours, and the summary is written from a transcript scrubbed of phone
numbers and emails. A guardrails layer that scrubs personal data from every
model call is planned.

### Any LLM
Agents run through the LLM Gateway, so each one can use the model that fits
the nonprofit's budget, privacy needs and languages, and switch later.

### Shared tools
Every tool is a reusable capability, built once and shared across agents. The
target interface is [MCP](https://modelcontextprotocol.io).

```
TOOL REGISTRY — built once, reused by every agent
────────────────────────────────────────────────────────
document_verification ← aid intake · eligibility checks · ...
pii_scrub             ← every agent
summary               ← every agent (plain-language recap for the person)
partner_handoff       ← intake · casework · ...
scheduling            ← volunteer management · appointments · ...
rag                   ← every agent with a knowledge base
[your tool]           ← immediately available to every future agent
```

`document_verification` (the death-certificate pipeline), `pii_scrub`,
`summary` and `partner_handoff` (the GiveLight bucket handoff) exist today in
`tools/`; the others are planned.

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
│  │                  Guardrails Layer                          │  │
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
│  │  AGENT   │     ROUTER                │   TOOL / MCP        │  │
│  │ REGISTRY │  Intent × Location ×      │   REGISTRY          │  │
│  │          │  Language routing         │                     │  │
│  └──────────┴────────────────┬──────────┴─────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼────────────────────────────────┐  │
│  │                  Agent Runtime                             │  │
│  │   Builds self-contained, isolated prompt                   │  │
│  │   Injects tool results · In-session memory only            │  │
│  └───────────────────────────┬────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼────────────────────────────────┐  │
│  │                  LLM API GATEWAY                           │  │
│  │   Single b2 identity for all LLM calls                     │  │
│  │   Isolation enforcement — strips all user signals          │  │
│  │   Prompt validation — rejects non-compliant prompts        │  │
│  │   Model routing — any LLM, chosen per agent                │  │
│  │   Retry logic · Rate limiting · Cost tracking              │  │
│  └───────────────────────────┬────────────────────────────────┘  │
└──────────────────────────────┼───────────────────────────────────┘
                               │ b2 identity only — no user signal
┌──────────────────────────────▼───────────────────────────────────┐
│                        MODEL LAYER                               │
│         LLMAdapter interface — model-agnostic, swappable         │
│           Gemini (live) · GPT · Claude · Llama · open models     │
│   Every model sees: b2 · Every model sees: isolated context      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Defining an Agent

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

The target design adds languages, location scope, session settings, persona,
safety rules and human escalation. The loader doesn't read these fields yet.
The example is a volunteer management agent, the next one planned:

```yaml
# agents/volunteer-management/agent.yaml

name: Volunteer Management Agent
version: "1.0"
description: >
  Volunteers asking about open shifts, signing up, changing or cancelling a
  shift, or asking what to bring and where to go.

languages: [en, es, id]

location_scope:
  type: organization        # this nonprofit's sites only

session:
  timeout_minutes: 60       # per agent — platform maximum is 4320 (72 hours)
  max_turns: 20
  auto_summary: true        # send a plain-language summary at session end
  summary_instructions: >   # optional — override the default summary prompt
    Confirm the shifts the volunteer signed up for or changed, with dates,
    times and addresses. Plain language.

persona:
  tone: warm, appreciative, practical
  response_length: concise
  voice_style: conversational, no jargon

tools:
  - scheduling
  - rag:knowledge/          # the nonprofit's volunteer handbook

safety:
  never_share: [other_volunteers_contact_details]
  escalate_to_staff: [safeguarding_concerns, complaints, injuries]

provider:                   # any LLM, through the LLM Gateway
  type: gemini
  model: gemini-2.5-flash

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
# tools/scheduling/tool.yaml

name: Scheduling Tool
id: scheduling
version: "1.0"
description: Finds open volunteer shifts and books, moves or cancels them.

input_schema:
  action:
    type: string
    enum: [find, book, move, cancel]
  location:
    type: object
    auto_inject: true           # runtime injects LocationContext automatically
    required: false
  date_range:
    type: object
    required: false

output_schema:
  shifts: list                  # site, role, start, end, spots left
  confirmation: string | null

mcp_endpoint: tools/scheduling/server.py
rate_limit: 100/minute
auth: nonprofit                 # each nonprofit connects its own calendar or roster
owner: contributor:your-github-handle
```

---

## Language Support

The people nonprofits serve write in their own languages, and their documents
come in many more. Agents reply in the language a person writes in, and
document checks read certificates in Indonesian, Bengali, Urdu and other
languages directly — no translation layer. Gemini, the model used today,
supports 100+ languages:

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

The people nonprofits serve often share their most sensitive information: IDs,
death certificates, family circumstances. These are the principles b2 is built
to, and the target for every agent:

- **People decide about people.** Agents gather, check and explain. Decisions
  about a person's eligibility or case are made by the nonprofit's staff.
- **Nothing is kept longer than the work needs.** Conversations expire, 72 hours
  at most, set per agent. When a session ends, it is deleted with no archive and
  no recovery path.
- **The nonprofit's data stays with the nonprofit.** What an agent hands over
  goes into the nonprofit's own systems. b2 can deliver work there but never
  read it back.
- **People hold their own memory.** At the end of a session, b2 sends a
  plain-language summary to the person. It lives in their own chat. If they
  reply to it later, b2 reads it as context and continues.
- **No PII is written that doesn't have to be.** Phone numbers are one-way
  hashed to session IDs on receipt, except where a nonprofit needs a way to
  contact the person back.
- **No location is ever stored.** Location context lives in session memory only.
  Never written to logs, databases, or analytics.
- **No beneficiary signal ever reaches the LLM.** All model calls flow through
  the LLM Gateway. The model provider sees only b2 — never a person's identity,
  session or location.
- **Every LLM call is fully isolated.** Prompts are constructed fresh for every
  call. No conversation from any other person, session or channel is present.
- **No profiles, no tracking, no behavioral data.** Usage is counted
  anonymously, as request counts only.
- **Guardrails scrub PII**, including precise location, from every message
  before it reaches any LLM or tool.

### Where the current implementation differs

- **Conversations are kept for up to 72 hours, in Firestore, not in memory.**
  Cloud Run runs several instances, so a WhatsApp conversation has to survive
  between them. Firestore holds each conversation's history, the agent handling
  it, and the latest uploaded image. Records are keyed by the user's WhatsApp
  number and expire 72 hours after the last message.
- **Death certificates are passed to GiveLight by design.** Each case goes
  through a staging bucket. It is deleted once GiveLight's script collects it,
  or after 7 days.
- **Phone numbers are not hashed yet.** Besides keying the Firestore records,
  the number goes to GiveLight in each case so their team can contact the
  family, and some log lines still include it.
- **The summary is sent after a certificate is verified,** not at the end of a
  session.
- **Not built yet:** the guardrails layer, the LLM Gateway, location detection,
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
| | Other providers (GPT, Claude, Llama, self-hosted open models) through the LLM Gateway | Planned |
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

### Built
- [x] WhatsApp channel through the central webhook, on Cloud Run (staging and
      production)
- [x] Router with sticky agents, and agents, tools and skills defined in YAML
- [x] **GiveLight intake agent**: orphan-aid applications over WhatsApp,
      death-certificate verification (fraud checks, story consistency,
      scoring) and handoff to GiveLight's reviewers
- [x] Firestore conversation state with a 72-hour expiry
- [x] End-to-end test harness and a labelled test set of real and fake
      certificates

### Next
- [ ] **Volunteer management agent**: finding and booking shifts, reminders
      and changes, questions answered from the nonprofit's own handbook
- [ ] Onboarding a new nonprofit: its own WhatsApp number, credentials and
      handoff destination, without code changes
- [ ] LLM Gateway with adapters for other model providers, so any agent can
      run on any LLM
- [ ] Guardrails layer: PII scrubbing on every model call, content safety,
      prompt-injection defense
- [ ] End-of-session summaries for every agent

### Later
- [ ] A library of pretrained nonprofit agents (intake and eligibility,
      volunteer management, case follow-up, FAQs) that organizations adapt
      instead of building
- [ ] More channels: SMS, Telegram, voice
- [ ] Tools over MCP, shared across agents and organizations
- [ ] Location-aware routing and tools
- [ ] A contributor guide and SDK for building new agents and tools

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

We are looking for contributors to build platform components and new agents
for nonprofits.

1. Browse [open issues](https://github.com/benevolentbandwidth/b2-platform/issues)
2. Comment to claim an issue
3. Build, test, PR

**The deal:** You build the component. The foundation provides infrastructure,
code review, and deployment. Your work goes straight to nonprofits and the
people they serve — in their language, on the channels they already use, with
their privacy intact.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Links

- [Benevolent Bandwidth Foundation](https://benevolentbandwidth.org)
- [Open Issues](https://github.com/benevolentbandwidth/b2-platform/issues)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)

---

*Built with purpose by the Benevolent Bandwidth Foundation.*
*Every tool built serves every agent. Every agent serves a nonprofit's mission.*
*Every conversation is private. Every decision about a person is made by a person.*
