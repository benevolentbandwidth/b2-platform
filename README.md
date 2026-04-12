# b2 — Shopify for AI Agents for Good

> One identity. Every skill. Every tool. Every channel. Every language. Every location. Zero data stored.

**b2** is an open-source Agentic OS Core Platform built by the [Benevolent Bandwidth Foundation](https://benevolentbandwidth.org). It distributes the power of agentic AI to underserved communities worldwide — through the channels they already use, on the devices they already own, in the languages they already speak, with knowledge of where they actually are.

A user in rural Kenya texts a number on WhatsApp. A farmer in Senegal sends a voice note. A mother in rural Indonesia sends an SMS from a basic phone. They all reach **@b2** — one identity — and get a genuinely useful, expert answer tailored to their language, their location, and their context, in under two seconds, from a narrow AI skill built by a contributor anywhere in the world.

No app to download. No account to create. No data stored. No friction.

---

## The Problem

AI has enormous potential to help the people who need it most — smallholder farmers, patients without healthcare access, workers navigating complex legal systems, students without teachers. But the delivery mechanisms are wrong:

- **App-based AI requires** a smartphone, a data plan, an app download, an account, and digital literacy most users don't have
- **Per-skill bots require** users to know which bot to contact for which problem — a discovery burden that kills adoption
- **Generic AI gives generic answers** — advice that ignores where a user actually is, which crops grow in their soil, which laws apply in their country, which clinics are nearby
- **Direct LLM access exposes users** — most platforms pass user identity directly to model providers, creating trackable profiles
- **Data-hungry systems** exploit the communities they claim to help

b2 solves all five.

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
    Gemini 1.5 Pro responds in Swahili with Kenya-specific farming advice
           ↓
User receives hyperlocal expert advice. Nothing stored. LLM learned nothing.
When session ends, b2 sends a plain-language summary. User holds their own memory.
If they reply to that summary later, b2 picks up where it left off.
```

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
│      Gemini 1.5 Pro (MVP) · GPT-4o · Claude · LLaMA             │
│   Every model sees: b2 · Every model sees: isolated context      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Defining a Skill Agent

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

Gemini 1.5 Pro understands and responds natively in 100+ languages. The intent classifier detects language from the message itself — no translation layer, no extra latency, no cost.

| Region | Languages |
|--------|-----------|
| East Africa | Swahili (sw), Amharic (am), Oromo (om), Somali (so) |
| West Africa | Hausa (ha), Yoruba (yo), Igbo (ig), French (fr) |
| North Africa + Middle East | Arabic (ar), Berber (ber), Persian (fa) |
| South Asia | Hindi (hi), Bengali (bn), Urdu (ur), Gujarati (gu), Punjabi (pa) |
| Southeast Asia | Indonesian (id), Filipino (fil), Vietnamese (vi), Khmer (km) |
| Latin America | Spanish (es), Portuguese (pt), Haitian Creole (ht), Quechua (qu) |
| + 80 more | All languages natively supported by Gemini 1.5 Pro |

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

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Channels | Meta Cloud API, Twilio, Telegram Bot API | Widest reach, lowest friction |
| Async pipeline | GCP Pub/Sub | Webhook returns 200 in <1s |
| Guardrails | NVIDIA NeMo Guardrails | Open source, production-proven |
| Tool protocol | Model Context Protocol (MCP) | Standard, composable, location-aware |
| Location detection | Gemini inference + geocoding + phone prefix | Three-source fallback chain |
| Geocoding | Google Maps Geocoding API | Place names → country/region |
| LLM API Gateway | b2 custom (single identity, isolation enforcement) | Privacy + contamination prevention |
| Intent + routing | b2 Agentic OS Core (custom) | Intent × location × language |
| LLM (MVP) | Gemini 1.5 Pro via Vertex AI | 100+ languages, GCP-native |
| LLM (adapters) | GPT-4o, Claude, LLaMA | Model-agnostic adapter interface |
| Knowledge / RAG | Vertex AI Vector Search | Semantic retrieval |
| Session state | In-memory only, max 72 hrs per skill | Privacy: nothing persists beyond TTL |
| Compute | GCP Cloud Run | Serverless, scales to zero |
| Language | Python 3.11 | FastAPI, async-first |

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
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── envelope.py            # Canonical message + LocationContext
│   │   ├── session.py             # Ephemeral session manager
│   │   ├── safety.py              # NeMo Guardrails integration
│   │   ├── worker.py              # Pub/Sub async processor
│   │   └── config.py
│   ├── intelligence/
│   │   ├── location.py            # Location detector
│   │   ├── classifier.py          # Intent + language classifier
│   │   ├── router.py              # Skill router
│   │   ├── runtime.py             # Skill agent runtime (prompt builder)
│   │   ├── gateway.py             # LLM API Gateway (NEW)
│   │   └── adapters/
│   │       ├── base.py            # LLMAdapter interface
│   │       ├── gemini.py          # Gemini 1.5 Pro (MVP)
│   │       ├── gpt4o.py           # Phase 1
│   │       └── claude.py          # Phase 1
│   ├── registries/
│   │   ├── agent_registry.py
│   │   ├── tool_registry.py
│   │   └── location_registry.py
│   └── channels/
│       ├── base.py
│       ├── whatsapp/
│       ├── sms/
│       └── telegram/
├── agents/
│   ├── farming-global/
│   └── farming-kenya/
├── tools/
│   ├── rag/
│   └── web_search/
├── tests/
├── scripts/
│   └── deploy.sh
├── Dockerfile
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
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
