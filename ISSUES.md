# b2 Platform — All 18 MVP GitHub Issues
# Complete descriptions ready to paste into GitHub Issues
# ─────────────────────────────────────────────────────────────────────────────


## Issue #1 — Canonical Message Envelope
**Labels:** `core` `good first issue`
**Depends on:** nothing
**Estimated effort:** 2–3 hours

### Summary
Define the universal message format that every channel normalizes to before anything else in the platform touches a message. This is the single most important file in the codebase — every other component depends on it.

### Background
b2 supports WhatsApp, SMS, Telegram, voice, and future channels. Each speaks a different protocol. The canonical envelope is the contract that decouples channels from intelligence. Once normalized, no other component ever needs to know which channel a message came from.

### What to build
`app/core/envelope.py`

```python
@dataclass
class LocationContext:
    """
    Detected per-session. Never written to any storage.
    Coordinates only from device signal — never from geocoding.
    """
    lat: float | None           # device signal only
    lng: float | None           # device signal only
    country_code: str | None    # ISO 3166-1 alpha-2 e.g. "KE"
    region: str | None          # e.g. "east-africa"
    city: str | None            # best-effort from geocoding
    source: str                 # "device" | "prompt" | "phone_prefix" | "none"
    confidence: float           # 0.0–1.0


@dataclass
class SessionContext:
    turn_count: int
    prior_skill: str | None
    conversation_history: list[dict]   # [{role, content}] current session only
    language: str


@dataclass
class CanonicalMessage:
    session_id: str             # sha256(channel_user_id + channel + date) — never a real ID
    channel: str                # "whatsapp" | "sms" | "telegram" | "voice"
    input_type: str             # "text" | "audio" | "image" | "video" | "document" | "location"
    text_content: str           # always present — transcripts for audio, captions for images
    media_url: str | None       # temporary, never persisted
    language_hint: str | None   # ISO 639-1, set by classifier
    location: LocationContext   # always present, confidence may be 0.0
    session_context: SessionContext
    timestamp: datetime
    raw_message_id: str | None  # for deduplication only

    @staticmethod
    def make_session_id(channel_user_id: str, channel: str) -> str:
        """One-way hash. Rotates daily. Cannot be reversed."""

    def to_agent_prompt(self) -> str:
        """Returns clean text for the LLM. Never includes session_id, channel, or coordinates."""


@dataclass
class CanonicalResponse:
    text: str
    skill_used: str
    confidence: float
    language: str
    session_id: str
    location_used: bool
    follow_up_hint: str | None
```

### Privacy contract (enforced in this file)
- `make_session_id` must use sha256 — no raw identifiers ever stored
- `lat`/`lng` only populated from device signal — never inferred or geocoded
- `LocationContext` is never written to any storage
- `to_agent_prompt()` never includes session_id, channel name, or coordinates
- No PII in any log statement in this file

### Acceptance criteria
- [ ] All dataclasses defined with correct types and docstrings
- [ ] `make_session_id()` is one-way sha256, rotates daily
- [ ] Same user same day → same session_id. Same user next day → different session_id
- [ ] `to_agent_prompt()` includes location context when confidence > 0.5, never includes coordinates
- [ ] Unit tests: hash consistency within a day, hash rotation across days, prompt with/without location
- [ ] Zero PII fields, zero coordinate fields in any log statement


---


## Issue #2 — Agent Registry + agent.yaml Spec
**Labels:** `registry` `core`
**Depends on:** #1
**Estimated effort:** 5–6 hours

### Summary
Build the agent registry — loads skill definitions from `agent.yaml` files, generates semantic embeddings for each skill description, and exposes `find_best_skill()` for the intent classifier. Location scope is a first-class routing dimension alongside intent and language.

### What to build
`app/registries/agent_registry.py`

```python
@dataclass
class LocationScope:
    type: str                           # "global" | "region" | "country" | "coordinates"
    regions: list[str]                  # e.g. ["east-africa", "west-africa"]
    countries: list[str]                # ISO 3166-1 alpha-2
    center: tuple[float, float] | None  # lat, lng for coordinates type
    radius_km: float | None

    def matches(self, location: LocationContext) -> tuple[bool, float]:
        """Returns (matches, precision_score). Higher = more specific match."""


@dataclass
class AgentDefinition:
    id: str
    name: str
    version: str
    description: str
    languages: list[str]
    location_scope: LocationScope
    session_timeout_minutes: int
    max_turns: int
    persona: dict
    location_aware: bool
    tools: list[str]
    safety: dict
    channels: list[str]
    owner: str
    embedding: np.ndarray | None        # generated on registration


class AgentRegistry:
    async def load(self, agents_dir: str) -> None
    async def register(self, yaml_path: Path) -> AgentDefinition
    async def find_best_skill(
        self,
        query_embedding: np.ndarray,
        location: LocationContext,
        language: str,
        threshold: float = 0.65,
        top_k: int = 5,
    ) -> list[tuple[str, float]]
    def get_agent(self, agent_id: str) -> AgentDefinition | None
    async def list_agents(self) -> list[AgentDefinition]
```

### The agent.yaml contract

```yaml
name: string                  # required
version: string               # required
description: string           # required — used for intent embedding

languages: list[str]          # required, ISO 639-1

location_scope:               # required
  type: global | region | country | coordinates
  regions: list[str]          # for type: region e.g. [east-africa]
  countries: list[str]        # for type: country, ISO 3166-1 alpha-2
  center: [lat, lng]          # for type: coordinates
  radius_km: number           # for type: coordinates

session:
  timeout_minutes: int        # required — per-skill, not global
  max_turns: int              # required — per-skill, not global

persona:
  tone: string
  response_length: concise | detailed
  voice_style: string
  location_aware: bool        # inject country/region into system prompt

tools: list[str]              # tool IDs from tool registry

safety:
  never_advise: list[str]
  always_refer: string | null

channels: all | list[str]
owner: string                 # "contributor:github-handle"
```

### Routing score formula
```
score = (intent_similarity × 0.60) + (location_bonus × 0.30) + (language_bonus × 0.10)

location_bonus:
  coordinates match within radius : 0.30
  country match                   : 0.20
  region match                    : 0.10
  global scope (always applies)   : 0.05
  no location signal              : all skills scored on intent only
```

### Routing guarantee
A `global` scope skill is always the fallback. The router never returns `no_match` solely due to location.

### Acceptance criteria
- [ ] All yaml fields parsed and validated — clear error on malformed yaml
- [ ] `find_best_skill()` applies three-factor scoring (intent × location × language)
- [ ] Global scope skill always wins when no location signal
- [ ] Country-scope skill beats global at same intent confidence
- [ ] `register()` works at runtime — new skill immediately routable after deploy
- [ ] Unit tests: global fallback, country match, region match, coordinate match, no location signal
- [ ] Integration test: two skills (global + country) routing correctly based on location


---


## Issue #3 — Tool / MCP Registry + tool.yaml Spec
**Labels:** `registry` `core`
**Depends on:** #1
**Estimated effort:** 5–7 hours

### Summary
Build the tool registry — loads tool definitions from `tool.yaml` files, validates MCP endpoints, and resolves tool dependencies for skill agents at runtime. Tools built for one skill are automatically available to all skills. This is the compounding value engine of the platform.

### What to build
`app/registries/tool_registry.py`

```python
@dataclass
class ToolDefinition:
    id: str
    name: str
    version: str
    description: str
    input_schema: dict              # JSON Schema
    output_schema: dict             # JSON Schema
    mcp_endpoint: str               # path to MCP server implementation
    rate_limit: str | None          # e.g. "100/minute"
    auth_required: bool
    owner: str
    has_location_input: bool        # true if any field declares auto_inject: true


class ToolRegistry:
    async def load(self, tools_dir: str) -> None
    async def register(self, yaml_path: Path) -> ToolDefinition
    def get_tool(self, tool_id: str) -> ToolDefinition | None
    async def resolve_dependencies(self, tool_ids: list[str]) -> dict[str, ToolDefinition]
    async def call_tool(
        self,
        tool_id: str,
        inputs: dict,
        location: LocationContext | None = None,
    ) -> dict
    async def list_tools(self) -> list[ToolDefinition]
```

### The tool.yaml contract

```yaml
name: string
id: string                    # unique slug e.g. "web_search"
version: string
description: string

input_schema:
  field_name:
    type: string | number | integer | boolean | object
    description: string
    required: boolean
    default: any | null
    auto_inject: boolean      # true = runtime injects LocationContext automatically

output_schema:
  field_name:
    type: ...
    description: string

mcp_endpoint: string          # path to MCP server file e.g. "tools/weather/server.py"
rate_limit: string | null     # e.g. "100/minute" or null for unlimited
auth: none | api_key | oauth

owner: string
```

### Location auto-injection
When a tool declares `auto_inject: true` on a location field:
1. Runtime checks `LocationContext.confidence > 0.3`
2. If yes: inject `LocationContext` before calling tool
3. If no: call tool without location — tool must handle null gracefully

### call_tool() contract
- Validates inputs against `input_schema` before calling
- Injects location when `auto_inject: true` and confidence > 0.3
- Validates outputs against `output_schema` after calling
- Enforces rate limits
- Raises `ToolNotFoundError`, `ToolInputError`, `ToolRateLimitError`, `ToolExecutionError`

### Acceptance criteria
- [ ] All yaml fields parsed and validated
- [ ] `call_tool()` injects location when `auto_inject: true` and confidence sufficient
- [ ] Tool called without location when confidence < 0.3 — no crash
- [ ] Input schema validated before call, output schema validated after
- [ ] Rate limiting enforced correctly
- [ ] Unit tests: with location, without location, below threshold, schema validation, rate limit
- [ ] Integration test: two skills sharing the same tool


---


## Issue #4 — LLM Adapter Interface + Gemini 1.5 Pro Implementation
**Labels:** `intelligence` `core`
**Depends on:** #1
**Estimated effort:** 4–5 hours

### Summary
Define the abstract `LLMAdapter` interface that makes b2 model-agnostic, then implement the first concrete adapter for Gemini 1.5 Pro via Vertex AI. The adapter is called only by the LLM API Gateway (#5) — never directly by any other component.

### What to build

`app/intelligence/adapters/base.py`

```python
from abc import ABC, abstractmethod

class LLMAdapter(ABC):
    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: list[dict],       # [{role, content}] — current session only
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str: ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def supported_languages(self) -> list[str]: ...
```

`app/intelligence/adapters/gemini.py`

```python
class GeminiAdapter(LLMAdapter):
    model_id = "gemini-1.5-pro"
    # generate() via Vertex AI GenerativeModel
    # embed() via text-embedding-004
    # supported_languages returns full Gemini 1.5 Pro list (100+ languages)
```

### Important design constraint
The adapter has zero knowledge of users, sessions, or locations. It receives a fully-formed system prompt and a message list from the gateway. That is all. All isolation and privacy enforcement happens in the Gateway (#5) before the adapter is ever called.

### Local dev fallback
Must work without Vertex AI credentials. When unavailable, log a warning and return a stub response. Never crash on startup due to missing credentials.

### Acceptance criteria
- [ ] `LLMAdapter` is a clean abstract base — no Gemini-specific code
- [ ] `GeminiAdapter` implements `generate()` and `embed()` via Vertex AI
- [ ] `supported_languages` returns the full documented Gemini 1.5 Pro language list
- [ ] Works with Application Default Credentials — no API key in code
- [ ] Local dev fallback does not crash — returns stub with clear warning log
- [ ] `EchoAdapter` test double demonstrates the interface works
- [ ] Unit tests mock Vertex AI — no real API calls in tests


---


## Issue #5 — LLM API Gateway
**Labels:** `intelligence` `core`
**Depends on:** #1, #4
**Estimated effort:** 5–7 hours

### Summary
Build the LLM API Gateway — the single chokepoint through which every LLM call in the platform flows. It enforces two guarantees: (1) every call originates from b2's identity, never a user's, and (2) every prompt is fully self-contained and isolated from every other conversation.

This is one of b2's most critical privacy components. No other component calls an LLM adapter directly — everything goes through this gateway.

### What to build
`app/intelligence/gateway.py`

```python
@dataclass
class IsolatedPrompt:
    """
    A fully validated, self-contained prompt ready for LLM submission.
    Contains everything the LLM needs. Contains nothing it should not have.
    """
    system_prompt: str
    messages: list[dict]        # current session history only
    model_id: str
    temperature: float
    max_tokens: int
    isolation_declared: bool    # system prompt contains isolation declaration
    no_user_signal: bool        # no session ID, phone, or channel in any field
    no_raw_location: bool       # no coordinates in any field
    history_bounded: bool       # history is current session only, max 10 turns


class LLMGateway:
    def build_prompt(
        self,
        skill: AgentDefinition,
        message: CanonicalMessage,
        tool_results: dict[str, any],
        session_history: list[dict],
    ) -> IsolatedPrompt

    async def call(
        self,
        prompt: IsolatedPrompt,
        adapter: LLMAdapter,
    ) -> str
```

### build_prompt() — the isolation protocol

Every prompt is constructed in this exact order. No exceptions.

```
1. ISOLATION DECLARATION (always first, always present)
   "This is a new, independent conversation. You have no memory of any
   previous conversation. Each request is entirely self-contained. Do not
   reference or infer anything from outside the information in this prompt."

2. SKILL CONTEXT (from agent.yaml — no user info)
   Name, description, tone, response length

3. LANGUAGE CONTEXT (language code only)
   "Respond in: {ISO 639-1 code}. Match the user's language exactly."

4. LOCATION CONTEXT (country/region only — never coordinates)
   "Geographic context: {country or region name}. Adapt where relevant."
   Only included when location.confidence > 0.5 and skill.location_aware = true

5. TOOL RESULTS (injected as context, not as user messages)
   "Relevant information retrieved: {tool output}"

6. SAFETY RULES (from agent.yaml)
   Never advise on X. For complex cases refer to Y.

7. CONVERSATION HISTORY (current session only, max 10 turns, clean [{role, content}])
```

### What is NEVER in any prompt

| Prohibited | Why |
|-----------|-----|
| Session ID (even hashed) | Could correlate requests |
| Phone number (even partial) | Direct PII |
| Channel name (whatsapp, sms, etc.) | Could correlate users by platform |
| Precise coordinates | Location PII |
| Any reference to other users | Cross-contamination |
| Any reference to other conversations | Cross-contamination |

### _validate_prompt() — compliance checker

```python
def _validate_prompt(prompt: IsolatedPrompt) -> None:
    full_text = prompt.system_prompt + str(prompt.messages)

    # Reject coordinate-precision numbers
    if re.search(r'-?\d{1,3}\.\d{4,}', full_text):
        raise PromptIsolationError("Prompt contains coordinate-precision numbers")

    # Reject session ID patterns
    if re.search(r'\b[0-9a-f]{16,}\b', full_text):
        raise PromptIsolationError("Prompt may contain a session identifier")

    # Reject phone number patterns
    if re.search(r'\+\d{7,15}', full_text):
        raise PromptIsolationError("Prompt contains a phone number pattern")

    # Reject channel names
    for channel in ["whatsapp", "telegram", "sms", "messenger", "viber"]:
        if channel in full_text.lower():
            raise PromptIsolationError(f"Prompt contains channel name: {channel}")
```

### call() — validation gate

```python
async def call(self, prompt: IsolatedPrompt, adapter: LLMAdapter) -> str:
    # Final validation — reject non-compliant prompts before submission
    if not all([
        prompt.isolation_declared,
        prompt.no_user_signal,
        prompt.no_raw_location,
        prompt.history_bounded,
    ]):
        raise PromptIsolationError("Prompt failed isolation validation")

    # Submit through adapter — adapter has no user context
    return await adapter.generate(
        system_prompt=prompt.system_prompt,
        messages=prompt.messages,
    )
```

### Acceptance criteria
- [ ] `build_prompt()` always starts with the isolation declaration
- [ ] Country/region included when location confident — coordinates never included
- [ ] History contains only current session turns, max 10
- [ ] `_validate_prompt()` rejects prompts containing: coordinates, phone patterns, session IDs, channel names
- [ ] `call()` raises `PromptIsolationError` on validation failure — never submits bad prompts
- [ ] Two concurrent calls for different users share zero context in system prompt (except skill description)
- [ ] Unit tests: isolation declaration present, coordinate rejection, phone rejection, channel name rejection
- [ ] Property-based test: any two `build_prompt()` calls with different sessions share zero unique tokens


---


## Issue #6 — Location Detector
**Labels:** `intelligence` `core`
**Depends on:** #1
**Estimated effort:** 5–6 hours

### Summary
Detect a `LocationContext` from three sources in priority order: device signal → prompt inference → phone prefix. Runs before intent classification on every message. Output is session-only — never stored.

### What to build
`app/intelligence/location.py`

```python
class LocationDetector:
    async def detect(
        self,
        message: CanonicalMessage,
        phone_number_prefix: str | None = None,
    ) -> LocationContext
```

### Three-source detection chain

**Source 1 — Device signal (confidence: 0.95)**
WhatsApp location message type delivers lat/lng directly. `input_type == "location"` → parse and return immediately with `source: "device"`.

**Source 2 — Prompt inference (confidence: 0.70–0.85)**
Fast Gemini call to extract location from message text. Geocode result to country/region only — discard coordinates immediately.

```python
EXTRACT_PROMPT = """Extract location from this message. JSON only.
No location: {"found": false}
Location found: {"found": true, "place": "Kisumu, Kenya", "country_code": "KE"}
Message: {text}"""
```

Detect signals like: "I'm in Nairobi", "farming in Kisumu", "near Lake Victoria", crop names as regional hints.

**Source 3 — Phone prefix (confidence: 0.40)**
Map country prefix to country code and region. Examples:
```python
PHONE_PREFIX_MAP = {
    "+254": ("KE", "east-africa"),
    "+255": ("TZ", "east-africa"),
    "+256": ("UG", "east-africa"),
    "+221": ("SN", "west-africa"),
    "+234": ("NG", "west-africa"),
    "+251": ("ET", "east-africa"),
    "+62":  ("ID", "southeast-asia"),
    "+91":  ("IN", "south-asia"),
    "+55":  ("BR", "latin-america"),
    "+52":  ("MX", "latin-america"),
    # minimum 20 countries across all target regions
}
```

**No signal (confidence: 0.0)**
Return `LocationContext(source="none", confidence=0.0)`. Router falls back to global skills gracefully.

### Privacy contract
- `lat`/`lng` only from source 1 — never from geocoding
- Geocoding returns country_code and region only — coordinates discarded immediately
- Gemini inference prompt must not include other PII from the message
- Logs may include country_code for routing debug — never coordinates, never city

### Acceptance criteria
- [ ] Device signal parsed from WhatsApp location message type
- [ ] Prompt inference extracts location from messages in English, Swahili, French, Arabic, Hindi
- [ ] Phone prefix map covers 20+ countries across all target regions
- [ ] `source: "none"` returned gracefully when all sources fail
- [ ] Full detection completes under 300ms
- [ ] Coordinates never appear in any log statement
- [ ] Unit tests: each source, all fallback paths, privacy contract for each source


---


## Issue #7 — Intent Classifier + Language Detection
**Labels:** `intelligence` `core`
**Depends on:** #1, #2, #4, #6
**Estimated effort:** 4–5 hours

### Summary
Classify intent, detect language, and pass `LocationContext` from the location detector to `find_best_skill()` for location-weighted routing. The location detector (#6) runs before this — the classifier receives `message.location` already populated.

### What to build
`app/intelligence/classifier.py`

```python
@dataclass
class ClassificationResult:
    agent_id: str | None
    confidence: float
    strategy: Literal["direct", "ambiguous", "no_match"]
    detected_language: str              # ISO 639-1
    location_used: bool
    location_scope_matched: str         # "coordinates" | "country" | "region" | "global" | "none"
    alternatives: list[tuple[str, float]]


class IntentClassifier:
    async def classify(
        self,
        message: CanonicalMessage,      # location already set by detector
        registry: AgentRegistry,
    ) -> ClassificationResult
```

### Language detection strategy
1. **Fast heuristic first** — marker words for Swahili, Hausa, Amharic, Arabic, Hindi, French. Resolves ~60% of cases in under 1ms.
2. **Gemini-native fallback** — for ambiguous cases, the LLM call itself handles it automatically since Gemini responds in the detected language.

### Routing strategies
- **Direct (conf ≥ 0.72):** route to top match immediately
- **Ambiguous (top 2 within 0.10 of each other):** return candidates for clarification
- **No match (all conf < 0.57):** return no_match — router handles graceful fallback

### Acceptance criteria
- [ ] Same farming message routes to `kenya-farming-agent` when location is KE, `farming-global` elsewhere
- [ ] Language detected correctly from short messages in 6+ languages
- [ ] `no_match` never triggered solely due to location — always falls back to global
- [ ] Classifier runs under 200ms for the heuristic path
- [ ] `location_used` and `location_scope_matched` correctly set
- [ ] Unit tests: direct route with location, ambiguous, no-match, language detection for 6+ languages


---


## Issue #8 — Skill Router (Intent × Location × Language)
**Labels:** `intelligence` `core`
**Depends on:** #2, #6, #7
**Estimated effort:** 3–4 hours

### Summary
The skill router takes a `ClassificationResult` and returns the resolved `AgentDefinition` plus the final routing decision, including localized messages for ambiguous and no-match cases.

### What to build
`app/intelligence/router.py`

```python
@dataclass
class RoutingDecision:
    agent: AgentDefinition | None
    confidence: float
    strategy: str                       # "direct" | "ambiguous" | "no_match"
    location_matched: bool
    alternatives: list[AgentDefinition]
    clarification_message: str | None   # localized


class SkillRouter:
    async def route(
        self,
        classification: ClassificationResult,
        registry: AgentRegistry,
        message: CanonicalMessage,
    ) -> RoutingDecision
```

### Localized messages
Clarification and no-match messages must be provided in at minimum: `en`, `sw`, `fr`, `ha`, `ar`, `hi`, `pt`, `id`.

Example no-match in Swahili:
> "Habari, mimi ni b2. Ninaweza kusaidia na kilimo, afya, elimu, na zaidi. Unahitaji nini?"

### Acceptance criteria
- [ ] Returns resolved `AgentDefinition` for direct routes
- [ ] Returns localized clarification message for ambiguous routes
- [ ] Returns localized fallback for no-match routes
- [ ] `location_matched` correctly reflects location's role in decision
- [ ] Unit tests: all three strategies, localized messages in 4+ languages


---


## Issue #9 — WhatsApp Channel Adapter
**Labels:** `channel`
**Depends on:** #1
**Estimated effort:** 6–8 hours

### Summary
Build the full WhatsApp channel adapter — inbound webhook handler, signature verification, message parser, media fetcher, Pub/Sub publisher, response sender. Extracts `LocationContext` from WhatsApp's native location message type.

### Critical constraint
**Must return HTTP 200 within 1 second.** Meta retries if you don't and will eventually disable the webhook. All processing is async via Pub/Sub after the 200 is sent.

### What to build

`app/channels/base.py` — ChannelAdapter interface
```python
class ChannelAdapter(ABC):
    @abstractmethod
    async def parse_inbound(self, request: Request) -> list[dict]: ...
    @abstractmethod
    async def send_response(self, session_id: str, text: str, recipient_id: str) -> bool: ...
    @abstractmethod
    def verify_webhook(self, request: Request) -> bool: ...
```

`app/channels/whatsapp/webhook.py`
- `GET /webhook/whatsapp` — Meta verification (echo hub.challenge if token matches)
- `POST /webhook/whatsapp` — Verify signature → parse → enqueue to Pub/Sub → return 200

`app/channels/whatsapp/sender.py`
- `send_text(session_id, text, phone_number)` — POST to Meta Graph API
- `mark_as_read(message_id)` — fire-and-forget

### Privacy contract
- Hash `wa_id` to `session_id` immediately on receipt using `CanonicalMessage.make_session_id()`
- Raw phone number used only in `sender.py` for the outbound API call — never stored anywhere
- No names from the `contacts` array ever used or stored

### Location extraction (from WhatsApp location message type)
```python
elif msg_type == "location":
    loc = msg.get("location", {})
    location = LocationContext(
        lat=loc.get("latitude"),
        lng=loc.get("longitude"),
        country_code=None,          # geocoded later by location detector
        region=None,
        city=loc.get("name"),       # display name only
        source="device",
        confidence=0.95,
    )
    text_content = f"[Location shared]"
    input_type = "location"
```

### Message types to handle
text, audio (→ flag for STT), image (→ flag for VLM), document, location, video.

### Acceptance criteria
- [ ] `GET /webhook/whatsapp` verifies token and echoes challenge
- [ ] `POST /webhook/whatsapp` returns 200 in under 200ms
- [ ] Signature verification rejects unsigned requests with 401
- [ ] All 6 message types parsed and mapped to correct `input_type`
- [ ] WhatsApp location type populates `LocationContext` with `source: "device"`
- [ ] `wa_id` hashed before any other processing — never in logs
- [ ] `send_text()` delivers response and marks message as read
- [ ] Unit tests: signature verify, each message type, send response
- [ ] Integration test: full round trip with test WhatsApp number


---


## Issue #10 — Skill Agent Runtime
**Labels:** `intelligence`
**Depends on:** #1, #2, #3, #4, #5
**Estimated effort:** 5–6 hours

### Summary
Build the skill agent runtime. Its job is resolving tool dependencies, calling tools, and passing results to the LLM API Gateway. The runtime does NOT build prompts or call LLMs directly — that is the gateway's exclusive responsibility.

### What to build
`app/intelligence/runtime.py`

```python
class AgentRuntime:
    async def respond(
        self,
        message: CanonicalMessage,
        agent: AgentDefinition,
        tool_registry: ToolRegistry,
        gateway: LLMGateway,
        adapter: LLMAdapter,
        session_history: list[dict],
    ) -> CanonicalResponse
```

### Runtime pipeline

```python
async def respond(...) -> CanonicalResponse:

    # 1. Resolve and call declared tools
    tool_results = {}
    for tool_id in agent.tools:
        if tool_id.startswith("rag:"):
            result = await tool_registry.call_tool(
                "rag",
                {"query": message.text_content, "knowledge_base_id": agent.id},
                location=message.location,
            )
            tool_results["knowledge"] = result
        else:
            result = await tool_registry.call_tool(
                tool_id,
                {"query": message.text_content},
                location=message.location,
            )
            tool_results[tool_id] = result

    # 2. Pass everything to gateway — gateway builds the isolated prompt
    response_text = await gateway.call(
        prompt=gateway.build_prompt(
            skill=agent,
            message=message,
            tool_results=tool_results,
            session_history=session_history,
        ),
        adapter=adapter,
    )

    # 3. Format for channel
    response_text = _format_for_channel(response_text, message.channel)

    return CanonicalResponse(
        text=response_text,
        skill_used=agent.id,
        confidence=1.0,
        language=message.language_hint or "en",
        session_id=message.session_id,
        location_used=message.location.confidence > 0.5,
    )
```

### The runtime's contract with the gateway
The runtime passes facts to the gateway. The gateway decides what enters the prompt. The runtime never constructs prompt strings and never calls an LLM adapter directly. This separation is structural — the adapter is not accessible from the runtime except through the gateway.

### Channel formatting
- `whatsapp`: light markdown supported. Max 4096 chars.
- `sms`: strip all markdown. Max 160 chars per segment.
- `voice`: no markdown, short sentences only.
- `telegram`: full markdown supported.

### Acceptance criteria
- [ ] Runtime calls tools and passes results to gateway — does not build prompts
- [ ] LLM adapter not directly accessible from runtime code
- [ ] Location auto-injection works for tools that declare `auto_inject: true`
- [ ] Channel formatting applied correctly for all 4 channels
- [ ] Fallback response returned in detected language on any failure
- [ ] Unit tests: tool resolution, gateway delegation, channel formatting for all 4 channels, fallback


---


## Issue #11 — Core Tool: RAG (Knowledge Retrieval)
**Labels:** `tool`
**Depends on:** #3
**Estimated effort:** 5–6 hours

### Summary
Build the RAG core tool — the most-used tool on the platform. Every skill with a knowledge base uses it. Chunks, embeds, and indexes documents from the skill's `/knowledge` folder at deploy time, then retrieves relevant passages at query time via Vertex AI Vector Search.

### What to build
`tools/rag/tool.yaml` — tool manifest
`tools/rag/server.py` — MCP server implementation
`tools/rag/indexer.py` — called at `b2 deploy` to index knowledge files

### Input / output schema
```yaml
input:
  query: string
  knowledge_base_id: string   # agent ID — used to namespace the index
  top_k: integer (default: 5)

output:
  passages: list[string]
  sources: list[string]       # source filenames
```

### Indexing pipeline (runs on deploy)
1. Read all `.md` and `.txt` files from `agents/{skill}/knowledge/`
2. Chunk into ~500-token segments with 50-token overlap
3. Embed each chunk via `GeminiAdapter.embed()`
4. Store in Vertex AI Vector Search namespaced by `agent_id`

### Retrieval pipeline (runs on every message where RAG is declared)
1. Embed the user's query
2. Query Vertex AI Vector Search for top-k nearest chunks
3. Return passages and source filenames

### Acceptance criteria
- [ ] `tool.yaml` valid and loads via tool registry
- [ ] Indexer processes `.md` and `.txt` files
- [ ] Chunking produces ~500-token segments with overlap
- [ ] Retrieval returns top-k passages in under 500ms
- [ ] Namespaced by agent ID — skill A cannot retrieve skill B's knowledge
- [ ] Unit tests: chunking, embedding (mocked), retrieval (mocked)
- [ ] Integration test: index a real document, retrieve a relevant passage


---


## Issue #12 — Core Tool: Web Search
**Labels:** `tool`
**Depends on:** #3
**Estimated effort:** 3–4 hours

### Summary
Build the web search core tool. Gives skill agents access to current information — market prices, weather news, recent research — that a static knowledge base cannot provide. Used by farming, health, legal, and many future skills.

### What to build
`tools/web_search/tool.yaml`
`tools/web_search/server.py`

### Input / output schema
```yaml
input:
  query: string
  num_results: integer (default: 5, max: 10)
  language: string | null     # ISO 639-1 for localized results

output:
  results:
    - title: string
      snippet: string
      url: string
      published_date: string | null
```

### Implementation
Use [Serper API](https://serper.dev) (free tier: 2,500 queries/month) or [Tavily](https://tavily.com). API key stored in GCP Secret Manager — never hardcoded or in environment variables.

### Acceptance criteria
- [ ] `tool.yaml` valid and loads via tool registry
- [ ] Returns structured results with title, snippet, url
- [ ] Language parameter used when supported by the API
- [ ] API key comes from Secret Manager only
- [ ] Handles API errors gracefully — returns empty results, never crashes
- [ ] Rate limiting prevents runaway costs
- [ ] Unit tests mock the API — no real calls in tests


---


## Issue #13 — Farming Skill Agents (Global + Kenya Reference Implementation)
**Labels:** `skill` `good first issue`
**Depends on:** #2, #3, #8, #10, #11, #12
**Estimated effort:** 5–6 hours

### Summary
Build the farming skill as two agents: a global reference implementation and a Kenya-specific example. Together they demonstrate the full location-aware skill model and serve as the template all future contributors follow.

### Agent 1: Global Farming Agent
`agents/farming-global/agent.yaml`

```yaml
name: Farming Agent
version: "1.0"
description: >
  Crop advice, pest identification, soil health, weather alerts,
  and market prices for smallholder farmers globally.
location_scope:
  type: global
languages: [sw, ha, am, om, en, fr, ar, pt, hi, id]
session:
  timeout_minutes: 10
  max_turns: 8
persona:
  tone: warm, practical, uses local farming examples where possible
  response_length: concise
  location_aware: true
tools:
  - web_search
  - rag:knowledge/
safety:
  never_advise: [pesticide_dosage_medical, financial_loans]
  always_refer: local_agricultural_extension_officer
channels: all
owner: contributor:benevolent-bandwidth-foundation
```

### Agent 2: Kenya Farming Agent
`agents/farming-kenya/agent.yaml`

```yaml
name: Kenya Farming Agent
version: "1.0"
description: >
  Crop advice for Kenyan smallholder farmers. Knows Kenyan crop
  calendars, KEPHIS regulations, county extension services, and
  local market prices.
location_scope:
  type: country
  countries: [KE]
languages: [sw, en]
session:
  timeout_minutes: 10
  max_turns: 8
persona:
  tone: warm, practical, uses Kenyan farming examples
  response_length: concise
  location_aware: true
tools:
  - web_search
  - rag:knowledge/
safety:
  never_advise: [pesticide_dosage_medical, financial_loans]
  always_refer: local_agricultural_extension_officer
channels: all
owner: contributor:benevolent-bandwidth-foundation
```

### Knowledge bases to seed

`agents/farming-global/knowledge/`:
- `crop_diseases.md` — maize, cassava, sorghum, beans common diseases
- `soil_health.md` — soil testing, deficiency symptoms, organic remedies
- `pest_management.md` — stem borer, fall armyworm, integrated pest management
- `market_and_storage.md` — post-harvest storage, cooperative selling
- `planting_calendars.md` — East Africa and West Africa seasonal guides

`agents/farming-kenya/knowledge/`:
- `kenya_crop_calendar.md` — monthly planting/harvest guide by county
- `kenya_extension_services.md` — county extension offices and contacts
- `kenya_market_prices.md` — major market locations and price patterns
- `kenya_regulations.md` — KEPHIS seed certification, fertilizer subsidies

### Test cases (must all pass end-to-end)
1. English, Kenya location: "My maize leaves are turning yellow" → Kenya agent, nitrogen advice
2. Swahili, Kenya location: "Mahindi yangu yanageuka manjano" → Kenya agent, Swahili response
3. English, Brazil location: "My maize has yellowing leaves" → Global agent
4. English, no location: "What crops should I plant?" → Global agent
5. Hausa, no location: "Masarar ta canza launin rawaya" → Global agent, Hausa response
6. Any user: "I need help" → clarification question, no hallucination

### Acceptance criteria
- [ ] Both `agent.yaml` files valid and register successfully
- [ ] Kenya agent selected when location is KE
- [ ] Global agent selected when no location signal
- [ ] All 6 test cases pass end-to-end
- [ ] Responses never recommend specific pesticide doses
- [ ] New contributor can use these as template in under 2 hours


---


## Issue #14 — Pub/Sub Async Message Worker
**Labels:** `infra` `core`
**Depends on:** #1, #7, #8, #9, #10
**Estimated effort:** 4–5 hours

### Summary
Build the Pub/Sub message worker — the async processing loop that pulls messages from `b2-inbound-messages`, runs the full pipeline, and sends responses via the appropriate channel adapter.

### What to build
`app/core/worker.py`

```python
class MessageWorker:
    async def process_message(self, envelope: dict) -> None
    def start(self) -> None     # subscribes to Pub/Sub
    def stop(self) -> None
```

### Pipeline (inside process_message)
1. Deserialize envelope from Pub/Sub
2. Get or create session (in-memory)
3. Build `CanonicalMessage` from envelope
4. Run `LocationDetector.detect()`
5. Run `IntentClassifier.classify()`
6. Run `SkillRouter.route()`
7. If `no_match` or `ambiguous`: send localized message via channel sender
8. If `direct`: resolve tool deps, call `AgentRuntime.respond()`
9. Send response via channel sender
10. Update session (in-memory only)
11. Ack the Pub/Sub message

### Error handling
- Any error → log with session_id only (no message content), nack for retry
- After 3 retries → send generic error in detected language, ack
- Message is always acked or nacked — never left hanging

### Acceptance criteria
- [ ] Worker subscribes to `b2-inbound-messages` on startup
- [ ] Full pipeline executes for a direct-route message
- [ ] `no_match` and `ambiguous` handled with localized responses
- [ ] Pub/Sub message always acked or nacked — no hanging messages
- [ ] Logs contain session_id only — never message content
- [ ] Unit tests mock Pub/Sub, classifier, router, runtime, sender
- [ ] Worker runs alongside FastAPI without blocking the event loop


---


## Issue #15 — Ephemeral Session Manager
**Labels:** `core`
**Depends on:** #1
**Estimated effort:** 3–4 hours

### Summary
Build the session manager — in-memory only, no persistence, session expires when conversation ends or times out. Per-skill timeout and max_turns from `agent.yaml`. When a session expires it is deleted from memory permanently — no recovery path.

### What to build
`app/core/session.py`

```python
@dataclass
class Session:
    session_id: str
    turn_count: int
    prior_skill: str | None
    history: list[dict]             # [{role, content}] — max 10 turns
    language: str
    created_at: datetime
    updated_at: datetime

    def add_turn(self, user_text: str, agent_response: str, skill_id: str) -> None
    def is_expired(self, timeout_minutes: int) -> bool
    def is_at_max_turns(self, max_turns: int) -> bool
    def get_history_for_llm(self, limit: int = 10) -> list[dict]


class SessionManager:
    async def get_or_create(self, session_id: str) -> Session
    async def save(self, session: Session) -> None
    async def delete(self, session_id: str) -> None
    async def cleanup_expired(self) -> int       # returns count deleted
```

### Key design points
- `is_expired()` and `is_at_max_turns()` take the skill's declared values as parameters — there is no global timeout config
- History capped at 10 turns — oldest discarded when exceeded
- `cleanup_expired()` runs periodically — deleted sessions are gone with no archive

### Acceptance criteria
- [ ] Session created on first message, retrieved on subsequent turns
- [ ] `is_expired()` respects per-skill timeout passed as parameter
- [ ] `is_at_max_turns()` respects per-skill max_turns passed as parameter
- [ ] History capped at 10 turns, oldest discarded
- [ ] Expired sessions deleted from memory — not archived anywhere
- [ ] Unit tests: create, retrieve, expire, max turns, history cap, cleanup


---


## Issue #16 — NeMo Guardrails Integration
**Labels:** `intelligence` `core`
**Depends on:** #1, #4
**Estimated effort:** 4–6 hours

### Summary
Integrate [NVIDIA NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) as the platform safety layer. Every message passes through input rails before the LLM is called, every response passes through output rails before delivery. Contributors cannot disable this.

### What to build
`app/core/safety.py`

```python
@dataclass
class GuardrailsResult:
    passed: bool
    modified_content: str | None    # if output was modified
    blocked_reason: str | None      # if blocked
    pii_detected: bool


class GuardrailsLayer:
    async def check_input(self, message: CanonicalMessage) -> GuardrailsResult
    async def check_output(self, response: str, message: CanonicalMessage) -> GuardrailsResult
```

### Rails for MVP
1. **PII input rail** — detect and redact phone numbers, names, addresses before LLM
2. **Content safety rail** — block harmful content
3. **Jailbreak detection rail** — detect and block prompt injection attempts
4. **Output moderation rail** — ensure responses are safe before delivery

### Configuration
Rails configured in `config/guardrails/` using NeMo's Colang format. Platform-level — no per-agent override possible.

### Integration point in runtime
```python
# Before LLM call (in gateway)
input_result = await guardrails.check_input(message)
if not input_result.passed:
    return safe_fallback_in_language(message.language_hint)

# After LLM call (in gateway)
output_result = await guardrails.check_output(llm_response, message)
response_text = output_result.modified_content or llm_response
```

### Acceptance criteria
- [ ] NeMo Guardrails installed and configured
- [ ] PII rail redacts phone numbers from user input
- [ ] Content safety rail blocks harmful content
- [ ] Jailbreak detection blocks prompt injection attempts
- [ ] Output moderation runs on every response
- [ ] `passed=False` returns graceful fallback — never an error message to user
- [ ] Guardrails cannot be disabled by any agent configuration
- [ ] Unit tests: each rail, pass case, block case, PII redaction


---


## Issue #17 — Cloud Run Deploy + GitHub Actions CI/CD
**Labels:** `infra`
**Depends on:** #14
**Estimated effort:** 3–4 hours

### Summary
Production deployment pipeline: Dockerfile, Cloud Run deploy script, GitHub Actions workflow that automatically builds and deploys on every merge to `main`.

### What to build
- `Dockerfile` — production container
- `scripts/deploy.sh` — manual deploy script
- `.github/workflows/deploy.yml` — CI/CD pipeline

### Cloud Run configuration
```
service:          b2-platform
project:          benevolent-bandwidth
region:           us-east1
service-account:  b2-app-service@benevolent-bandwidth.iam.gserviceaccount.com
min-instances:    0
max-instances:    10
memory:           1Gi
cpu:              1
timeout:          60s
concurrency:      80
allow-unauthenticated: true
```

### Secrets injection
Secrets come from Secret Manager via `--set-secrets` flag. Never from environment files or workflow secrets:
```
WHATSAPP_VERIFY_TOKEN=WHATSAPP_VERIFY_TOKEN:latest
WHATSAPP_ACCESS_TOKEN=WHATSAPP_ACCESS_TOKEN:latest
WHATSAPP_PHONE_NUMBER_ID=WHATSAPP_PHONE_NUMBER_ID:latest
```

### GitHub Actions workflow
Trigger: push to `main`
Steps: authenticate via Workload Identity Federation → run tests → build + push image → deploy → smoke test `/health`

### Acceptance criteria
- [ ] `Dockerfile` builds and runs locally
- [ ] `deploy.sh` validates `--project=benevolent-bandwidth` before running
- [ ] GitHub Actions deploys automatically on merge to main
- [ ] All secrets from Secret Manager — never from files or workflow secrets
- [ ] Workload Identity Federation used — no service account key JSON in repo
- [ ] Smoke test hits `/health` after every deploy


---


## Issue #18 — CONTRIBUTING.md + Local Dev Guide
**Labels:** `docs` `good first issue`
**Depends on:** nothing — write in parallel with everything else
**Estimated effort:** 3–4 hours

### Summary
Write the contributor documentation — the document every contributor reads before writing their first line of code. If it is unclear or incomplete, contributors give up. This is one of the most important files in the repo.

### Sections required

1. **Welcome** — who we are, what b2 is, why it matters. One paragraph. Link to README.

2. **How to get involved** — find an issue, claim it, what to expect during review.

3. **Local development setup** — step-by-step for macOS and Linux. Prerequisites, clone, virtualenv, `.env.example` walkthrough, `pip install`, running tests, running the server. Must work for someone who has never used GCP.

4. **How to write a skill agent** — `agent.yaml` field by field, knowledge base setup, local test, deploy. Use farming-global as the reference example.

5. **How to write a tool** — `tool.yaml` and `server.py`, MCP interface, auto_inject location, simple example (echo tool), deploy.

6. **How to run tests** — `pytest tests/unit/` and `pytest tests/integration/`. What mocking is provided. How to write a new test.

7. **Pull request process** — branch naming (`feature/`, `fix/`, `tool/`, `skill/`), PR template, review timeline (72 hours).

8. **Code style** — Black, isort, type hints required, docstrings on public methods, pre-commit setup.

9. **The privacy rules** — the five privacy constraints from the README explained with code examples. What is allowed and what is not. This section is required reading.

10. **Questions** — Discord link, GitHub Discussions, email.

### Tone
Write for a developer who cares about the mission but has never contributed to this project before. Warm but precise. Every step testable. No assumptions.

### Acceptance criteria
- [ ] Local dev setup works end-to-end on a fresh machine — test it yourself
- [ ] Skill agent walkthrough produces a valid `agent.yaml` that loads
- [ ] Tool walkthrough produces a valid `tool.yaml` that loads
- [ ] Privacy section has code examples for each of the five constraints
- [ ] A developer can go from zero to first PR in under 2 hours
- [ ] Reviewed by at least one other contributor before merge


---


## Dependency Map

```
Stream A — assign on day one, all in parallel:
  #1  Canonical envelope       ← unblocks everything
  #4  LLM adapter              ← unblocks gateway and classifier
  #18 CONTRIBUTING.md          ← no dependencies

Stream B — after #1 complete:
  #2  Agent registry
  #3  Tool registry
  #6  Location detector
  #9  WhatsApp adapter
  #15 Session manager

Stream C — after Stream A + B:
  #5  LLM API Gateway          (needs #1, #4)
  #7  Intent classifier        (needs #1, #2, #4, #6)
  #11 Core tool: RAG           (needs #3)
  #12 Core tool: web_search    (needs #3)

Stream D — after Stream C:
  #8  Skill router             (needs #2, #6, #7)
  #10 Agent runtime            (needs #1, #2, #3, #4, #5)
  #16 NeMo Guardrails          (needs #1, #4)

Stream E — after Stream D:
  #13 Farming skill agents     (needs #2, #3, #8, #10, #11, #12)
  #14 Pub/Sub worker           (needs #1, #7, #8, #9, #10)

Final:
  #17 Cloud Run CI/CD          (needs #14)
```
