# Capability Registry Design

> **Status:** Draft
> **Date:** 2026-03-18
> **Repos affected:** access-agent, access-qa-bot, qa-bot-core, cyberteam_drupal (embedding only)

## Problem

Users have no way to discover what the chatbot can do. Capabilities are expanding (announcements CRUD, tickets, XDMoD, allocations) but the UI shows hardcoded options that don't reflect reality. Internally, there's no structured way to track which capabilities are active when evaluating system performance.

## Goals

1. **User discovery** — users understand what the chatbot can do through UI buttons, welcome context, and conversational hints
2. **Internal observability** — every interaction is logged with the capability it exercised, enabling performance analysis as capabilities shift
3. **Single source of truth** — one structured definition drives the UI, the agent's self-knowledge, and analytics
4. **Security** — never expose internal architecture; filter by auth status server-side; protect user-specific data

## Non-Goals

- OAuth/MCP client capability discovery (separate spec)
- Full user profile system (future, but we design extension points)
- Per-site capability configuration (all sites share the same agent)

---

## Architecture

### Capability Data Model

```python
@dataclass
class Capability:
    id: str              # Opaque identifier: "search_announcements"
    label: str           # User-facing: "Search announcements"
    description: str     # User-facing: "Find ACCESS news and announcements"
    category: str        # Groups into UI sections: "explore"
    requires_auth: bool  # Filter for anonymous vs authenticated
    enabled: bool        # Toggle without redeploy
```

```python
@dataclass
class Category:
    id: str              # "explore"
    label: str           # "Explore resources"
    order: int           # Display order in UI
```

### Where Capabilities Are Defined

**Domain agent capabilities** live in the domain config alongside the system prompt:

```python
ANNOUNCEMENTS_CONFIG = DomainAgentConfig(
    name="announcements",
    capabilities=[
        Capability(
            id="search_announcements",
            label="Search announcements",
            description="Find ACCESS news and announcements",
            category="explore",
            requires_auth=False,
        ),
        Capability(
            id="manage_announcements",
            label="Manage your announcements",
            description="Create, update, and delete announcements you've authored",
            category="content",
            requires_auth=True,
        ),
    ],
    ...
)
```

**Domain agent capabilities — JSM:**

```python
JSM_CONFIG = DomainAgentConfig(
    name="jsm",
    capabilities=[
        Capability(
            id="open_ticket",
            label="Open a help ticket",
            description="Create a support ticket for technical issues",
            category="support",
            requires_auth=False,
        ),
        Capability(
            id="report_login_problem",
            label="Report a login problem",
            description="Get help with ACCESS or resource login issues",
            category="support",
            requires_auth=False,
        ),
        Capability(
            id="report_security",
            label="Report a security issue",
            description="Report a security concern to the ACCESS team",
            category="support",
            requires_auth=False,
        ),
    ],
    ...
)
```

**General pipeline capabilities** (non-domain tools) are defined in a separate registry file since they're used by the general RAG+tools pipeline, not a specific domain agent:

```python
GENERAL_CAPABILITIES = [
    Capability("ask_question", "Ask a question", "Get answers about ACCESS resources, policies, and services", "general", True),
    Capability("check_allocations", "Check allocations", "Look up allocation details and status", "explore", True),
    Capability("search_software", "Search software", "Find software available on ACCESS resources", "explore", True),
    Capability("check_system_status", "Check system status", "See current outages and resource status", "explore", True),
    Capability("browse_events", "Browse events", "Find upcoming trainings, workshops, and office hours", "explore", True),
    Capability("browse_affinity_groups", "Browse affinity groups", "Explore community affinity groups", "explore", True),
    Capability("check_usage", "Check usage (XDMoD)", "View resource usage and performance data", "analytics", True),
    Capability("search_nsf_awards", "Search NSF awards", "Look up NSF award information", "explore", True),
]
```

> **Note:** Most capabilities require auth because the primary RAG pipeline (UKY) requires authenticated access. If anonymous RAG access becomes available in the future, these flags can be flipped without code changes. Only the support capabilities (tickets, security reports) are available to anonymous users.

All capabilities are aggregated into a single `CapabilityRegistry` class at startup, providing one place to query the full list. Domain configs and `GENERAL_CAPABILITIES` feed into it; consumers only interact with the registry.

**Required code changes:**
- `DomainAgentConfig` dataclass (`src/agent/domains/config.py`) gains an optional `capabilities: list[Capability] = field(default_factory=list)` field
- All existing domain configs (announcements, jsm) must be updated to include their capabilities
- New `CapabilityRegistry` class with methods: `get_capabilities(authenticated: bool)`, `get_by_id(id: str)`, `get_system_prompt_section(authenticated: bool)`, `get_capability_for_query(domain: str | None, tools_used: list[str]) -> str`

**Categories:**

```python
CATEGORIES = [
    Category("general", "Ask a question", 0),
    Category("support", "Get help", 1),
    Category("content", "Manage content", 2),
    Category("explore", "Explore resources", 3),
    Category("analytics", "Check usage", 4),
]
```

### Enable/Disable

The `enabled` flag on each capability defaults to `True`. To disable a capability without redeploying, set the environment variable `DISABLED_CAPABILITIES` to a comma-separated list of capability IDs:

```
DISABLED_CAPABILITIES=manage_announcements,check_usage
```

The agent reads this at startup and sets `enabled=False` on matching capabilities. Disabled capabilities are omitted from all API responses and the agent's self-knowledge.

---

## API Endpoints

### `GET /api/v1/capabilities`

**Purpose:** Fast, generic capability list for initial UI load.

**Authentication:** Optional. If a valid JWT cookie is present, includes `requires_auth=True` capabilities. Otherwise, only public capabilities.

**Response:**

```json
{
  "categories": [
    {
      "id": "general",
      "label": "Ask a question",
      "order": 0,
      "capabilities": [
        {
          "id": "ask_question",
          "label": "Ask a question",
          "description": "Get answers about ACCESS resources, policies, and services"
        }
      ]
    },
    {
      "id": "support",
      "label": "Get help",
      "order": 1,
      "capabilities": [
        {
          "id": "open_ticket",
          "label": "Open a help ticket",
          "description": "Create a support ticket for technical issues"
        },
        {
          "id": "report_security",
          "label": "Report a security issue",
          "description": "Report a security concern to the ACCESS team"
        }
      ]
    }
  ],
  "is_authenticated": false
}
```

**For anonymous users**, auth-required capabilities are included but marked as locked, along with a login URL. This encourages authentication by showing what's available:

```json
{
  "categories": [
    {
      "id": "content",
      "label": "Manage content",
      "order": 2,
      "capabilities": [
        {
          "id": "manage_announcements",
          "label": "Manage your announcements",
          "description": "Create, update, and delete announcements you've authored",
          "requires_auth": true,
          "locked": true
        }
      ]
    }
  ],
  "is_authenticated": false,
  "login_url": "/login?redirect=..."
}
```

The `locked` field is `true` for auth-required capabilities when the user is anonymous, and absent (or `false`) when authenticated. The `login_url` is included only for anonymous users. The UI can render locked capabilities with a lock icon and link to login.

**Security:**
- No internal details (tool names, MCP servers, system architecture)
- Only enabled capabilities included
- Capability IDs are opaque (no implementation leakage)
- Auth-required capabilities are visible to anonymous users (to encourage login) but not actionable

**Performance:** No external calls. Built from in-memory registry. Should respond in <10ms.

### `GET /api/v1/capabilities/personalized`

**Purpose:** User-specific context for enhanced UI and agent behavior. Called lazily after page load.

**Authentication:** Required. Returns 401 for anonymous users.

**Response:**

```json
{
  "user": {
    "access_id": "apasquale@access-ci.org"
  },
  "highlighted_capabilities": [
    {
      "id": "manage_announcements",
      "label": "Manage Pegasus announcements",
      "reason": "You coordinate the Pegasus affinity group"
    }
  ],
  "context": {
    "is_coordinator": true,
    "coordinated_groups": [
      {"name": "Pegasus", "id": "pegasus"}
    ],
    "active_allocations": [
      {"resource": "Delta", "project": "TG-CIS123456"}
    ]
  }
}
```

**What populates the context (today):**
- `is_coordinator` + `coordinated_groups`: From the `mcp_my_affinity_groups` Drupal view via jsonapi_views
- `active_allocations`: From the user's `field_cider_resources` entity reference field in Drupal, fetched via JSON:API on the user entity

**Extension points (future):**
- `conversation_summary`: Summary of prior conversations
- `researcher_profile`: Interests, skills from Drupal profile
- `recent_tickets`: Open tickets from JSM

**Security:**
- Requires valid JWT cookie authentication
- Response is scoped to the authenticated user only
- User-specific data (allocations, groups) fetched server-side, never from client input
- Rate limited: results cached in Redis per-user-hash with 5-minute TTL

**Performance:** Makes external calls (MCP servers, Drupal). Target <2s response time. Called lazily, never blocks initial UI load. External calls are made in parallel where possible.

**Error handling:** Each context source (coordinator status, allocations) is fetched independently. If one fails, the others still populate. The response includes whatever succeeded — a partial response is better than no response. If all calls fail, returns `{"highlighted_capabilities": [], "context": {}}` with a 200 (not a 500), so the UI degrades gracefully.

**Caching:** Results are cached in-memory (`cachetools.TTLCache`) keyed by SHA-256 hash of the user's ACCESS ID, with a 5-minute TTL. Cache is checked before making external calls.

---

## UI Integration

### Repo: access-qa-bot (feature branch)

The hardcoded options in `main-menu-flow.ts` are replaced by dynamic options fetched from the capabilities endpoint.

**On load:**
1. `AccessQABot` calls `GET /api/v1/capabilities`
2. Categories are rendered as option buttons (replacing the hardcoded 4 options)
3. "Ask a question" category may be omitted as a button since typing is the natural default — TBD based on testing

**After load (lazy):**
1. If user is authenticated, call `GET /api/v1/capabilities/personalized`
2. If `highlighted_capabilities` are returned, show them as additional contextual suggestions (e.g., a subtle "You can also: Manage Pegasus announcements" hint)

**Discovery button:**
A final button — "Show my options" — is always shown after the category buttons, visually distinct (outlined instead of filled). When clicked, it sends as a message and the agent responds with a personalized capability summary. For authenticated users with context, the response highlights relevant capabilities ("I see you coordinate Pegasus and have allocations on Delta..."). For anonymous users, it lists all capabilities with a login nudge for locked ones. This is the main discovery mechanism — category buttons are for users who already know what they want, this button is for exploration.

**When a category button is clicked:**
- The button text is sent as the user's message (same as current behavior)
- The agent classifies and routes normally
- Example: clicking "Get help" sends "Get help" which the agent interprets as a help/ticket request

### Repo: qa-bot-core

Minimal changes:
- `QABot` accepts an optional `capabilitiesEndpoint` prop
- The flow system supports dynamically loaded options (may already work since options are evaluated as functions in react-chatbotify)

### Repo: cyberteam_drupal

No code changes — the embedding already passes `qaEndpoint` which the chatbot uses to derive the capabilities URL.

---

## Agent Self-Knowledge

The agent's system prompt includes a condensed capability summary so it can:

1. **Answer "what can you do?"** — returns a well-organized list by category
2. **Give contextual hints** — "I can also help you create an announcement" after showing announcements
3. **Avoid promising things it can't do** — the capability list is the source of truth

The capability summary is injected into the base system prompt at startup, not the domain prompts. Format:

```
## What You Can Do
You have the following capabilities. When relevant, mention these to help users discover features.

**Explore resources:** Search announcements, Check allocations, Search software, Check system status, Browse events, Browse affinity groups, Search NSF awards
**Get help:** Open a help ticket, Report a security issue, Report a login problem
**Manage content (requires login):** Manage your announcements
**Check usage (requires login):** Check usage (XDMoD)
```

For authenticated users, the personalized context is appended **only if already cached** (from a prior `/api/v1/capabilities/personalized` call or a previous query in the same session). The agent never blocks on a personalized context fetch during query processing. If personalized context is not yet available, the agent proceeds with base capabilities only — personalized hints will appear in subsequent turns once the context is populated.

```
## About This User
- Coordinates affinity groups: Pegasus
- Has active allocations on: Delta (TG-CIS123456)
```

---

## Logging & Analytics

### Capability-to-Query Mapping

The classify node (`src/agent/nodes/classify.py`) already determines `query_type` and `domain`. We extend `QueryClassification` in `src/agent/state.py` with a `capability_id` field:

```python
class QueryClassification(TypedDict):
    query_type: str          # "static", "dynamic", "combined"
    domain: str | None       # "announcements", "jsm", None
    capability_id: str | None  # "manage_announcements", "open_ticket", etc.
    ...
```

**Mapping rules:**
- Domain agent queries: The classify node sets `capability_id` based on `domain` + intent. For example, `domain="announcements"` with a write intent maps to `manage_announcements`; a read intent maps to `search_announcements`.
- General pipeline queries: Default to `ask_question`. If tools are used, the primary tool determines the capability (e.g., allocations tools → `check_allocations`). A mapping from MCP server names to capability IDs is maintained in the registry (internal only, never exposed).
- Combined queries: Use the primary capability (the one the user's intent most closely matches). Secondary capabilities are logged separately as `secondary_capability_ids`.

**Edge cases:**
- If classification is ambiguous, `capability_id` is set to the most specific match. `ask_question` is the fallback.
- A single query exercising multiple capabilities (rare) logs the primary and lists secondaries.

### Query-Level Logging

The existing `usage_logger` already logs `tools_used` per query. We add:

- `capability_id`: The primary capability exercised (e.g., `manage_announcements`)
- `secondary_capability_ids`: Optional list of additional capabilities exercised
- `category`: The category (e.g., `content`)
- `was_authenticated`: Whether the user was logged in
- `was_personalized`: Whether personalized context was available

### Database Migration

The `usage_logs` table requires new columns. Since the project uses `Base.metadata.create_all()` (which only creates tables, not columns), we need an explicit migration:

```sql
ALTER TABLE usage_logs ADD COLUMN IF NOT EXISTS capability_id VARCHAR(64);
ALTER TABLE usage_logs ADD COLUMN IF NOT EXISTS secondary_capability_ids TEXT;  -- JSON array
ALTER TABLE usage_logs ADD COLUMN IF NOT EXISTS category VARCHAR(32);
ALTER TABLE usage_logs ADD COLUMN IF NOT EXISTS was_authenticated BOOLEAN DEFAULT FALSE;
ALTER TABLE usage_logs ADD COLUMN IF NOT EXISTS was_personalized BOOLEAN DEFAULT FALSE;
ALTER TABLE usage_logs ADD COLUMN IF NOT EXISTS rating VARCHAR(16);             -- "helpful" or "not_helpful"
ALTER TABLE usage_logs ADD COLUMN IF NOT EXISTS rating_feedback TEXT;           -- optional free-text feedback
```

This runs as a startup migration check — if the columns don't exist, add them. Existing rows get NULL for the new columns, which is acceptable for historical data.

### Capability-Level Metrics

With `capability_id` in every log entry, reporting can answer:
- Which capabilities are most/least used?
- What's the success rate per capability?
- How do authenticated vs. anonymous usage patterns differ?
- When capabilities are added/removed, how does usage shift?

---

## Response Metadata & Ratings

### The Problem

Rating buttons currently appear after every agent response and always route to UKY RAG. This is wrong for several reasons:
- Ratings appear after clarifying questions and mid-conversation turns where they're meaningless
- Non-RAG responses (ticket creation, announcements, tool results) have no appropriate rating destination
- A single rating system can't evaluate both "was this answer accurate?" (RAG) and "did this workflow complete successfully?" (domain agent)

### Response Metadata

The agent's `/api/v1/query` response includes metadata that tells the UI how to handle the response:

```json
{
  "answer": "...",
  "metadata": {
    "capability_id": "ask_question",
    "is_final_response": true,
    "rating_target": "uky_rag",
    "query_id": "uuid-here"
  }
}
```

**`is_final_response`** — whether this response completes the user's request. The UI shows rating buttons only when this is `true`.

True when:
- A RAG answer was synthesized and delivered
- A domain agent completed its task (announcement created, ticket opened)
- A tool query returned results and the agent summarized them

False when:
- The agent is asking a clarifying question
- It's presenting options or a preview for confirmation
- It's mid-workflow (e.g., gathering fields for announcement creation)

**`rating_target`** — where the rating should be routed:
- `"uky_rag"` — UKY RAG pipeline responses (existing rating endpoint)
- `"agent"` — domain agent and general tool responses (stored in our usage_logs)
- `null` — no rating appropriate (edge case)

**`capability_id`** — which capability was exercised. Stored with the rating for analytics.

### How the Agent Determines Metadata

The `is_final_response` flag is set at two points:
- **General pipeline:** The synthesis node (`src/agent/nodes/synthesize.py`) is the terminal node — any response from synthesis is final.
- **Domain agents:** The domain agent executor (`src/agent/nodes/domain_agent.py`) returns the final answer after its react loop completes. This response is also final. Intermediate turns within the domain agent loop (clarifying questions, previews) are streamed but not flagged as final.

Both paths set `is_final_response=True` on the response that gets returned to the user via `/api/v1/query`.

The `rating_target` is determined by the query path:
- If UKY RAG was the primary answer source → `"uky_rag"`
- If the query was handled by a domain agent or general tools without RAG → `"agent"`
- For combined queries (RAG + tools), use `"uky_rag"` since RAG quality is the primary concern

### UI Rating Flow

1. Agent response arrives with metadata
2. If `is_final_response` is `false`, no rating buttons shown
3. If `is_final_response` is `true`, show thumbs up/down buttons
4. User clicks a rating → UI sends rating to the appropriate endpoint based on `rating_target`:
   - `"uky_rag"` → existing UKY rating endpoint (preserves current behavior)
   - `"agent"` → `POST /api/v1/rating` on the access-agent (new endpoint)
5. Rating is stored with `query_id`, `capability_id`, `rating`, and optional `feedback` text

### Agent Rating Storage

New endpoint `POST /api/v1/rating`:

```json
{
  "query_id": "uuid",
  "rating": "helpful" | "not_helpful",
  "feedback": "optional text"
}
```

Ratings are stored in the existing `usage_logs` table (new columns: `rating`, `rating_feedback`). The `query_id` is the existing `question_id` already generated client-side and sent with each query — no new ID needed. The rating endpoint validates that the `query_id` exists in `usage_logs` before accepting the rating. Anonymous ratings are accepted (since anonymous users can use support capabilities) but require a valid `query_id` to prevent spoofing.

---

## Security Considerations

1. **Server-side auth enforcement** — auth-required capabilities are visible to anonymous users (to encourage login) but marked as `locked`. The actual enforcement happens at the `/api/v1/query` endpoint — locked capabilities cannot be exercised without a valid JWT cookie.

2. **No internal details exposed** — capability IDs are opaque labels. No tool names, MCP server names, endpoint URLs, or system architecture in any response.

3. **Existing internal endpoints must be restricted** — the current `GET /api/v1/tools` and `GET /api/v1/catalog` endpoints return MCP server names and tool details. These must be restricted to admin access (require an API key header) or removed. They are not needed by the chatbot UI and should never be publicly accessible.

4. **Personalized endpoint requires authentication** — returns 401 for anonymous. The JWT cookie is validated the same way as `/api/v1/query`.

5. **User context is fetched server-side** — the personalized endpoint calls MCP servers and Drupal internally. The client never sends user context; it only receives it.

6. **Rate limiting and caching** — the personalized endpoint caches results in-memory using a TTL cache (e.g., `cachetools.TTLCache`), keyed by user hash with a 5-minute TTL. This prevents abuse and reduces load on MCP servers/Drupal. Sufficient for single-instance deployment; can be migrated to Redis if scaling requires shared cache.

7. **Graceful degradation** — if the personalized endpoint's external calls fail (MCP server down, Drupal unreachable), return a partial response with whatever succeeded. Never block the UI. The `highlighted_capabilities` array is empty if context is unavailable.

8. **Disabled capabilities are invisible** — they don't appear in any response, not even as `"enabled": false`. Their existence is not disclosed.

9. **CORS** — same policy as existing `/api/v1/query` endpoint.

---

## Branch Strategy

This work happens on feature branches:
- `access-agent`: `feature/capability-registry`
- `access-qa-bot`: `feature/dynamic-capabilities`
- `qa-bot-core`: `feature/capabilities-endpoint-prop` (if needed)

The `access-qa-bot` main branch remains available for the upcoming UKY RAG endpoint parameters feature, which can ship to production independently.

---

## Resolved Design Decisions

1. **"Ask a question" does not need a button** — typing is the natural default. The text input is enabled from the start (changing current behavior where typing is disabled until a button is clicked).
2. **Category buttons send the label as a message** — the agent responds conversationally with sub-capabilities rather than the UI expanding to show nested buttons. Simpler, more flexible.
3. **Welcome message includes the AI disclaimer** — combined into one message shown on load, rather than a separate transition step.
4. **Most capabilities require auth** — the UKY RAG pipeline requires login, so all RAG-dependent capabilities are `requires_auth=True`. Only support capabilities (tickets, security reports) are available anonymously. The `requires_auth` flag can be flipped if anonymous RAG access becomes available.
5. **Ratings are contextual** — shown only on final responses, routed to the appropriate backend based on the response source (UKY RAG vs. agent).

## Additional Resolved Decisions

6. **Personalization is agent context, not UI clutter** — The personalized endpoint feeds the agent's system prompt, not the button area. The agent uses it to give smarter responses (e.g., knowing which resources the user has, which groups they coordinate) without surfacing it all in the UI. This scales naturally as personalization grows.
7. **Subtle discovery hints via placeholder text** — The input placeholder rotates through contextual suggestions drawn from the personalized context (e.g., "Try: 'check my usage on Delta'", "Try: 'show my announcements'"). This is non-intrusive and encourages exploration.
8. **Personalized welcome greeting** — The agent greets the user by name or username to signal it knows who they are. Depends on UKY PII policy (see open questions). Fallback: greet without name but use personalized placeholder text.
9. **First-visit vs. returning** — First visit gets a slightly more descriptive welcome ("I can help you with ACCESS questions, manage your announcements, check your resource usage, and more."). Return visits get a shorter greeting.
10. **Ratings include optional free-text feedback** — thumbs up/down plus an optional text field for details.
11. **Anonymous users see the chatbot** — they can see all capabilities (locked ones with login prompt) and use the support capabilities (tickets, security reports).

## Open Questions

1. **UKY PII policy** — Can the user's name be included in the system prompt sent to the LLM? Need to check with UKY. Fallback: use ACCESS ID username portion or no name at all.
2. **Placeholder text mechanics** — How does the chatbot UI rotate placeholder text? Need to check react-chatbotify capabilities. May require a qa-bot-core change.
