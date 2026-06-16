"""Application configuration from environment variables."""

from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment."""

    # Environment
    ENVIRONMENT: Literal["local", "docker", "production"] = "local"
    DEBUG: bool = False

    # OpenTelemetry / Observability (Honeycomb)
    OTEL_ENABLED: bool = True
    OTEL_SERVICE_NAME: str = "access-agent"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = ""  # e.g., https://api.honeycomb.io
    OTEL_EXPORTER_OTLP_HEADERS: str = ""  # e.g., x-honeycomb-team=xxx
    HONEYCOMB_DATASET: str = "access-ci"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # CORS - comma-separated list of allowed origins for production
    ALLOWED_ORIGINS: str = ""

    # LLM Provider (for agent workflow: planning, evaluation, synthesis)
    LLM_PROVIDER: Literal["openai", "vllm", "access_ai"] = "openai"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"

    # vLLM (self-hosted)
    VLLM_BASE_URL: str = "http://localhost:8080/v1"
    VLLM_API_KEY: str = ""
    VLLM_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # ACCESS AI (custom endpoint)
    ACCESS_AI_BASE_URL: str = "https://access-ai-grace1-external.ccs.uky.edu/access/chat/api/"
    ACCESS_AI_API_KEY: str = ""

    # UKY RAG Endpoints (dual RAG: general ACCESS Q&A + XDMoD)
    # Falls back to ACCESS_AI_API_KEY if not set
    UKY_RAG_API_KEY: str = ""
    UKY_RAG_GENERAL_URL: str = "https://access-ai-grace1-external.ccs.uky.edu/access/chat/api/"
    UKY_RAG_XDMOD_URL: str = "https://access-ai-grace1-external.ccs.uky.edu/access/xdmod/chat/api/"
    UKY_RAG_TIMEOUT: float = 60.0
    UKY_RAG_ENABLED: bool = True

    # UKY chat-mcp retrieve-docs endpoint: returns ranked chunks without
    # UKY-side synthesis, so the agent synthesizes from raw excerpts.
    # The sibling /api/ endpoint also returns chunks but additionally
    # spends UKY compute on a synthesized response we discard.
    UKY_CHATMCP_URL: str = (
        "https://access-ai-grace1-external.ccs.uky.edu/access/chat-mcp/api/retrieve-docs"
    )
    UKY_CHATMCP_API_KEY: str = ""

    @property
    def uky_rag_api_key_resolved(self) -> str:
        """Resolve UKY RAG API key, falling back to ACCESS_AI_API_KEY."""
        return self.UKY_RAG_API_KEY or self.ACCESS_AI_API_KEY

    # Resource Provider section cache
    RP_CACHE_TTL_SECONDS: int = 1800  # 30 minutes
    DRUPAL_RESOURCE_GROUPS_URL: str = "https://support.access-ci.org/api/1.0/resource-groups"

    # Database (checkpointing)
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/langgraph"

    # JWT Authentication (ES256 + JWKS)
    # Comma-separated list of "issuer=jwks_url" pairs.
    # Example: "https://support.access-ci.org=https://support.access-ci.org/.well-known/jwks.json"
    TRUSTED_JWKS_URLS: str = ""
    ALLOW_BODY_ACTING_USER: bool = True  # Transition: accept acting_user from body

    # Turnstile bot protection (Cloudflare)
    # When TURNSTILE_SECRET_KEY is empty, Turnstile is disabled (current behavior).
    TURNSTILE_SITE_KEY: str = ""
    TURNSTILE_SECRET_KEY: str = ""
    TURNSTILE_MODE: Literal["immediate", "deferred"] = "deferred"
    TURNSTILE_FREE_QUERIES: int = 3  # Queries before challenge (deferred mode only)
    TURNSTILE_SESSION_TTL: int = 3600  # Seconds a verified session lasts

    @property
    def turnstile_enabled(self) -> bool:
        """Turnstile is active only when a secret key is configured."""
        return bool(self.TURNSTILE_SECRET_KEY)

    # Capability registry
    # ENABLED_CAPABILITIES: comma-separated capability IDs to enable.
    # If empty, all capabilities are candidates. If set, only listed IDs
    # are candidates; everything else is filtered out. DISABLED_CAPABILITIES
    # is then applied on top (deny wins).
    ENABLED_CAPABILITIES: str = ""
    # DISABLED_CAPABILITIES: comma-separated capability IDs to disable.
    # Always wins over ENABLED_CAPABILITIES. Disabled capabilities are
    # removed from UI discovery, system prompt, MCP tool catalog, RAG
    # routing, domain agent routing, and usage attribution.
    DISABLED_CAPABILITIES: str = ""
    # READ_ONLY: when True, force-disables all write-capable capabilities
    # (manage_announcements, open_ticket, report_login_problem, report_security)
    # by adding them to the disabled set at registry build time. Intended for
    # staging environments, the smoke test window, and any deploy where
    # inadvertent writes would be unacceptable. Overrides nothing else.
    READ_ONLY: bool = False

    # MCP Servers. With both unset (the default), the catalog is built by
    # live-aggregating each MCP server's /tools endpoint at startup and cached
    # in memory (see CatalogAggregator). MCP_CATALOG_PATH is an optional escape
    # hatch to load a static catalog file instead; MCP_CATALOG_URL to fetch one
    # from an HTTP endpoint. (A retired n8n webhook previously generated such a
    # file; that path is no longer used.)
    MCP_CATALOG_URL: str = ""
    MCP_CATALOG_PATH: str | None = None

    # max_tokens budget for the tool_calling_loop's react agent. Reasoning
    # models (Qwen3, Kimi, DeepSeek-R1) consume part of the budget on
    # chain-of-thought before emitting user-visible content, so this default
    # is sized larger than what a non-reasoning model strictly needs.
    MAX_TOKENS_LOOP: int = 6000

    # SummarizationMiddleware thresholds.
    #
    # Fire summarization when accumulated message tokens cross
    # SUMMARIZATION_TRIGGER_TOKENS, then preserve only the most recent
    # SUMMARIZATION_KEEP_TOKENS worth of messages verbatim — everything
    # older gets summarized into a single note.
    #
    # KEEP must be a TOKEN budget, not a message count: in tool-heavy agents
    # a single ToolMessage (e.g. list_all_software for Anvil) can be 30k+
    # tokens by itself. With keep=("messages", 20), three such results
    # within the last 20 messages would pin the keep-window at ~70k tokens
    # and post-compaction state could still overflow the model context.
    # With keep=("tokens", N) the window itself is bounded — giant tool
    # results either fit within the budget or get summarized too.
    #
    # Sizing: trigger should be ~2-3x keep so there's room to grow between
    # compactions. Total post-compaction state ≈ SUMMARIZATION_KEEP_TOKENS
    # + ~500 token summary; budget the rest of the model context for
    # system prompt + tool schemas + new-turn growth.
    SUMMARIZATION_TRIGGER_TOKENS: int = 24000
    SUMMARIZATION_KEEP_TOKENS: int = 8000

    # MCP Server base host (configurable, defaults to production IP)
    MCP_SERVER_HOST: str = "localhost"

    # MCP API key for servers that require authentication (write operations)
    MCP_API_KEY: str = ""

    # Reporting (GA4 + Email/Slack delivery)
    GA4_PROPERTY_ID: str = ""
    GA4_CREDENTIALS_FILE: str = ""
    MAILGUN_API_KEY: str = ""
    MAILGUN_DOMAIN: str = ""
    REPORT_EMAIL_FROM: str = "reports@mg.sweetandfizzy.com"
    REPORT_EMAIL_TO: str = ""
    REPORT_SLACK_WEBHOOK_URL: str = ""

    AGENT_VERSION: str = ""  # OTEL service.version; stamped at build (Plan A2)
    DEPLOY_ENV: str = ""  # staging | prod

    # Eval pipeline
    EVAL_JUDGE_BASE_URL: str = ""  # Empty = use default OpenAI. Set for on-premise LLM.
    EVAL_JUDGE_API_KEY: str = ""  # Falls back to OPENAI_API_KEY if empty
    EVAL_JUDGE_MODEL: str = "gpt-4o-mini"

    # Argilla (human review)
    ARGILLA_URL: str = "http://localhost:6900"
    ARGILLA_API_KEY: str = ""
    ARGILLA_EVAL_DATASET: str = "eval-production"

    # MCP Server port mappings
    @property
    def mcp_server_urls(self) -> dict[str, str]:
        """Get MCP server URLs based on environment."""
        if self.ENVIRONMENT == "docker":
            # Docker network uses service names
            return {
                "system-status": "http://mcp-system-status:3000",
                "compute-resources": "http://mcp-compute-resources:3000",
                "software-discovery": "http://mcp-software-discovery:3000",
                "announcements": "http://mcp-announcements:3000",
                "allocations": "http://mcp-allocations:3000",
                "nsf-awards": "http://mcp-nsf-awards:3000",
                "events": "http://mcp-events:3000",
                "affinity-groups": "http://mcp-affinity-groups:3000",
                "xdmod": "http://mcp-xdmod:3000",
                "xdmod-data": "http://mcp-xdmod-data:3000",
                "jsm": "http://mcp-jsm:3000",
            }
        if self.ENVIRONMENT == "production":
            # Production MCP servers at 45.79.215.140
            host = self.MCP_SERVER_HOST
            return {
                "system-status": f"http://{host}:3003",
                "compute-resources": f"http://{host}:3002",
                "software-discovery": f"http://{host}:3004",
                "announcements": f"http://{host}:3009",
                "allocations": f"http://{host}:3006",
                "nsf-awards": f"http://{host}:3007",
                "events": f"http://{host}:3010",
                "affinity-groups": f"http://{host}:3011",
                "xdmod": f"http://{host}:3005",
                "xdmod-data": f"http://{host}:3008",
                "jsm": f"http://{host}:3012",
            }
        # Local development uses localhost with same port mapping
        return {
            "system-status": "http://localhost:3003",
            "compute-resources": "http://localhost:3002",
            "software-discovery": "http://localhost:3004",
            "announcements": "http://localhost:3009",
            "allocations": "http://localhost:3006",
            "nsf-awards": "http://localhost:3007",
            "events": "http://localhost:3010",
            "affinity-groups": "http://localhost:3011",
            "xdmod": "http://localhost:3005",
            "xdmod-data": "http://localhost:3008",
            "jsm": "http://localhost:3012",
        }

    # Servers that require API key authentication for tool calls
    @property
    def mcp_servers_requiring_api_key(self) -> set[str]:
        """Servers that perform write operations and require API key auth."""
        return {"jsm", "announcements", "events"}

    # extra=ignore: .env is shared with docker-compose and may contain vars for
    # other services (e.g., XDMOD_API_TOKEN for mcp-xdmod-data). Rejecting unknown
    # vars would break when the .env has entries not declared in this Settings class.
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
