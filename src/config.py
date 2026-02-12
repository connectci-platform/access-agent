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

    # Database (checkpointing)
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/langgraph"

    # RAG Settings - using access-qa-service for verified Q&A retrieval
    QA_SERVICE_URL: str = "http://localhost:8001"
    RAG_TOP_K: int = 3

    # Query-type-specific similarity thresholds
    RAG_THRESHOLD_STATIC: float = 0.85  # High threshold for static queries (confident answers)
    RAG_THRESHOLD_COMBINED: float = 0.75  # Moderate threshold for combined queries (augment tools)
    RAG_THRESHOLD_FALLBACK: float = 0.65  # Lower threshold for fallback scenarios

    # Legacy compatibility (uses static threshold)
    RAG_SIMILARITY_THRESHOLD: float = 0.85

    # JWT Authentication (ES256 + JWKS)
    # Comma-separated list of "issuer=jwks_url" pairs.
    # Example: "https://support.access-ci.org=https://support.access-ci.org/.well-known/jwks.json"
    TRUSTED_JWKS_URLS: str = ""
    ALLOW_BODY_ACTING_USER: bool = True  # Transition: accept acting_user from body

    # MCP Servers
    MCP_CATALOG_URL: str = "http://localhost:5678/webhook/generate-mcp-catalog"
    MCP_CATALOG_PATH: str | None = None

    # Retry Settings
    MAX_RETRIES_PER_TOOL: int = 2
    MAX_RETRIES_TOTAL: int = 5
    TIMEOUT_BUDGET_MS: int = 120000

    # Quality Loop
    MAX_QUALITY_ATTEMPTS: int = 3

    # Content length limits for LLM processing
    # Modern LLMs have 128K+ context windows, so these can be generous
    MAX_TOOL_RESULT_LENGTH: int = 50000  # Max chars per tool result in evaluate
    MAX_SINGLE_RESULT_LENGTH: int = 20000  # Max chars for a single result before truncating

    # Token budget for synthesis - tool results exceeding this will be condensed first
    # Default 80K leaves room for prompts and response within gpt-4o's 128K limit
    # For smaller models (e.g., Mistral-7B with 32K), set to ~20000
    SYNTHESIS_TOKEN_BUDGET: int = 80000

    # MCP Server base host (configurable, defaults to production IP)
    MCP_SERVER_HOST: str = "localhost"

    # MCP API key for servers that require authentication (write operations)
    MCP_API_KEY: str = ""

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
                "xdmod-charts": "http://mcp-xdmod-charts:3000",
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
                "xdmod-charts": f"http://{host}:3005",
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
            "xdmod-charts": "http://localhost:3005",
            "xdmod-data": "http://localhost:3008",
            "jsm": "http://localhost:3012",
        }

    # Servers that require API key authentication for tool calls
    @property
    def mcp_servers_requiring_api_key(self) -> set[str]:
        """Servers that perform write operations and require API key auth."""
        return {"jsm", "announcements", "events"}

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
