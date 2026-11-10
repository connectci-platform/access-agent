"""MCP server URLs: MCP_BASE_URL sends every server through one proxy, and unset
leaves the per-environment host and port URLs alone."""

from src.config import Settings


def _mk(**overrides):
    # _env_file=None: tests must not inherit the developer's .env.
    return Settings(_env_file=None, **overrides)


def test_base_url_puts_every_server_under_its_name():
    urls = _mk(MCP_BASE_URL="https://mcp.access-ci.org/").mcp_server_urls
    assert urls["compute-resources"] == "https://mcp.access-ci.org/compute-resources"
    assert urls["xdmod-data"] == "https://mcp.access-ci.org/xdmod-data"
    assert all(url.startswith("https://mcp.access-ci.org/") for url in urls.values())


def test_base_url_covers_the_same_servers_as_the_port_map():
    assert set(_mk(MCP_BASE_URL="https://mcp.access-ci.org").mcp_server_urls) == set(
        _mk().mcp_server_urls
    )


def test_unset_base_url_keeps_production_host_and_ports():
    urls = _mk(
        ENVIRONMENT="production",
        MCP_SERVER_HOST="10.0.0.5",
        EVAL_JUDGE_BASE_URL="https://llm.ccs.uky.edu/v1",
        LLM_PROVIDER="vllm",
    ).mcp_server_urls
    assert urls["compute-resources"] == "http://10.0.0.5:3002"


def test_unset_base_url_keeps_local_ports():
    assert _mk().mcp_server_urls["compute-resources"] == "http://localhost:3002"
