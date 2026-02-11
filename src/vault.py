"""HashiCorp Vault client for secret management.

Provides the JWT signing secret used to validate ``SESSaccess_auth`` cookies.
Falls back to the ``JWT_SECRET`` environment variable when Vault is
unavailable or not configured.
"""

from __future__ import annotations

import logging
import time

import hvac

logger = logging.getLogger(__name__)

# In-memory cache to avoid hitting Vault on every request.
# Typed as (secret_value, fetched_at_timestamp).
_cache: tuple[str, float] | None = None
CACHE_TTL = 300  # seconds (5 minutes)


class VaultClient:
    """Thin wrapper around the ``hvac`` Vault client."""

    def __init__(self, addr: str, token: str) -> None:
        self.client = hvac.Client(url=addr, token=token)

    def get_jwt_secret(self, path: str = "access/jwt") -> str | None:
        """Read the JWT signing key from Vault KV v2.

        Args:
            path: KV v2 path (relative to the ``secret/`` mount).

        Returns:
            The signing key string, or ``None`` if Vault is unavailable.
        """
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            value = response["data"]["data"].get("signing_key")
            return str(value) if value is not None else None
        except Exception:
            logger.warning("Failed to read JWT secret from Vault", exc_info=True)
            return None


def get_jwt_secret(
    vault_addr: str,
    vault_token: str,
    vault_secret_path: str,
    env_fallback: str,
) -> str:
    """Return the JWT signing secret, with caching and env-var fallback.

    Resolution order:
    1. In-memory cache (if still within TTL)
    2. Vault KV v2
    3. ``JWT_SECRET`` environment variable

    Args:
        vault_addr: Vault server address.
        vault_token: Vault authentication token.
        vault_secret_path: KV v2 path for the JWT secret.
        env_fallback: Value of the ``JWT_SECRET`` env var (fallback).

    Returns:
        The signing secret string.

    Raises:
        RuntimeError: If no secret is available from any source.
    """
    global _cache
    now = time.time()

    # 1. Check cache
    if _cache is not None:
        cached_secret, fetched_at = _cache
        if (now - fetched_at) < CACHE_TTL:
            return cached_secret

    # 2. Try Vault (only if configured)
    if vault_addr and vault_token:
        client = VaultClient(vault_addr, vault_token)
        secret = client.get_jwt_secret(path=vault_secret_path)
        if secret:
            _cache = (secret, now)
            return secret
        # Vault failed — fall through to env var, but keep stale cache if available
        if _cache is not None:
            logger.warning("Vault unavailable; using cached JWT secret")
            return _cache[0]

    # 3. Env-var fallback
    if env_fallback:
        return env_fallback

    msg = "No JWT signing secret available. " "Set JWT_SECRET env var or configure Vault."
    raise RuntimeError(msg)
