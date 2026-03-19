"""JWT cookie authentication for ACCESS QA Bot.

Extracts user identity from an ES256-signed JWT cookie (``SESSaccess_auth``)
set by ACCESS sites (Drupal, Django, etc.) on ``.access-ci.org``.

Each issuing site signs with its own EC P-256 private key. This module
validates tokens by fetching the issuer's public key from its JWKS endpoint
(``/.well-known/jwks.json``). No shared secret is needed.

The route handler is responsible for body-based fallback (transition period)
using the already-parsed ``QueryRequest.acting_user`` field — this module
never reads the request body, avoiding double-consumption of the ASGI
body stream.

See: access-qa-planning/08-qa-bot-authentication.md
"""

from __future__ import annotations

import logging
import ssl
from typing import TYPE_CHECKING, Any

import jwt
from jwt import PyJWKClient

if TYPE_CHECKING:
    from fastapi import Request

logger = logging.getLogger(__name__)

# Module-level cache of PyJWKClient instances, keyed by issuer URL.
# Built once at startup via configure_trusted_issuers().
_jwks_clients: dict[str, PyJWKClient] = {}


def configure_trusted_issuers(
    trusted_jwks_urls: dict[str, str],
    *,
    environment: str = "production",
) -> None:
    """Initialize JWKS clients for each trusted issuer.

    Call this once at application startup.

    Args:
        trusted_jwks_urls: Mapping of issuer URL to JWKS endpoint URL.
            Example: ``{"https://support.access-ci.org":
            "https://support.access-ci.org/.well-known/jwks.json"}``
        environment: Current environment. Non-production environments skip
            TLS certificate verification for JWKS endpoints (needed for
            DDEV self-signed certs). JWT signature verification is unaffected.
    """
    _jwks_clients.clear()

    # In local/docker environments, allow self-signed certs for JWKS fetch.
    ssl_context: ssl.SSLContext | None = None
    if environment in ("local", "docker"):
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        logger.warning("JWKS SSL verification disabled (environment=%s)", environment)

    for issuer, jwks_url in trusted_jwks_urls.items():
        _jwks_clients[issuer] = PyJWKClient(
            jwks_url,
            cache_keys=True,
            ssl_context=ssl_context,
        )
        logger.info("Registered trusted issuer: %s -> %s", issuer, jwks_url)

    if not _jwks_clients:
        logger.warning(
            "No trusted JWKS issuers configured. "
            "JWT cookie authentication will not work. "
            "Set TRUSTED_JWKS_URLS in environment."
        )


def get_acting_user_from_cookie(
    request: Request,
) -> tuple[str | None, bool]:
    """Extract the acting user from the ``SESSaccess_auth`` JWT cookie.

    Validates the ES256 signature using the issuer's public key fetched
    from their JWKS endpoint.

    Args:
        request: The incoming FastAPI request.

    Returns:
        A tuple of ``(user, cookie_was_present)``:
        - ``("jsmith@access-ci.org", True)`` — valid cookie
        - ``(None, True)`` — cookie present but invalid/expired
        - ``(None, False)`` — no cookie sent
    """
    token = request.cookies.get("SESSaccess_auth")
    if not token:
        return None, False

    user = _decode_jwt(token)
    return user, True


def _decode_jwt(token: str) -> str | None:
    """Decode and validate an ES256 JWT, returning the ``sub`` claim.

    1. Reads the unverified ``iss`` claim to identify the issuer.
    2. Looks up the JWKS client for that issuer.
    3. Fetches the public key matching the ``kid`` in the JWT header.
    4. Verifies the signature, expiration, and issuer.
    """
    if not _jwks_clients:
        logger.warning("No trusted issuers configured; cannot validate JWT")
        return None

    try:
        # Peek at the unverified payload to get the issuer.
        unverified: dict[str, Any] = jwt.decode(
            token,
            options={"verify_signature": False},
        )
        issuer = unverified.get("iss")
        if not issuer:
            logger.warning("JWT cookie missing 'iss' claim")
            return None

        # Find the JWKS client for this issuer.
        jwks_client = _jwks_clients.get(issuer)
        if jwks_client is None:
            logger.warning("JWT from untrusted issuer: %s", issuer)
            return None

        # Fetch the signing key using the kid from the JWT header.
        signing_key = jwks_client.get_signing_key_from_jwt(token)

        # Verify signature, expiration, and issuer.
        payload: dict[str, Any] = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],
            issuer=list(_jwks_clients.keys()),
        )

        sub = payload.get("sub")
        if sub:
            return str(sub)
        logger.warning("JWT cookie missing 'sub' claim")
        return None

    except jwt.ExpiredSignatureError:
        logger.warning("Expired JWT cookie")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Invalid JWT cookie")
        return None
    except Exception:
        logger.warning("Failed to validate JWT cookie", exc_info=True)
        return None
