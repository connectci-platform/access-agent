"""JWT cookie authentication for ACCESS QA Bot.

Extracts user identity from an ES256-signed JWT cookie (``SESSaccess_auth``)
set by ACCESS sites (Drupal, Django, etc.) on ``.access-ci.org``.

Each issuing site signs with its own EC P-256 private key. This module
validates tokens by fetching the issuer's public key from its JWKS endpoint
(``/.well-known/jwks.json``). No shared secret is needed.

Identity is cookie-only: there is no body-supplied identity fallback. This
module never reads the request body, avoiding double-consumption of the
ASGI body stream.

See: access-qa-planning/08-qa-bot-authentication.md
"""

from __future__ import annotations

import logging
import ssl
import time
from typing import TYPE_CHECKING, Any

import jwt
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientConnectionError, PyJWKClientError

if TYPE_CHECKING:
    from fastapi import Request

logger = logging.getLogger(__name__)

# Module-level cache of PyJWKClient instances, keyed by issuer URL.
# Built once at startup via configure_trusted_issuers().
_jwks_clients: dict[str, PyJWKClient] = {}

# A JWT's `kid` header is attacker-controlled, and PyJWKClient.get_signing_key
# forces a live outbound JWKS fetch on ANY cache miss (not just real
# rotation) before raising. Without a limit, an anonymous caller varying
# `kid` per request forces one outbound fetch per request — amplification
# against the issuer's JWKS endpoint plus exhaustion of our own outbound/
# timeout budget. This bounds forced refreshes to at most one per issuer per
# cooldown window; see `_get_signing_key` below. Reuses PyJWKClient's own
# cache lifespan (`lifespan=300` default) as the cooldown so a suppressed
# refresh never outlives the cache entry it would have replaced.
JWKS_REFRESH_COOLDOWN_S = 300

# Per-issuer timestamp of the last forced (refresh=True) JWKS fetch. Reset
# alongside _jwks_clients in configure_trusted_issuers(). Single-process —
# same documented pattern as _jwks_clients itself.
_last_forced_refresh: dict[str, float] = {}


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
    _last_forced_refresh.clear()

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


def _get_signing_key(jwks_client: PyJWKClient, issuer: str, token: str) -> jwt.PyJWK:
    """Resolve the signing key for `token`, rate-limiting forced refreshes.

    Mirrors ``PyJWKClient.get_signing_key``'s match-then-refresh-then-match
    shape, but only allows the refresh (the live outbound HTTPS fetch) once
    per issuer per ``JWKS_REFRESH_COOLDOWN_S``. The `kid` driving the lookup
    is an attacker-controlled JWT header field, so an unknown kid must be
    cheap when we've already refreshed this issuer recently — otherwise a
    caller varying `kid` per request forces one outbound fetch per request
    (see module docstring on `JWKS_REFRESH_COOLDOWN_S`).

    Legitimate key rotation is unaffected: the first unknown kid after the
    cooldown elapses still forces exactly one refresh and picks up new keys.

    Raises ``PyJWKClientError`` (from the final ``match_kid`` miss, or
    propagated from the underlying fetch) on anything that isn't a resolved
    key — identical in shape to what ``get_signing_key`` would raise, so the
    caller's existing exception handling is unchanged.
    """
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")
    if not kid:
        # No kid at all can never match anything in the set; same terminal
        # outcome as an unmatched kid, without spending a lookup on it.
        raise PyJWKClientError("Unable to find a signing key: JWT header missing 'kid'")

    keys = jwks_client.get_signing_keys()  # cached — never fetches here
    key = jwks_client.match_kid(keys, kid)
    if key is not None:
        return key

    now = time.time()
    last = _last_forced_refresh.get(issuer, 0.0)
    if now - last < JWKS_REFRESH_COOLDOWN_S:
        # Attacker-controlled input (kid); DEBUG only — must not become a
        # log-spam or alerting oracle for an anonymous caller.
        logger.debug(
            "AUTH_INFRA: suppressing forced JWKS refresh for issuer %r "
            "(kid miss within %ss cooldown)",
            issuer,
            JWKS_REFRESH_COOLDOWN_S,
        )
        raise PyJWKClientError(f'Unable to find a signing key that matches: "{kid}"')

    keys = jwks_client.get_signing_keys(refresh=True)
    _last_forced_refresh[issuer] = now
    key = jwks_client.match_kid(keys, kid)
    if key is not None:
        return key
    raise PyJWKClientError(f'Unable to find a signing key that matches: "{kid}"')


def _decode_jwt(token: str) -> str | None:
    """Decode and validate an ES256 JWT, returning the ``sub`` claim.

    1. Reads the unverified ``iss`` claim to identify the issuer.
    2. Looks up the JWKS client for that issuer.
    3. Fetches the public key matching the ``kid`` in the JWT header.
    4. Verifies the signature, expiration, and issuer.
    """
    if not _jwks_clients:
        # Infrastructure, not user state: every logged-in user silently
        # degrades to anonymous until TRUSTED_JWKS_URLS is set. Logged at
        # ERROR so a misconfigured deploy is visible rather than looking
        # like a site full of logged-out users.
        logger.error(
            "AUTH_INFRA: no trusted issuers configured (TRUSTED_JWKS_URLS empty); "
            "all authenticated users are being treated as anonymous"
        )
        return None

    try:
        # Peek at the unverified payload to get the issuer.
        unverified: dict[str, Any] = jwt.decode(
            token,
            options={"verify_signature": False},
        )
        issuer = unverified.get("iss")
        if not issuer:
            logger.warning("AUTH_USER: JWT cookie missing 'iss' claim; treating as anonymous")
            return None

        # Find the JWKS client for this issuer.
        jwks_client = _jwks_clients.get(issuer)
        if jwks_client is None:
            # Ambiguous by nature: either a token from somewhere we don't
            # trust, or a deploy whose TRUSTED_JWKS_URLS is missing a real
            # issuer. The issuer is logged so the two can be told apart.
            # %r, not %s: the issuer is attacker-controlled, and an unquoted
            # newline in it would forge log lines — including fake AUTH_INFRA
            # records, defeating this very classification.
            logger.warning(
                "AUTH_USER: JWT from untrusted issuer %r; treating as anonymous "
                "(if this issuer is legitimate, it is missing from TRUSTED_JWKS_URLS)",
                issuer,
            )
            return None

        # Fetch the signing key using the kid from the JWT header. Rate-limits
        # the forced-refresh-on-miss (see _get_signing_key) instead of calling
        # jwks_client.get_signing_key_from_jwt directly, which would force a
        # live outbound fetch on every unknown kid.
        signing_key = _get_signing_key(jwks_client, issuer, token)

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
        logger.warning("AUTH_USER: JWT cookie missing 'sub' claim; treating as anonymous")
        return None

    except jwt.ExpiredSignatureError:
        # User state: the ordinary end of a session (rolling 18h TTL), so
        # INFO — an idle overnight tab is expected, not an incident.
        logger.info("AUTH_USER: expired JWT cookie; treating as anonymous")
        return None
    except jwt.InvalidTokenError:
        # User state: malformed, tampered, or wrongly-signed token.
        logger.warning("AUTH_USER: invalid JWT cookie; treating as anonymous")
        return None
    except PyJWKClientConnectionError:
        # Infrastructure: the JWKS endpoint was unreachable (down, timeout,
        # TLS failure, HTTP error), so we never got to check the token. Not
        # the user's fault and not an auth decision. Kept fail-open so an
        # upstream outage degrades rather than locking everyone out, but
        # ERROR so it is not mistaken for a wave of logged-out users (#243).
        logger.exception(
            "AUTH_INFRA: JWKS endpoint unreachable; token could not be verified "
            "and the user is being treated as anonymous"
        )
        return None
    except PyJWKClientError:
        # Key lookup failed against a REACHABLE JWKS — usually a `kid` that
        # isn't in the key set. Deliberately WARNING, not ERROR: the kid is
        # attacker-controlled, so an anonymous caller could otherwise
        # generate ERROR-level alerts at will. It is also the expected,
        # benign state during key rotation.
        logger.warning("AUTH_INFRA: JWKS key lookup failed (unknown key id); treating as anonymous")
        return None
    except Exception:
        # Unclassified — treat as infrastructure until proven otherwise.
        # Also catches malformed/non-JWKS response bodies (JSONDecodeError,
        # PyJWKSetError), which are infrastructure but not PyJWKClientError.
        logger.exception("AUTH_INFRA: unexpected failure validating JWT cookie")
        return None
