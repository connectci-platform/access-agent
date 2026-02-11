"""JWT cookie authentication for ACCESS QA Bot.

Extracts user identity from a signed JWT cookie (``SESSaccess_auth``) set by
Drupal on ``.access-ci.org``.

The route handler is responsible for body-based fallback (transition period)
using the already-parsed ``QueryRequest.acting_user`` field — this module
never reads the request body, avoiding double-consumption of the ASGI
body stream.

See: access-qa-planning/08-qa-bot-authentication.md
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import jwt

if TYPE_CHECKING:
    from fastapi import Request

logger = logging.getLogger(__name__)


def get_acting_user_from_cookie(
    request: Request,
    jwt_secret: str,
) -> tuple[str | None, bool]:
    """Extract the acting user from the ``SESSaccess_auth`` JWT cookie.

    Args:
        request: The incoming FastAPI request.
        jwt_secret: Shared HMAC-SHA256 signing secret.

    Returns:
        A tuple of ``(user, cookie_was_present)``:
        - ``("jsmith@access-ci.org", True)`` — valid cookie
        - ``(None, True)`` — cookie present but invalid/expired
        - ``(None, False)`` — no cookie sent
    """
    token = request.cookies.get("SESSaccess_auth")
    if not token:
        return None, False

    user = _decode_jwt(token, jwt_secret)
    return user, True


def _decode_jwt(token: str, secret: str) -> str | None:
    """Decode and validate a JWT, returning the ``sub`` claim."""
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
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
