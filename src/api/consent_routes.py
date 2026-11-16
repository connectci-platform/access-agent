"""The signed-in user's personalization setting, for the full-screen client (D8-3124).

GET reads it for the Settings switch. POST records a choice from the switch
or from the offer card: allow or deny. "Not now" needs no call, because the
agent records the ask itself before it offers. Both act only for the
verified caller. The answer is stored in the user's ACCESS profile; this
process's cached copy is replaced at once, so the next message follows it.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..agent.consent import Consent, consent_for, record_consent
from .identity import Identity, require_identity

consent_router = APIRouter()

PATH = "/me/assistant-consent"


class ConsentChoice(BaseModel):
    consent: Literal["allow", "deny"]


def _body(consent: Consent) -> dict[str, str | None]:
    return {
        "consent": consent.state,
        "asked_at": consent.asked_at.isoformat() if consent.asked_at else None,
    }


def _unavailable() -> JSONResponse:
    return JSONResponse(
        {"error": "Your personalization setting couldn't be reached. Try again shortly."},
        status_code=503,
        headers={"Cache-Control": "no-store"},
    )


@consent_router.get(PATH, response_model=None)
async def get_assistant_consent(identity: Identity) -> JSONResponse:
    consent = await consent_for(require_identity(identity))
    if consent is None:
        return _unavailable()
    return JSONResponse(_body(consent), headers={"Cache-Control": "no-store"})


@consent_router.post(PATH, response_model=None)
async def post_assistant_consent(choice: ConsentChoice, identity: Identity) -> JSONResponse:
    consent = await record_consent(require_identity(identity), choice.consent)
    if consent is None:
        return _unavailable()
    return JSONResponse(_body(consent), headers={"Cache-Control": "no-store"})
