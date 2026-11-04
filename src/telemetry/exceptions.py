"""Record unhandled endpoint exceptions on the active server span."""

from opentelemetry import trace
from starlette.types import ASGIApp, Receive, Scope, Send


class RecordExceptionMiddleware:
    """Attach an ``exception`` event to the current span, then re-raise.

    FastAPI's native server span records only error.type and ERROR status, and
    sends exception detail to the OTel logs signal, which we don't export.
    Sitting inside the span and outside the exception handlers, this sees only
    exceptions no handler turned into a response, so each is recorded once and
    the HTTP response is unchanged. Only exception.type reaches Honeycomb: the
    export filter drops the message and traceback (telemetry/content_filter.py).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await self.app(scope, receive, send)
        except Exception as exc:
            trace.get_current_span().record_exception(exc)
            raise
