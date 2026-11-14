"""Validate and re-encode chat uploads (D8-3086).

`process_upload` checks content only: decode, pixel cap, page cap, text cap, NUL
bytes. Per-kind byte limits are enforced at ingress, before this runs. Images
and PDFs are read in their own processes (image_extract.py, pdf_extract.py),
held to a timeout and a memory limit, at most ``UPLOAD_DECODE_CONCURRENCY`` at
a time across the agent.
"""

import asyncio
import base64
import contextlib
import json
import re
import sys
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.agent.upload_window import fit_file_text
from src.config import settings

PDF_EXTRACT_PATH = Path(__file__).with_name("pdf_extract.py")
IMAGE_EXTRACT_PATH = Path(__file__).with_name("image_extract.py")
_IMAGE_TYPES = frozenset({"image/png", "image/jpeg"})
_semaphore = asyncio.Semaphore(settings.UPLOAD_DECODE_CONCURRENCY)


@dataclass
class Processed:
    kind: Literal["image", "pdf", "text"]
    name: str
    mime_type: str
    data: bytes
    text: str | None = None
    text_truncated: bool = False
    no_text: bool = False
    width: int | None = None
    height: int | None = None
    pages: int | None = None


@dataclass
class Rejected:
    name: str
    reason: str  # a short phrase a user reads: "<name> <reason>"


def clean_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 ._()-]", "_", name)[:100]
    return cleaned or "file"


# `mime_type` is the client's claim and is not trusted; the content decides.
async def process_upload(
    kind: str,
    mime_type: str,  # noqa: ARG001
    name: str,
    data: bytes,
) -> Processed | Rejected:
    name = clean_filename(name)
    if kind == "image":
        return await _in_slot(name, _process_image(name, data))
    if kind == "pdf":
        return await _in_slot(name, _process_pdf(name, data))
    if kind == "text":
        return _process_text(name, data)
    return Rejected(name, "isn't a supported file type")


BUSY = "wasn't read because the service is busy. Try sharing it again in a moment"


async def _in_slot(
    name: str, work: Coroutine[Any, Any, Processed | Rejected]
) -> Processed | Rejected:
    """Run ``work`` in a decode slot, or refuse the file when none frees up
    within ``UPLOAD_DECODE_WAIT_S``."""
    try:
        async with asyncio.timeout(settings.UPLOAD_DECODE_WAIT_S):
            await _semaphore.acquire()
    except TimeoutError:
        work.close()
        return Rejected(name, BUSY)
    try:
        return await work
    finally:
        _semaphore.release()


class _ChildError(Exception):
    """The extractor couldn't start, or exited with an error."""


async def _run_child(script: Path, args: list[int], data: bytes, timeout: float) -> bytes:
    """Run an extractor on ``data`` and return what it wrote.

    Raises TimeoutError past ``timeout`` and _ChildError when it can't start
    or exits with an error. On a timeout, a cancel or any other error, the
    child is killed and reaped before this returns.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            str(script),
            *(str(arg) for arg in args),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError as exc:  # fork or exec failed (EAGAIN, ENOMEM)
        raise _ChildError from exc
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(data), timeout)
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
    if proc.returncode != 0:
        raise _ChildError
    return stdout


def _too_many_pixels(name: str) -> Rejected:
    millions = settings.UPLOAD_MAX_PIXELS // 1_000_000
    return Rejected(name, f"is over {millions} million pixels")


async def _process_image(name: str, data: bytes) -> Processed | Rejected:
    unreadable = Rejected(name, "couldn't be read as an image")
    try:
        stdout = await _run_child(
            IMAGE_EXTRACT_PATH,
            [
                settings.UPLOAD_MAX_PIXELS,
                settings.UPLOAD_IMAGE_MAX_EDGE,
                settings.UPLOAD_PNG_MAX_BYTES,
                settings.UPLOAD_IMAGE_RLIMIT_AS,
            ],
            data,
            settings.UPLOAD_IMAGE_TIMEOUT_S,
        )
    except TimeoutError:
        return Rejected(name, "took too long to read")
    except _ChildError:
        return unreadable
    try:
        result = json.loads(stdout)
        if result.get("refused") == "pixels":
            return _too_many_pixels(name)
        mime_type = str(result["mime_type"])
        encoded = base64.b64decode(result["data"], validate=True)
        width, height = int(result["width"]), int(result["height"])
    except (ValueError, KeyError, TypeError, AttributeError):
        return unreadable
    if mime_type not in _IMAGE_TYPES:
        return unreadable
    return Processed(
        kind="image",
        name=name,
        mime_type=mime_type,
        data=encoded,
        width=width,
        height=height,
    )


async def _process_pdf(name: str, data: bytes) -> Processed | Rejected:
    unreadable = Rejected(name, "couldn't be read as a PDF")
    try:
        stdout = await _run_child(
            PDF_EXTRACT_PATH,
            [
                settings.UPLOAD_PDF_MAX_PAGES,
                settings.UPLOAD_TEXT_MAX_CHARS,
                settings.UPLOAD_PDF_RLIMIT_AS,
            ],
            data,
            settings.UPLOAD_PDF_TIMEOUT_S,
        )
    except TimeoutError:
        return Rejected(name, "took too long to read")
    except _ChildError:
        return unreadable
    try:
        result = json.loads(stdout)
        text = str(result["text"])
        pages = int(result["pages"])
        truncated = bool(result["truncated"])
    except (ValueError, KeyError, TypeError):
        return unreadable
    text, cut = fit_file_text(name, text)
    return Processed(
        kind="pdf",
        name=name,
        mime_type="application/pdf",
        data=data,
        text=text,
        text_truncated=truncated or cut,
        no_text=not text.strip(),
        pages=pages,
    )


def _process_text(name: str, data: bytes) -> Processed | Rejected:
    if b"\x00" in data:
        return Rejected(name, "isn't a text file")
    text, cut = fit_file_text(name, data.decode("utf-8", errors="replace"))
    return Processed(
        kind="text",
        name=name,
        mime_type="text/plain",
        data=data,
        text=text,
        text_truncated=cut,
    )
