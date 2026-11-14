"""Validate and re-encode chat uploads (D8-3086).

`process_upload` checks content only: decode, pixel cap, page cap, text cap, NUL
bytes. Per-kind byte limits are enforced at ingress, before this runs.
"""

import asyncio
import contextlib
import io
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from src.agent.upload_window import fit_file_text
from src.config import settings

# Pillow only offers a module-global pixel cap, so it is set once, here. Nothing
# else in the agent opens images.
Image.MAX_IMAGE_PIXELS = settings.UPLOAD_MAX_PIXELS

PDF_EXTRACT_PATH = Path(__file__).with_name("pdf_extract.py")
_IMAGE_FORMATS = ["PNG", "JPEG", "GIF", "WEBP"]
_JPEG_QUALITY = 90
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
        return await _decode_in_thread(name, data)
    if kind == "pdf":
        async with _semaphore:
            return await _process_pdf(name, data)
    if kind == "text":
        return _process_text(name, data)
    return Rejected(name, "isn't a supported file type")


async def _decode_in_thread(name: str, data: bytes) -> Processed | Rejected:
    """Hold a decode slot until the worker thread ends, even if the caller is cancelled.

    A cancelled await does not stop a running thread. Releasing the slot at
    cancellation would let dropped connections stack up decodes without limit.
    """
    await _semaphore.acquire()
    task = asyncio.ensure_future(asyncio.to_thread(_process_image, name, data))
    task.add_done_callback(lambda _: _semaphore.release())
    return await asyncio.shield(task)


def _flatten_to_8bit(im: Image.Image) -> Image.Image:
    if im.mode.startswith("I;16") or im.mode == "I":
        im = im.point(lambda v: v / 256).convert("L")
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        return im.convert("RGBA")
    return im.convert("RGB")


def _encode(im: Image.Image) -> tuple[bytes, str]:
    out = io.BytesIO()
    im.save(out, "PNG", optimize=True)
    if out.tell() <= settings.UPLOAD_PNG_MAX_BYTES:
        return out.getvalue(), "image/png"
    if im.mode == "RGBA":
        flat = Image.new("RGB", im.size, "white")
        flat.paste(im, mask=im.getchannel("A"))
        im = flat
    out = io.BytesIO()
    im.save(out, "JPEG", quality=_JPEG_QUALITY)
    return out.getvalue(), "image/jpeg"


def _too_many_pixels(name: str) -> Rejected:
    millions = settings.UPLOAD_MAX_PIXELS // 1_000_000
    return Rejected(name, f"is over {millions} million pixels")


def _process_image(name: str, data: bytes) -> Processed | Rejected:
    try:
        # Pillow only warns between 1x and 2x the cap and raises above 2x. A
        # warnings filter is process-global and unsafe with parallel decodes, so
        # the cap is enforced here, on the header size, before load().
        opened = Image.open(io.BytesIO(data), formats=_IMAGE_FORMATS)
        if opened.width * opened.height > settings.UPLOAD_MAX_PIXELS:
            return _too_many_pixels(name)
        opened.seek(0)
        opened.load()
        im = _flatten_to_8bit(opened)
        edge = settings.UPLOAD_IMAGE_MAX_EDGE
        im.thumbnail((edge, edge))
        encoded, mime = _encode(im)
    except Image.DecompressionBombError:
        return _too_many_pixels(name)
    except Exception:
        return Rejected(name, "couldn't be read as an image")
    return Processed(
        kind="image",
        name=name,
        mime_type=mime,
        data=encoded,
        width=im.width,
        height=im.height,
    )


async def _process_pdf(name: str, data: bytes) -> Processed | Rejected:
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            str(PDF_EXTRACT_PATH),
            str(settings.UPLOAD_PDF_MAX_PAGES),
            str(settings.UPLOAD_TEXT_MAX_CHARS),
            str(settings.UPLOAD_PDF_RLIMIT_AS),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError:  # fork or exec failed (EAGAIN, ENOMEM)
        return Rejected(name, "couldn't be read as a PDF")
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(data), settings.UPLOAD_PDF_TIMEOUT_S)
    except TimeoutError:
        return Rejected(name, "took too long to read")
    finally:
        # Timeout, cancellation or any other error: never leave the child running.
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
    if proc.returncode != 0:
        return Rejected(name, "couldn't be read as a PDF")
    try:
        result = json.loads(stdout)
        text = str(result["text"])
        pages = int(result["pages"])
        truncated = bool(result["truncated"])
    except (ValueError, KeyError, TypeError):
        return Rejected(name, "couldn't be read as a PDF")
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
