"""Validate and re-encode chat uploads (D8-3086).

`process_upload` checks content only: decode, pixel cap, page cap, text cap, NUL
bytes. Per-kind byte limits are enforced at ingress, before this runs.
"""

import asyncio
import io
import json
import re
import resource
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

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
    async with _semaphore:
        if kind == "image":
            return await asyncio.to_thread(_process_image, name, data)
        if kind == "pdf":
            return await _process_pdf(name, data)
        if kind == "text":
            return _process_text(name, data)
    return Rejected(name, "isn't a supported file type")


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


def _process_image(name: str, data: bytes) -> Processed | Rejected:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            # open() checks im.size against MAX_IMAGE_PIXELS before any decoding.
            opened = Image.open(io.BytesIO(data), formats=_IMAGE_FORMATS)
        opened.seek(0)
        opened.load()
        im = _flatten_to_8bit(opened)
        edge = settings.UPLOAD_IMAGE_MAX_EDGE
        im.thumbnail((edge, edge))
        encoded, mime = _encode(im)
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        millions = settings.UPLOAD_MAX_PIXELS // 1_000_000
        return Rejected(name, f"is over {millions} million pixels")
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


def _limit_memory() -> None:
    """Runs in the extractor's child process before exec."""
    try:
        limit = settings.UPLOAD_PDF_RLIMIT_AS
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError):
        pass  # macOS can refuse RLIMIT_AS; the timeout still applies


async def _process_pdf(name: str, data: bytes) -> Processed | Rejected:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        str(PDF_EXTRACT_PATH),
        str(settings.UPLOAD_PDF_MAX_PAGES),
        str(settings.UPLOAD_TEXT_MAX_CHARS),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        preexec_fn=_limit_memory,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(data), settings.UPLOAD_PDF_TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return Rejected(name, "took too long to read")
    if proc.returncode != 0:
        return Rejected(name, "couldn't be read as a PDF")
    try:
        result = json.loads(stdout)
        text = str(result["text"])
        pages = int(result["pages"])
        truncated = bool(result["truncated"])
    except (ValueError, KeyError, TypeError):
        return Rejected(name, "couldn't be read as a PDF")
    return Processed(
        kind="pdf",
        name=name,
        mime_type="application/pdf",
        data=data,
        text=text,
        text_truncated=truncated,
        no_text=not text.strip(),
        pages=pages,
    )


def _process_text(name: str, data: bytes) -> Processed | Rejected:
    if b"\x00" in data:
        return Rejected(name, "isn't a text file")
    text = data.decode("utf-8", errors="replace")
    limit = settings.UPLOAD_TEXT_MAX_CHARS
    return Processed(
        kind="text",
        name=name,
        mime_type="text/plain",
        data=data,
        text=text[:limit],
        text_truncated=len(text) > limit,
    )
