"""Decode a shared image and re-encode it for the model.

Runs as its own process (`python -I image_extract.py MAX_PIXELS MAX_EDGE
PNG_MAX_BYTES RLIMIT_AS_BYTES`), as pdf_extract.py does, so a slow or
memory-hungry image can be killed on a timeout and held to a memory limit. It
reads the image from stdin and writes JSON to stdout: `{mime_type, width,
height, data}` with the re-encoded image in base64, or `{refused: "pixels"}`
or `{refused: "unreadable"}`. It must import only the standard library and
Pillow.
"""

import base64
import io
import json
import resource
import sys
from typing import Any, BinaryIO

_IMAGE_FORMATS = ["PNG", "JPEG", "GIF", "WEBP"]
_JPEG_QUALITY = 90


def _flatten_to_8bit(im: Any) -> Any:
    """RGB, or RGBA when the image has transparency. An image already in that
    mode is returned as is: a copy of a 25-million-pixel image is 100MB."""
    if im.mode.startswith("I;16") or im.mode == "I":
        im = im.point(lambda v: v / 256).convert("L")
    target = "RGBA" if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info else "RGB"
    return im if im.mode == target else im.convert(target)


def _encode(im: Any, png_max_bytes: int) -> tuple[bytes, str]:
    from PIL import Image

    out = io.BytesIO()
    im.save(out, "PNG", optimize=True)
    if out.tell() <= png_max_bytes:
        return out.getvalue(), "image/png"
    if im.mode == "RGBA":
        flat = Image.new("RGB", im.size, "white")
        flat.paste(im, mask=im.getchannel("A"))
        im = flat
    out = io.BytesIO()
    im.save(out, "JPEG", quality=_JPEG_QUALITY)
    return out.getvalue(), "image/jpeg"


def decode(data: bytes, max_pixels: int, max_edge: int, png_max_bytes: int) -> dict[str, Any]:
    from PIL import Image, ImageOps  # after the memory limit is set

    # Pillow only warns between 1x and 2x its own cap and raises above 2x, so
    # the cap is enforced here, on the header size, before load().
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        opened = Image.open(io.BytesIO(data), formats=_IMAGE_FORMATS)
        if opened.width * opened.height > max_pixels:
            return {"refused": "pixels"}
        opened.seek(0)
        # A JPEG decodes at the smallest scale still past the long edge, which
        # saves memory and time. Other formats ignore it.
        opened.draft("RGB", (max_edge, max_edge))
        opened.load()
        # A phone photo is stored sideways with an EXIF note to turn it, which
        # re-encoding drops, so it is turned here and measured after.
        ImageOps.exif_transpose(opened, in_place=True)
        im = _flatten_to_8bit(opened)
        im.thumbnail((max_edge, max_edge))
        encoded, mime = _encode(im, png_max_bytes)
    except Image.DecompressionBombError:
        return {"refused": "pixels"}
    except Exception:
        return {"refused": "unreadable"}
    return {
        "mime_type": mime,
        "width": im.width,
        "height": im.height,
        "data": base64.b64encode(encoded).decode(),
    }


def limit_memory(limit: int) -> None:
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError):
        if sys.platform != "darwin":  # macOS has no usable RLIMIT_AS
            raise


def main(argv: list[str], stdin: BinaryIO, stdout: BinaryIO) -> None:
    result = decode(stdin.read(), int(argv[1]), int(argv[2]), int(argv[3]))
    stdout.write(json.dumps(result).encode())


if __name__ == "__main__":
    limit_memory(int(sys.argv[4]))
    main(sys.argv, sys.stdin.buffer, sys.stdout.buffer)
