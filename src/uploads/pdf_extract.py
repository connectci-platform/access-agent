"""Pull the text layer out of a PDF.

Runs as its own process (`python -I pdf_extract.py MAX_PAGES MAX_CHARS`) so a
hostile PDF can be killed on a timeout and held to a memory limit. It reads the
PDF from stdin and writes `{pages, text, truncated}` as JSON to stdout. It must
import only the standard library and pypdf.
"""

import io
import json
import sys
from typing import Any, BinaryIO

from pypdf import PdfReader


def extract(data: bytes, max_pages: int, max_chars: int) -> dict[str, Any]:
    """`pages` is the document's page count. Only the first `max_pages` are read."""
    reader = PdfReader(io.BytesIO(data))
    pages = len(reader.pages)
    parts: list[str] = []
    size = 0
    truncated = pages > max_pages
    for page in reader.pages[:max_pages]:
        text = page.extract_text() or ""
        parts.append(text)
        size += len(text)
        if size > max_chars:
            truncated = True
            break
    return {
        "pages": pages,
        "text": "\n".join(parts)[:max_chars],
        "truncated": truncated,
    }


def main(argv: list[str], stdin: BinaryIO, stdout: BinaryIO) -> None:
    result = extract(stdin.read(), int(argv[1]), int(argv[2]))
    stdout.write(json.dumps(result).encode())


if __name__ == "__main__":
    main(sys.argv, sys.stdin.buffer, sys.stdout.buffer)
