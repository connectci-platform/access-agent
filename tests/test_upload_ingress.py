"""Message blocks are parsed into typed parts at ingress (D8-3086)."""

import base64
from typing import Any

from src.api.thread_routes import IncomingUpload, _extract_text_content, parse_message

REF = "0123456789abcdef" * 2


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _image(name: str = "Screenshot.png", raw: bytes = b"png", mime: str = "image/png"):
    return {"type": "image", "mimeType": mime, "data": _b64(raw), "metadata": {"name": name}}


def _file(name: str, raw: bytes, mime: str):
    return {"type": "file", "mimeType": mime, "data": _b64(raw), "metadata": {"filename": name}}


def test_image_block_becomes_upload():
    parsed = parse_message([{"type": "text", "text": "hi"}, _image()], anonymous=False)
    assert parsed.uploads == [IncomingUpload("image", "image/png", "Screenshot.png", b"png")]
    assert parsed.text == "hi"
    assert parsed.dropped == []


def test_pdf_and_text_file_blocks_become_uploads():
    content = [
        _file("a.pdf", b"%PDF", "application/pdf"),
        _file("run.log", b"hello", "text/plain"),
    ]
    parsed = parse_message(content, anonymous=False)
    assert [(u.kind, u.mime_type, u.name, u.data) for u in parsed.uploads] == [
        ("pdf", "application/pdf", "a.pdf", b"%PDF"),
        ("text", "text/plain", "run.log", b"hello"),
    ]


def test_string_content_is_the_text():
    assert parse_message("hello", anonymous=True).text == "hello"


def test_non_list_content_is_empty():
    parsed = parse_message({"type": "text", "text": "x"}, anonymous=True)
    assert parsed.text == "" and parsed.uploads == []


def test_image_url_and_unknown_blocks_are_dropped():
    content = [
        {"type": "image_url", "image_url": {"url": "http://example.com/x.png"}},
        {"type": "video", "metadata": {"name": "clip.mp4"}},
        {
            "type": "file",
            "mimeType": "application/zip",
            "data": _b64(b"x"),
            "metadata": {"filename": "a.zip"},
        },
        {
            "type": "image",
            "mimeType": "image/svg+xml",
            "data": _b64(b"x"),
            "metadata": {"name": "a.svg"},
        },
        {"type": "image", "mimeType": "image/png", "url": "http://example.com/x.png"},
        "not a block",
    ]
    parsed = parse_message(content, anonymous=False)
    assert parsed.uploads == []
    assert parsed.dropped == [
        "a file was left out: that kind of file isn't supported.",
        "clip.mp4 was left out: that kind of file isn't supported.",
        "a.zip was left out: that kind of file isn't supported.",
        "a.svg was left out: that kind of file isn't supported.",
        "a file was left out: that kind of file isn't supported.",
    ]


def test_names_are_cleaned_and_blank_names_fall_back():
    content = [
        {"type": "video", "metadata": {"name": "../x\n.mp4"}},
        {"type": "video", "metadata": {"name": "  "}},
        {"type": "video", "metadata": "nope"},
    ]
    parsed = parse_message(content, anonymous=False)
    assert parsed.dropped[0].startswith(".._x_.mp4 was left out")
    assert parsed.dropped[1].startswith("a file was left out")
    assert parsed.dropped[2].startswith("a file was left out")


def test_bad_base64_is_dropped():
    block = _image()
    block["data"] = "not base64!!"
    parsed = parse_message([block], anonymous=False)
    assert parsed.uploads == []
    assert parsed.dropped == ["Screenshot.png was left out: it couldn't be read."]


def test_third_file_for_anonymous_is_dropped():
    parsed = parse_message([_image("a.png"), _image("b.png"), _image("c.png")], anonymous=True)
    assert [u.name for u in parsed.uploads] == ["a.png", "b.png"]
    assert parsed.dropped == ["c.png was left out: at most 2 files per message."]


def test_signed_in_allows_five_files():
    parsed = parse_message([_image(f"{i}.png") for i in range(6)], anonymous=False)
    assert len(parsed.uploads) == 5
    assert parsed.dropped == ["5.png was left out: at most 5 files per message."]


def test_image_over_the_anonymous_limit_is_dropped_before_decoding():
    big = _image(raw=b"x" * (6 * 1024 * 1024))
    assert parse_message([big], anonymous=True).dropped == [
        "Screenshot.png was left out: images can be at most 5 MB."
    ]
    assert len(parse_message([big], anonymous=False).uploads) == 1


def test_file_just_over_the_limit_is_dropped_after_decoding():
    # Small enough that the base64 length passes the cheap pre-check.
    limit = 100 * 1024
    block = _file("big.txt", b"x" * (limit + 1), "text/plain")
    assert parse_message([block], anonymous=True).dropped == [
        "big.txt was left out: text files can be at most 100 KB."
    ]
    exact = _file("ok.txt", b"x" * limit, "text/plain")
    assert len(parse_message([exact], anonymous=True).uploads) == 1


def test_pdf_limit_names_pdfs():
    big = _file("a.pdf", b"x" * (5 * 1024 * 1024 + 1), "application/pdf")
    assert parse_message([big], anonymous=True).dropped == [
        "a.pdf was left out: PDFs can be at most 5 MB."
    ]


def test_total_limit_drops_the_file_that_crosses_it():
    chunk = b"x" * (4 * 1024 * 1024 + 512 * 1024)
    content = [_file("a.pdf", chunk, "application/pdf"), _file("b.pdf", chunk, "application/pdf")]
    parsed = parse_message(content, anonymous=True)
    assert [u.name for u in parsed.uploads] == ["a.pdf"]
    assert parsed.dropped == ["b.pdf was left out: files can total at most 8 MB per message."]


def test_query_excludes_file_content():
    content = [
        {"type": "text", "text": "what is this"},
        _file("notes.txt", b"secret file body", "text/plain"),
        {"type": "text", "text": "please"},
    ]
    parsed = parse_message(content, anonymous=False)
    assert parsed.text == "what is this please"
    assert "secret" not in parsed.text


def test_non_string_and_empty_text_blocks_are_skipped():
    content = [{"type": "text", "text": 5}, {"type": "text", "text": ""}, {"type": "text"}]
    assert parse_message(content, anonymous=False).text == ""


def test_reference_ids_are_validated_and_deduplicated():
    content: list[Any] = [
        {"type": "attachment", "id": REF, "url": "http://example.com", "mimeType": "x"},
        {"type": "attachment", "id": REF},
        {"type": "attachment", "id": REF.upper()},
        {"type": "attachment", "id": "short"},
        {"type": "attachment", "id": 7},
        {"type": "attachment"},
    ]
    parsed = parse_message(content, anonymous=False)
    assert parsed.ref_ids == [REF]
    assert parsed.uploads == [] and parsed.dropped == []


def test_confirm_ids_are_still_extracted():
    content = [
        {"type": "text", "text": "Confirm"},
        {"type": "confirm", "preview_ids": [REF, "bad"]},
    ]
    parsed = parse_message(content, anonymous=False)
    assert parsed.confirm_ids == [REF]
    assert parsed.text == "Confirm"
    assert parsed.dropped == []


def test_extract_text_content_is_the_parsed_text():
    assert _extract_text_content([{"type": "text", "text": "a"}, _image()]) == "a"
