"""Upload content checks and re-encoding (D8-3086)."""

import asyncio
import base64
import io
import json
import os
import sys

import pytest
from PIL import Image

from src.agent.upload_window import shown_in_full
from src.api.thread_routes import _upload_payload
from src.config import settings
from src.tool_attachments import upload_reference
from src.uploads import image_extract, pdf_extract, process
from src.uploads.process import Processed, Rejected, clean_filename, process_upload


def _png(im: Image.Image) -> bytes:
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


def _open(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def _pdf(page_texts: list[str]) -> bytes:
    """A minimal PDF with one Helvetica text line per page."""
    objs: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(page_texts)))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>".encode())
    font = 3 + 2 * len(page_texts)
    for i, text in enumerate(page_texts):
        stream = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode() if text else b""
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
            f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {4 + 2 * i} 0 R >>".encode()
        )
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1,
        xref,
    )
    return bytes(out)


async def _image(data: bytes, name: str = "a.png") -> Processed:
    result = await process_upload("image", "image/png", name, data)
    assert isinstance(result, Processed)
    return result


async def test_16bit_png_becomes_8bit():
    im = Image.new("I;16", (4, 4))
    for x in range(4):
        for y in range(4):
            im.putpixel((x, y), 40000)
    assert _open(_png(im)).mode.startswith("I;16")

    result = await _image(_png(im))

    out = _open(result.data)
    assert out.mode in ("L", "RGB")
    assert out.getpixel((0, 0)) in (156, (156, 156, 156))  # 40000 / 256


async def test_large_png_downscaled_to_1600():
    result = await _image(_png(Image.new("RGB", (3200, 1000), "blue")))

    assert (result.width, result.height) == (1600, 500)
    assert _open(result.data).size == (1600, 500)
    assert result.mime_type == "image/png"
    assert result.kind == "image"


async def test_over_pixel_cap_rejected():
    result = await process_upload(
        "image", "image/png", "big.png", _png(Image.new("L", (6000, 5000)))
    )

    assert result == Rejected("big.png", "is over 25 million pixels")


async def test_decompression_bomb_error_rejected():
    # More than twice the cap makes Pillow raise an error rather than warn.
    result = await process_upload(
        "image", "image/png", "bomb.png", _png(Image.new("1", (9000, 9000)))
    )

    assert isinstance(result, Rejected)
    assert "million pixels" in result.reason


async def test_eps_bytes_declared_png_rejected():
    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\n"
    result = await process_upload("image", "image/png", "x.png", eps)

    assert result == Rejected("x.png", "couldn't be read as an image")


async def test_tiff_bytes_declared_png_rejected():
    out = io.BytesIO()
    Image.new("RGB", (4, 4)).save(out, "TIFF")
    result = await process_upload("image", "image/png", "x.png", out.getvalue())

    assert result == Rejected("x.png", "couldn't be read as an image")


async def test_gif_uses_first_frame():
    frames = [Image.new("RGB", (8, 8), c) for c in ("red", "blue")]
    out = io.BytesIO()
    frames[0].save(out, "GIF", save_all=True, append_images=frames[1:])

    result = await _image(out.getvalue(), "a.gif")

    assert _open(result.data).convert("RGB").getpixel((0, 0)) == (255, 0, 0)


async def test_exif_removed():
    src = io.BytesIO()
    exif = Image.Exif()
    exif[0x010E] = "secret description"
    Image.new("RGB", (8, 8), "red").save(src, "JPEG", exif=exif)
    assert _open(src.getvalue()).getexif()

    result = await _image(src.getvalue(), "a.jpg")

    assert not _open(result.data).getexif()
    assert b"secret description" not in result.data


async def test_transparency_kept_as_rgba():
    result = await _image(_png(Image.new("RGBA", (8, 8), (255, 0, 0, 0))))

    assert _open(result.data).mode == "RGBA"
    assert _open(result.data).getpixel((0, 0))[3] == 0


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
async def test_big_png_falls_back_to_jpeg(mode):
    channels = len(mode)
    noise = Image.frombytes(mode, (1000, 1000), os.urandom(1000 * 1000 * channels))
    assert len(_png(noise)) > settings.UPLOAD_PNG_MAX_BYTES

    result = await _image(_png(noise))

    assert result.mime_type == "image/jpeg"
    assert _open(result.data).format == "JPEG"


async def test_unknown_kind_rejected():
    result = await process_upload("video", "video/mp4", "a.mp4", b"x")

    assert result == Rejected("a.mp4", "isn't a supported file type")


async def test_pdf_text_extracted_and_capped_at_40000():
    long_line = "A" * 30000
    result = await process_upload("pdf", "application/pdf", "doc.pdf", _pdf([long_line, long_line]))

    assert isinstance(result, Processed)
    assert result.pages == 2
    assert result.text is not None
    assert len(result.text) == 40000
    assert result.text.startswith("AAA")
    assert result.text_truncated is True
    assert result.no_text is False
    assert result.mime_type == "application/pdf"


async def test_pdf_short_text_not_truncated_and_keeps_original_bytes():
    data = _pdf(["Hello ACCESS"])
    result = await process_upload("pdf", "application/pdf", "doc.pdf", data)

    assert isinstance(result, Processed)
    assert result.text is not None and "Hello ACCESS" in result.text
    assert result.text_truncated is False
    assert result.data == data


async def test_pdf_page_cap_truncates():
    data = _pdf(["page"] * 31)
    result = await process_upload("pdf", "application/pdf", "doc.pdf", data)

    assert isinstance(result, Processed)
    assert result.pages == 31
    assert result.text_truncated is True
    assert result.text is not None
    assert result.text.count("page") == 30


async def test_pdf_without_text_marks_no_text():
    result = await process_upload("pdf", "application/pdf", "scan.pdf", _pdf([""]))

    assert isinstance(result, Processed)
    assert result.no_text is True
    assert result.pages == 1


async def test_garbage_pdf_rejected():
    result = await process_upload("pdf", "application/pdf", "x.pdf", b"not a pdf")

    assert result == Rejected("x.pdf", "couldn't be read as a PDF")


async def test_extractor_output_missing_fields_rejected(tmp_path, monkeypatch):
    script = tmp_path / "bad.py"
    script.write_text("print('{}')")
    monkeypatch.setattr(process, "PDF_EXTRACT_PATH", script)

    result = await process_upload("pdf", "application/pdf", "x.pdf", b"x")

    assert result == Rejected("x.pdf", "couldn't be read as a PDF")


async def test_pdf_timeout_kills_extractor(tmp_path, monkeypatch):
    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(60)\n")
    monkeypatch.setattr(process, "PDF_EXTRACT_PATH", script)
    monkeypatch.setattr(settings, "UPLOAD_PDF_TIMEOUT_S", 0.1)
    procs = []
    real = asyncio.create_subprocess_exec

    async def spy(*args, **kwargs):
        procs.append(await real(*args, **kwargs))
        return procs[-1]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)

    result = await process_upload("pdf", "application/pdf", "x.pdf", b"x")

    assert result == Rejected("x.pdf", "took too long to read")
    assert procs[0].returncode is not None  # killed and reaped


def test_extractor_runs_as_isolated_script():
    import subprocess

    done = subprocess.run(
        [sys.executable, "-I", str(process.PDF_EXTRACT_PATH), "30", "100", "1000000000"],
        input=_pdf(["Hello"]),
        capture_output=True,
        check=True,
    )

    assert json.loads(done.stdout) == {"pages": 1, "text": "Hello", "truncated": False}


def test_extract_main_writes_json():
    out = io.BytesIO()
    pdf_extract.main(["x", "30", "100"], io.BytesIO(_pdf(["Hi"])), out)

    assert json.loads(out.getvalue())["text"] == "Hi"


def _refuse(*_args):
    raise ValueError("current limit exceeds maximum limit")


def test_pixel_cap_does_not_depend_on_pillow_warning(monkeypatch):
    big = _png(Image.new("L", (6000, 5000)))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)
    real_open = Image.open

    def silent_open(*args, **kwargs):
        Image.MAX_IMAGE_PIXELS = None  # Pillow stays silent
        return real_open(*args, **kwargs)

    monkeypatch.setattr(Image, "open", silent_open)

    assert image_extract.decode(big, 25_000_000, 1600, 2 * 1024 * 1024) == {"refused": "pixels"}


def _spy_children(monkeypatch) -> list[asyncio.subprocess.Process]:
    procs: list[asyncio.subprocess.Process] = []
    real = asyncio.create_subprocess_exec

    async def spy(*args, **kwargs):
        procs.append(await real(*args, **kwargs))
        return procs[-1]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    return procs


def _sleeper(tmp_path) -> str:
    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(60)\n")
    return str(script)


async def test_image_timeout_kills_the_decoder(tmp_path, monkeypatch):
    monkeypatch.setattr(process, "IMAGE_EXTRACT_PATH", _sleeper(tmp_path))
    monkeypatch.setattr(settings, "UPLOAD_IMAGE_TIMEOUT_S", 0.1)
    procs = _spy_children(monkeypatch)

    result = await process_upload("image", "image/png", "slow.jpg", b"x")

    assert result == Rejected("slow.jpg", "took too long to read")
    assert procs[0].returncode is not None  # killed and reaped


async def test_cancelled_image_decode_kills_the_child_and_frees_its_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(process, "IMAGE_EXTRACT_PATH", _sleeper(tmp_path))
    monkeypatch.setattr(process, "_semaphore", asyncio.Semaphore(1))
    procs = _spy_children(monkeypatch)
    task = asyncio.create_task(process_upload("image", "", "a.png", b"x"))
    while not procs:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert procs[0].returncode is not None
    assert process._semaphore._value == 1


async def test_image_decoder_is_held_to_its_memory_limit(monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_IMAGE_RLIMIT_AS", 1024 * 1024 * 1024)
    argv: list[tuple] = []
    real = asyncio.create_subprocess_exec

    async def spy(*args, **kwargs):
        argv.append(args)
        return await real(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)

    await _image(_png(Image.new("RGB", (4, 4))))

    assert argv[0][2] == str(process.IMAGE_EXTRACT_PATH)
    assert argv[0][-1] == str(1024 * 1024 * 1024)


def test_image_decoder_runs_as_isolated_script():
    import subprocess

    done = subprocess.run(
        [sys.executable, "-I", str(process.IMAGE_EXTRACT_PATH), "100", "1600", "1000000", "1"],
        input=_png(Image.new("RGB", (20, 10), "red")),
        capture_output=True,
        check=True,
    )

    assert json.loads(done.stdout) == {"refused": "pixels"}


def test_image_decoder_main_writes_json():
    out = io.BytesIO()
    image_extract.main(
        ["x", "100", "1600", "1000000"], io.BytesIO(_png(Image.new("L", (5, 4)))), out
    )

    result = json.loads(out.getvalue())
    assert (result["mime_type"], result["width"], result["height"]) == ("image/png", 5, 4)


@pytest.mark.parametrize("module", [pdf_extract, image_extract])
def test_each_decoder_sets_its_memory_limit(monkeypatch, module):
    calls = []
    monkeypatch.setattr(module.resource, "setrlimit", lambda *a: calls.append(a))

    module.limit_memory(123)

    assert calls == [(module.resource.RLIMIT_AS, (123, 123))]


@pytest.mark.parametrize("module", [pdf_extract, image_extract])
def test_each_decoder_tolerates_no_limit_on_macos_only(monkeypatch, module):
    monkeypatch.setattr(module.resource, "setrlimit", _refuse)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    module.limit_memory(123)

    monkeypatch.setattr(module.sys, "platform", "linux")
    with pytest.raises(ValueError):
        module.limit_memory(123)


@pytest.mark.parametrize(
    "output",
    [
        "{}",
        '{"refused": "unreadable"}',
        "[]",
        '{"mime_type": "image/gif", "data": "", "width": 1, "height": 1}',
        '{"mime_type": "image/png", "data": "**", "width": 1, "height": 1}',
        "not json",
    ],
)
async def test_unexpected_decoder_output_is_unreadable(tmp_path, monkeypatch, output):
    script = tmp_path / "bad.py"
    script.write_text(f"import sys\nsys.stdout.write({output!r})\n")
    monkeypatch.setattr(process, "IMAGE_EXTRACT_PATH", script)

    result = await process_upload("image", "image/png", "x.png", b"x")

    assert result == Rejected("x.png", "couldn't be read as an image")


async def test_a_decoder_that_crashes_is_unreadable(tmp_path, monkeypatch):
    script = tmp_path / "crash.py"
    script.write_text("raise MemoryError\n")
    monkeypatch.setattr(process, "IMAGE_EXTRACT_PATH", script)

    result = await process_upload("image", "image/png", "x.png", b"x")

    assert result == Rejected("x.png", "couldn't be read as an image")


async def test_image_spawn_failure_is_unreadable(monkeypatch):
    async def fail(*_args, **_kwargs):
        raise OSError("fork: Cannot allocate memory")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail)

    result = await process_upload("image", "", "x.png", b"x")

    assert result == Rejected("x.png", "couldn't be read as an image")


async def test_cancelled_pdf_extraction_kills_the_child(tmp_path, monkeypatch):
    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(60)\n")
    monkeypatch.setattr(process, "PDF_EXTRACT_PATH", script)
    procs = []
    real = asyncio.create_subprocess_exec

    async def spy(*args, **kwargs):
        procs.append(await real(*args, **kwargs))
        return procs[-1]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    task = asyncio.create_task(process_upload("pdf", "", "x.pdf", b"x"))
    while not procs:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert procs[0].returncode is not None


async def test_subprocess_spawn_failure_is_rejected(monkeypatch):
    async def fail(*_args, **_kwargs):
        raise OSError("fork: Cannot allocate memory")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail)

    result = await process_upload("pdf", "", "x.pdf", b"x")

    assert result == Rejected("x.pdf", "couldn't be read as a PDF")


async def test_latin1_text_decodes_with_replacement():
    result = await process_upload("text", "text/plain", "a.log", b"caf\xe9 ok")

    assert isinstance(result, Processed)
    assert result.text == "caf� ok"
    assert result.data == b"caf\xe9 ok"
    assert result.text_truncated is False


async def test_text_capped_at_40000():
    result = await process_upload("text", "text/plain", "a.log", b"x" * 40001)

    assert isinstance(result, Processed)
    assert result.text is not None and len(result.text) == 40000
    assert result.text_truncated is True


async def test_nul_text_rejected():
    result = await process_upload("text", "text/plain", "a.bin", b"ab\x00cd")

    assert result == Rejected("a.bin", "isn't a text file")


def test_clean_filename():
    assert clean_filename("run_md.sh") == "run_md.sh"
    assert clean_filename("My Report (1).pdf") == "My Report (1).pdf"
    assert clean_filename("../etc/pa\nss<wd>") == ".._etc_pa_ss_wd_"
    assert clean_filename("é.png") == "_.png"
    assert clean_filename("") == "file"
    assert len(clean_filename("a" * 300)) == 100


async def test_concurrency_is_bounded():
    assert process._semaphore._value == settings.UPLOAD_DECODE_CONCURRENCY


def test_extract_stops_reading_pages_past_the_char_cap():
    result = pdf_extract.extract(_pdf(["A" * 60, "B" * 60, "C" * 60]), 30, 100)

    assert result["pages"] == 3
    assert result["truncated"] is True
    assert result["text"] == ("A" * 60 + "\n" + "B" * 60)[:100]


# A Slurm log: 40,000 characters of dense lines, which estimate past one
# message's token cap before the character cap is reached.
_SLURM_LOG = "".join(
    f"[2026-11-12T08:{i % 60:02d}:{i % 60:02d}.{i:06d}] slurmstepd: task/cgroup: "
    f"job {4100000 + i} step {i % 7} rss={i * 37}K vmem={i * 91}K\n"
    for i in range(2000)
)[:40_000]
_CSV = "".join(
    f"{i},{i * 3.14159:.5f},{i * 7919 % 100003},node{i % 512:03d}\n" for i in range(4000)
)
_HEX = "".join(f"{(i * 2654435761) % 2**32:08x}" for i in range(5000))
_CJK = "资源分配作业调度计算节点超级计算机" * 2500


@pytest.mark.parametrize("text", [_SLURM_LOG, _CSV, _HEX, _CJK], ids=["log", "csv", "hex", "cjk"])
@pytest.mark.parametrize("name", ["slurm-4100000.out", "n" * 100])
async def test_a_dense_text_file_alone_is_always_shown_in_full(text, name):
    result = await process_upload("text", "text/plain", name, text.encode())

    assert isinstance(result, Processed)
    assert result.text_truncated is True
    ref = upload_reference("a" * 32, _upload_payload(result))
    assert shown_in_full([ref]) == [True]


async def test_a_dense_file_is_cut_at_the_end_of_a_line():
    result = await process_upload("text", "text/plain", "slurm.out", _SLURM_LOG.encode())

    assert isinstance(result, Processed) and result.text is not None
    assert _SLURM_LOG.startswith(result.text + "\n")
    assert len(result.text) > 10_000


async def test_a_dense_pdf_alone_is_always_shown_in_full():
    pages = [_HEX[i : i + 1500] for i in range(0, 30_000, 1500)]
    result = await process_upload("pdf", "application/pdf", "dump.pdf", _pdf(pages))

    assert isinstance(result, Processed)
    assert result.text_truncated is True
    ref = upload_reference("a" * 32, _upload_payload(result))
    assert shown_in_full([ref]) == [True]


async def test_a_text_file_that_fits_is_not_cut():
    result = await process_upload("text", "text/plain", "a.log", _SLURM_LOG[:2000].encode())

    assert isinstance(result, Processed)
    assert result.text == _SLURM_LOG[:2000]
    assert result.text_truncated is False


# The decoder runs in its own process above; these run it in this one, so
# every branch is measured.
def _decoded(data: bytes) -> dict:
    return image_extract.decode(data, 25_000_000, 1600, settings.UPLOAD_PNG_MAX_BYTES)


def test_decode_in_process_covers_each_outcome():
    sixteen = Image.new("I;16", (4, 4))
    assert _open(base64.b64decode(_decoded(_png(sixteen))["data"])).mode in ("L", "RGB")

    clear = _decoded(_png(Image.new("RGBA", (8, 8), (255, 0, 0, 0))))
    assert _open(base64.b64decode(clear["data"])).mode == "RGBA"

    for mode in ("RGB", "RGBA"):
        noise = Image.frombytes(mode, (1000, 1000), os.urandom(1000 * 1000 * len(mode)))
        assert _decoded(_png(noise))["mime_type"] == "image/jpeg"

    assert _decoded(_png(Image.new("1", (9000, 9000)))) == {"refused": "pixels"}
    assert _decoded(b"%!PS-Adobe-3.0 EPSF-3.0\n") == {"refused": "unreadable"}


def _rotated_jpeg() -> bytes:
    """A 40x20 JPEG, left half red, whose EXIF says to turn it 90 degrees clockwise."""
    im = Image.new("RGB", (40, 20), "blue")
    im.paste((255, 0, 0), (0, 0, 20, 20))
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW to display
    out = io.BytesIO()
    im.save(out, "JPEG", exif=exif, quality=95)
    return out.getvalue()


async def test_exif_orientation_is_applied_before_re_encoding():
    result = await _image(_rotated_jpeg(), "phone.jpg")

    assert (result.width, result.height) == (20, 40)
    out = _open(result.data).convert("RGB")
    assert out.size == (20, 40)
    red, blue = out.getpixel((10, 5)), out.getpixel((10, 35))
    assert red[0] > 200 and red[2] < 60  # the left half is now the top
    assert blue[2] > 200 and blue[0] < 60
    assert not out.getexif().get(0x0112)


def test_decode_in_process_applies_exif_orientation():
    result = _decoded(_rotated_jpeg())

    assert (result["width"], result["height"]) == (20, 40)


@pytest.mark.parametrize(("kind", "name"), [("image", "a.png"), ("pdf", "a.pdf")])
async def test_a_file_waiting_too_long_for_a_slot_is_refused(monkeypatch, kind, name):
    monkeypatch.setattr(process, "_semaphore", asyncio.Semaphore(1))
    monkeypatch.setattr(settings, "UPLOAD_DECODE_WAIT_S", 0.05)
    await process._semaphore.acquire()  # someone else's decode

    result = await process_upload(kind, "", name, b"x")

    assert result == Rejected(name, process.BUSY)
    assert result.reason == (
        "wasn't read because the service is busy. Try sharing it again in a moment"
    )
    process._semaphore.release()
    assert process._semaphore._value == 1  # the refused wait took no slot
