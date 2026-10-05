"""One table of document formats shared by every stage."""

import pytest

from tools.fake_image_detector import file_formats as F

_SAMPLES = {
    "jpeg": b"\xff\xd8\xff\xe0 rest",
    "png": b"\x89PNG\r\n\x1a\n rest",
    "webp": b"RIFF\x00\x00\x00\x00WEBPVP8 ",
    "pdf": b"%PDF-1.7\n",
    "tiff": b"II*\x00 rest",
    "heic": b"\x00\x00\x00\x18ftypheic rest",
    "heif": b"\x00\x00\x00\x18ftypmif1 rest",
}


@pytest.mark.parametrize(("name", "data"), _SAMPLES.items())
def test_sniff_identifies_each_supported_format(name, data):
    assert F.sniff(data).name == name


def test_big_endian_tiff_is_recognised():
    assert F.sniff(b"MM\x00* rest") is F.TIFF


@pytest.mark.parametrize("data", [b"", b"hello world", b"\x00\x00\x00\x18ftypavif rest"])
def test_unrecognised_bytes_are_not_guessed(data):
    """AVIF shares the ftyp box with HEIC; it must not be mistaken for it."""
    assert F.sniff(data) is None
    assert F.mime_type_of(data) == F.UNKNOWN_MIME_TYPE


def test_pdf_is_labelled_as_pdf_not_jpeg():
    """Every Gemini caller used to fall back to image/jpeg for anything else."""
    assert F.mime_type_of(_SAMPLES["pdf"]) == "application/pdf"


@pytest.mark.parametrize(
    ("declared", "expected"),
    [("image/jpeg", F.JPEG), ("IMAGE/JPEG", F.JPEG), ("image/jpg", F.JPEG),
     ("application/pdf; charset=binary", F.PDF), ("image/tiff", F.TIFF), (None, None), ("text/plain", None)],
)
def test_by_mime(declared, expected):
    assert F.by_mime(declared) is expected


def test_gemini_cannot_read_tiff():
    assert not F.TIFF.gemini_readable
    assert all(f.gemini_readable for f in F.FORMATS if f is not F.TIFF)


def _tiff(mode="RGB", pages=1) -> bytes:
    import io

    from PIL import Image

    frames = [Image.new(mode, (40, 30)) for _ in range(pages)]
    out = io.BytesIO()
    frames[0].save(out, format="TIFF", save_all=pages > 1, append_images=frames[1:])
    return out.getvalue()


@pytest.mark.parametrize("mode", ["RGB", "L", "1", "CMYK"])
def test_tiff_is_converted_to_png_for_gemini(mode):
    """Gemini rejects TIFF, so it gets a PNG copy (CMYK etc. converted to RGB)."""
    data, mime = F.gemini_payload(_tiff(mode))
    assert mime == "image/png"
    assert F.sniff(data) is F.PNG


def test_multipage_tiff_sends_first_page_and_says_so(caplog):
    caplog.set_level("WARNING", logger=F.__name__)
    data, mime = F.gemini_payload(_tiff(pages=3))
    assert mime == "image/png"
    assert "pages=3" in caplog.text


def test_readable_formats_pass_through_untouched():
    jpeg = _SAMPLES["jpeg"]
    assert F.gemini_payload(jpeg) == (jpeg, "image/jpeg")


@pytest.mark.filterwarnings("ignore:Corrupt EXIF data")
def test_corrupt_tiff_falls_back_to_original_bytes():
    """A TIFF that cannot be decoded is sent as-is; Gemini rejects it and the
    stage reports itself unavailable, which escalates to a human."""
    broken = b"II*\x00 not really a tiff"
    assert F.gemini_payload(broken) == (broken, "image/tiff")


def _image_bytes(fmt: str, size: tuple[int, int], exif_orientation: int | None = None) -> bytes:
    import io

    from PIL import Image

    img = Image.new("RGB", size, (180, 170, 160))
    out = io.BytesIO()
    kwargs = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif.tobytes()
    img.save(out, format=fmt, **kwargs)
    return out.getvalue()


def _size(data: bytes) -> tuple[int, int]:
    import io

    from PIL import Image

    return Image.open(io.BytesIO(data)).size


@pytest.mark.parametrize(("pil_format", "mime"), [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_large_images_are_shrunk_for_gemini_in_their_own_format(pil_format, mime):
    """Full-resolution scans timed out. Shrunk copies keep their format, so no
    new compression artefacts look like tampering."""
    data, sent_mime = F.gemini_payload(_image_bytes(pil_format, (2480, 3508)))

    assert sent_mime == mime
    assert max(_size(data)) == F.GEMINI_MAX_EDGE


def test_images_within_the_limit_are_sent_untouched():
    original = _image_bytes("JPEG", (1200, 1600))
    assert F.gemini_payload(original) == (original, "image/jpeg")


def test_phone_rotation_is_applied_before_shrinking():
    """Shrinking drops EXIF; a sideways phone photo must not reach Gemini sideways."""
    rotated = _image_bytes("JPEG", (3000, 1000), exif_orientation=6)  # 90 degrees
    data, _ = F.gemini_payload(rotated)

    width, height = _size(data)
    assert height > width
    assert height == F.GEMINI_MAX_EDGE


def test_full_size_can_be_requested():
    original = _image_bytes("PNG", (2480, 3508))
    assert F.gemini_payload(original, max_edge=None) == (original, "image/png")


def test_large_tiff_becomes_a_shrunk_png():
    data, mime = F.gemini_payload(_image_bytes("TIFF", (2480, 3508)))
    assert mime == "image/png"
    assert max(_size(data)) == F.GEMINI_MAX_EDGE


def test_black_and_white_scans_are_smoothed_when_shrunk():
    """Pillow shrinks 1-bit images with nearest-neighbour, breaking up text;
    converting to greyscale first keeps it legible."""
    import io

    from PIL import Image

    # Fine black/white stripes one pixel wide: nearest-neighbour turns them into
    # pure black or white; proper smoothing yields grey.
    stripes = Image.new("1", (4000, 100))
    stripes.putdata([(x % 2) * 255 for _ in range(100) for x in range(4000)])
    out = io.BytesIO()
    stripes.save(out, format="TIFF")

    data, mime = F.gemini_payload(out.getvalue())
    shrunk = Image.open(io.BytesIO(data))

    assert mime == "image/png"
    assert shrunk.mode == "L"
    assert max(shrunk.size) == F.GEMINI_MAX_EDGE
    pixels = sorted(shrunk.tobytes())  # mode L: one byte per pixel
    assert 64 < pixels[len(pixels) // 2] < 192  # mostly grey, not black/white
