"""Document file formats: one table for detecting, labelling and naming files.

Every stage that needs a file's type goes through here, so the format the
document stage recognises, the MIME type sent to Gemini and the extension used
in Drive cannot drift apart. Detection reads the bytes rather than trusting a
declared MIME type.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileFormat:
    name: str
    mime_type: str
    extension: str
    # Gemini accepts PNG, JPEG, WEBP, HEIC, HEIF and PDF. Anything else is
    # rejected by the API, so its stages will report themselves unavailable.
    gemini_readable: bool


JPEG = FileFormat("jpeg", "image/jpeg", ".jpg", True)
PNG = FileFormat("png", "image/png", ".png", True)
WEBP = FileFormat("webp", "image/webp", ".webp", True)
HEIC = FileFormat("heic", "image/heic", ".heic", True)
HEIF = FileFormat("heif", "image/heif", ".heif", True)
PDF = FileFormat("pdf", "application/pdf", ".pdf", True)
TIFF = FileFormat("tiff", "image/tiff", ".tiff", False)

FORMATS: tuple[FileFormat, ...] = (JPEG, PNG, WEBP, HEIC, HEIF, PDF, TIFF)

# Label for bytes we could not identify. Honest, unlike defaulting to JPEG.
UNKNOWN_MIME_TYPE = "application/octet-stream"

# ISO-BMFF major brands (bytes 8-12, after "ftyp" at 4-8).
_HEIC_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis"}
_HEIF_BRANDS = {b"mif1", b"msf1"}

_BY_MIME = {f.mime_type: f for f in FORMATS} | {"image/jpg": JPEG, "image/tif": TIFF}


def sniff(data: bytes) -> FileFormat | None:
    """Identify a file from its leading bytes, or None if unrecognised."""
    if data[:2] == b"\xff\xd8":
        return JPEG
    if data[:4] == b"\x89PNG":
        return PNG
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return WEBP
    if data[:4] == b"%PDF":
        return PDF
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return TIFF
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in _HEIC_BRANDS:
            return HEIC
        if brand in _HEIF_BRANDS:
            return HEIF
    return None


def by_mime(mime_type: str | None) -> FileFormat | None:
    """Look up a declared MIME type (case- and parameter-insensitive)."""
    if not mime_type:
        return None
    return _BY_MIME.get(mime_type.split(";", 1)[0].strip().lower())


def mime_type_of(data: bytes) -> str:
    """MIME type to declare when sending these bytes to an API."""
    fmt = sniff(data)
    return fmt.mime_type if fmt else UNKNOWN_MIME_TYPE


# Longest side, in pixels, of an image sent to Gemini. Full-resolution scans
# (~2500x3500, 2-3 MB) timed out; 2000 px keeps handwriting legible (about
# 170 dpi on A4). Smaller images are sent untouched.
GEMINI_MAX_EDGE = 2000

# How a shrunk image is re-saved: in its own format, so no new compression
# artefacts appear that a fraud check could mistake for tampering.
_SAVE_AS = {
    "jpeg": ("JPEG", {"quality": 92}),
    "png": ("PNG", {}),
    "webp": ("WEBP", {"quality": 92}),
}
# PNG can hold these Pillow modes as-is; anything else (CMYK, 16-bit, ...) is
# converted to RGB first.
_PNG_MODES = {"1", "L", "LA", "P", "RGB", "RGBA"}


def _savable(frame, pil_format: str):
    if pil_format == "JPEG":
        return frame if frame.mode in ("RGB", "L", "CMYK") else frame.convert("RGB")
    if pil_format == "WEBP":
        return frame if frame.mode in ("RGB", "RGBA") else frame.convert("RGB")
    return frame if frame.mode in _PNG_MODES else frame.convert("RGB")


def _prepare(data: bytes, fmt: FileFormat, max_edge: int | None) -> tuple[bytes, str]:
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as img:
        too_big = max_edge is not None and max(img.size) > max_edge
        if fmt.gemini_readable and not too_big:
            return data, fmt.mime_type

        pages = getattr(img, "n_frames", 1)
        if pages > 1:
            logger.warning("file_formats.multipage pages=%d sending first page to Gemini", pages)
        img.seek(0)
        # Shrinking drops EXIF, including a phone photo's rotation, so apply it
        # first or a sideways photo reaches Gemini sideways.
        frame = ImageOps.exif_transpose(img)
        if too_big:
            if frame.mode == "1":
                # Pillow forces nearest-neighbour on 1-bit images, which breaks
                # up thin strokes in black-and-white scans; greyscale lets
                # LANCZOS smooth them so text stays legible.
                frame = frame.convert("L")
            frame.thumbnail((max_edge, max_edge), Image.LANCZOS)

        if fmt.gemini_readable and fmt.name in _SAVE_AS:
            (pil_format, options), target = _SAVE_AS[fmt.name], fmt
        else:  # TIFF, or a format Pillow cannot write (HEIC)
            (pil_format, options), target = ("PNG", {}), PNG
        out = io.BytesIO()
        _savable(frame, pil_format).save(out, format=pil_format, **options)
        return out.getvalue(), target.mime_type


def gemini_payload(data: bytes, *, max_edge: int | None = GEMINI_MAX_EDGE) -> tuple[bytes, str]:
    """Bytes and MIME type to send to Gemini.

    - Images Gemini reads that fit within max_edge pass through untouched.
    - Larger ones are shrunk to fit, in their own format.
    - Formats Gemini rejects (TIFF) become PNG, first page only.
    - PDFs pass through.

    Only the copy sent to Gemini changes: forensic checks and the Drive upload
    keep the original bytes. If an image cannot be decoded (corrupt, or HEIC
    without a Pillow plugin) the original is sent as-is.
    """
    fmt = sniff(data)
    if fmt is None:
        return data, UNKNOWN_MIME_TYPE
    if fmt is PDF:
        return data, fmt.mime_type
    try:
        return _prepare(data, fmt, max_edge)
    except Exception as exc:  # Pillow missing, corrupt file, decompression bomb, HEIC
        logger.warning("file_formats.prepare_failed format=%s error=%s", fmt.name, exc)
        return data, fmt.mime_type
