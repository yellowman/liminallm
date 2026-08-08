"""Getting text out of uploaded files, which are attacker-controlled bytes.

Every parser behind this — Pillow's C decoders, pypdf, expat, tesseract,
poppler — is treated as compromisable, so extraction runs in a disposable
rlimited child. These drive the real parsers (all present here: Pillow, pypdf,
tesseract, pdftoppm) rather than stubbing them, because a stub cannot tell you
the sandbox actually held.
"""

from __future__ import annotations

import zipfile

import pytest

from liminallm.service.extract import (
    _PH_CLOSE,
    _PH_OPEN,
    MAX_IMAGE_PIXELS,
    ExtractError,
    _compose_method,
    _placeholder,
    _strip_markers,
    extract_text,
    ocr_available,
    parse_reader_order,
    rasterizer_available,
)

# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def test_a_text_file_comes_back_verbatim(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello, world\nsecond line")
    out = extract_text(f)
    assert out["text"] == "hello, world\nsecond line"
    assert out["method"] == "text"


def test_utf8_survives(tmp_path):
    f = tmp_path / "u.md"
    f.write_text("héllo — ünïcode ✓", encoding="utf-8")
    assert "ünïcode" in extract_text(f)["text"]


def test_binary_bytes_are_refused_rather_than_stored_as_garbage(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(bytes(range(256)) * 40)
    with pytest.raises(ExtractError, match="binary"):
        extract_text(f)


def test_an_empty_file_is_refused(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    with pytest.raises(ExtractError):
        extract_text(f)


def test_legacy_doc_says_what_to_do_instead(tmp_path):
    f = tmp_path / "old.doc"
    f.write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 200)
    with pytest.raises(ExtractError, match="docx"):
        extract_text(f)


# ---------------------------------------------------------------------------
# Placeholder markers: file content must not be able to forge a vision slot
# ---------------------------------------------------------------------------


def test_a_file_cannot_forge_an_image_slot(tmp_path):
    """The markers are private-use codepoints, and a document that contains
    them would otherwise be able to point at, or blank out, a real slot."""
    f = tmp_path / "hostile.txt"
    f.write_text(f"before {_PH_OPEN}0{_PH_CLOSE} after {_placeholder(7)} end")
    text = extract_text(f)["text"]
    assert _PH_OPEN not in text and _PH_CLOSE not in text
    assert "before" in text and "end" in text


def test_strip_markers_removes_both_ends():
    assert _strip_markers(f"a{_PH_OPEN}1{_PH_CLOSE}b") == "a1b"


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------


def _text_pdf(path, body="The quick brown fox jumps over the lazy dog."):
    """A one-page PDF with a real text layer, written by hand.

    Hand-built rather than generated so the test does not need a PDF writer
    dependency; pypdf reads it the same way it reads any other.
    """
    stream = f"BT /F1 18 Tf 40 700 Td ({body}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n".encode()
        + b"%%EOF\n"
    )
    path.write_bytes(bytes(out))
    return path


def test_a_pdf_with_a_text_layer_is_read_without_ocr(tmp_path):
    """The cheapest tier: pypdf, no rasterizing, no OCR."""
    out = extract_text(_text_pdf(tmp_path / "doc.pdf"))
    assert "quick brown fox" in out["text"]
    assert out["method"] == "pdf"


def test_a_pdf_that_is_not_a_pdf_is_refused(tmp_path):
    f = tmp_path / "lying.pdf"
    f.write_bytes(b"not a pdf at all, just bytes pretending" * 20)
    with pytest.raises(ExtractError):
        extract_text(f)


def test_a_truncated_pdf_is_refused_not_crashed(tmp_path):
    full = _text_pdf(tmp_path / "whole.pdf").read_bytes()
    f = tmp_path / "cut.pdf"
    f.write_bytes(full[: len(full) // 2])
    with pytest.raises(ExtractError):
        extract_text(f)


# ---------------------------------------------------------------------------
# Images and OCR
# ---------------------------------------------------------------------------


def _image_of_text(path, words="HELLO WORLD", size=(600, 200)):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    # The default bitmap font is small but tesseract reads it when scaled up.
    draw.text((20, 80), words, fill="black")
    img = img.resize((size[0] * 3, size[1] * 3), Image.LANCZOS)
    img.save(path)
    return path


@pytest.mark.skipif(not ocr_available(), reason="tesseract not installed")
@pytest.mark.slow
def test_an_image_of_text_is_read_by_ocr(tmp_path):
    out = extract_text(_image_of_text(tmp_path / "sign.png"))
    assert out["method"] == "ocr"
    assert "HELLO" in out["text"].upper()


@pytest.mark.slow
def test_an_unreadable_image_says_what_to_install(tmp_path):
    """A blank image yields nothing, and the refusal carries the remedy rather
    than storing an empty note."""
    from PIL import Image

    f = tmp_path / "blank.png"
    Image.new("RGB", (300, 300), "white").save(f)
    with pytest.raises(ExtractError) as caught:
        extract_text(f)
    assert "tesseract" in str(caught.value).lower()


@pytest.mark.slow
def test_a_decompression_bomb_is_refused_before_it_allocates(tmp_path):
    """A header claiming more pixels than MAX_IMAGE_PIXELS must raise inside
    the child rather than allocating gigabytes. Pillow's own default is to
    warn and allocate anyway."""
    from PIL import Image

    f = tmp_path / "bomb.png"
    side = int(MAX_IMAGE_PIXELS**0.5) + 2000
    # A solid image compresses to a few KB while claiming ~70MP.
    Image.new("L", (side, side), 0).save(f, optimize=True)
    assert f.stat().st_size < 2 * 1024 * 1024, "fixture must stay small on disk"

    with pytest.raises(ExtractError):
        extract_text(f)


# ---------------------------------------------------------------------------
# Office documents
# ---------------------------------------------------------------------------


def _docx(path, paragraphs=("First paragraph.", "Second paragraph.")):
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    doc = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", doc)
    return path


def test_a_docx_yields_its_paragraphs(tmp_path):
    out = extract_text(_docx(tmp_path / "d.docx"))
    assert "First paragraph." in out["text"]
    assert "Second paragraph." in out["text"]
    assert out["method"].startswith("docx")


def test_a_docx_that_is_not_a_zip_is_refused(tmp_path):
    f = tmp_path / "fake.docx"
    f.write_bytes(b"PK\x03\x04" + b"\x00" * 400)
    with pytest.raises(ExtractError):
        extract_text(f)


def test_an_xml_bomb_in_a_docx_does_not_hang_the_server(tmp_path):
    """Billion laughs. Whether expat refuses the entities or the child is
    killed by its own limits, what must not happen is the parent blocking."""
    bomb = (
        '<?xml version="1.0"?><!DOCTYPE lolz ['
        '<!ENTITY lol "lol">'
        + "".join(
            f'<!ENTITY lol{i} "{"&lol%d;" % (i - 1) * 10}">' for i in range(1, 10)
        )
        + "]>"
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>&lol9;</w:t></w:r></w:p></w:body></w:document>"
    )
    f = tmp_path / "bomb.docx"
    with zipfile.ZipFile(f, "w") as zf:
        zf.writestr("word/document.xml", bomb)
    with pytest.raises(ExtractError):
        extract_text(f)


def test_a_zip_bomb_is_bounded_by_the_size_cap(tmp_path):
    """A docx whose inner XML expands enormously: the child's file-size and
    memory rlimits are what stop it, not a check on the archive."""
    f = tmp_path / "big.docx"
    with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", "<w:t>" + "A" * (80 * 1024 * 1024) + "</w:t>")
    assert f.stat().st_size < 1024 * 1024, "fixture must stay small on disk"
    with pytest.raises(ExtractError):
        extract_text(f)


# ---------------------------------------------------------------------------
# The reader roster
# ---------------------------------------------------------------------------


def test_the_default_roster_tries_ocr_before_vision():
    """Cheapest and most faithful first: OCR quotes the document, vision reads
    it. Order is a deployment choice, but this is the shipped one."""
    assert parse_reader_order(None) == ("ocr", "vision")
    assert parse_reader_order("") == ("ocr", "vision")


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("vision", ("vision",)),
        ("vision,ocr", ("vision", "ocr")),
        (" ocr , vision ", ("ocr", "vision")),
        (",,,", ("ocr", "vision")),
    ],
)
def test_the_roster_can_be_reordered(spec, expected):
    assert parse_reader_order(spec) == expected


def test_an_unknown_reader_name_does_not_crash_extraction(tmp_path):
    """A typo in EXTRACT_READERS should not take text files down with it."""
    f = tmp_path / "n.txt"
    f.write_text("plain text is read without any reader at all")
    assert extract_text(f, readers=("nosuchreader",))["text"].startswith("plain")


# ---------------------------------------------------------------------------
# How the method string is composed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "container, mech, had_text, expected",
    [
        ("text", None, True, "text"),
        ("pdf", None, True, "pdf"),
        # "+" when the container also had its own text, "-" when it did not:
        # a scanned PDF read entirely by OCR is not the same as a PDF whose
        # text layer was supplemented.
        ("pdf", "ocr", True, "pdf+ocr"),
        ("pdf", "ocr", False, "pdf-ocr"),
        ("docx", "vision", False, "docx-vision"),
        ("image", "ocr", False, "ocr"),
        ("image", None, False, "unread"),
    ],
)
def test_the_method_says_how_the_text_was_obtained(container, mech, had_text, expected):
    assert _compose_method(container, mech, had_text) == expected


# ---------------------------------------------------------------------------
# Capability probes
# ---------------------------------------------------------------------------


def test_the_probes_answer_without_raising():
    """Both are called to decide whether a deployment can read images at all,
    so they must answer on a box with nothing installed."""
    assert isinstance(ocr_available(), bool)
    assert isinstance(rasterizer_available(), bool)


def test_unsandboxed_parsing_agrees_with_sandboxed(tmp_path):
    """sandbox=False is the debugging path; if it diverged, debugging would
    be chasing a different program."""
    f = tmp_path / "same.txt"
    f.write_text("identical either way")
    assert extract_text(f, sandbox=False)["text"] == extract_text(f)["text"]


def test_parsing_really_happens_in_a_child_process(tmp_path):
    """The whole security model rests on this. If extraction ran in-process,
    a Pillow or pypdf compromise would be an API-process compromise."""
    import inspect

    from liminallm.service import extract

    source = inspect.getsource(extract.extract_text)
    assert "run_in_sandbox(" in source
