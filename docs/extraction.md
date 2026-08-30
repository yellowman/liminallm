# file extraction: parsers, readers, and the sandbox

Implementation detail behind SPEC §19.5 (uploads and the vault) and §21.2
(sandboxing untrusted work). The SPEC states the contract - extraction is
sandboxed, unreadable files are refused with a remedy, vision passes never
run in the parsing child. This file records how the current implementation
meets it.

## the extractor ladder

One shared extractor (`service/extract.py`), used by `POST /v1/notes/from-file`
and by RAG ingestion alike. Tiers run cheapest and most faithful first:

- **text bytes** decode directly.
- **`.docx` / `.odt`** extract natively - stdlib zip + xml with a
  decompression budget. Legacy `.doc` is refused with a save-as remedy.
- **pdf** goes through pypdf.

Containers are text, image, or both - decided per page or attachment, the
same rule for pdf and docx/odt alike:

- a pdf page whose text layer holds no real words is rasterized via poppler
  when present (this path reads jbig2/ccitt scans; embedded-image extraction
  is the poppler-less fallback) and spliced back beside the text pages;
- a document's content-bearing embedded images (a size floor drops logos and
  bullets) are read the same way and land beside the typed paragraphs.

Methods compose accordingly and are recorded in the note's provenance:
`pdf+ocr`, `docx-vision`, and so on.

## image readers

Images (png, jpg including cmyk, webp, gif, tiff including multi-page, bmp -
pillow normalizes all of them to what tesseract expects), and scanned pdfs
via their embedded page images, walk a configurable reader roster: the
`extract_readers` admin setting, default `ocr,vision`.

Readers are a registry (`extract.register_reader`), so another OCR engine, a
dedicated OCR model, or a model on new hardware (for example a loom-hosted
reader once its pjrt plugin lands - see docs/jax_backend.md) is a
registration, not a rewrite.

Built-ins:

- **`ocr`** - tesseract. Auto-detected, `liminallm[ocr]` extra; technically
  optional, practically required. Deterministic, free per call, and quotes
  rather than paraphrases.
- **`vision`** - the configured model. One bounded call, with the image
  framed as DATA to read. The capability is probed per backend, never
  assumed: API backends use openai-compatible content parts; a local
  multimodal model implements `transcribe_image`.

"ocr"-kind readers yield to the next reader when they find less than a
document's worth of text; "vision" readers are deliberate readings, accepted
as-is. Files nothing can read are refused with the reason and the remedy,
never stored as garbage.

## the sandbox

Uploads are attacker-controlled bytes, and every parser in the ladder -
pillow's C decoders, pypdf, expat, tesseract + leptonica, poppler - has a
CVE history. All parsing therefore runs in a disposable rlimited child
(`service/sandbox.py`):

- memory, cpu, and file-size caps, inherited by tesseract / pdftoppm
  grandchildren;
- a wall-clock kill;
- a hard pixel ceiling, so decompression bombs raise instead of allocating.

The model's vision pass never runs in that child - it needs the network, but
it never parses. The child hands extracted image bytes back over the pipe as
pending slots (private-use-area markers, stripped from all extracted content
so a file cannot forge a slot), and the parent fills them.

Honest limit, stated in the SPEC and repeated here: the child shares the
server's uid. The sandbox converts api-process compromise into compromise of
a short-lived capped process, not into nothing; the container/vm
recommendation from the interpreter section is the outer wall.
