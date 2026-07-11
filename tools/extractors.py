from __future__ import annotations

from pathlib import Path


def extract_upload_content(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".png", ".jpg", ".jpeg"}:
        return _extract_image(path)
    return f"Unsupported file type: {path.name}"


def _extract_pdf(path: Path) -> str:
    try:
        import pdfplumber

        parts = []
        with pdfplumber.open(path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(f"Page {index}:\n{text}")
        if parts:
            return "\n\n".join(parts)
    except Exception:
        pass

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"Page {index}:\n{text}")
        return "\n\n".join(pages) or f"No text extracted from PDF: {path.name}"
    except Exception as exc:
        return f"Could not extract PDF text from {path.name}: {exc}"


def _extract_image(path: Path) -> str:
    try:
        from PIL import Image

        image = Image.open(path)
        metadata = f"Image file: {path.name}; format={image.format}; size={image.size[0]}x{image.size[1]}"
    except Exception as exc:
        metadata = f"Image file: {path.name}; metadata unavailable: {exc}"

    try:
        import pytesseract
        from PIL import Image

        text = pytesseract.image_to_string(Image.open(path)).strip()
        if text:
            return f"{metadata}\n\nOCR text:\n{text}"
    except Exception:
        pass

    return (
        f"{metadata}\n\nNo OCR text extracted. The radiology agent can discuss limitations, "
        "but formal image interpretation should be performed by a qualified clinician."
    )
