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

    sections = [metadata]

    ocr_text = ""
    try:
        import pytesseract
        from PIL import Image

        ocr_text = pytesseract.image_to_string(Image.open(path)).strip()
    except Exception:
        ocr_text = ""
    if ocr_text:
        sections.append(f"OCR text:\n{ocr_text}")

    vision = _describe_image_with_vision(path)
    if vision:
        sections.append(f"AI vision review (not a formal radiology interpretation):\n{vision}")

    if not ocr_text and not vision:
        sections.append(
            "No OCR text extracted and AI vision review was unavailable. "
            "Formal image interpretation should be performed by a qualified clinician."
        )

    return "\n\n".join(sections)


def _describe_image_with_vision(path: Path) -> str:
    """Send the image to a vision-capable model for a radiology-style description.

    Returns an empty string if vision is unavailable (missing key, model error, etc.);
    the caller falls back to OCR/metadata so analysis still proceeds.
    """
    import base64
    import mimetypes
    import os

    try:
        from litellm import completion
    except Exception:
        return ""

    try:
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        data_url = f"data:{mime};base64,{b64}"

        response = completion(
            model=os.getenv("LITELLM_VISION_MODEL", "openai/gpt-4o"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a radiology support assistant reviewing a medical image "
                        "(such as an X-ray, scan, prescription photo, or lab report photo). "
                        "Describe what is visible objectively: the type of image, body region, "
                        "and any notable findings or abnormalities. Do NOT give a definitive "
                        "diagnosis. State clearly that this is not a formal radiology "
                        "interpretation and that a board-certified radiologist should confirm."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this medical image for a clinician."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0,
            max_tokens=600,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return ""
