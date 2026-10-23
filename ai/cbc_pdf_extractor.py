
from pathlib import Path
from pypdf import PdfReader


def extract_cbc_pdf_text(pdf_path: str) -> str:
    """Extract text from a text-based PDF CBC report."""
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError("CBC PDF report was not found.")

    try:
        reader = PdfReader(str(path))

        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        ).strip()

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "Could not read the CBC PDF report."
        ) from exc

    if not text:
        raise ValueError(
            "No selectable text found. This PDF may be scanned; OCR is not implemented yet."
        )

    return text

def inspect_cbc_pdf(pdf_path: str) -> dict:
    """Extract PDF text and report whether it contains usable text."""
    text = extract_cbc_pdf_text(pdf_path)

    return {
        "text_length": len(text),
        "text_preview": text[:3000],
        "needs_manual_review": True,
        "message": (
            "Extracted text must be reviewed before measurements "
            "and laboratory reference ranges are used for analysis."
        ),
    }
