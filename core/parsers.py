"""Text extraction from TXT, PDF, DOCX, and HTML documents."""
import re
from pathlib import Path


def extract_text(file_path: str, original_name: str) -> str:
    """Return plain text extracted from the given file."""
    ext = Path(original_name).suffix.lower()
    path = Path(file_path)

    if ext == ".txt":
        return path.read_text(encoding="utf-8-sig").strip()

    if ext == ".pdf":
        import pypdf

        parts: list[str] = []
        with open(file_path, "rb") as fh:
            reader = pypdf.PdfReader(fh)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()

    if ext == ".docx":
        from docx import Document

        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()

    if ext in (".html", ".htm"):
        from bs4 import BeautifulSoup

        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "meta", "link", "noscript"]):
            tag.decompose()
        raw = soup.get_text(separator="\n")
        return re.sub(r"\n{3,}", "\n\n", raw).strip()

    raise ValueError(f"Unsupported file type: {ext}")
