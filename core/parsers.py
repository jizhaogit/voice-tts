"""Text extraction from TXT, PDF, DOCX, and HTML documents."""
import re
from pathlib import Path


def extract_text(file_path: str, original_name: str) -> str:
    """Return plain text extracted from the given file."""
    ext = Path(original_name).suffix.lower()
    path = Path(file_path)

    if ext == ".txt":
        # Detect encoding from BOM first, then try common Chinese encodings.
        # Windows Notepad writes UTF-16 LE BOM (FF FE) or UTF-8 BOM (EF BB BF).
        raw = path.read_bytes()
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
            enc_order = ("utf-16",)          # has BOM — trust it
        elif raw[:3] == b"\xef\xbb\xbf":
            enc_order = ("utf-8-sig",)       # UTF-8 BOM
        else:
            enc_order = ("utf-8", "gbk", "gb18030", "big5", "cp936")

        for enc in enc_order:
            try:
                text = raw.decode(enc).lstrip("\ufeff").strip()
                if text:
                    return text
            except (UnicodeDecodeError, LookupError):
                continue

        # Final fallback — replace undecodable bytes rather than crash
        return raw.decode("utf-8", errors="replace").strip()

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
