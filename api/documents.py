"""Document upload, text extraction, and storage."""
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from core.db import read_db, write_db
from core.parsers import extract_text

router = APIRouter()
DOCS_DIR = Path("data/documents")
ALLOWED_EXT = {".txt", ".pdf", ".docx", ".html", ".htm"}


def _strip_text(doc: dict) -> dict:
    """Return document metadata without the (potentially huge) extracted_text."""
    return {k: v for k, v in doc.items() if k != "extracted_text"}


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("/")
def list_documents():
    db = read_db()
    return [_strip_text(d) for d in db["documents"]]


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            400, f"Supported: {', '.join(ALLOWED_EXT)}. Got: {ext}"
        )

    doc_id = str(uuid.uuid4())
    stored_path = DOCS_DIR / f"{doc_id}{ext}"

    with open(stored_path, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        text = extract_text(str(stored_path), file.filename)
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Text extraction failed: {exc}")

    word_count = len(text.split())

    doc = {
        "id": doc_id,
        "original_name": file.filename,
        "stored_file": stored_path.name,
        "size_bytes": stored_path.stat().st_size,
        "word_count": word_count,
        "char_count": len(text),
        "extracted_text": text,
        "uploaded_at": datetime.utcnow().isoformat(),
    }

    db = read_db()
    db["documents"].append(doc)
    write_db(db)
    return _strip_text(doc)


# ── Get text ──────────────────────────────────────────────────────────────────

@router.get("/{doc_id}/text")
def get_document_text(doc_id: str):
    db = read_db()
    doc = next((d for d in db["documents"] if d["id"] == doc_id), None)
    if not doc:
        raise HTTPException(404, "Document not found")
    return {"text": doc["extracted_text"], "word_count": doc["word_count"]}


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    db = read_db()
    doc = next((d for d in db["documents"] if d["id"] == doc_id), None)
    if not doc:
        raise HTTPException(404, "Document not found")

    path = DOCS_DIR / doc["stored_file"]
    path.unlink(missing_ok=True)

    db["documents"] = [d for d in db["documents"] if d["id"] != doc_id]
    write_db(db)
    return {"success": True}
