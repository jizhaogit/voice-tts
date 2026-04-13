"""Voice profile management.

A 'voice' in CosyVoice 2 is defined by:
  - A reference audio clip (3-15 sec, clear speech, no music/noise)
  - The exact transcript of that clip (ref_text)
  - Language tag: en | zh | ja

No model training needed — inference is zero-shot.
"""
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from core.db import read_db, write_db

router = APIRouter()
VOICES_DIR = Path("data/voices")
ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".webm"}


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("/")
def list_voices():
    return read_db()["voices"]


# ── Create ───────────────────────────────────────────────────────────────────

@router.post("/")
async def create_voice(
    name: str = Form(...),
    ref_text: str = Form(...),
    language: str = Form("en"),
    description: str = Form(""),
    file: UploadFile = File(...),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_AUDIO:
        raise HTTPException(
            400, f"Audio only ({', '.join(ALLOWED_AUDIO)}). Got: {ext}"
        )

    voice_id = str(uuid.uuid4())
    audio_path = VOICES_DIR / f"{voice_id}{ext}"

    with open(audio_path, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    voice = {
        "id": voice_id,
        "name": name.strip(),
        "description": description.strip(),
        "ref_text": ref_text.strip(),
        "language": language,
        "audio_file": audio_path.name,
        "size_bytes": audio_path.stat().st_size,
        "created_at": datetime.utcnow().isoformat(),
    }

    db = read_db()
    db["voices"].append(voice)
    write_db(db)
    return voice


# ── Auto-transcribe ───────────────────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe_reference(
    language: str = Form("en"),
    file: UploadFile = File(...),
):
    """Transcribe an audio file with Whisper to auto-fill the reference text."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_AUDIO:
        raise HTTPException(400, "Audio files only")

    tmp_path = VOICES_DIR / f"tmp_{uuid.uuid4()}{ext}"
    try:
        with open(tmp_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)

        from core.tts import transcribe_audio

        text = transcribe_audio(str(tmp_path), language)
        return {"text": text}
    except Exception as exc:
        raise HTTPException(500, str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Stream reference audio ────────────────────────────────────────────────────

@router.get("/{voice_id}/audio")
def get_voice_audio(voice_id: str):
    db = read_db()
    voice = next((v for v in db["voices"] if v["id"] == voice_id), None)
    if not voice:
        raise HTTPException(404, "Voice not found")
    path = VOICES_DIR / voice["audio_file"]
    if not path.exists():
        raise HTTPException(404, "Audio file missing from disk")
    ext = path.suffix.lower()
    media = "audio/wav" if ext == ".wav" else "audio/mpeg"
    return FileResponse(str(path), media_type=media)


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/{voice_id}")
def delete_voice(voice_id: str):
    db = read_db()
    voice = next((v for v in db["voices"] if v["id"] == voice_id), None)
    if not voice:
        raise HTTPException(404, "Voice not found")

    path = VOICES_DIR / voice["audio_file"]
    path.unlink(missing_ok=True)

    db["voices"] = [v for v in db["voices"] if v["id"] != voice_id]
    write_db(db)
    return {"success": True}
