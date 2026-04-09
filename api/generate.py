"""TTS generation jobs — async inference using F5-TTS."""
import threading
import uuid
from datetime import datetime
from pathlib import Path

import urllib.parse

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.db import read_db, write_db

router = APIRouter()
GENERATED_DIR = Path("data/generated")


class GenerateRequest(BaseModel):
    voice_id: str
    document_id: str
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    remove_silence: bool = False


# ── Background job runner ────────────────────────────────────────────────────

def _update_job(job_id: str, updates: dict) -> None:
    db = read_db()
    job = next((j for j in db["jobs"] if j["id"] == job_id), None)
    if job:
        job.update(updates)
        write_db(db)


def _run_job(job_id: str, voice: dict, text: str, speed: float, remove_silence: bool = False) -> None:
    from core.tts import generate_speech

    def progress(current: int, total: int):
        _update_job(job_id, {"processed_chunks": current, "total_chunks": total})

    try:
        _update_job(job_id, {"status": "processing"})

        ref_audio = str(Path("data/voices") / voice["audio_file"])
        audio_bytes, fmt = generate_speech(
            ref_audio_path=ref_audio,
            ref_text=voice["ref_text"],
            gen_text=text,
            speed=speed,
            remove_silence=remove_silence,
            progress_cb=progress,
        )

        out_path = GENERATED_DIR / f"{job_id}.{fmt}"
        out_path.write_bytes(audio_bytes)

        _update_job(
            job_id,
            {
                "status": "ready",
                "output_file": out_path.name,
                "format": fmt,
                "file_size_bytes": len(audio_bytes),
                "completed_at": datetime.utcnow().isoformat(),
            },
        )
    except Exception as exc:
        _update_job(job_id, {"status": "failed", "error": str(exc)})


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/")
def list_jobs():
    db = read_db()
    return list(reversed(db["jobs"]))


@router.post("/")
def create_job(req: GenerateRequest):
    db = read_db()
    voice = next((v for v in db["voices"] if v["id"] == req.voice_id), None)
    doc = next((d for d in db["documents"] if d["id"] == req.document_id), None)

    if not voice:
        raise HTTPException(404, "Voice not found")
    if not doc:
        raise HTTPException(404, "Document not found")
    if not doc.get("extracted_text", "").strip():
        raise HTTPException(400, "Document has no extractable text")

    job = {
        "id": str(uuid.uuid4()),
        "voice_id": req.voice_id,
        "document_id": req.document_id,
        "voice_name": voice["name"],
        "document_name": doc["original_name"],
        "language": voice.get("language", "en"),
        "speed": req.speed,
        "remove_silence": req.remove_silence,
        "status": "pending",
        "total_chunks": 0,
        "processed_chunks": 0,
        "output_file": None,
        "format": None,
        "file_size_bytes": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }

    db["jobs"].append(job)
    write_db(db)

    text = doc["extracted_text"]
    threading.Thread(
        target=_run_job, args=(job["id"], voice, text, req.speed, req.remove_silence), daemon=True
    ).start()

    return {"job_id": job["id"], "status": "pending"}


@router.get("/{job_id}")
def get_job(job_id: str):
    db = read_db()
    job = next((j for j in db["jobs"] if j["id"] == job_id), None)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/{job_id}/audio")
def download_audio(job_id: str, inline: bool = False):
    db = read_db()
    job = next((j for j in db["jobs"] if j["id"] == job_id), None)
    if not job or job["status"] != "ready":
        raise HTTPException(404, "Audio not ready")

    path = GENERATED_DIR / job["output_file"]
    if not path.exists():
        raise HTTPException(404, "Audio file missing from disk")

    fmt = job.get("format", "mp3")
    media = "audio/mpeg" if fmt == "mp3" else "audio/wav"
    raw_name = (
        f"{job['voice_name']}-{Path(job['document_name']).stem}.{fmt}"
        .replace(" ", "_")
        .replace("/", "-")
    )

    # HTTP headers are latin-1 only.  Voice / document names may contain
    # Chinese, Japanese, etc.  Use RFC 5987 encoding so all characters work.
    disposition = "inline" if inline else "attachment"
    try:
        raw_name.encode("latin-1")          # pure ASCII / latin-1 → simple form
        cd = f'{disposition}; filename="{raw_name}"'
    except UnicodeEncodeError:
        # Strip to ASCII for the fallback, keep full name in filename*
        ascii_name = raw_name.encode("ascii", errors="ignore").decode() or f"audio.{fmt}"
        encoded    = urllib.parse.quote(raw_name, safe="")
        cd = f'{disposition}; filename="{ascii_name}"; filename*=UTF-8\'\'{encoded}'

    # Do NOT pass filename= to FileResponse — it creates a duplicate
    # Content-Disposition header that confuses browsers (wrong MIME / name).
    return FileResponse(str(path), media_type=media,
                        headers={"Content-Disposition": cd})


@router.delete("/{job_id}")
def delete_job(job_id: str):
    db = read_db()
    job = next((j for j in db["jobs"] if j["id"] == job_id), None)
    if not job:
        raise HTTPException(404, "Job not found")

    if job.get("output_file"):
        (GENERATED_DIR / job["output_file"]).unlink(missing_ok=True)

    db["jobs"] = [j for j in db["jobs"] if j["id"] != job_id]
    write_db(db)
    return {"success": True}
