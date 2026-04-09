"""Voice TTS Studio — FastAPI application entry point."""
import logging
import os
import sys
import threading
import time
import warnings
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Suppress known harmless noise before any heavy imports ───────────────────

# 1) HuggingFace symlink warning (Windows without Developer Mode)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# 2) pydub ffmpeg/ffprobe "Couldn't find" RuntimeWarnings
warnings.filterwarnings("ignore", message=".*ffmpeg.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*ffprobe.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*avconv.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*avprobe.*", category=RuntimeWarning)

# 3) Windows asyncio ConnectionResetError (WinError 10054) — browser closes
#    keep-alive connections; completely harmless but floods the console.
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


# ── Wire up bundled ffmpeg so pydub (used internally by f5-tts) works ────────
def _configure_ffmpeg() -> None:
    """
    imageio-ffmpeg ships a self-contained ffmpeg binary.  We expose it as
    'ffmpeg.exe' in the runtime folder so every subprocess (Whisper,
    torchaudio, pydub) can find it by its conventional name on PATH —
    without requiring a system-wide ffmpeg install.

    Background: imageio-ffmpeg names its binary e.g. 'ffmpeg-win64-v7.0.exe',
    NOT 'ffmpeg.exe'.  Simply adding its directory to PATH is not enough
    because Whisper / torchaudio call subprocess(['ffmpeg', ...]) by name.
    Copying it to runtime\\ffmpeg.exe (next to python.exe) solves this.
    """
    try:
        import imageio_ffmpeg  # bundled binary, installed via requirements.txt
        ffmpeg_src = Path(imageio_ffmpeg.get_ffmpeg_exe())

        # ── Place a canonically-named alias in runtime\ ───────────────────────
        runtime_dir = _PROJECT_ROOT / "runtime"
        ffmpeg_alias = runtime_dir / "ffmpeg.exe"

        if runtime_dir.exists() and not ffmpeg_alias.exists():
            import shutil
            shutil.copy2(str(ffmpeg_src), str(ffmpeg_alias))

        # Use the alias if it exists, otherwise fall back to the original path
        ffmpeg_exe = ffmpeg_alias if ffmpeg_alias.exists() else ffmpeg_src

        # ── Tell pydub's AudioSegment to use this binary ──────────────────────
        from pydub import AudioSegment
        AudioSegment.converter = str(ffmpeg_exe)
        # Modern ffmpeg handles ffprobe-style queries; point both at same binary
        AudioSegment.ffprobe = str(ffmpeg_exe)

        # ── Add runtime\ to PATH so all subprocesses find 'ffmpeg' ───────────
        runtime_str = str(runtime_dir)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if runtime_str not in path_parts:
            os.environ["PATH"] = runtime_str + os.pathsep + os.environ.get("PATH", "")

        print(f"  ffmpeg : {ffmpeg_exe}")
    except Exception as exc:
        # Non-fatal — WAV reference audio still works; only non-WAV formats
        # (MP3, M4A, WebM) will fail when f5-tts tries to load them.
        print(f"  ⚠ ffmpeg not configured ({exc}). Upload WAV reference audio "
              f"or re-run run.bat to install imageio-ffmpeg.")


_configure_ffmpeg()

# ── Late imports (after env vars and ffmpeg are set) ─────────────────────────
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

# Ensure data directories exist
for _folder in ["data/voices", "data/documents", "data/generated"]:
    Path(_folder).mkdir(parents=True, exist_ok=True)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Reset jobs that were interrupted by a previous server crash/restart."""
    from core.db import read_db, write_db

    db = read_db()
    changed = False
    for job in db["jobs"]:
        if job["status"] in ("pending", "processing"):
            job["status"] = "failed"
            job["error"] = "Server restarted while job was running"
            changed = True
    if changed:
        write_db(db)

    yield


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    from api.voices import router as voices_router
    from api.documents import router as documents_router
    from api.generate import router as generate_router

    app = FastAPI(
        title="Voice TTS Studio",
        description="Zero-shot voice cloning and document TTS using F5-TTS",
        lifespan=lifespan,
    )

    app.include_router(voices_router,   prefix="/api/voices",    tags=["voices"])
    app.include_router(documents_router, prefix="/api/documents", tags=["documents"])
    app.include_router(generate_router,  prefix="/api/generate",  tags=["generate"])

    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    def root():
        return RedirectResponse(url="/static/index.html")

    return app


# ── Entry point ───────────────────────────────────────────────────────────────
def open_browser(port: int) -> None:
    time.sleep(1.8)
    webbrowser.open(f"http://localhost:{port}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"\n  Voice TTS Studio  →  http://localhost:{port}\n")

    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
