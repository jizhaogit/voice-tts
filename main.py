"""Voice TTS Studio — FastAPI application entry point."""
import logging
import os
import subprocess
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
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=".*ffmpeg.*",  category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*ffprobe.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*avconv.*",  category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*avprobe.*", category=RuntimeWarning)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


# ── ffmpeg setup (needed to load non-WAV reference audio) ────────────────────
def _configure_ffmpeg() -> None:
    try:
        import imageio_ffmpeg
        ffmpeg_src = Path(imageio_ffmpeg.get_ffmpeg_exe())
        runtime_dir   = _PROJECT_ROOT / "runtime"
        ffmpeg_alias  = runtime_dir / "ffmpeg.exe"
        if runtime_dir.exists() and not ffmpeg_alias.exists():
            import shutil
            shutil.copy2(str(ffmpeg_src), str(ffmpeg_alias))
        ffmpeg_exe = ffmpeg_alias if ffmpeg_alias.exists() else ffmpeg_src
        from pydub import AudioSegment
        AudioSegment.converter = str(ffmpeg_exe)
        AudioSegment.ffprobe   = str(ffmpeg_exe)
        runtime_str = str(runtime_dir)
        if runtime_str not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = runtime_str + os.pathsep + os.environ.get("PATH", "")
        print(f"  ffmpeg : {ffmpeg_exe}")
    except Exception as exc:
        print(f"  ⚠ ffmpeg not configured ({exc}). WAV reference audio still works.")

_configure_ffmpeg()

# ── Late imports ──────────────────────────────────────────────────────────────
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

for _folder in ["data/voices", "data/documents", "data/generated"]:
    Path(_folder).mkdir(parents=True, exist_ok=True)


# ── TTS model pre-loader ──────────────────────────────────────────────────────

def _preload_tts() -> None:
    """Load the active TTS engine at startup so the first request is instant."""
    import traceback
    engine = os.getenv("TTS_ENGINE", "cosyvoice2").lower().strip()
    _log = _PROJECT_ROOT / "data" / f"{engine}.log"
    print(f"  [..] Pre-loading TTS engine: {engine}")
    try:
        if engine == "indextts":
            from core.tts import _get_indextts
            _get_indextts()
        else:
            from core.tts import _get_cosyvoice
            _get_cosyvoice()
    except Exception as exc:
        msg = traceback.format_exc()
        print(f"  ⚠  {engine} preload failed: {exc}")
        print(f"     Full traceback written to: {_log}")
        print("      The model will load on the first TTS request instead.")
        try:
            _log.parent.mkdir(parents=True, exist_ok=True)
            with open(_log, "w", encoding="utf-8") as _f:
                _f.write(f"{engine} load error\n{'='*60}\n")
                _f.write(msg)
        except Exception:
            pass


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the active TTS engine in the background (errors written to data/{engine}.log)
    threading.Thread(target=_preload_tts, daemon=True).start()

    # Reset any interrupted jobs from a previous crash
    from core.db import read_db, write_db
    db = read_db()
    changed = False
    for job in db["jobs"]:
        if job["status"] in ("pending", "processing"):
            job["status"] = "failed"
            job["error"]  = "Server restarted while job was running"
            changed = True
    if changed:
        write_db(db)

    yield


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    from api.voices    import router as voices_router
    from api.documents import router as documents_router
    from api.generate  import router as generate_router

    app = FastAPI(
        title="Voice TTS Studio",
        description="Zero-shot voice cloning with CosyVoice 2",
        lifespan=lifespan,
    )

    app.include_router(voices_router,    prefix="/api/voices",    tags=["voices"])
    app.include_router(documents_router, prefix="/api/documents", tags=["documents"])
    app.include_router(generate_router,  prefix="/api/generate",  tags=["generate"])

    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    def root():
        return RedirectResponse(url="/static/index.html")

    @app.get("/api/config")
    def get_config():
        """Return runtime configuration visible to the frontend."""
        engine = os.getenv("TTS_ENGINE", "cosyvoice2").lower().strip()
        labels = {
            "cosyvoice2": {"name": "CosyVoice 2", "langs": "EN · ZH · JA"},
            "indextts":   {"name": "IndexTTS 1.5", "langs": "EN · ZH"},
        }
        info = labels.get(engine, {"name": engine, "langs": "EN · ZH"})
        return {"engine": engine, "engine_name": info["name"], "engine_langs": info["langs"]}

    return app


# ── Entry point ───────────────────────────────────────────────────────────────
def open_browser(port: int) -> None:
    time.sleep(2.5)
    webbrowser.open(f"http://localhost:{port}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"\n  Voice TTS Studio  →  http://localhost:{port}\n")

    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
