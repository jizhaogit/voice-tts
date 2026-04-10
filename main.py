"""Voice TTS Studio — FastAPI application entry point."""
import logging
import os
import socket
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


# ── GPT-SoVITS API server management ─────────────────────────────────────────

_GPTSOVITS_PORT = int(os.getenv("GPTSOVITS_PORT", 9880))
_gptsovits_proc: subprocess.Popen | None = None


def _gptsovits_is_running() -> bool:
    """Return True if something is already listening on the GPT-SoVITS port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", _GPTSOVITS_PORT)) == 0


def _start_gptsovits() -> None:
    global _gptsovits_proc

    gts_dir = _PROJECT_ROOT / "gpt-sovits"
    if not (gts_dir / "api_v2.py").exists():
        print("  ⚠ GPT-SoVITS not found at gpt-sovits/api_v2.py.")
        print("    Run run.bat to install it automatically.")
        return

    if _gptsovits_is_running():
        print(f"  [OK] GPT-SoVITS already running on port {_GPTSOVITS_PORT}.")
        return

    # Prefer the portable runtime Python; fall back to the current interpreter
    runtime_py = _PROJECT_ROOT / "runtime" / "python.exe"
    py_exe = str(runtime_py) if runtime_py.exists() else sys.executable

    log_path = _PROJECT_ROOT / "data" / "gptsovits.log"
    log_file = log_path.open("a", encoding="utf-8")

    # Force UTF-8 mode so GPT-SoVITS works on Chinese Windows (cp950/GBK default
    # encoding breaks simplified Chinese characters inside the subprocess).
    gts_env = os.environ.copy()
    gts_env["PYTHONUTF8"]               = "1"
    gts_env["PYTHONIOENCODING"]         = "utf-8"
    gts_env["PYTHONLEGACYWINDOWSSTDIO"] = "0"

    print(f"  [..] Starting GPT-SoVITS API server on port {_GPTSOVITS_PORT} ...")
    _gptsovits_proc = subprocess.Popen(
        [py_exe,
         "-X", "utf8",          # force UTF-8 mode — most reliable, beats env vars
         "api_v2.py",
         "-a", "127.0.0.1",
         "-p", str(_GPTSOVITS_PORT)],
        cwd=str(gts_dir),
        env=gts_env,
        stdout=log_file,
        stderr=log_file,
    )

    # Wait up to 120 s for the server to accept connections
    for elapsed in range(120):
        time.sleep(1)
        if _gptsovits_is_running():
            print(f"  [OK] GPT-SoVITS ready (started in {elapsed + 1}s).")
            print(f"       Log: {log_path}")
            return
        if _gptsovits_proc.poll() is not None:
            print(f"  [ERROR] GPT-SoVITS process exited early (code {_gptsovits_proc.returncode}).")
            print(f"          Check {log_path} for details.")
            _gptsovits_proc = None
            return

    print(f"  [!] GPT-SoVITS did not become ready within 120 s.")
    print(f"      Check {log_path} for details.")


def _stop_gptsovits() -> None:
    global _gptsovits_proc
    if _gptsovits_proc and _gptsovits_proc.poll() is None:
        print("  [..] Stopping GPT-SoVITS API server ...")
        _gptsovits_proc.terminate()
        try:
            _gptsovits_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _gptsovits_proc.kill()
        _gptsovits_proc = None
        print("  [OK] GPT-SoVITS stopped.")


_watchdog_active = False

def _watchdog() -> None:
    """Restart GPT-SoVITS automatically if it crashes."""
    global _watchdog_active
    _watchdog_active = True
    while _watchdog_active:
        time.sleep(5)
        if not _watchdog_active:
            break
        if not _gptsovits_is_running():
            print("  [!] GPT-SoVITS is not responding — restarting ...")
            _start_gptsovits()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start GPT-SoVITS then launch watchdog to auto-restart on crash
    threading.Thread(target=_start_gptsovits, daemon=False).start()
    threading.Thread(target=_watchdog, daemon=True).start()

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

    global _watchdog_active
    _watchdog_active = False
    _stop_gptsovits()


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    from api.voices    import router as voices_router
    from api.documents import router as documents_router
    from api.generate  import router as generate_router

    app = FastAPI(
        title="Voice TTS Studio",
        description="Zero-shot voice cloning with GPT-SoVITS",
        lifespan=lifespan,
    )

    app.include_router(voices_router,    prefix="/api/voices",    tags=["voices"])
    app.include_router(documents_router, prefix="/api/documents", tags=["documents"])
    app.include_router(generate_router,  prefix="/api/generate",  tags=["generate"])

    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    def root():
        return RedirectResponse(url="/static/index.html")

    return app


# ── Entry point ───────────────────────────────────────────────────────────────
def open_browser(port: int) -> None:
    time.sleep(2.5)          # give GPT-SoVITS a head-start
    webbrowser.open(f"http://localhost:{port}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"\n  Voice TTS Studio  →  http://localhost:{port}\n")

    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
