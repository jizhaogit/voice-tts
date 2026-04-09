"""Voice TTS Studio — FastAPI application entry point."""
import os
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

# Ensure all data directories exist on startup
for folder in ["data/voices", "data/documents", "data/generated"]:
    Path(folder).mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Reset any jobs that were left pending/processing from a previous run."""
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

    yield  # app runs here


def create_app() -> FastAPI:
    from api.voices import router as voices_router
    from api.documents import router as documents_router
    from api.generate import router as generate_router

    app = FastAPI(
        title="Voice TTS Studio",
        description="Zero-shot voice cloning and document TTS using F5-TTS",
        lifespan=lifespan,
    )

    app.include_router(voices_router, prefix="/api/voices", tags=["voices"])
    app.include_router(documents_router, prefix="/api/documents", tags=["documents"])
    app.include_router(generate_router, prefix="/api/generate", tags=["generate"])

    # Serve frontend
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    def root():
        return RedirectResponse(url="/static/index.html")

    return app


def open_browser(port: int):
    time.sleep(1.8)
    webbrowser.open(f"http://localhost:{port}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"\n  Voice TTS Studio  →  http://localhost:{port}\n")

    # Open browser in background thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="warning")
