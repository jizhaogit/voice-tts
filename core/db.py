"""Simple JSON file database with thread-safe read/write."""
import json
from pathlib import Path
from threading import Lock

DB_PATH = Path("data/db.json")
_lock = Lock()
_INITIAL: dict = {"voices": [], "documents": [], "jobs": []}


def read_db() -> dict:
    with _lock:
        if not DB_PATH.exists():
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            DB_PATH.write_text(json.dumps(_INITIAL, indent=2, ensure_ascii=False))
            return {k: list(v) for k, v in _INITIAL.items()}
        try:
            return json.loads(DB_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {k: list(v) for k, v in _INITIAL.items()}


def write_db(data: dict) -> None:
    with _lock:
        DB_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
