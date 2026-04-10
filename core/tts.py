"""GPT-SoVITS wrapper — zero-shot voice cloning via local API server.

Architecture
------------
GPT-SoVITS runs as a separate process (started by main.py) that listens
on http://127.0.0.1:9880.  This module calls that API with the reference
audio path + transcript and receives raw WAV bytes in return.

GPT-SoVITS handles text chunking internally (text_split_method="cut5"),
so none of the F5-TTS chunking/alignment complexity is needed here.
"""
_TTS_VERSION = "2026-04-09-h"

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Console + file tee  (keeps full log in  data/tts_run.log)
# ---------------------------------------------------------------------------

class _Tee:
    """Mirror every print() to both the real stdout and a UTF-8 log file."""

    def __init__(self, log_path: Path):
        self._log_path = log_path
        self._real_stdout = sys.stdout
        self._file = None

    def __enter__(self):
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._log_path.open("a", encoding="utf-8", buffering=1)
        import datetime
        self._file.write(
            f"\n{'='*60}\n"
            f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
            f"tts.py {_TTS_VERSION}\n"
            f"{'='*60}\n"
        )
        sys.stdout = self
        return self

    def write(self, data: str):
        self._real_stdout.write(data)
        if self._file:
            self._file.write(data)

    def flush(self):
        self._real_stdout.flush()
        if self._file:
            self._file.flush()

    def __exit__(self, *_):
        sys.stdout = self._real_stdout
        if self._file:
            self._file.close()
            self._file = None

    def __getattr__(self, name: str):
        return getattr(self._real_stdout, name)


# ---------------------------------------------------------------------------
# ffmpeg helpers  (needed to convert MP3/M4A reference audio → WAV)
# ---------------------------------------------------------------------------

def _get_ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    runtime_ffmpeg = Path(__file__).parent.parent / "runtime" / "ffmpeg.exe"
    if runtime_ffmpeg.exists():
        return str(runtime_ffmpeg)
    return None


def _to_wav(audio_path: str, target_sr: int = 24000) -> tuple[str, bool]:
    """Convert any audio file to a 24 kHz mono WAV.

    Returns (path, is_temp).  Caller must delete the file when is_temp=True.
    WAV files are returned unchanged (is_temp=False).
    """
    src = Path(audio_path)
    if src.suffix.lower() == ".wav":
        return audio_path, False

    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe is None:
        print("  ⚠ ffmpeg not found — using audio as-is. "
              "Re-run run.bat to install imageio-ffmpeg.")
        return audio_path, False

    tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
    try:
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", str(src),
             "-ar", str(target_sr), "-ac", "1", "-f", "wav", str(tmp_wav)],
            capture_output=True,
            check=True,
        )
        return str(tmp_wav), True
    except Exception as exc:
        print(f"  ⚠ Audio conversion failed: {exc}")
        return audio_path, False


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def _detect_lang(text: str) -> str:
    """Return GPT-SoVITS language tag for the dominant script in *text*."""
    zh = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    ja = sum(1 for c in text if "\u3040" <= c <= "\u30ff")
    en = sum(1 for c in text if c.isascii() and c.isalpha())
    if zh >= en and zh >= ja:
        return "zh"
    if ja > en:
        return "ja"
    return "en"


# ---------------------------------------------------------------------------
# Public API  (same signature as the old F5-TTS wrapper)
# ---------------------------------------------------------------------------

_GPTSOVITS_URL = os.getenv("GPTSOVITS_URL", "http://127.0.0.1:9880")


def generate_speech(
    ref_audio_path: str,
    ref_text: str,
    gen_text: str,
    speed: float = 1.0,
    remove_silence: bool = False,
    progress_cb=None,
) -> tuple[bytes, str]:
    """Generate speech and return (wav_bytes, 'wav')."""
    _log_path = Path(__file__).parent.parent / "data" / "tts_run.log"
    with _Tee(_log_path):
        return _generate_speech_inner(
            ref_audio_path, ref_text, gen_text, speed, remove_silence, progress_cb
        )


def _generate_speech_inner(
    ref_audio_path: str,
    ref_text: str,
    gen_text: str,
    speed: float = 1.0,
    remove_silence: bool = False,
    progress_cb=None,
) -> tuple[bytes, str]:
    import requests as _requests

    print(f"  [tts.py {_TTS_VERSION}]  engine=GPT-SoVITS")

    gen_text = gen_text.strip()
    if not gen_text:
        raise ValueError("gen_text is empty — nothing to synthesise.")

    wav_ref, wav_is_temp = _to_wav(ref_audio_path)

    try:
        text_lang   = _detect_lang(gen_text)
        prompt_lang = _detect_lang(ref_text)

        print(f"  text_lang={text_lang}  prompt_lang={prompt_lang}  "
              f"speed={speed}  chars={len(gen_text)}")

        if progress_cb:
            progress_cb(0, 1)

        payload = {
            "text":               gen_text,
            "text_lang":          text_lang,
            "ref_audio_path":     str(Path(wav_ref).resolve()),
            "prompt_text":        ref_text,
            "prompt_lang":        prompt_lang,
            "top_k":              5,
            "top_p":              1.0,
            "temperature":        1.0,
            "text_split_method":  "cut5",   # GPT-SoVITS handles chunking
            "batch_size":         1,
            "speed_factor":       float(speed),
            "split_bucket":       True,
            "fragment_interval":  0.3,
            "seed":               -1,
            "media_type":         "wav",
            "streaming_mode":     False,
            "parallel_infer":     True,
            "repetition_penalty": 1.35,
        }

        print(f"  → POST {_GPTSOVITS_URL}/tts ...")
        resp = _requests.post(
            f"{_GPTSOVITS_URL}/tts",
            json=payload,
            timeout=600,          # allow up to 10 min for long documents
        )

        if resp.status_code != 200:
            # Try to surface a helpful error message
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:300]
            raise RuntimeError(
                f"GPT-SoVITS API returned HTTP {resp.status_code}: {detail}"
            )

        audio_bytes = resp.content
        print(f"  [OK] {len(audio_bytes):,} bytes received.")

        if progress_cb:
            progress_cb(1, 1)

        return audio_bytes, "wav"

    finally:
        if wav_is_temp:
            Path(wav_ref).unlink(missing_ok=True)
