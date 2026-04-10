"""GPT-SoVITS wrapper — zero-shot voice cloning via local API server.

Architecture
------------
GPT-SoVITS runs as a separate process (started by main.py) that listens
on http://127.0.0.1:9880.  This module splits the text into sentences
itself, calls GPT-SoVITS once per sentence with cut0 (no internal
splitting), then stitches the audio together with natural pauses.
This gives full control over pause length and avoids mid-sentence stops.
"""
_TTS_VERSION = "2026-04-10-a"

import io
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

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
# Reference audio trimmer
# ---------------------------------------------------------------------------

_REF_MIN_SEC = 3.0
_REF_MAX_SEC = 10.0
_REF_TARGET_SEC = 9.0   # trim long clips to this length


def _ensure_ref_duration(wav_path: str) -> tuple[str, bool]:
    """Ensure the reference WAV is within GPT-SoVITS's 3–10 s requirement.

    - If duration is already in range: return as-is (is_temp=False).
    - If duration > 10 s: trim to 9 s and return a temp file (is_temp=True).
    - If duration < 3 s: raise ValueError so the user gets a clear message.
    """
    import soundfile as sf
    info = sf.info(wav_path)
    dur  = info.duration

    if _REF_MIN_SEC <= dur <= _REF_MAX_SEC:
        return wav_path, False

    if dur < _REF_MIN_SEC:
        raise ValueError(
            f"Reference audio is only {dur:.1f}s — GPT-SoVITS requires at least "
            f"{_REF_MIN_SEC:.0f}s. Please upload a longer clip in the Voice tab."
        )

    # Too long — trim with ffmpeg
    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe is None:
        print(f"  ⚠ Ref audio is {dur:.1f}s (max {_REF_MAX_SEC}s) but ffmpeg not "
              f"found to trim it. Generation may fail.")
        return wav_path, False

    print(f"  ⚠ Ref audio is {dur:.1f}s — trimming to {_REF_TARGET_SEC}s for GPT-SoVITS.")
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    try:
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", wav_path,
             "-t", str(_REF_TARGET_SEC),
             "-ar", "24000", "-ac", "1", "-f", "wav", str(tmp)],
            capture_output=True, check=True,
        )
        return str(tmp), True
    except Exception as exc:
        print(f"  ⚠ Trim failed: {exc} — using original (may be rejected by GPT-SoVITS).")
        return wav_path, False


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
# Text splitting  (our own — bypasses GPT-SoVITS internal splitting)
# ---------------------------------------------------------------------------

# Silence to append after each segment (seconds)
_PAUSE_SENTENCE  = 0.40   # after 。！？.!?
_PAUSE_CLAUSE    = 0.15   # after ，,;；
_PAUSE_PARAGRAPH = 0.70   # after a blank line / paragraph break


def _split_segments(text: str) -> list[tuple[str, float]]:
    """Split *text* into (segment, pause_after_seconds) pairs.

    Splitting rules (in priority order):
      1. Blank lines → paragraph break (longest pause)
      2. Sentence-ending punctuation → sentence pause
      3. Clause punctuation (comma/semicolon) only when segment > 60 chars
         to avoid over-splitting short phrases

    Returns only segments that contain at least one speakable character.
    """
    def _speakable(s: str) -> bool:
        return any(c.isalpha() or c.isdigit() for c in s)

    results: list[tuple[str, float]] = []

    # Step 1: split on paragraph breaks first
    paragraphs = re.split(r'\n{2,}', text.strip())

    for p_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        # Step 2: split paragraph into sentences at 。！？.!?
        # Keep the punctuation attached to the preceding segment.
        raw = re.split(r'(?<=[。！？.!?])\s*', para)
        sentences = [s.strip() for s in raw if s.strip()]

        for s_idx, sent in enumerate(sentences):
            if not _speakable(sent):
                continue

            is_last_sent = (s_idx == len(sentences) - 1)
            is_last_para = (p_idx == len(paragraphs) - 1)

            # Determine base pause
            if is_last_sent and not is_last_para:
                base_pause = _PAUSE_PARAGRAPH
            elif sent[-1] in "。！？.!?":
                base_pause = _PAUSE_SENTENCE
            else:
                base_pause = _PAUSE_CLAUSE

            # Step 3: if sentence is long, sub-split at clause punctuation
            if len(sent) > 60:
                clauses = re.split(r'(?<=[，,;；])\s*', sent)
                clauses = [c.strip() for c in clauses if c.strip() and _speakable(c)]
                for c_idx, clause in enumerate(clauses):
                    is_last_clause = (c_idx == len(clauses) - 1)
                    pause = base_pause if is_last_clause else _PAUSE_CLAUSE
                    results.append((clause, pause))
            else:
                results.append((sent, base_pause))

    return results


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


def _call_gptsovits(
    session,
    ref_path: str,
    ref_text: str,
    gen_text: str,
    text_lang: str,
    prompt_lang: str,
    speed: float,
) -> np.ndarray:
    """Call GPT-SoVITS API for a single segment. Returns float32 audio array."""
    import soundfile as sf

    payload = {
        "text":               gen_text,
        "text_lang":          text_lang,
        "ref_audio_path":     ref_path,
        "prompt_text":        ref_text,
        "prompt_lang":        prompt_lang,
        "top_k":              5,
        "top_p":              1.0,
        "temperature":        1.0,
        "text_split_method":  "cut0",   # we handle splitting ourselves
        "batch_size":         1,
        "speed_factor":       float(speed),
        "split_bucket":       False,
        "fragment_interval":  0.0,
        "seed":               -1,
        "media_type":         "wav",
        "streaming_mode":     False,
        "parallel_infer":     True,
        "repetition_penalty": 1.35,
    }

    resp = session.post(f"{_GPTSOVITS_URL}/tts", json=payload, timeout=300)

    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:200]
        raise RuntimeError(f"GPT-SoVITS HTTP {resp.status_code}: {detail}")

    audio_arr, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
    if audio_arr.ndim > 1:
        audio_arr = audio_arr.mean(axis=1)   # stereo → mono

    # GPT-SoVITS v2 prepends the reference audio to every output chunk.
    # Strip it by skipping the first ref_duration worth of samples.
    try:
        ref_info     = sf.info(ref_path)
        ref_samples  = int(ref_info.duration * sr)
        if len(audio_arr) > ref_samples + sr:   # only strip if result is longer than ref alone
            audio_arr = audio_arr[ref_samples:]
    except Exception:
        pass   # if info fails, keep audio as-is

    return audio_arr, sr


def _generate_speech_inner(
    ref_audio_path: str,
    ref_text: str,
    gen_text: str,
    speed: float = 1.0,
    remove_silence: bool = False,
    progress_cb=None,
) -> tuple[bytes, str]:
    import requests as _requests
    import soundfile as sf

    print(f"  [tts.py {_TTS_VERSION}]  engine=GPT-SoVITS")

    gen_text = gen_text.strip()
    if not gen_text:
        raise ValueError("gen_text is empty — nothing to synthesise.")

    # Check GPT-SoVITS server is reachable before doing any work
    import socket as _socket
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
        _s.settimeout(2)
        if _s.connect_ex(("127.0.0.1", int(os.getenv("GPTSOVITS_PORT", 9880)))) != 0:
            raise RuntimeError(
                "GPT-SoVITS server is not running on port 9880.\n"
                "It should restart automatically in a few seconds — please try again.\n"
                "If the problem persists, restart run.bat."
            )

    wav_ref, wav_is_temp = _to_wav(ref_audio_path)

    # Trim/validate reference audio to GPT-SoVITS's 3–10 s requirement
    wav_ref2, wav_is_temp2 = _ensure_ref_duration(wav_ref)
    if wav_is_temp2 and wav_ref2 != wav_ref:
        if wav_is_temp:
            Path(wav_ref).unlink(missing_ok=True)
        wav_ref      = wav_ref2
        wav_is_temp  = True

    try:
        ref_path    = str(Path(wav_ref).resolve())
        text_lang   = _detect_lang(gen_text)
        prompt_lang = _detect_lang(ref_text)

        segments = _split_segments(gen_text)
        total    = len(segments)
        print(f"  text_lang={text_lang}  prompt_lang={prompt_lang}  "
              f"speed={speed}  segments={total}")

        if progress_cb:
            progress_cb(0, total)

        sr = 24000
        parts: list[np.ndarray] = []

        with _requests.Session() as session:
            for i, (seg_text, pause_sec) in enumerate(segments):
                print(f"  [{i+1}/{total}] {len(seg_text)} chars  "
                      f"pause={pause_sec:.2f}s  {seg_text[:50]!r}")

                audio_arr, sr = _call_gptsovits(
                    session, ref_path, ref_text,
                    seg_text, text_lang, prompt_lang, speed,
                )
                parts.append(audio_arr)

                # Add natural pause after segment
                if pause_sec > 0 and i < total - 1:
                    silence = np.zeros(int(sr * pause_sec), dtype=np.float32)
                    parts.append(silence)

                if progress_cb:
                    progress_cb(i + 1, total)

        # Stitch all segments into one WAV
        full_audio = np.concatenate(parts) if parts else np.zeros(sr, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, full_audio, sr, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        print(f"  [OK] {len(wav_bytes):,} bytes  ({len(full_audio)/sr:.1f}s)")
        return wav_bytes, "wav"

    finally:
        if wav_is_temp:
            Path(wav_ref).unlink(missing_ok=True)
