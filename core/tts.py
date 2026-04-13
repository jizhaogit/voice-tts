"""CosyVoice 2 TTS engine — zero-shot voice cloning, runs in-process.

CosyVoice 2 is loaded once on first use and reused for all requests.
It returns clean generated audio with NO reference audio prepended,
so no stripping logic is needed at all.
"""
_TTS_VERSION = "2026-04-13-a"

import io
import os
import re
import subprocess
import sys
import tempfile
import threading
import warnings
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    """Return language tag for the dominant script in *text*."""
    zh = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    ja = sum(1 for c in text if "\u3040" <= c <= "\u30ff")
    en = sum(1 for c in text if c.isascii() and c.isalpha())
    if zh >= en and zh >= ja:
        return "zh"
    if ja > en:
        return "ja"
    return "en"


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

# Silence to append after each segment (seconds)
_PAUSE_SENTENCE  = 0.40   # after 。！？.!?
_PAUSE_CLAUSE    = 0.15   # after ，,;；
_PAUSE_PARAGRAPH = 0.70   # after a blank line / paragraph break

# Minimum characters per CosyVoice 2 call.
# Short segments produce so little speech that merging short segments
# together ensures each call generates enough audio.
_MIN_SEGMENT_CHARS = 20


def _speakable(s: str) -> bool:
    return any(c.isalpha() or c.isdigit() for c in s)


def _split_segments(text: str) -> list[tuple[str, float]]:
    """Split *text* into (segment, pause_after_seconds) pairs.

    Splitting rules (in priority order):
      1. Blank lines → paragraph break (longest pause)
      2. Sentence-ending punctuation → sentence pause
      3. Clause punctuation (comma/semicolon) only when segment > 60 chars
         to avoid over-splitting short phrases

    Returns only segments that contain at least one speakable character.
    """
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


def _merge_short_segments(
    segments: list[tuple[str, float]],
    min_chars: int = _MIN_SEGMENT_CHARS,
    max_chars: int = 80,
) -> list[tuple[str, float]]:
    """Merge consecutive short segments so each CosyVoice 2 call is long enough.

    Short segments (< min_chars) are accumulated into a buffer and flushed only
    when the buffer reaches min_chars or a paragraph boundary is hit.
    The pause of the *last* merged piece is kept for the combined segment.
    Merging never crosses paragraph breaks (_PAUSE_PARAGRAPH) and never exceeds
    max_chars so CosyVoice 2 doesn't choke on very long inputs.
    """
    if not segments:
        return segments

    result: list[tuple[str, float]] = []
    acc_text  = ""
    acc_pause = 0.0

    for text, pause in segments:
        if not acc_text:
            acc_text  = text
            acc_pause = pause
            continue

        # Flush buffer before a paragraph break — never merge across paragraphs
        if acc_pause >= _PAUSE_PARAGRAPH:
            result.append((acc_text, acc_pause))
            acc_text  = text
            acc_pause = pause
            continue

        # Flush buffer if merging would exceed the per-call character limit
        if len(acc_text) + len(text) > max_chars:
            result.append((acc_text, acc_pause))
            acc_text  = text
            acc_pause = pause
            continue

        # Merge into buffer; keep the new (later) pause
        acc_text  = acc_text + text
        acc_pause = pause

        # Flush once buffer is long enough
        if len(acc_text) >= min_chars:
            result.append((acc_text, acc_pause))
            acc_text  = ""
            acc_pause = 0.0

    if acc_text:
        result.append((acc_text, acc_pause))

    return result


# ---------------------------------------------------------------------------
# CosyVoice 2 model singleton
# ---------------------------------------------------------------------------
_CV_LOCK  = threading.Lock()
_CV_MODEL = None   # loaded lazily on first TTS call


def _get_cosyvoice():
    """Return the loaded CosyVoice2 instance (loads on first call, ~30-60 s)."""
    global _CV_MODEL
    if _CV_MODEL is not None:
        return _CV_MODEL

    with _CV_LOCK:
        if _CV_MODEL is not None:
            return _CV_MODEL

        cv_dir = _PROJECT_ROOT / "cosyvoice"
        if not cv_dir.exists():
            raise RuntimeError(
                "CosyVoice 2 source not found at cosyvoice/.\n"
                "Run run.bat to install it automatically."
            )

        # Add CosyVoice package directories to sys.path
        for extra in [str(cv_dir),
                      str(cv_dir / "third_party" / "Matcha-TTS")]:
            if extra not in sys.path:
                sys.path.insert(0, extra)

        model_dir = cv_dir / "pretrained_models" / "CosyVoice2-0.5B"
        if not model_dir.exists():
            raise RuntimeError(
                "CosyVoice2-0.5B model not found at cosyvoice/pretrained_models/.\n"
                "Run run.bat to download the model (~2.5 GB)."
            )

        from cosyvoice.cli.cosyvoice import CosyVoice2
        import torch
        fp16 = torch.cuda.is_available()
        print(f"  [..] Loading CosyVoice2-0.5B  (fp16={fp16}) ...")
        _CV_MODEL = CosyVoice2(
            str(model_dir),
            load_jit=False,
            load_trt=False,
            fp16=fp16,
        )
        print(f"  [OK] CosyVoice2-0.5B ready  (sr={_CV_MODEL.sample_rate} Hz)")
        return _CV_MODEL


# ---------------------------------------------------------------------------
# Per-segment synthesis
# ---------------------------------------------------------------------------

def _call_cosyvoice(
    model,
    prompt_speech_16k,   # torch.Tensor returned by load_wav()
    ref_text:  str,
    gen_text:  str,
    speed:     float = 1.0,
) -> tuple[np.ndarray, int]:
    """Synthesise one text segment. Returns (float32 array, sample_rate)."""
    sr     = model.sample_rate
    chunks: list[np.ndarray] = []

    for result in model.inference_zero_shot(
        gen_text,
        ref_text,
        prompt_speech_16k,
        stream=False,
        speed=speed,
    ):
        audio = result["tts_speech"]          # torch.Tensor  (1, N)
        chunks.append(audio.squeeze(0).float().cpu().numpy())

    if not chunks:
        # Model returned nothing — substitute proportional silence
        sil_sec = min(2.0, max(0.3, len(gen_text) * 0.20))
        print(f"  [!] no audio returned — using {sil_sec:.2f}s silence")
        return np.zeros(int(sr * sil_sec), dtype=np.float32), sr

    audio_arr = np.concatenate(chunks).astype(np.float32)
    print(f"  [dbg] output={len(audio_arr)/sr:.2f}s  sr={sr}")
    return audio_arr, sr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_speech(
    ref_audio_path: str,
    ref_text:       str,
    gen_text:       str,
    speed:          float = 1.0,
    remove_silence: bool  = False,
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
    ref_text:       str,
    gen_text:       str,
    speed:          float = 1.0,
    remove_silence: bool  = False,
    progress_cb=None,
) -> tuple[bytes, str]:
    import soundfile as sf

    print(f"  [tts.py {_TTS_VERSION}]  engine=CosyVoice2")

    gen_text = gen_text.strip()
    if not gen_text:
        raise ValueError("gen_text is empty — nothing to synthesise.")

    # Load model (cached after first call)
    model = _get_cosyvoice()

    # Convert reference audio to WAV if needed, then load at 16 kHz
    wav_ref, wav_is_temp = _to_wav(ref_audio_path)
    try:
        ref_path = str(Path(wav_ref).resolve())

        # Load reference at 16 kHz — CosyVoice 2 requires this sample rate
        from cosyvoice.utils.file_utils import load_wav
        prompt_speech_16k = load_wav(ref_path, 16000)

        text_lang   = _detect_lang(gen_text)
        prompt_lang = _detect_lang(ref_text)

        segments = _split_segments(gen_text)
        segments = _merge_short_segments(segments)
        total    = len(segments)
        print(f"  text_lang={text_lang}  prompt_lang={prompt_lang}  "
              f"speed={speed}  segments={total}")

        if progress_cb:
            progress_cb(0, total)

        sr    = model.sample_rate
        parts: list[np.ndarray] = []

        for i, (seg_text, pause_sec) in enumerate(segments):
            print(f"  [{i+1}/{total}] {len(seg_text)} chars  "
                  f"pause={pause_sec:.2f}s  {seg_text[:60]!r}")

            audio_arr, sr = _call_cosyvoice(
                model, prompt_speech_16k, ref_text, seg_text, speed
            )
            parts.append(audio_arr)

            if pause_sec > 0 and i < total - 1:
                parts.append(np.zeros(int(sr * pause_sec), dtype=np.float32))

            if progress_cb:
                progress_cb(i + 1, total)

        full_audio = (np.concatenate(parts)
                      if parts else np.zeros(sr, dtype=np.float32))
        buf = io.BytesIO()
        sf.write(buf, full_audio, sr, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        print(f"  [OK] {len(wav_bytes):,} bytes  ({len(full_audio)/sr:.1f}s)")
        return wav_bytes, "wav"

    finally:
        if wav_is_temp:
            Path(wav_ref).unlink(missing_ok=True)
