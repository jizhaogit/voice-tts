"""F5-TTS wrapper — zero-shot voice cloning for EN / ZH / JA.

F5-TTS uses a reference audio clip + its transcript to clone a voice.
No fine-tuning required: inference is zero-shot.

Model download (~1.2 GB) happens automatically on first call to infer().
"""
import io
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

# Lazy-loaded singletons
_tts_instance = None


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

def _get_ffmpeg_exe() -> str | None:
    """Return the full path to a working ffmpeg binary, or None.

    Tries imageio-ffmpeg (bundled pip package) first, then falls back to the
    copy that main.py places in runtime\ffmpeg.exe.  We intentionally do NOT
    rely on PATH so that Whisper / torchaudio subprocess calls work regardless
    of whether the directory was added to the environment.
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    # Fallback: main.py copies the binary here on startup
    runtime_ffmpeg = Path(__file__).parent.parent / "runtime" / "ffmpeg.exe"
    if runtime_ffmpeg.exists():
        return str(runtime_ffmpeg)

    return None


def _to_wav(audio_path: str, target_sr: int = 24000) -> tuple[str, bool]:
    """Convert any audio file to a 24 kHz mono WAV using the bundled ffmpeg.

    WAV files are returned as-is.  For any other format (MP3, M4A, WebM, OGG,
    FLAC, AAC …) we call ffmpeg by its *full path* — no PATH lookup, so
    WinError 2 cannot occur.

    Returns:
        (path_to_wav, is_temp)
        is_temp=True  → caller must delete the file when done.
        is_temp=False → caller must NOT delete (it's the original file).
    """
    src = Path(audio_path)
    if src.suffix.lower() == ".wav":
        return audio_path, False

    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe is None:
        # No ffmpeg available at all — pass original path and hope the
        # downstream library can handle it (e.g. soundfile with native codec).
        print("  ⚠ ffmpeg not found — attempting to use audio as-is. "
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
        print(f"  ⚠ Audio conversion to WAV failed: {exc}")
        return audio_path, False


def _print_cuda_warning(err: Exception, torch) -> None:
    """Print a targeted fix hint based on the CUDA error type."""
    msg = str(err)
    print(f"\n  {'='*60}")
    print(f"  ⚠  CUDA ERROR: {err.__class__.__name__}")
    print(f"  {'='*60}")

    if "no kernel image" in msg.lower():
        # Classic architecture mismatch — common with brand-new GPUs
        gpu_name = "unknown GPU"
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_name = f"{gpu_name} (sm_{props.major}{props.minor})"
        except Exception:
            pass
        torch_ver = getattr(torch, "__version__", "?")
        cuda_built = getattr(torch.version, "cuda", "?")
        print(f"  GPU   : {gpu_name}")
        print(f"  PyTorch {torch_ver} was built for CUDA {cuda_built}")
        print(f"  → This GPU's architecture is not supported by that build.")
        print()
        if "sm_10" in gpu_name or "sm_9" in gpu_name:
            print("  FIX: You have an RTX 50xx / RTX 40xx (Blackwell / Ada) GPU.")
            print("       Re-run  run.bat  — it will install PyTorch 2.7+ cu128")
            print("       which adds sm_100 (Blackwell) kernel support.")
        else:
            print("  FIX: Re-run  run.bat  — it will detect your GPU and install")
            print("       the correct PyTorch CUDA build automatically.")
    elif "out of memory" in msg.lower():
        print("  GPU ran out of VRAM. Falling back to CPU for this session.")
        print("  Try a shorter document or reduce chunk size in .env:")
        print("    TTS_CHUNK_SIZE=150")
    else:
        print(f"  Detail: {msg}")
        print("  Falling back to CPU. Re-run run.bat to repair GPU support.")

    print(f"  {'='*60}\n")


def _warn_if_gpu_present(torch) -> None:
    """Warn when CUDA is unavailable despite an NVIDIA GPU being present."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        gpu_name = result.stdout.strip()
        if result.returncode == 0 and gpu_name:
            cuda_built = getattr(torch.version, "cuda", None)
            print(f"\n  ⚠  GPU detected ({gpu_name}) but PyTorch cannot use it.")
            if cuda_built is None:
                print("     A CPU-only PyTorch build is installed.")
            else:
                print(f"     PyTorch CUDA build: {cuda_built}")
            print("     Re-run  run.bat  to reinstall the GPU-compatible build.\n")
    except Exception:
        pass  # nvidia-smi not found = genuine CPU-only machine, stay silent


def _get_tts():
    global _tts_instance
    if _tts_instance is None:
        import torch
        from f5_tts.api import F5TTS

        model_name = os.getenv("F5_MODEL", "F5TTS_v1_Base")

        if torch.cuda.is_available():
            # Quick sanity-check before loading the full model.
            # Catches "no kernel image" (architecture mismatch) and other
            # CUDA runtime errors before we waste time loading weights.
            try:
                torch.zeros(1, device="cuda")
                device = "cuda"
            except Exception as cuda_err:
                _print_cuda_warning(cuda_err, torch)
                device = "cpu"
        else:
            # torch.cuda.is_available() == False can mean either:
            #   (a) no NVIDIA GPU present — normal CPU machine
            #   (b) CPU-only PyTorch was installed despite having a GPU
            _warn_if_gpu_present(torch)
            device = "cpu"

        print(f"  Loading F5-TTS ({model_name}) on {device}...")
        try:
            _tts_instance = F5TTS(model=model_name, device=device)
        except Exception:
            if device == "cuda":
                # Model load itself failed on GPU — try CPU as last resort
                print("  ⚠ Model failed to load on GPU, retrying on CPU...")
                _tts_instance = F5TTS(model=model_name, device="cpu")
            else:
                raise

        print(f"  F5-TTS ready ({device.upper()}).")
    return _tts_instance


# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int | None = None) -> list[str]:
    """Split text into chunks ≤ max_chars on sentence boundaries.

    Handles English, Chinese (。！？), and Japanese (。！？) punctuation.
    """
    if max_chars is None:
        max_chars = int(os.getenv("TTS_CHUNK_SIZE", 250))

    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split on sentence-ending punctuation (EN + CJK)
    raw_sentences = re.split(r'(?<=[.!?。！？])\s*', text.strip())

    chunks: list[str] = []
    current = ""

    # Punctuation-only pattern — these are not speakable and cause F5-TTS to crash
    _punct_only = re.compile(r'^[\s\W]+$')

    for sent in raw_sentences:
        sent = sent.strip()
        if not sent or _punct_only.match(sent):
            continue
        if len(current) + len(sent) + 1 > max_chars:
            if current:
                chunks.append(current)
            # Sentence is itself too long — hard-split
            while len(sent) > max_chars:
                chunks.append(sent[:max_chars])
                sent = sent[max_chars:]
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent

    if current:
        chunks.append(current)

    return chunks or [text[:max_chars]]


# ── Core inference ────────────────────────────────────────────────────────────

def _infer_chunk(
    tts,
    ref_file: str,
    ref_text: str,
    gen_text: str,
    speed: float,
    remove_silence: bool,
    _depth: int = 0,
) -> tuple[np.ndarray, int]:
    """Infer a single text chunk; auto-splits on tensor-size mismatch.

    F5-TTS occasionally raises:
      RuntimeError: Sizes of tensors must match except in dimension 2.
    This happens when the mel-spectrogram lengths of the reference audio and
    the generated segment are incompatible.  The fix is to split the offending
    chunk into two smaller pieces and try again (recursively, up to depth 4).
    """
    try:
        audio_arr, sr, _ = tts.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=gen_text,
            speed=speed,
            remove_silence=remove_silence,
        )
        return audio_arr.flatten(), sr

    except RuntimeError as exc:
        if "Sizes of tensors must match" not in str(exc):
            raise
        if _depth >= 4 or len(gen_text) <= 10:
            # Give up splitting — skip this tiny piece rather than crash
            print(f"  ⚠ Skipping unresolvable chunk ({len(gen_text)} chars): {gen_text[:40]!r}")
            return np.zeros(100, dtype=np.float32), 24000

        # Split at the nearest word boundary to the midpoint
        mid = len(gen_text) // 2
        split_at = gen_text.rfind(" ", 1, mid) or gen_text.find(" ", mid) or mid
        left  = gen_text[:split_at].strip()
        right = gen_text[split_at:].strip()

        if not left or not right:
            # No sensible split — hard-split
            left, right = gen_text[:mid], gen_text[mid:]

        print(f"  ↳ tensor mismatch — splitting chunk into {len(left)} + {len(right)} chars")
        arrays, sr = [], 24000
        for part in (left, right):
            arr, sr = _infer_chunk(tts, ref_file, ref_text, part, speed,
                                   remove_silence, _depth + 1)
            arrays.append(arr)
        return np.concatenate(arrays), sr


def generate_speech(
    ref_audio_path: str,
    ref_text: str,
    gen_text: str,
    speed: float = 1.0,
    remove_silence: bool = True,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[bytes, str]:
    """Generate speech audio from text using a reference voice.

    Returns (audio_bytes, extension) where extension is 'mp3' or 'wav'.
    progress_cb(current_chunk, total_chunks) is called after each chunk.
    """
    tts = _get_tts()
    chunks = chunk_text(gen_text)
    total = len(chunks)
    all_audio: list[np.ndarray] = []
    sample_rate = 24000

    # Convert reference audio to WAV up-front using the bundled ffmpeg binary
    # (full path — no PATH lookup).  This prevents [WinError 2] from torchaudio
    # or f5-tts trying to shell out to 'ffmpeg' by name for non-WAV formats.
    wav_ref, wav_is_temp = _to_wav(ref_audio_path)
    try:
        for i, chunk in enumerate(chunks):
            audio_arr, sr = _infer_chunk(
                tts, wav_ref, ref_text, chunk, speed, remove_silence
            )
            sample_rate = sr
            all_audio.append(audio_arr)
            if progress_cb:
                progress_cb(i + 1, total)
    finally:
        if wav_is_temp:
            Path(wav_ref).unlink(missing_ok=True)

    combined: np.ndarray = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]

    # Convert float32 → int16 PCM
    audio_int16 = (combined * 32767).clip(-32768, 32767).astype(np.int16)

    # Encode to MP3 via lameenc (pure Python, no ffmpeg needed)
    try:
        import lameenc

        enc = lameenc.Encoder()
        enc.set_bit_rate(128)
        enc.set_in_sample_rate(sample_rate)
        enc.set_channels(1)
        enc.set_quality(2)  # 2 = highest quality
        mp3_bytes = enc.encode(audio_int16.tobytes()) + enc.flush()
        return mp3_bytes, "mp3"
    except Exception:
        # Fallback to WAV (browsers support it fine)
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, combined, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "wav"


# ── Auto-transcription ────────────────────────────────────────────────────────

def transcribe_audio(audio_path: str, language: str = "en") -> str:
    """Transcribe an audio file using OpenAI Whisper.

    Uses the 'base' model (~150 MB download on first call).
    Returns the transcript string.
    """
    import whisper

    # Whisper calls ffmpeg by name via subprocess for non-WAV files.
    # Convert first using the full ffmpeg path to avoid [WinError 2].
    wav_path, wav_is_temp = _to_wav(audio_path)
    try:
        model = whisper.load_model("base")
        lang = language if language != "auto" else None
        result = model.transcribe(wav_path, language=lang)
        return result["text"].strip()
    finally:
        if wav_is_temp:
            Path(wav_path).unlink(missing_ok=True)
