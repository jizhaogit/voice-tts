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
    Guarantees no chunk shorter than MIN_CHUNK chars reaches F5-TTS, which
    crashes on single characters.
    """
    if max_chars is None:
        max_chars = int(os.getenv("TTS_CHUNK_SIZE", 250))

    MIN_CHUNK = 8  # shorter than this → merge into the previous chunk

    def _has_speakable(s: str) -> bool:
        """Return True if s contains at least one letter or digit.

        Uses str.isalpha() / str.isdigit() which are unambiguous about CJK
        characters — avoids regex \\W edge-cases with certain Unicode blocks.
        """
        return any(c.isalpha() or c.isdigit() for c in s)

    # Normalise line endings and whitespace
    text = re.sub(r'\r\n?', '\n', text).strip()

    # Split AFTER sentence-ending punctuation (EN + CJK)
    raw_sentences = re.split(r'(?<=[.!?。！？])\s*', text)

    chunks: list[str] = []
    current = ""

    for sent in raw_sentences:
        sent = sent.strip()
        if not sent or not _has_speakable(sent):
            continue

        if len(current) + len(sent) + 1 > max_chars:
            if current:
                chunks.append(current)
                current = ""
            # Sentence itself is longer than max_chars — hard-split at word
            # boundaries (EN) or fixed width (CJK).  Ensure no leftover is
            # shorter than MIN_CHUNK by absorbing it into the last split piece.
            while len(sent) > max_chars:
                split_at = max_chars
                # For English, try to break at a space
                space = sent.rfind(" ", MIN_CHUNK, max_chars)
                if space > 0:
                    split_at = space
                chunks.append(sent[:split_at].strip())
                sent = sent[split_at:].strip()
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent

    if current:
        chunks.append(current)

    # ── Post-pass: merge any chunk shorter than MIN_CHUNK into its neighbour ──
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < MIN_CHUNK:
            # Absorb short tail into previous chunk (may exceed max_chars
            # slightly but avoids F5-TTS single-char crashes)
            merged[-1] = (merged[-1] + " " + chunk).strip()
        else:
            merged.append(chunk)

    return merged or [text[:max_chars]]


# ── Post-processing: pitch-preserving speed change ───────────────────────────

def _apply_speed(audio: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """Time-stretch audio using ffmpeg's atempo filter (pitch-preserving).

    Operates entirely on temp files with the bundled ffmpeg binary — no PATH
    lookup required.  Returns the original array unchanged on any error.

    atempo operates in the range 0.5–2.0; values outside that range are
    achieved by chaining multiple atempo stages.
    """
    if abs(speed - 1.0) < 0.02:
        return audio

    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe is None:
        print("  ⚠ ffmpeg not available — speed adjustment skipped.")
        return audio

    import soundfile as sf

    # Build chained atempo filter string so any speed in [0.25, 4.0] works
    def _atempo_chain(s: float) -> str:
        stages: list[str] = []
        while s < 0.5 - 1e-6:
            stages.append("atempo=0.5")
            s /= 0.5
        while s > 2.0 + 1e-6:
            stages.append("atempo=2.0")
            s /= 2.0
        stages.append(f"atempo={s:.4f}")
        return ",".join(stages)

    tmp_in  = Path(tempfile.mktemp(suffix=".wav"))
    tmp_out = Path(tempfile.mktemp(suffix=".wav"))
    try:
        sf.write(str(tmp_in), audio, sample_rate, format="WAV", subtype="PCM_16")
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", str(tmp_in),
             "-af", _atempo_chain(speed),
             "-ar", str(sample_rate), "-ac", "1", str(tmp_out)],
            capture_output=True, check=True,
        )
        result, _ = sf.read(str(tmp_out), dtype="float32")
        return result
    except Exception as exc:
        print(f"  ⚠ Speed adjustment failed ({exc}) — returning at 1.0×.")
        return audio
    finally:
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)


# ── Core inference ────────────────────────────────────────────────────────────

def _infer_chunk(
    tts,
    ref_file: str,
    ref_text: str,
    gen_text: str,
    speed: float,
    remove_silence: bool = False,
    _depth: int = 0,
) -> tuple[np.ndarray, int]:
    """Infer one text chunk; retries with varied speeds and splits on mismatch.

    WHY speed-jitter works
    ─────────────────────
    F5-TTS computes the target mel duration as:
        duration = ref_mel_len + round(ref_mel_len / ref_bytes * gen_bytes / speed)

    When the reference text is long (360+ bytes for 120 Chinese chars) but the
    gen chunk is short (< 30 chars = 90 bytes), the target_mel_len is tiny.
    Due to block-padding inside the diffusion model, the actual output tensor
    may not match this requested size → RuntimeError "Sizes of tensors must
    match except in dimension 2."

    Varying the speed slightly changes the computed duration, shifting it into
    a block-aligned value. Trying 7 candidate speeds at each depth almost
    always finds one that works without splitting.
    """
    # Hard guard: skip text that contains no letters or digits at all.
    # '。', '，', '…' etc. are not speakable and always crash F5-TTS.
    if not any(c.isalpha() or c.isdigit() for c in gen_text):
        return np.zeros(100, dtype=np.float32), 24000

    # Speeds to try — jitter covers ±40 % to find a block-aligned mel length
    _SPEED_CANDIDATES = [1.0, 0.85, 1.15, 0.70, 1.30, 0.55, 1.50]

    for try_speed in _SPEED_CANDIDATES:
        try:
            audio_arr, sr, _ = tts.infer(
                ref_file=ref_file,
                ref_text=ref_text,
                gen_text=gen_text,
                speed=try_speed,
                remove_silence=remove_silence,
            )
            return audio_arr.flatten(), sr
        except RuntimeError as exc:
            if "Sizes of tensors must match" not in str(exc):
                raise   # unrelated error — propagate immediately

    # All speed variants failed → split and recurse
    if _depth >= 3 or len(gen_text) <= 4:
        print(f"  ⚠ Skipping chunk ({len(gen_text)} chars): {gen_text[:40]!r}")
        return np.zeros(100, dtype=np.float32), 24000

    # Split at the nearest word boundary (EN) or mid-point (CJK)
    mid = len(gen_text) // 2
    split_at = gen_text.rfind(" ", 1, mid) or gen_text.find(" ", mid) or mid
    left  = gen_text[:split_at].strip()
    right = gen_text[split_at:].strip()
    if not left or not right:
        left, right = gen_text[:mid], gen_text[mid:]

    print(f"  ↳ splitting chunk into {len(left)} + {len(right)} chars")
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
    remove_silence: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[bytes, str]:
    """Generate speech audio from text using a reference voice.

    Returns (audio_bytes, extension) where extension is 'mp3' or 'wav'.
    progress_cb(current_chunk, total_chunks) is called after each chunk.
    """
    tts = _get_tts()

    # ── Debug: show what text arrived and what chunks were produced ───────────
    char_count = len(gen_text)
    preview    = gen_text[:80].replace('\n', '↵')
    print(f"  gen_text: {char_count} chars — {preview!r}")

    chunks = chunk_text(gen_text)
    total = len(chunks)
    if chunks:
        print(f"  Chunks  : {total}  |  first={chunks[0][:60]!r}")
    else:
        print("  ⚠ No chunks produced — document may be empty or corrupt.")
    if not chunks:
        raise ValueError(
            "No speakable text found. The document may be empty, corrupt, "
            "or encoded in a format that could not be read. "
            "Please delete and re-upload the document."
        )
    all_audio: list[np.ndarray] = []
    sample_rate = 24000

    # Convert reference audio to WAV up-front using the bundled ffmpeg binary
    # (full path — no PATH lookup).  This prevents [WinError 2] from torchaudio
    # or f5-tts trying to shell out to 'ffmpeg' by name for non-WAV formats.
    wav_ref, wav_is_temp = _to_wav(ref_audio_path)
    try:
        # Always generate at speed=1.0 inside F5-TTS.
        # Passing non-1.0 values changes the internal mel-spectrogram target
        # duration and frequently causes tensor-size mismatches (especially
        # speed < 0.8).  We apply the user's speed afterwards via ffmpeg atempo
        # which is pitch-preserving and works reliably for any value 0.5–2.0.
        for i, chunk in enumerate(chunks):
            audio_arr, sr = _infer_chunk(
                tts, wav_ref, ref_text, chunk, speed=1.0, remove_silence=remove_silence
            )
            sample_rate = sr
            all_audio.append(audio_arr)
            if progress_cb:
                progress_cb(i + 1, total)
    finally:
        if wav_is_temp:
            Path(wav_ref).unlink(missing_ok=True)

    combined: np.ndarray = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]

    # Apply user-requested speed via pitch-preserving ffmpeg atempo
    if abs(speed - 1.0) > 0.02:
        print(f"  Applying {speed}× time-stretch via ffmpeg atempo…")
        combined = _apply_speed(combined, sample_rate, speed)

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
