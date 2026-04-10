"""F5-TTS wrapper — zero-shot voice cloning for EN / ZH / JA.

F5-TTS uses a reference audio clip + its transcript to clone a voice.
No fine-tuning required: inference is zero-shot.

Model download (~1.2 GB) happens automatically on first call to infer().
"""
_TTS_VERSION = "2026-04-09-g"  # bump this to confirm which copy is running
import io
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

# Lazy-loaded singletons
_tts_instance = None


# ── Console + file tee ────────────────────────────────────────────────────────

class _Tee:
    """Write every print() to both the real stdout and a log file.

    Usage (as a context manager):
        with _Tee(log_path):
            ...  # all print() output goes to console AND log file
    """
    def __init__(self, log_path: Path):
        self._log_path = log_path
        self._real_stdout = sys.stdout
        self._file = None

    def __enter__(self):
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._log_path.open("a", encoding="utf-8", buffering=1)
        import datetime
        self._file.write(f"\n{'='*60}\n"
                         f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
                         f"tts.py {_TTS_VERSION}\n"
                         f"{'='*60}\n")
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

    # Proxy any attribute access the real stdout needs (e.g. .encoding)
    def __getattr__(self, name: str):
        return getattr(self._real_stdout, name)


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

def chunk_text(
    text: str,
    max_chars: int | None = None,
    min_chars: int | None = None,
) -> list[str]:
    """Split text into chunks ≤ max_chars on sentence boundaries.

    Handles English, Chinese (。！？…), and Japanese (。！？) punctuation.
    Also splits on newline runs so single stray characters on their own lines
    (e.g. '的\n了' formatting artifacts in Chinese novels) are absorbed into
    adjacent sentences rather than becoming orphan chunks.

    min_chars controls the minimum size of any chunk produced.  Caller should
    pass ``len(ref_text)`` so every chunk is at least as long as the voice
    reference transcript — this is the key constraint for F5-TTS to produce
    good audio without tensor-size mismatches.
    """
    if max_chars is None:
        max_chars = int(os.getenv("TTS_CHUNK_SIZE", 250))

    def _has_speakable(s: str) -> bool:
        """Return True if s contains at least one letter or digit."""
        return any(c.isalpha() or c.isdigit() for c in s)

    # Normalise line endings and whitespace
    text = re.sub(r'\r\n?', '\n', text).strip()

    # Split AFTER sentence-ending punctuation (EN + CJK) or at blank lines /
    # newline runs.  This catches single-character lines like '的\n了' that
    # appear as formatting artefacts in plain-text Chinese novels.
    raw_sentences = re.split(r'(?<=[.!?。！？…])\s*|\n+', text)

    # ── First pass: accumulate sentences into max_chars chunks ────────────────
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
            # Sentence itself longer than max_chars — hard-split
            while len(sent) > max_chars:
                split_at = max_chars
                # For English, prefer a space boundary
                space = sent.rfind(" ", max(1, max_chars // 4), max_chars)
                if space > 0:
                    split_at = space
                chunks.append(sent[:split_at].strip())
                sent = sent[split_at:].strip()
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent

    if current:
        chunks.append(current)

    # ── Second pass: merge short chunks so every output ≥ min_chars ──────────
    # F5-TTS needs gen_text to be at least as long as ref_text (in characters)
    # to produce a mel-duration that the U-Net block alignment can satisfy.
    # We merge forward (accumulate into a buffer and flush when big enough) so
    # even the very first chunk ends up long enough.
    MIN_CHUNK = min_chars if min_chars is not None else 40

    merged: list[str] = []
    buf = ""
    for chunk in chunks:
        buf = (buf + " " + chunk).strip() if buf else chunk
        if len(buf) >= MIN_CHUNK:
            merged.append(buf)
            buf = ""

    # Flush any leftover that never reached MIN_CHUNK
    if buf:
        if merged:
            # Absorb into the last chunk (may push it slightly over max_chars,
            # but avoids sending a tiny fragment to F5-TTS)
            merged[-1] = (merged[-1] + " " + buf).strip()
        elif _has_speakable(buf):
            merged.append(buf)   # only content in the document — keep it

    # Only fall back to raw text if it contains speakable content.
    if not merged:
        raw_fallback = text[:max_chars].strip()
        if _has_speakable(raw_fallback):
            return [raw_fallback]
        return []          # nothing speakable at all
    return merged


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

# F5-TTS switches to multi-batch (internal chaining) when ref_mel + gen_mel
# exceeds this threshold.  Multi-batch mode has a cache-corruption bug: the
# intermediate buffer from the first sub-batch leaks into the model state, so
# every subsequent tts.infer() call for the SAME session sees a wrong ref_mel
# (e.g. 2005 instead of 1226) and fails unconditionally.
# We keep total_mel BELOW this limit so F5-TTS always uses 1 batch.
_F5_SINGLE_BATCH_THRESHOLD = 1990   # conservative; observed threshold ~2100


def _get_ref_mel(ref_file: str) -> int:
    """Return the approximate mel-frame count of the reference WAV file.

    F5-TTS uses hop_length=256 at 24 kHz → 93.75 frames/second ≈ 100 fps.
    We read the actual audio duration via soundfile (fast, header-only).
    """
    try:
        import soundfile as sf
        info = sf.info(ref_file)
        return max(1, int(info.duration * 24000 / 256))   # exact hop_length
    except Exception:
        return 1200   # fallback ≈ 13 s clip


def _valid_speeds_for_alignment(
    ref_mel: int,
    gen_mel_base: float,
    block_sizes: tuple = (2, 4, 8, 16),
    speed_lo: float = 0.70,
    speed_hi: float = 1.30,
) -> list[float]:
    """Return speeds (sorted by closeness to 1.0) that produce a total
    mel-length (ref_mel + gen_mel) divisible by each candidate block size.

    F5-TTS's U-Net requires total_mel % block_size == 0.  Varying speed
    changes gen_mel by integer steps, so we can analytically find the exact
    speeds that satisfy this constraint without brute-force search.
    """
    seen: set[float] = set()
    results: list[float] = []

    for bs in block_sizes:
        required_rem = (-ref_mel) % bs          # gen_mel must have this remainder
        base_k = int(gen_mel_base) // bs
        for offset in range(-8, 12):            # search ±8 multiples away
            k = base_k + offset
            if k <= 0:
                continue
            gen_mel = k * bs + required_rem
            if gen_mel <= 0:
                continue
            if (ref_mel + gen_mel) % bs != 0:   # sanity check
                continue
            speed = round(gen_mel_base / gen_mel, 4)
            if speed_lo <= speed <= speed_hi and speed not in seen:
                seen.add(speed)
                results.append(speed)

    results.sort(key=lambda s: abs(s - 1.0))   # try near-1.0 first
    return results


def _trim_ref_text(ref_text: str, gen_text: str, factor: float = 2.5) -> str:
    """Shorten ref_text so its UTF-8 byte length ≤ factor × gen_text bytes.

    WHY THIS MATTERS
    ────────────────
    F5-TTS estimates the target mel duration as:
        total = ref_mel + round(ref_mel / ref_bytes × gen_bytes / speed)

    When ref_bytes >> gen_bytes (long ref transcript, short gen chunk), the
    gen portion of the duration is tiny.  The diffusion U-Net requires the
    total duration to be block-aligned (divisible by 2^n).  With a tiny gen
    contribution, no speed value in the normal 0.5–2.0 range hits a valid
    alignment → RuntimeError "Sizes of tensors must match…"

    Trimming ref_text makes ref_bytes ≈ factor × gen_bytes.  At factor=2.5
    and speed=1.0 the ratio becomes reasonable and block-alignment is usually
    satisfied immediately.  Voice quality is largely preserved because F5-TTS
    captures timbre from the mel-spectrogram of the audio, not text alignment.

    Cuts at the last sentence-ending punctuation within the byte budget so the
    trimmed text still ends cleanly.
    """
    gen_bytes = len(gen_text.encode("utf-8"))
    ref_bytes = len(ref_text.encode("utf-8"))
    target_bytes = int(gen_bytes * factor)

    if ref_bytes <= target_bytes:
        return ref_text  # already within budget — no change needed

    # Slice UTF-8 bytes, then decode safely (ignore any partial multi-byte char)
    shortened = ref_text.encode("utf-8")[:target_bytes].decode("utf-8", errors="ignore").strip()

    # Prefer ending at a sentence boundary so the trimmed text reads cleanly
    for punct in ("。", "！", "？", "…", ".", "!", "?"):
        cut = shortened.rfind(punct)
        if cut >= max(4, len(shortened) // 2):
            return shortened[: cut + 1].strip()

    return shortened or ref_text[:20]


def _infer_chunk(
    tts,
    ref_file: str,
    ref_text: str,
    gen_text: str,
    speed: float,
    remove_silence: bool = False,
    _depth: int = 0,
    _ref_mel: int | None = None,
) -> tuple[np.ndarray, int]:
    """Infer one text chunk with multi-pass speed search.

    ROOT CAUSE OF FAILURES (discovered from tts_run.log analysis)
    ──────────────────────────────────────────────────────────────
    F5-TTS has two failure modes:

    1. Multi-batch chaining bug: when ref_mel + gen_mel > ~2000, F5-TTS splits
       the sequence internally into sub-batches and chains them.  The chaining
       code has a cache-corruption bug — an intermediate 2005-frame buffer leaks
       into model state, so every subsequent tts.infer() call in the session
       sees ref_mel=2005 instead of the actual ref_mel (e.g. 1226).  This
       causes ALL subsequent chunks to fail, producing a 0-second output file.

       Fix: keep ref_mel + gen_mel < _F5_SINGLE_BATCH_THRESHOLD (1990) so
       F5-TTS always uses a single inference batch.  Chunk sizes are set
       accordingly in generate_speech() before calling us.

    2. U-Net block alignment: the diffusion U-Net requires total_mel to be
       divisible by its downsampling factor (empirically 2, 4, 8, or 16).
       At speed=1.0 the alignment is usually satisfied, but not always.

       Fix: analytically compute the exact speeds that produce a block-aligned
       total, then fall back to a fine 0.01-step grid covering all residues.
    """
    # Hard guard: skip text that contains no letters or digits at all.
    if not any(c.isalpha() or c.isdigit() for c in gen_text):
        return np.zeros(100, dtype=np.float32), 24000

    speakable_chars = sum(1 for c in gen_text if c.isalpha() or c.isdigit())
    if speakable_chars < 4:
        print(f"  ⚠ Skipping too-short chunk ({speakable_chars} speakable chars): {gen_text!r}")
        return np.zeros(100, dtype=np.float32), 24000

    # Read actual ref audio mel length once (expensive syscall; cached via arg)
    ref_mel = _ref_mel if _ref_mel is not None else _get_ref_mel(ref_file)

    # ── Speed candidates ──────────────────────────────────────────────────────
    # Priority 1: analytically exact speeds for common block sizes (2, 4, 8, 16)
    gen_mel_base = ref_mel * len(gen_text) / max(1, len(ref_text))
    _EXACT = _valid_speeds_for_alignment(ref_mel, gen_mel_base)

    # Priority 2: coarse candidates that catch easy cases fast
    _SPARSE = [1.00, 0.97, 1.03, 0.94, 1.06, 0.91, 1.09, 0.88, 1.12]

    # Priority 3: fine 0.01-step grid — guarantees covering every residue mod 16
    # Sorted by distance from 1.0 so near-unity speeds are tried first.
    _FINE = sorted(
        [round(0.70 + i * 0.01, 2) for i in range(61)],
        key=lambda s: abs(s - 1.0),
    )

    # ── Error filter ──────────────────────────────────────────────────────────
    _SIZE_MISMATCH_HINTS = (
        "sizes of tensors must match",
        "size mismatch", "shape mismatch",
        "expected size", "the size of tensor", "dimension",
    )

    _first_exc_logged: list[bool] = [False]

    def _is_size_mismatch(exc: RuntimeError) -> bool:
        return any(h in str(exc).lower() for h in _SIZE_MISMATCH_HINTS)

    def _try_speeds(use_ref_text: str, candidates: list) -> tuple[np.ndarray, int] | None:
        seen: set[float] = set()
        for try_speed in candidates:
            s = round(try_speed, 3)
            if s in seen:
                continue
            seen.add(s)
            try:
                audio_arr, sr, _ = tts.infer(
                    ref_file=ref_file,
                    ref_text=use_ref_text,
                    gen_text=gen_text,
                    speed=try_speed,
                    remove_silence=remove_silence,
                )
                return audio_arr.flatten(), sr
            except RuntimeError as exc:
                if not _first_exc_logged[0]:
                    print(f"  [dbg] RuntimeError speed={try_speed:.3f}: {exc!r}")
                    _first_exc_logged[0] = True
                if not _is_size_mismatch(exc):
                    raise
        return None

    # ── Pass 1: exact + sparse speeds ────────────────────────────────────────
    result = _try_speeds(ref_text, _EXACT + _SPARSE)
    if result:
        return result

    # ── Pass 2: full fine grid ────────────────────────────────────────────────
    print(f"  ↳ exact+sparse failed — fine grid ({len(gen_text)} chars, ref_mel={ref_mel})")
    result = _try_speeds(ref_text, _FINE)
    if result:
        return result

    # ── Pass 3: trim ref_text + fine grid ────────────────────────────────────
    trimmed_ref = _trim_ref_text(ref_text, gen_text, factor=1.2)
    if trimmed_ref != ref_text:
        gen_mel_base2 = ref_mel * len(gen_text) / max(1, len(trimmed_ref))
        exact2 = _valid_speeds_for_alignment(ref_mel, gen_mel_base2)
        print(f"  ↳ trimming ref_text {len(ref_text)}→{len(trimmed_ref)} chars + fine grid")
        result = _try_speeds(trimmed_ref, exact2 + _FINE)
        if result:
            return result

    # ── Pass 4: split chunk and recurse ──────────────────────────────────────
    if _depth >= 2 or len(gen_text) <= 8:
        print(f"  ⚠ Skipping unresolvable chunk ({len(gen_text)} chars): {gen_text[:40]!r}")
        return np.zeros(100, dtype=np.float32), 24000

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
                               remove_silence, _depth + 1, ref_mel)
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
    All console output is also mirrored to  data/tts_run.log  so that
    long runs whose top lines scroll off the console window can be reviewed.
    """
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
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[bytes, str]:
    tts = _get_tts()

    # ── Debug: confirm version + show what text arrived ───────────────────────
    print(f"  [tts.py {_TTS_VERSION}]")
    char_count = len(gen_text)
    preview    = gen_text[:80].replace('\n', '↵')
    print(f"  gen_text: {char_count} chars — {preview!r}")

    # ── Convert ref audio to WAV so we can read its mel length ────────────────
    wav_ref, wav_is_temp = _to_wav(ref_audio_path)

    # ── Compute safe chunk sizes based on actual ref audio length ─────────────
    # F5-TTS enters multi-batch chaining mode when ref_mel + gen_mel exceeds
    # ~2100 frames.  That chaining code has a cache-corruption bug: on its
    # first failure it stores a stale 2005-frame value in model state, making
    # every subsequent tts.infer() call fail (observed in tts_run.log).
    #
    # Fix: cap chunk size so even at the lowest speed (0.70) the total stays
    # below _F5_SINGLE_BATCH_THRESHOLD.  This guarantees single-batch mode.
    #
    #   gen_mel = ref_mel × gen_chars / ref_chars / speed
    #   At speed=0.70: gen_mel_max = (threshold − ref_mel) × 0.70
    #   max_chars = gen_mel_max × ref_chars / ref_mel
    ref_mel  = _get_ref_mel(wav_ref)

    # ── Auto-trim ref_text to match actual audio speaking rate ────────────────
    # F5-TTS uses ref_text length to estimate the speaker's talking speed.
    # If ref_text has more characters than the audio actually contains speech,
    # F5-TTS thinks the speaker is faster than they really are and generates
    # compressed, unintelligible audio.
    #
    # Chinese speech: ~6 chars / second is a realistic average.
    # We allow up to 8 chars/sec before trimming (30 % headroom).
    _CHARS_PER_SEC_MAX = 8.0          # above this → ref_text is likely too long
    audio_sec = ref_mel * 256 / 24000  # hop_length=256, sr=24000
    max_ref_chars_by_dur = int(audio_sec * _CHARS_PER_SEC_MAX)
    if len(ref_text) > max_ref_chars_by_dur:
        trimmed = ref_text.encode("utf-8")[:max_ref_chars_by_dur * 3]\
                          .decode("utf-8", errors="ignore").strip()
        for punct in ("。", "！", "？", ".", "!", "?"):
            cut = trimmed.rfind(punct)
            if cut >= len(trimmed) // 2:
                trimmed = trimmed[:cut + 1].strip()
                break
        print(f"  ⚠ ref_text too long for {audio_sec:.1f}s audio "
              f"({len(ref_text)} chars → trimming to {len(trimmed)}).")
        print(f"     For best quality, manually shorten your ref_text to "
              f"~{int(audio_sec * 6)} chars in the Voice tab.")
        ref_text = trimmed

    ref_chars = max(1, len(ref_text))
    head_room = max(100, _F5_SINGLE_BATCH_THRESHOLD - ref_mel)   # gen frames allowed
    safe_max  = max(20, int(head_room * 0.70 * ref_chars / ref_mel))

    env_max   = int(os.getenv("TTS_CHUNK_SIZE", 250))
    max_chars = min(safe_max, env_max)
    min_chunk = max(20, min(max_chars, ref_chars))    # also keep chunks ≥ ref_text length
    print(f"  ref_mel={ref_mel}  ref_chars={ref_chars}  safe_max={safe_max}"
          f"  → max_chars={max_chars}  min_chunk={min_chunk}")

    chunks = chunk_text(gen_text, max_chars=max_chars, min_chars=min_chunk)
    total = len(chunks)
    if chunks:
        print(f"  Chunks  : {total}  |  first={chunks[0][:60]!r}")
    else:
        print("  ⚠ No speakable chunks — document contains no letters or digits.")
        print(f"     First 120 chars of raw text: {gen_text[:120]!r}")

    if not chunks:
        raise ValueError(
            "No speakable text found in this document. "
            f"The extracted text starts with: {gen_text[:80]!r}. "
            "If this looks wrong (garbled characters or only punctuation), "
            "please DELETE the document in the Documents tab and re-upload it — "
            "the app will re-extract the text with the latest encoding fixes."
        )
    all_audio: list[np.ndarray] = []
    sample_rate = 24000

    try:
        for i, chunk in enumerate(chunks):
            audio_arr, sr = _infer_chunk(
                tts, wav_ref, ref_text, chunk,
                speed=1.0, remove_silence=remove_silence,
                _ref_mel=ref_mel,
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
