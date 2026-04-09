"""F5-TTS wrapper — zero-shot voice cloning for EN / ZH / JA.

F5-TTS uses a reference audio clip + its transcript to clone a voice.
No fine-tuning required: inference is zero-shot.

Model download (~1.2 GB) happens automatically on first call to infer().
"""
import io
import os
import re
from pathlib import Path
from typing import Callable

import numpy as np

# Lazy-loaded singletons
_tts_instance = None


def _get_tts():
    global _tts_instance
    if _tts_instance is None:
        import torch
        from f5_tts.api import F5TTS

        model_name = os.getenv("F5_MODEL", "F5TTS_v1_Base")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading F5-TTS ({model_name}) on {device}...")
        _tts_instance = F5TTS(model=model_name, device=device)
        print("  F5-TTS ready.")
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

    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
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

    for i, chunk in enumerate(chunks):
        audio_arr, sr, _ = tts.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text,
            gen_text=chunk,
            speed=speed,
            remove_silence=remove_silence,
        )
        sample_rate = sr
        all_audio.append(audio_arr.flatten())
        if progress_cb:
            progress_cb(i + 1, total)

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

    model = whisper.load_model("base")
    lang = language if language != "auto" else None
    result = model.transcribe(audio_path, language=lang)
    return result["text"].strip()
