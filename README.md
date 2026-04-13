# Voice TTS Studio

Portable Windows app for zero-shot voice cloning and text-to-speech generation.
Powered by **CosyVoice 2** (Alibaba) — runs entirely on your local machine, no cloud API or account required.

---

## Features

- **Zero-shot voice cloning** — clone any voice from a 3–10 second audio sample, no GPU training needed
- **Language support** — English, Chinese, Japanese
- **Three-page web UI** — Voices, Documents, Generate
- **Voice recording** — record directly from your microphone or upload an audio file
- **Audio formats** — MP3, WAV, M4A, OGG, FLAC, AAC accepted as reference audio
- **Whisper auto-transcribe** — automatically fills in the reference text from your uploaded audio clip
- **Document library** — import TXT, PDF, DOCX, or HTML files (up to 50 MB)
- **Speed control** — 0.5× to 2.0× playback speed
- **Audio output** — download generated audio as MP3 or WAV
- **Generation history** — browse and replay past generations
- **GPU acceleration** — CUDA auto-detected (NVIDIA); falls back to CPU automatically

---

## Requirements

| Item | Detail |
|------|--------|
| OS | Windows 10 / 11 (64-bit) |
| Disk space | ~10 GB free (runtime + model) |
| Internet | Required for first-time setup only |
| GPU | NVIDIA GPU optional — CUDA 11 or 12 supported |

---

## Quick Start

1. Copy the project folder to your machine
2. Double-click **`run.bat`**
3. First run automatically installs everything (10–20 min depending on internet speed):
   - Portable Python 3.11 runtime
   - PyTorch — CUDA build if a compatible GPU is found, CPU otherwise
   - All Python dependencies
   - CosyVoice 2 source code + Matcha-TTS submodule
   - CosyVoice2-0.5B pretrained model (~4.8 GB)
4. Your browser opens at **http://localhost:7860**

> **HuggingFace blocked?** The installer automatically falls back to ModelScope for all model downloads.

> **Subsequent launches** are fast — `run.bat` skips every step that is already complete.

---

## Usage

### Step 1 — Create a Voice

1. Go to the **Voices** page
2. Click **Create New Voice**
3. Enter a voice name and select a language (or use Auto-detect)
4. Upload a clear 3–10 second audio clip, or record from your microphone
5. Click **Auto-transcribe** to fill in the reference text automatically, or type it manually
6. Click **Save**

> **Tip:** Use a clip with clear speech and no background noise or music for the best cloning quality.

### Step 2 — Upload a Document

1. Go to the **Documents** page
2. Drop or select a file — TXT, PDF, DOCX, or HTML are all supported
3. The document appears in your library with a text preview

### Step 3 — Generate Audio

1. Go to the **Generate** page
2. Select a voice and a document
3. Adjust speed if needed (default 1.0×)
4. Click **Generate Audio**
5. Download the result as MP3 or WAV when complete

> **First generation** takes 30–60 seconds while the model loads into memory. All subsequent generations are fast.

---

## Configuration

Copy `.env.example` to `.env` to customise settings:

```
PORT=7860
HOST=0.0.0.0
TTS_CHUNK_SIZE=250
```

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `7860` | Web server port |
| `HOST` | `0.0.0.0` | Bind address (`127.0.0.1` to restrict to localhost) |
| `TTS_CHUNK_SIZE` | `250` | Characters per TTS segment — lower = more natural pauses |

---

## Project Structure

```
voice-tts/
├── api/
│   ├── documents.py        # Document upload and text extraction
│   ├── generate.py         # TTS generation jobs (async)
│   └── voices.py           # Voice profile management
├── core/
│   ├── tts.py              # CosyVoice 2 engine — in-process, no HTTP server
│   ├── db.py               # Lightweight JSON database
│   └── parsers.py          # PDF / DOCX / HTML text extraction
├── static/
│   ├── index.html          # Single-page web UI
│   ├── app.js              # Frontend logic
│   └── style.css           # Styling
├── cosyvoice/              # Auto-downloaded: CosyVoice 2 source + pretrained model
├── runtime/                # Auto-downloaded: portable Python 3.11
├── data/                   # Created at runtime: voices, documents, generated audio
├── main.py                 # FastAPI application entry point
├── run.bat                 # One-click launcher and auto-installer
├── setup_cosyvoice.py      # CosyVoice 2 setup script (code + deps + model)
└── requirements.txt        # Python package dependencies
```

---

## Logs

| File | Contents |
|------|----------|
| `data/tts_run.log` | TTS generation details for each request |
| `data/cosyvoice2.log` | Model load errors on startup (if any) |

---

## Notes

- `run.bat` is safe to re-run at any time — it detects what is already installed and skips those steps
- Chinese text normalisation uses WeTextProcessing; a compatible stub is created automatically on Windows if the native build fails
- The CosyVoice 2 model runs **in-process** (no separate server process) — it loads once and stays in memory until the app is stopped
- Reference audio longer than 10 seconds is trimmed automatically; shorter than 3 seconds will produce lower quality results
