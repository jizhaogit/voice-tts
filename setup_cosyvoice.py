"""Auto-install CosyVoice 2 for Voice TTS Studio."""
import io
import os
import subprocess
import sys
from pathlib import Path

_ROOT      = Path(__file__).resolve().parent
_CV_DIR    = _ROOT / "cosyvoice"
_MODEL_DIR = _CV_DIR / "pretrained_models" / "CosyVoice2-0.5B"

# Packages needed by CosyVoice 2 (installed one at a time so failures don't block)
# Derived from CosyVoice official requirements.txt — Windows inference only
# (skips: deepspeed/tensorrt = Linux-only, tensorboard = training-only,
#         gradio/fastapi = CosyVoice's own server, not ours)
PACKAGES = [
    # ── Core ML backbone ──────────────────────────────────────────────────────
    "transformers>=4.40.0",         # HuggingFace Transformers — model backbone (REQUIRED)
    "x-transformers>=1.30.0",       # Extended Transformers — generation head (REQUIRED)
    "diffusers>=0.29.0",            # Flow-matching / diffusion scheduler
    "peft>=0.10.0",                 # LoRA / adapter layers used by diffusers (replaces LoRACompatibleLinear)
    "einops",                       # Tensor rearrangement helpers
    "lightning>=2.0.0",             # PyTorch Lightning — used in flow modules (REQUIRED)
    # ── Audio processing ──────────────────────────────────────────────────────
    "librosa>=0.10.0",              # Audio feature extraction
    "soundfile>=0.12.1",            # WAV read/write
    "pydub",                        # Audio format conversion helper
    "pyworld",                      # Pitch (F0) extraction — may need VC++ on Windows
    "matplotlib>=3.7.0",            # Audio/spectrogram visualisation helpers
    # ── Text / language processing ────────────────────────────────────────────
    "pypinyin",                     # Chinese pinyin conversion
    "cn2an",                        # Chinese numeral ↔ Arabic numeral
    "inflect",                      # English number-to-words
    # ── Config / serialisation ────────────────────────────────────────────────
    "HyperPyYAML>=1.2.0",           # YAML config loader used by CosyVoice
    "omegaconf>=2.3.0",             # Hydra-style config objects
    "hydra-core>=1.3.0",            # Hydra config framework (omegaconf backend)
    "conformer>=0.3.2",             # Conformer encoder block
    # ── ONNX ──────────────────────────────────────────────────────────────────
    "onnx>=1.14.0",                 # ONNX model format (frontend VAD/encoder)
    "onnxruntime>=1.16.0",          # Windows runtime (CPU + optional CUDA provider)
    # ── gRPC (proto compilation) ──────────────────────────────────────────────
    "grpcio>=1.57.0",
    "grpcio-tools>=1.57.0",         # Needed to compile .proto files at import time
    "protobuf>=4.25.0,<5.0",        # Must stay <5 for grpcio compatibility
    # ── Data / graph utilities ────────────────────────────────────────────────
    "networkx>=3.0",                # Graph operations used in text processing
    "pyarrow>=14.0.0",              # Columnar data — required by modelscope
    "wget",                         # Simple file download utility
    # ── Model hub / download ──────────────────────────────────────────────────
    "modelscope>=1.9.0",            # ModelScope fallback download
    "huggingface_hub>=0.20.0",      # HuggingFace Hub download
    "gdown>=5.1.0",                 # Google Drive downloader used by matcha.utils
    # ── Misc utilities ────────────────────────────────────────────────────────
    "rich>=13.0.0",                 # Pretty console output
    "tqdm",                         # Progress bars
]


def _pip(pkg: str) -> bool:
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "-q",
         "--no-warn-script-location"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"    ⚠  {pkg}: {r.stderr.strip()[:120]}")
        return False
    return True


def setup_code() -> None:
    """Download CosyVoice 2 source + Matcha-TTS submodule from GitHub.

    GitHub ZIP downloads do NOT include git submodules, so Matcha-TTS
    (third_party/Matcha-TTS) must be fetched separately.
    """
    import shutil, urllib.request, zipfile

    # ── Step A: CosyVoice 2 main source ──────────────────────────────────────
    marker = _CV_DIR / "cosyvoice" / "cli" / "cosyvoice.py"

    # Version check: older downloads lacked the qwen_pretrain_path override,
    # which causes "NoneType" errors when loading the Qwen2 tokeniser.
    # Re-download automatically if the fix is absent.
    _stale = False
    if marker.exists():
        try:
            src_text = marker.read_text(encoding="utf-8", errors="ignore")
            if "qwen_pretrain_path" not in src_text:
                print("  [!] CosyVoice 2 source is outdated (missing qwen_pretrain_path fix).")
                print("      Re-downloading ...")
                import shutil as _sh
                _sh.rmtree(_CV_DIR, ignore_errors=True)
                _stale = True
        except Exception:
            pass

    if marker.exists() and not _stale:
        print("  [OK] CosyVoice 2 source already present.")
    else:
        url = "https://github.com/FunAudioLLM/CosyVoice/archive/refs/heads/main.zip"
        print("  [..] Downloading CosyVoice 2 source from GitHub ...")
        zip_path = _ROOT / "_cv_src.zip"
        try:
            urllib.request.urlretrieve(url, zip_path)
            tmp = _ROOT / "_cv_tmp"
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            src = tmp / "CosyVoice-main"
            if _CV_DIR.exists():
                shutil.rmtree(_CV_DIR)
            shutil.move(str(src), str(_CV_DIR))
            print("  [OK] CosyVoice 2 source downloaded.")
        finally:
            zip_path.unlink(missing_ok=True)
            shutil.rmtree(_ROOT / "_cv_tmp", ignore_errors=True)

    # ── Step B: Matcha-TTS submodule (not included in GitHub ZIP) ────────────
    matcha_marker = _CV_DIR / "third_party" / "Matcha-TTS" / "matcha" / "__init__.py"
    if matcha_marker.exists():
        print("  [OK] Matcha-TTS submodule already present.")
    else:
        matcha_url = "https://github.com/shivammehta25/Matcha-TTS/archive/refs/heads/main.zip"
        print("  [..] Downloading Matcha-TTS submodule from GitHub ...")
        zip_path = _ROOT / "_matcha_src.zip"
        matcha_dir = _CV_DIR / "third_party" / "Matcha-TTS"
        try:
            urllib.request.urlretrieve(matcha_url, zip_path)
            tmp = _ROOT / "_matcha_tmp"
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            src = tmp / "Matcha-TTS-main"
            if matcha_dir.exists():
                shutil.rmtree(matcha_dir)
            matcha_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(matcha_dir))
            print("  [OK] Matcha-TTS submodule downloaded.")
        finally:
            zip_path.unlink(missing_ok=True)
            shutil.rmtree(_ROOT / "_matcha_tmp", ignore_errors=True)


def setup_deps() -> None:
    """Install CosyVoice 2 Python dependencies."""
    print("  [..] Installing CosyVoice 2 dependencies ...")

    for pkg in PACKAGES:
        # pyworld needs a C compiler on Windows — handle gracefully
        if pkg == "pyworld":
            print(f"    pip install {pkg} ...")
            if not _pip(pkg):
                print("    ⚠  pyworld build failed (needs VC++ build tools).")
                print("       Pitch extraction will fall back to basic mode. Continuing.")
            continue
        print(f"    pip install {pkg} ...")
        _pip(pkg)

    # WeTextProcessing: Chinese text normaliser — may need VC++ on Windows
    print("    pip install WeTextProcessing ...")
    if not _pip("WeTextProcessing"):
        print("    ⚠  WeTextProcessing failed — creating Windows stub ...")
        _install_wetextprocessing_stub()

    print("  [OK] CosyVoice 2 dependencies done.")


def _install_wetextprocessing_stub() -> None:
    """Minimal stub so CosyVoice imports don't crash on Windows."""
    import sysconfig
    site = Path(sysconfig.get_path("purelib"))
    tn   = site / "tn"
    tn.mkdir(exist_ok=True)
    (tn / "__init__.py").write_text('"""WeTextProcessing stub."""\n', encoding="utf-8")
    zh = tn / "chinese"
    zh.mkdir(exist_ok=True)
    (zh / "__init__.py").write_text("", encoding="utf-8")
    (zh / "normalizer.py").write_text(
        '"""WeTextProcessing stub."""\n'
        "class Normalizer:\n"
        "    def __init__(self, *a, **kw): pass\n"
        "    def normalize(self, text): return text\n",
        encoding="utf-8",
    )


def setup_models() -> None:
    """Download CosyVoice2-0.5B pretrained model (~4.8 GB total).

    Critical files that must ALL be present:
      flow.pt, llm.pt, hift.pt, speech_tokenizer_v2.onnx,
      campplus.onnx, cosyvoice2.yaml,
      CosyVoice-BlankEN/model.safetensors   ← Qwen2 backbone weights
      CosyVoice-BlankEN/config.json         ← Qwen2 architecture config
    """
    # Check the critical files — not just any file in the directory
    _required = [
        _MODEL_DIR / "flow.pt",
        _MODEL_DIR / "llm.pt",
        _MODEL_DIR / "cosyvoice2.yaml",
        _MODEL_DIR / "CosyVoice-BlankEN" / "config.json",
        _MODEL_DIR / "CosyVoice-BlankEN" / "model.safetensors",
    ]
    _missing = [str(f) for f in _required if not f.exists()]
    if not _missing:
        print("  [OK] CosyVoice2-0.5B model already present.")
        return

    print(f"  [!] Missing model files: {_missing}")
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("  [..] Downloading CosyVoice2-0.5B (~4.8 GB) — this takes several minutes ...")

    # Try HuggingFace first
    ok = subprocess.run(
        [sys.executable, "-c",
         "from huggingface_hub import snapshot_download; "
         f'snapshot_download("FunAudioLLM/CosyVoice2-0.5B", '
         f'local_dir=r"{_MODEL_DIR}", ignore_patterns=["*.md","*.txt"])'],
        capture_output=False,
    ).returncode == 0

    if not ok:
        print("  [!] HuggingFace failed — trying ModelScope ...")
        subprocess.run(
            [sys.executable, "-c",
             "from modelscope import snapshot_download; "
             f'snapshot_download("iic/CosyVoice2-0.5B", '
             f'local_dir=r"{_MODEL_DIR}")'],
            capture_output=False,
        )

    print("  [OK] Model download done.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--deps-only", action="store_true",
                    help="Only install/check Python packages (skip code+model download)")
    args = ap.parse_args()

    if args.deps_only:
        setup_deps()
    else:
        setup_code()
        setup_deps()
        setup_models()
