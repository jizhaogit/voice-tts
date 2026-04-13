"""Auto-install IndexTTS 1.5 for Voice TTS Studio."""
import subprocess
import sys
from pathlib import Path

_ROOT      = Path(__file__).resolve().parent
_IT_DIR    = _ROOT / "indextts"
_MODEL_DIR = _IT_DIR / "checkpoints"

# IndexTTS Python package + core dependencies
PACKAGES = [
    "indextts",                 # Main package (includes infer.py)
    "vector-quantize-pytorch",  # VQ used by the vocoder
    "transformers>=4.40.0",     # Already installed for CosyVoice — ensure version
    "librosa",                  # Audio processing
    "soundfile",                # WAV I/O
    "g2p-en",                   # English grapheme-to-phoneme
    "jieba",                    # Chinese word segmentation
    "sentencepiece",            # BPE tokenizer
    "einops",                   # Tensor ops
    "numba",                    # JIT acceleration (used by BigVGAN)
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


def setup_deps() -> None:
    """Install IndexTTS Python dependencies."""
    print("  [..] Installing IndexTTS dependencies ...")
    for pkg in PACKAGES:
        print(f"    pip install {pkg} ...")
        _pip(pkg)
    print("  [OK] IndexTTS dependencies done.")


def setup_models() -> None:
    """Download IndexTTS-1.5 pretrained model (~5.9 GB)."""
    # Critical files that must be present
    _required = [
        _MODEL_DIR / "config.yaml",
        _MODEL_DIR / "gpt.pth",
        _MODEL_DIR / "bigvgan_discriminator.pth",
        _MODEL_DIR / "bigvgan_generator.pth",
        _MODEL_DIR / "bpe.model",
    ]
    _missing = [f.name for f in _required if not f.exists()]
    if not _missing:
        print("  [OK] IndexTTS-1.5 model already present.")
        return

    print(f"  [!] Missing: {_missing}")
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("  [..] Downloading IndexTTS-1.5 (~5.9 GB) — this takes several minutes ...")

    # Try HuggingFace first
    ok = subprocess.run(
        [sys.executable, "-c",
         "from huggingface_hub import snapshot_download; "
         f'snapshot_download("IndexTeam/IndexTTS-1.5", '
         f'local_dir=r"{_MODEL_DIR}", ignore_patterns=["*.md"])'],
        capture_output=False,
    ).returncode == 0

    if not ok:
        print("  [!] HuggingFace failed — trying ModelScope ...")
        subprocess.run(
            [sys.executable, "-c",
             "from modelscope import snapshot_download; "
             f'snapshot_download("IndexTeam/IndexTTS-1.5", '
             f'local_dir=r"{_MODEL_DIR}")'],
            capture_output=False,
        )

    print("  [OK] IndexTTS-1.5 model download done.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--deps-only", action="store_true",
                    help="Only install/check Python packages (skip model download)")
    args = ap.parse_args()

    if args.deps_only:
        setup_deps()
    else:
        setup_deps()
        setup_models()
