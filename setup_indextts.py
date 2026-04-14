"""Auto-install IndexTTS 1.5 for Voice TTS Studio."""
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_ROOT      = Path(__file__).resolve().parent
_IT_DIR    = _ROOT / "indextts"
_MODEL_DIR = _IT_DIR / "checkpoints"

# IndexTTS source (not on PyPI — install directly from GitHub zip)
_INDEXTTS_GITHUB_ZIP = (
    "https://github.com/index-tts/index-tts/archive/refs/heads/main.zip"
)

# Additional dependencies
PACKAGES = [
    "vector-quantize-pytorch",  # VQ used by the vocoder
    "transformers>=4.40.0",     # Already installed for CosyVoice — ensure version
    "librosa",                  # Audio processing
    "soundfile",                # WAV I/O
    "g2p-en",                   # English grapheme-to-phoneme
    "jieba",                    # Chinese word segmentation
    "sentencepiece",            # BPE tokenizer
    "einops",                   # Tensor ops
    "numba",                    # JIT acceleration (used by BigVGAN)
    "inflect",                  # Number-to-words for English TN
    "unidecode",                # Unicode text normaliser
]


def _pip(pkg: str, silent_fail: bool = False) -> bool:
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "-q",
         "--no-warn-script-location"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        if not silent_fail:
            first_line = next(
                (l.strip() for l in r.stderr.splitlines() if l.strip()), "build failed"
            )
            print(f"    ⚠  {pkg}: {first_line[:120]}")
        return False
    return True


def _download_zip(url: str, dest: Path) -> bool:
    """Download *url* to *dest* using urllib (browser-like, no git required)."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},  # avoid GitHub throttling pip UA
        )
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        return True
    except Exception as exc:
        print(f"    ⚠  Download failed: {exc}")
        return False


def _install_indextts() -> bool:
    """Install indextts from GitHub (not on PyPI, no git required)."""
    # Check if already importable
    check = subprocess.run(
        [sys.executable, "-c", "import indextts"],
        capture_output=True,
    )
    if check.returncode == 0:
        print("    [OK] indextts already installed.")
        return True

    print("    Downloading indextts source from GitHub ...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="indextts_"))
    zip_path = tmp_dir / "indextts.zip"

    try:
        if not _download_zip(_INDEXTTS_GITHUB_ZIP, zip_path):
            return False

        print("    Extracting ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        # GitHub zips extract to a folder named "<repo>-<branch>"
        candidates = [d for d in tmp_dir.iterdir()
                      if d.is_dir() and d.name != "__MACOSX"]
        if not candidates:
            print("    ⚠  Zip extraction produced no folders.")
            return False
        src_dir = candidates[0]

        print(f"    pip install from {src_dir.name} ...")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", str(src_dir),
             "-q", "--no-warn-script-location"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            first_line = next(
                (l.strip() for l in r.stderr.splitlines() if l.strip()), "install failed"
            )
            print(f"    ⚠  pip install failed: {first_line[:120]}")
            return False

        print("    [OK] indextts installed.")
        return True

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def setup_deps() -> None:
    """Install IndexTTS Python dependencies."""
    print("  [..] Installing IndexTTS dependencies ...")

    # IndexTTS itself — GitHub only
    _install_indextts()

    # Supporting packages
    for pkg in PACKAGES:
        print(f"    pip install {pkg} ...")
        _pip(pkg)

    print("  [OK] IndexTTS dependencies done.")


def setup_models() -> None:
    """Download IndexTTS-1.5 pretrained model (~5.9 GB)."""
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
