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
PACKAGES = [
    "conformer",
    "diffusers>=0.30.0",
    "einops",
    "grpcio",
    "HyperPyYAML",
    "inflect",
    "librosa",
    "omegaconf",
    "pypinyin",
    "rich",
    "soundfile",
    "tqdm",
    "cn2an",
    "pydub",
    "protobuf",
    "onnxruntime-gpu",
    "modelscope",
    "huggingface_hub",
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
    """Download CosyVoice 2 source from GitHub."""
    marker = _CV_DIR / "cosyvoice" / "cli" / "cosyvoice.py"
    if marker.exists():
        print("  [OK] CosyVoice 2 source already present.")
        return

    import shutil, urllib.request, zipfile

    url = "https://github.com/FunAudioLLM/CosyVoice/archive/refs/heads/main.zip"
    print(f"  [..] Downloading CosyVoice 2 source from GitHub ...")
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


def setup_deps() -> None:
    """Install CosyVoice 2 Python dependencies."""
    print("  [..] Installing CosyVoice 2 dependencies ...")
    for pkg in PACKAGES:
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
    """Download CosyVoice2-0.5B pretrained model (~2.5 GB)."""
    if _MODEL_DIR.exists() and any(_MODEL_DIR.iterdir()):
        print("  [OK] CosyVoice2-0.5B model already present.")
        return

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("  [..] Downloading CosyVoice2-0.5B (~2.5 GB) — this takes a few minutes ...")

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
