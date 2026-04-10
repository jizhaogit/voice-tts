"""
GPT-SoVITS one-time setup: downloads code + pretrained models.
Called automatically by run.bat on first launch.
Safe to re-run: each step is skipped if already complete.

Usage:
  python setup_gptsovits.py            # full setup (code + deps + models)
  python setup_gptsovits.py --deps-only  # only install / repair dependencies
"""
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT       = Path(__file__).parent
GTS_DIR    = ROOT / "gpt-sovits"
MODELS_DIR = GTS_DIR / "GPT_SoVITS" / "pretrained_models"

# GitHub source (latest main branch)
GITHUB_ZIP = "https://github.com/RVC-Boss/GPT-SoVITS/archive/refs/heads/main.zip"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pip_install(pkg: str) -> bool:
    """Install a single package. Returns True on success."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg,
         "--no-warn-script-location", "--quiet", "--disable-pip-version-check"],
        capture_output=True,
    )
    return result.returncode == 0


def _hf_download():
    """Return (hf_hub_download, snapshot_download), installing the package if needed."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        return hf_hub_download, snapshot_download
    except ImportError:
        print("  [..] Installing huggingface_hub...")
        _pip_install("huggingface_hub")
        from huggingface_hub import hf_hub_download, snapshot_download
        return hf_hub_download, snapshot_download


# ---------------------------------------------------------------------------
# Step 1 — Download & extract GPT-SoVITS code
# ---------------------------------------------------------------------------

def setup_code():
    if (GTS_DIR / "api_v2.py").exists():
        print("  [OK] GPT-SoVITS code already present.")
        return

    zip_path = ROOT / "_gptsovits_tmp.zip"
    print("  [..] Downloading GPT-SoVITS source (~50 MB) ...")
    try:
        urllib.request.urlretrieve(GITHUB_ZIP, str(zip_path))
    except Exception as exc:
        print(f"  [ERROR] Download failed: {exc}")
        print("          Check your internet connection and try again.")
        sys.exit(1)

    print("  [..] Extracting ...")
    with zipfile.ZipFile(str(zip_path)) as zf:
        zf.extractall(str(ROOT))
    zip_path.unlink(missing_ok=True)

    for candidate in ROOT.iterdir():
        if candidate.is_dir() and candidate.name.startswith("GPT-SoVITS"):
            shutil.move(str(candidate), str(GTS_DIR))
            break

    if not (GTS_DIR / "api_v2.py").exists():
        print("  [ERROR] Extraction failed — api_v2.py not found in gpt-sovits/.")
        sys.exit(1)

    print("  [OK] GPT-SoVITS code ready.")


# ---------------------------------------------------------------------------
# Step 2 — Install ALL GPT-SoVITS Python dependencies
# ---------------------------------------------------------------------------

def setup_deps():
    """Install every package GPT-SoVITS needs, one at a time.

    Packages that need a C++ compiler (pyopenjtalk, g2pk2, ko_pron) and
    Korean/Japanese-only packages are skipped on Windows.
    torch/torchaudio/torchvision are skipped — already installed by run.bat
    with the correct GPU build.
    opencc is replaced with opencc-python-reimplemented (no compiler needed).
    """

    # Detect platform for conditional packages
    is_windows = sys.platform == "win32"
    is_x86_64  = platform.machine().lower() in ("amd64", "x86_64")

    # Full list derived from GPT-SoVITS requirements.txt + known missing packages.
    # Format: package_spec  (pip install argument, may include version constraints)
    PACKAGES = [
        # Core ML — version-pinned to match what GPT-SoVITS expects
        "numpy<2.0",
        "scipy",
        "pytorch-lightning>=2.4",
        "transformers>=4.43,<=4.50",
        "peft<0.18.0",
        "accelerate",
        "einops",
        "torchmetrics<=1.5",
        "x-transformers",
        "rotary-embedding-torch",

        # Audio processing
        "librosa==0.10.2",
        "numba",
        "ffmpeg-python",
        "soundfile",
        "av>=11",

        # Chinese NLP
        "jieba",
        "jieba_fast",
        "pypinyin",
        "cn2an",
        "split-lang",
        "fast-langdetect>=0.3.1",
        "LangSegment",
        "opencc-python-reimplemented",  # replaces opencc (no compiler needed)
        "ToJyutping",

        # English NLP
        "g2p-en",
        "wordsegment",

        # Model / inference
        "sentencepiece",
        "huggingface_hub",
        "modelscope",
        "funasr==1.0.27",
        "ctranslate2>=4.0,<5",

        # Utilities
        "tensorboard",
        "tqdm",
        "chardet",
        "PyYAML",
        "psutil",
        "Pillow",
        "pydantic<=2.10.6",
        "gradio<5",
    ]

    # onnxruntime: GPU variant on x86_64 Windows, CPU variant elsewhere
    if is_x86_64:
        PACKAGES.append("onnxruntime-gpu")
    else:
        PACKAGES.append("onnxruntime")

    # Korean packages only on non-Windows (they require system libs)
    if not is_windows:
        PACKAGES += ["g2pk2", "ko_pron", "python-mecab-ko"]

    total   = len(PACKAGES)
    failed  = []

    print(f"  [..] Installing {total} GPT-SoVITS dependencies (one-time) ...")
    for i, pkg in enumerate(PACKAGES, 1):
        label = pkg.split(">=")[0].split("<=")[0].split("<")[0].split(">")[0].split("==")[0]
        print(f"       [{i:>2}/{total}] {label} ...", end=" ", flush=True)
        ok = _pip_install(pkg)
        print("[OK]" if ok else "[SKIP]")
        if not ok:
            failed.append(pkg)

    if failed:
        print(f"\n  [!] {len(failed)} package(s) could not be installed:")
        for p in failed:
            print(f"        - {p}")
        print("      Applying shims for known build-only packages ...")

    # jieba_fast: C extension, often fails to build on Windows.
    # Create a shim module so GPT-SoVITS can still import it using jieba.
    _install_jieba_fast_shim()

    print("\n  [OK] Dependencies ready.")


def _install_jieba_fast_shim():
    """If jieba_fast is not importable, create a shim package that wraps jieba.

    GPT-SoVITS imports both  'jieba_fast'  and  'jieba_fast.posseg',
    so the shim must be a package (directory) not a single .py file.
    """
    try:
        import jieba_fast.posseg  # noqa: F401 — already works
        return
    except ImportError:
        pass

    import sysconfig
    site_packages = sysconfig.get_path("purelib")
    if not site_packages:
        print("  [!] Could not locate site-packages — jieba_fast shim skipped.")
        return

    # Remove any leftover flat-file shim from a previous run
    flat = Path(site_packages) / "jieba_fast.py"
    if flat.exists():
        flat.unlink()

    pkg = Path(site_packages) / "jieba_fast"
    pkg.mkdir(exist_ok=True)

    # __init__.py — mirrors the jieba top-level API
    (pkg / "__init__.py").write_text(
        '"""jieba_fast shim: C extension unavailable, using jieba as fallback."""\n'
        "from jieba import *  # noqa\n"
        "import jieba as _j\n"
        "cut             = _j.cut\n"
        "lcut            = _j.lcut\n"
        "cut_for_search  = _j.cut_for_search\n"
        "lcut_for_search = _j.lcut_for_search\n"
        "load_userdict   = _j.load_userdict\n"
        "add_word        = _j.add_word\n"
        "initialize      = _j.initialize\n",
        encoding="utf-8",
    )

    # posseg.py — mirrors jieba.posseg (used as  import jieba_fast.posseg as psg)
    (pkg / "posseg.py").write_text(
        '"""jieba_fast.posseg shim — delegates to jieba.posseg."""\n'
        "from jieba.posseg import *  # noqa\n"
        "from jieba.posseg import cut\n",
        encoding="utf-8",
    )

    print("  [OK] jieba_fast shim package installed (uses jieba as fallback).")


# ---------------------------------------------------------------------------
# Step 3 — Download pretrained models (~3.5 GB total, one-time)
# ---------------------------------------------------------------------------

def setup_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    gpt_dir = MODELS_DIR / "gsv-v2final-pretrained"
    gpt_dir.mkdir(exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    hf_hub_download, snapshot_download = _hf_download()

    models = [
        (
            "SoVITS acoustic model", "~300 MB",
            gpt_dir / "s2G2333k.pth",
            lambda: hf_hub_download(
                repo_id="lj1995/GPT-SoVITS",
                filename="gsv-v2final-pretrained/s2G2333k.pth",
                local_dir=str(MODELS_DIR),
            ),
        ),
        (
            "GPT language model", "~1.5 GB",
            gpt_dir / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            lambda: hf_hub_download(
                repo_id="lj1995/GPT-SoVITS",
                filename="gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
                local_dir=str(MODELS_DIR),
            ),
        ),
        (
            "Chinese BERT", "~1.3 GB",
            MODELS_DIR / "chinese-roberta-wwm-ext-large" / "config.json",
            lambda: snapshot_download(
                repo_id="hfl/chinese-roberta-wwm-ext-large",
                local_dir=str(MODELS_DIR / "chinese-roberta-wwm-ext-large"),
            ),
        ),
        (
            "Chinese HuBERT", "~380 MB",
            MODELS_DIR / "chinese-hubert-base" / "config.json",
            lambda: snapshot_download(
                repo_id="TencentGameMate/chinese-hubert-base",
                local_dir=str(MODELS_DIR / "chinese-hubert-base"),
            ),
        ),
    ]

    for name, size, check_path, download_fn in models:
        if check_path.exists():
            print(f"  [OK] {name} already present.")
            continue
        print(f"  [..] Downloading {name} ({size}) ...")
        try:
            download_fn()
            print(f"  [OK] {name} downloaded.")
        except Exception as exc:
            print(f"  [ERROR] Failed to download {name}: {exc}")
            print("          If you are in China, try setting:")
            print("            set HF_ENDPOINT=https://hf-mirror.com")
            print("          then re-run run.bat.")
            sys.exit(1)

    print("  [OK] All pretrained models ready.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    deps_only = "--deps-only" in sys.argv

    if deps_only:
        print()
        print("  ── GPT-SoVITS dependency check ─────────────")
        setup_deps()
        print()
    else:
        print()
        print("  ════════════════════════════════════════════")
        print("    GPT-SoVITS  —  First-time Setup")
        print("  ════════════════════════════════════════════")
        print()
        print("  ── Step 1/3 : Source code ──────────────────")
        setup_code()
        print()
        print("  ── Step 2/3 : Python dependencies ──────────")
        setup_deps()
        print()
        print("  ── Step 3/3 : Pretrained models (~3.5 GB) ──")
        print("  NOTE: Large download — please wait.")
        print()
        setup_models()
        print()
        print("  ════════════════════════════════════════════")
        print("    GPT-SoVITS setup complete!")
        print("  ════════════════════════════════════════════")
        print()
