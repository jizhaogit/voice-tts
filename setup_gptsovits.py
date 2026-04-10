"""
GPT-SoVITS one-time setup: downloads code + pretrained models.
Called automatically by run.bat on first launch.
Safe to re-run: each step is skipped if already complete.
"""
import os
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

def _pip(*args):
    subprocess.run(
        [sys.executable, "-m", "pip", *args,
         "--no-warn-script-location", "--quiet", "--disable-pip-version-check"],
        check=False,
    )


def _hf_download():
    """Return (hf_hub_download, snapshot_download), installing the package if needed."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        return hf_hub_download, snapshot_download
    except ImportError:
        print("  [..] Installing huggingface_hub...")
        _pip("install", "huggingface_hub")
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

    # GitHub zips extract as  GPT-SoVITS-main/
    for candidate in ROOT.iterdir():
        if candidate.is_dir() and candidate.name.startswith("GPT-SoVITS"):
            shutil.move(str(candidate), str(GTS_DIR))
            break

    if not (GTS_DIR / "api_v2.py").exists():
        print("  [ERROR] Extraction failed — api_v2.py not found in gpt-sovits/.")
        sys.exit(1)

    print("  [OK] GPT-SoVITS code ready.")


# ---------------------------------------------------------------------------
# Step 2 — Install GPT-SoVITS Python dependencies
# ---------------------------------------------------------------------------

def setup_deps():
    req = GTS_DIR / "requirements.txt"
    if not req.exists():
        print("  [!] requirements.txt not found in gpt-sovits/ — skipping.")
        return

    # GPT-SoVITS pins old torch/torchaudio/torchvision versions that conflict
    # with the GPU build installed by run.bat.  Strip those lines before installing.
    _SKIP_PREFIXES = ("torch", "torchaudio", "torchvision", "pyopenjtalk")

    lines = req.read_text(encoding="utf-8", errors="ignore").splitlines()
    filtered = [
        ln for ln in lines
        if ln.strip() and not ln.strip().startswith("#")
        and not any(ln.strip().lower().startswith(p) for p in _SKIP_PREFIXES)
    ]

    filtered_req = GTS_DIR / "_requirements_filtered.txt"
    filtered_req.write_text("\n".join(filtered), encoding="utf-8")

    print(f"  [..] Installing {len(filtered)} GPT-SoVITS dependencies ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(filtered_req),
         "--no-warn-script-location", "--quiet", "--disable-pip-version-check"],
    )
    filtered_req.unlink(missing_ok=True)

    if result.returncode != 0:
        print("  [!] Some deps failed — continuing anyway.")
    else:
        print("  [OK] Dependencies installed.")


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
        # (description, size, check_path, download_fn)
        (
            "SoVITS acoustic model",
            "~300 MB",
            gpt_dir / "s2G2333k.pth",
            lambda: hf_hub_download(
                repo_id="lj1995/GPT-SoVITS",
                filename="gsv-v2final-pretrained/s2G2333k.pth",
                local_dir=str(MODELS_DIR),
            ),
        ),
        (
            "GPT language model",
            "~1.5 GB",
            gpt_dir / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            lambda: hf_hub_download(
                repo_id="lj1995/GPT-SoVITS",
                filename="gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
                local_dir=str(MODELS_DIR),
            ),
        ),
        (
            "Chinese BERT",
            "~1.3 GB",
            MODELS_DIR / "chinese-roberta-wwm-ext-large" / "config.json",
            lambda: snapshot_download(
                repo_id="hfl/chinese-roberta-wwm-ext-large",
                local_dir=str(MODELS_DIR / "chinese-roberta-wwm-ext-large"),
            ),
        ),
        (
            "Chinese HuBERT",
            "~380 MB",
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
