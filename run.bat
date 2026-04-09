@echo off
setlocal EnableDelayedExpansion
title Voice TTS Studio
:: Always run relative to this script's folder
cd /d "%~dp0"

echo.
echo  =====================================================
echo    Voice TTS Studio  [Portable]
echo  =====================================================
echo.

:: ════════════════════════════════════════════════════════
:: STEP 1 — Locate or download a Python runtime
:: ════════════════════════════════════════════════════════

set PYTHON=runtime\python.exe

if exist "%PYTHON%" (
    echo  [OK] Portable Python runtime found.
    :: Ensure pip is available (repairs broken first-run installs)
    runtime\python.exe -m pip --version >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo  [..] pip not found — repairing...
        :: Re-apply site-packages fix (idempotent)
        for %%f in (runtime\python3*._pth) do (
            powershell -NoProfile -Command ^
                "(Get-Content '%%f') -replace '#import site','import site' | Set-Content '%%f'"
        )
        powershell -NoProfile -Command ^
            "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' " ^
            "-OutFile 'get-pip.py' -UseBasicParsing"
        runtime\python.exe get-pip.py --no-warn-script-location --quiet
        del /f /q get-pip.py >nul 2>&1
        echo  [OK] pip repaired.
    )
    echo.
    goto :check_packages
)

:: ── No bundled runtime yet — set it up now ───────────────
echo  [..] First-time setup: downloading portable Python runtime...
echo      This happens once and takes about 1-2 minutes.
echo.

powershell -NoProfile -Command "exit 0" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] PowerShell is required but was not found.
    pause & exit /b 1
)

set PY_VER=3.11.9
set PY_ZIP=python-%PY_VER%-embed-amd64.zip
set PY_URL=https://www.python.org/ftp/python/%PY_VER%/%PY_ZIP%

echo  [..] Downloading Python %PY_VER% embeddable package (~8 MB)...
powershell -NoProfile -Command ^
    "try { Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_ZIP%' -UseBasicParsing } " ^
    "catch { Write-Host $_.Exception.Message; exit 1 }"
if %ERRORLEVEL% neq 0 (
    echo.
    echo  [ERROR] Download failed. Check your internet connection.
    if exist "%PY_ZIP%" del /f /q "%PY_ZIP%"
    pause & exit /b 1
)

echo  [..] Extracting runtime...
if exist runtime rmdir /s /q runtime
mkdir runtime
powershell -NoProfile -Command ^
    "Expand-Archive -Path '%PY_ZIP%' -DestinationPath 'runtime' -Force"
del /f /q "%PY_ZIP%" >nul 2>&1

if not exist "runtime\python.exe" (
    echo  [ERROR] Extraction failed — python.exe not found in runtime\.
    pause & exit /b 1
)

:: Enable site-packages so pip-installed packages are importable
echo  [..] Configuring runtime...
for %%f in (runtime\python3*._pth) do (
    powershell -NoProfile -Command ^
        "(Get-Content '%%f') -replace '#import site','import site' | Set-Content '%%f'"
)

:: Bootstrap pip
echo  [..] Installing pip into runtime...
powershell -NoProfile -Command ^
    "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' " ^
    "-OutFile 'get-pip.py' -UseBasicParsing"
runtime\python.exe get-pip.py --no-warn-script-location --quiet
del /f /q get-pip.py >nul 2>&1

echo  [OK] Portable Python %PY_VER% runtime is ready.
echo.

:: ════════════════════════════════════════════════════════
:: STEP 2 — Install Python packages (first run or missing)
:: ════════════════════════════════════════════════════════

:check_packages
runtime\python.exe -c "import f5_tts" >nul 2>&1
if %ERRORLEVEL%==0 (
    echo  [OK] Packages already installed.
    echo.
    goto :setup_env
)

echo  [..] Installing packages — this takes several minutes the first time...
echo.

:: ── PyTorch: GPU vs CPU ───────────────────────────────────
nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    echo  [GPU] NVIDIA GPU detected — installing PyTorch with CUDA 12.1 ^(~2.5 GB^)...
    runtime\python.exe -m pip install torch torchaudio ^
        --index-url https://download.pytorch.org/whl/cu121 ^
        --no-warn-script-location --quiet
) else (
    echo  [CPU] No GPU detected — installing PyTorch CPU build ^(~200 MB^)...
    runtime\python.exe -m pip install torch torchaudio ^
        --index-url https://download.pytorch.org/whl/cpu ^
        --no-warn-script-location --quiet
)

:: ── Application dependencies ──────────────────────────────
echo  [..] Installing application dependencies...
runtime\python.exe -m pip install -r requirements.txt ^
    --no-warn-script-location ^
    --disable-pip-version-check ^
    --quiet

:: ── Verify (do NOT rely on pip exit code) ─────────────────
runtime\python.exe -c "import f5_tts" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo  [ERROR] Package installation failed.
    echo  Try running manually:
    echo    runtime\python.exe -m pip install -r requirements.txt
    pause & exit /b 1
)

echo.
echo  [OK] All packages installed and verified.
echo.

:: ════════════════════════════════════════════════════════
:: STEP 3 — First-run environment setup
:: ════════════════════════════════════════════════════════

:setup_env
if not exist ".env" (
    if exist ".env.example" copy .env.example .env >nul
    echo  [OK] Created .env from template.
    echo.
)

for %%d in (data data\voices data\documents data\generated) do (
    if not exist "%%d" mkdir "%%d"
)

if not exist "data\db.json" (
    echo {"voices":[],"documents":[],"jobs":[]}>data\db.json
)

:: ════════════════════════════════════════════════════════
:: STEP 4 — Launch the application
:: ════════════════════════════════════════════════════════

echo  =====================================================
echo  [..] Starting Voice TTS Studio...
echo  =====================================================
echo.
echo    Browser will open at:  http://localhost:7860
echo.
echo    NOTE: First TTS generation will download the
echo    F5-TTS model (~1.2 GB) — one-time only.
echo.
echo    Keep this window open while using the app.
echo    Press Ctrl+C to stop the server.
echo.

set PYTHONPATH=%~dp0;%PYTHONPATH%
runtime\python.exe main.py %*

echo.
echo  Server stopped.
pause
