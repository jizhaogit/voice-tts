@echo off
setlocal EnableDelayedExpansion
title Voice TTS Studio
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
    runtime\python.exe -m pip --version >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo  [..] pip not found — repairing...
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

:: ── No bundled runtime yet — download it ─────────────────
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

echo  [..] Configuring runtime...
for %%f in (runtime\python3*._pth) do (
    powershell -NoProfile -Command ^
        "(Get-Content '%%f') -replace '#import site','import site' | Set-Content '%%f'"
)

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
runtime\python.exe -c "import requests, soundfile, fastapi" >nul 2>&1
if %ERRORLEVEL% neq 0 goto :install_packages

:: CUDA compatibility smoke-test (catches version mismatch on existing installs)
nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    runtime\python.exe -c "import torch; torch.zeros(1).cuda()" >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo  [!] PyTorch CUDA mismatch detected -- reinstalling with correct build...
        echo.
        goto :install_packages
    )
)
echo  [OK] Packages already installed.
echo.
goto :check_gptsovits

:install_packages
echo  [..] Installing packages -- this takes several minutes the first time...
echo.

:: ── Detect CUDA version and pick matching PyTorch build ──────────────────────
set "TORCH_IDX=cpu"
nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    powershell -NoProfile -Command "$t='cpu'; try { $s=(nvidia-smi 2>&1 | Out-String); if($s -match 'CUDA Version:\s*12\.[89]'){$t='cu128'} elseif($s -match 'CUDA Version:\s*1[3-9]\.'){$t='cu128'} elseif($s -match 'CUDA Version:\s*12\.[4-7]'){$t='cu124'} elseif($s -match 'CUDA Version:\s*12\.'){$t='cu121'} elseif($s -match 'CUDA Version:\s*11\.'){$t='cu118'} } catch {}; $t" > cuda_detect.tmp 2>nul
    set /p TORCH_IDX=<cuda_detect.tmp
    del /f /q cuda_detect.tmp >nul 2>&1
)

if "%TORCH_IDX%"=="cpu" (
    echo  [CPU] No compatible GPU found -- installing PyTorch CPU build...
    runtime\python.exe -m pip install torch torchaudio ^
        --index-url https://download.pytorch.org/whl/cpu ^
        --no-warn-script-location --quiet
    goto :after_torch
)

echo  [GPU] GPU detected -- installing PyTorch %TORCH_IDX% build ^(~2.5 GB^)...

if "%TORCH_IDX%"=="cu128" (
    set "IS_BLACKWELL=0"
    powershell -NoProfile -Command "$r='0'; try { $caps=(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>&1); foreach($c in $caps){ if([float]$c.Trim() -ge 10.0){$r='1'; break} } } catch {}; $r" > blackwell_detect.tmp 2>nul
    set /p IS_BLACKWELL=<blackwell_detect.tmp
    del /f /q blackwell_detect.tmp >nul 2>&1

    if "!IS_BLACKWELL!"=="1" (
        echo  [..] Blackwell GPU detected -- requiring PyTorch ^>=2.7 ...
        runtime\python.exe -m pip install "torch>=2.7.0" torchaudio ^
            --index-url https://download.pytorch.org/whl/cu128 ^
            --no-warn-script-location --quiet
    ) else (
        runtime\python.exe -m pip install torch torchaudio ^
            --index-url https://download.pytorch.org/whl/cu128 ^
            --no-warn-script-location --quiet
    )

    runtime\python.exe -c "import torch; torch.zeros(1).cuda()" >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo  [!] cu128 GPU test failed -- trying PyTorch nightly...
        runtime\python.exe -m pip install --pre torch torchaudio ^
            --index-url https://download.pytorch.org/whl/nightly/cu128 ^
            --no-warn-script-location --quiet
    )
    goto :after_torch
)

runtime\python.exe -m pip install torch torchaudio ^
    --index-url https://download.pytorch.org/whl/%TORCH_IDX% ^
    --no-warn-script-location --quiet
if %ERRORLEVEL% neq 0 (
    echo  [!] GPU build failed -- falling back to CPU build...
    runtime\python.exe -m pip install torch torchaudio ^
        --index-url https://download.pytorch.org/whl/cpu ^
        --no-warn-script-location --quiet
    set "TORCH_IDX=cpu"
)

:after_torch
echo  [..] Installing application dependencies...
runtime\python.exe -m pip install -r requirements.txt ^
    --no-warn-script-location ^
    --disable-pip-version-check ^
    --quiet

runtime\python.exe -c "import requests, soundfile, fastapi" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo  [ERROR] Package installation failed.
    echo  Try running manually:
    echo    runtime\python.exe -m pip install -r requirements.txt
    pause & exit /b 1
)

echo.
echo  [OK] All packages installed and verified ^(%TORCH_IDX%^).
echo.

:: ════════════════════════════════════════════════════════
:: STEP 3 — GPT-SoVITS: download code + pretrained models
::           Skipped automatically if already complete.
:: ════════════════════════════════════════════════════════

:check_gptsovits
:: Ensure GPT-SoVITS core dependencies are present
:: (these are sometimes missed by its requirements.txt install)
runtime\python.exe -c "import ffmpeg" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  [..] Installing ffmpeg-python ^(required by GPT-SoVITS^)...
    runtime\python.exe -m pip install ffmpeg-python --no-warn-script-location --quiet
    echo  [OK] ffmpeg-python installed.
)
runtime\python.exe -c "import pytorch_lightning" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  [..] Installing pytorch-lightning ^(required by GPT-SoVITS^)...
    runtime\python.exe -m pip install pytorch-lightning --no-warn-script-location --quiet
    echo  [OK] pytorch-lightning installed.
)

if exist "gpt-sovits\api_v2.py" (
    if exist "gpt-sovits\GPT_SoVITS\pretrained_models\gsv-v2final-pretrained\s2G2333k.pth" (
        echo  [OK] GPT-SoVITS code and models already present.
        echo.
        goto :setup_env
    )
)

echo  =====================================================
echo   GPT-SoVITS  First-time Setup
echo   Downloading ~3.5 GB of models  ^(one-time only^)
echo   This will take several minutes depending on your
echo   internet speed.  Please wait...
echo  =====================================================
echo.

runtime\python.exe setup_gptsovits.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo  [ERROR] GPT-SoVITS setup failed.
    echo  If you are in China and HuggingFace is blocked, set:
    echo    set HF_ENDPOINT=https://hf-mirror.com
    echo  then close this window and run run.bat again.
    pause & exit /b 1
)
echo.

:: ════════════════════════════════════════════════════════
:: STEP 4 — First-run environment setup
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

set HF_HUB_DISABLE_SYMLINKS_WARNING=1

:: ════════════════════════════════════════════════════════
:: STEP 5 — Launch the application
:: ════════════════════════════════════════════════════════

echo  =====================================================
echo  [..] Starting Voice TTS Studio...
echo  =====================================================
echo.
echo    Browser:  http://localhost:7860
echo.
echo    GPT-SoVITS API server starts automatically.
echo    First generation loads models into VRAM (~30 s).
echo.
echo    Keep this window open while using the app.
echo    Press Ctrl+C to stop.
echo.

:: Free port 7860 if a previous instance is still running
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":7860 " ^| findstr "LISTENING"') do (
    echo  [..] Stopping previous instance on port 7860 ^(PID %%P^)...
    taskkill /PID %%P /F >nul 2>&1
)

set PYTHONPATH=%~dp0;%PYTHONPATH%
runtime\python.exe main.py %*

echo.
echo  Server stopped.
pause
