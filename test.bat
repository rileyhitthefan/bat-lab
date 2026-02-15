@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ---------------------------
REM Test no db
REM - Reuses batlabenv if present
REM - Installs requirements ONLY if key imports missing
REM ---------------------------

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "VENV_DIR=batlabenv"
set "BATLAB_NO_DB=1"
set "PORT=8501"
set "HOST=127.0.0.1"
set "URL=http://%HOST%:%PORT%"

REM Optional: pass --force to reinstall requirements
set "FORCE=0"
if /I "%~1"=="--force" set "FORCE=1"

echo.
echo =============================
echo Test no db
echo Root: %ROOT%
echo URL : %URL%
echo FORCE REINSTALL: %FORCE%
echo =============================
echo.

if not exist "%ROOT%app.py" (
  echo [ERROR] app.py not found in %ROOT%
  pause
  exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found in PATH.
  pause
  exit /b 1
)

REM ---- Create venv if missing ----
if exist "%ROOT%%VENV_DIR%\Scripts\activate.bat" (
  echo [OK] Using existing venv: %VENV_DIR%
) else (
  echo [INFO] Creating venv: %VENV_DIR%
  python -m venv "%ROOT%%VENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Failed to create venv.
    pause
    exit /b 1
  )
)

REM ---- Activate venv ----
call "%ROOT%%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate venv.
  pause
  exit /b 1
)

REM ---- Ensure pip is usable ----
python -m pip --version >nul 2>nul
if errorlevel 1 (
  echo [ERROR] pip not available in venv.
  pause
  exit /b 1
)

REM ---- Check if key deps exist; install if missing ----
set "NEED_INSTALL=0"

if "%FORCE%"=="1" (
  set "NEED_INSTALL=1"
) else (
  REM Pick imports that are required immediately by app.py
  python -c "import scipy" >nul 2>nul
  if errorlevel 1 set "NEED_INSTALL=1"

  python -c "import streamlit" >nul 2>nul
  if errorlevel 1 set "NEED_INSTALL=1"
)

if "%NEED_INSTALL%"=="1" (
  if exist "%ROOT%requirements.txt" (
    echo [INFO] Installing/repairing dependencies from requirements.txt...
    python -m pip install --upgrade pip
    pip install -r "%ROOT%requirements.txt"
    if errorlevel 1 (
      echo [ERROR] pip install -r requirements.txt failed.
      echo         Scroll up for the first error line.
      pause
      exit /b 1
    )
  ) else (
    echo [ERROR] requirements.txt not found; cannot install dependencies.
    pause
    exit /b 1
  )
) else (
  echo [OK] Key dependencies present; skipping pip install.
)

echo.
echo [INFO] Launching Streamlit in a new window...
echo.

python -m streamlit run app.py --server.address %HOST% --server.port %PORT%
timeout /t 2 >nul

endlocal
