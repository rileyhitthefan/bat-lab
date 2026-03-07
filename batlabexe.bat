@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ---------------------------
REM BatLab app starter (with DB)
REM - Double-click or use shortcut to start app
REM - Reuses batlabenv if present
REM - Installs requirements ONLY if key imports missing
REM ---------------------------

set "ROOT=%~dp0"
cd /d "%ROOT%"

REM ---- Create/refresh Desktop shortcut that points to this script ----
set "SHORTCUT_NAME=BatLab App"
set "DESKTOP=%USERPROFILE%\Desktop"
if exist "%DESKTOP%\%SHORTCUT_NAME%.lnk" goto :after_shortcut

echo [INFO] Creating Desktop shortcut "%SHORTCUT_NAME%.lnk"...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:USERPROFILE + '\Desktop\BatLab App.lnk');" ^
  "$s.TargetPath = '%~f0';" ^
  "$s.WorkingDirectory = '%ROOT%';" ^
  "$s.IconLocation = '%SystemRoot%\System32\SHELL32.dll, 2';" ^
  "$s.Save()"

:after_shortcut

set "VENV_DIR=batlabenv"
set "PORT=8501"
set "HOST=127.0.0.1"
set "URL=http://%HOST%:%PORT%"

REM ---- MySQL connection defaults (used by src/db/connection.py) ----
REM You can override any of these by setting environment variables before running.
if not defined MYSQL_HOST     set "MYSQL_HOST=localhost"
if not defined MYSQL_PORT     set "MYSQL_PORT=3306"
if not defined MYSQL_USER     set "MYSQL_USER=root"
if not defined MYSQL_PASSWORD set "MYSQL_PASSWORD=root@1234"
if not defined MYSQL_DATABASE set "MYSQL_DATABASE=batlab_schema"

REM Optional: pass --force to reinstall requirements
set "FORCE=0"
if /I "%~1"=="--force" set "FORCE=1"

echo.
echo =============================
echo Bat-Lab App
echo Root: %ROOT%
echo URL : %URL%
echo MySQL: %MYSQL_HOST%:%MYSQL_PORT%  user=%MYSQL_USER%  db=%MYSQL_DATABASE%
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

  REM DB is required; ensure mysql-connector-python is installed
  python -c "import mysql.connector" >nul 2>nul
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

REM ---- DB connectivity check (fail fast if DB unreachable) ----
echo.
echo [INFO] Checking MySQL connectivity...
python -c "exec('''import sys\nfrom src.db import get_connection_params\nimport mysql.connector\np=get_connection_params()\nprint('[INFO] DB host=',p['host'],'port=',p['port'],'user=',p['user'],'database=',p['database'])\ntry:\n    conn=mysql.connector.connect(**p)\n    cur=conn.cursor()\n    cur.execute('SELECT 1')\n    cur.fetchone()\n    conn.close()\n    print('[OK] DB connection OK')\nexcept Exception as e:\n    print('[ERROR] DB connection failed:', e)\n    sys.exit(1)\n''')"
if errorlevel 1 (
  echo.
  echo [ERROR] Database check failed. Fix DB settings or start MySQL, then re-run.
  pause
  exit /b 1
)

echo.
echo [INFO] Launching Streamlit in a new window...
echo.

python -m streamlit run app.py --server.address %HOST% --server.port %PORT%
timeout /t 2 >nul

endlocal
