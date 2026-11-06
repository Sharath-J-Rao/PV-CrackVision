@echo off
REM One-click launcher (Windows). Needs no setup: downloads Node.js (if missing) and all dependencies by itself.
setlocal
cd /d "%~dp0"

REM --- 1. Node.js 20+ : use system Node if new enough, else a private copy in .\.node
set NEEDNODE=1
where node >nul 2>nul && (for /f "usebackq" %%v in (`node -p "+process.versions.node.split(/\./)[0]"`) do if %%v GEQ 20 set NEEDNODE=0)
if "%NEEDNODE%"=="1" (
  if not exist ".node\node.exe" (
    echo Node.js 20+ not found - downloading a private copy...
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\get-node.ps1"
    if errorlevel 1 goto :fail
  )
  set "PATH=%~dp0.node;%PATH%"
)

REM --- 2. Dependencies (first run only; re-runs if package.json is newer than the install)
if not exist node_modules\.installed (
  echo Installing dependencies - first run only, may take a minute...
  call npm install --no-audit --no-fund
  if errorlevel 1 goto :fail
  echo ok> node_modules\.installed
)

REM --- 3. Config + launch
if not exist .env.local (
  copy .env.example .env.local >nul
  echo Created .env.local - add GEMINI_API_KEY there for Gemini, or leave empty to run offline.
)
if "%PORT%"=="" set PORT=3000
start "" /b cmd /c "timeout /t 7 /nobreak >nul & start http://localhost:%PORT%"
call npm run dev
goto :eof

:fail
echo.
echo Setup failed - see messages above. Check your internet connection and run again.
timeout /t 30
exit /b 1
