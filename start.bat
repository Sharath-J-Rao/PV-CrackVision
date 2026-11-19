@echo off
REM One-click launcher (Windows). Waits for the local API health check before opening the browser.
setlocal
cd /d "%~dp0"

REM --- 1. Node.js 20.19+ or 22.12+ : use system Node if supported, else a private copy in .\.node
set NEEDNODE=1
where node >nul 2>nul
if not errorlevel 1 (
  node scripts\node-runtime.mjs >nul 2>nul
  if not errorlevel 1 set NEEDNODE=0
)
if "%NEEDNODE%"=="1" (
  if not exist ".node\node.exe" (
    echo Supported Node.js version not found - downloading a private Node 22 copy...
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\get-node.ps1"
    if errorlevel 1 goto :fail
  )
  set "PATH=%~dp0.node;%PATH%"
)

REM --- 2. Dependencies (first run only)
if not exist node_modules\.installed (
  echo Installing dependencies - first run only, may take a few minutes...
  call npm install --no-audit --no-fund
  if errorlevel 1 goto :fail
  echo ok> node_modules\.installed
)

REM Optional CPU ONNX Runtime is needed only after a trained cell.onnx is installed in the shared data folder.
for /f "usebackq delims=" %%D in (`node scripts\data-paths.mjs`) do set "DATA_ROOT=%%D"
set "NEED_ONNX_RUNTIME=0"
if exist "%DATA_ROOT%\models\cell.onnx" set "NEED_ONNX_RUNTIME=1"
for /f "usebackq delims=" %%M in (`node scripts\data-paths.mjs --has-cell-model`) do if "%%M"=="true" set "NEED_ONNX_RUNTIME=1"
REM onnxruntime-node is now a regular dependency (installed by npm install); this is only a safety net.
if "%NEED_ONNX_RUNTIME%"=="1" if not exist "node_modules\onnxruntime-node\package.json" (
  echo Installing optional ONNX Runtime CPU package for the trained cell detector...
  call npm install onnxruntime-node --no-audit --no-fund --no-save --no-package-lock
  if errorlevel 1 goto :fail
)

REM --- 3. Config + launch
if not exist .env.local (
  copy .env.example .env.local >nul
  echo Created .env.local - optional settings only; the app runs fully offline.
)
if "%PORT%"=="" set "PORT=3000"

REM Do not open Firefox after a fixed delay. Poll /api/health and open only after Express is ready.
start "" /b powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\open-when-ready.ps1" -Port %PORT% -TimeoutSeconds 180
call npm run dev
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo PV CrackVision did not start. Read the error above; the browser will not be opened unless /api/health responds.
echo Check Node.js, npm dependencies, and whether port %PORT% is already in use.
pause
exit /b 1
