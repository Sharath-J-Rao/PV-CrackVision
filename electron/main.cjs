'use strict';

const { app, BrowserWindow, dialog, shell, Menu } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const net = require('node:net');
const { pathToFileURL } = require('node:url');
const { resolveDataRoot } = require('./data-root.cjs');

const APP_ID = 'com.sharathjrao.pvcrackvision';
const PRODUCT_NAME = 'PV CrackVision';
let mainWindow = null;
let logFile = null;
let serverPort = 0;

// Set Electron's persistent profile before the app emits `ready`, and align it with
// the existing server-side config-file convention (%APPDATA%\PV CrackVision).
const USER_DATA_DIR = path.join(app.getPath('appData'), PRODUCT_NAME);
app.setPath('userData', USER_DATA_DIR);
app.setAppUserModelId(APP_ID);
process.env.PV_CRACKVISION_CONFIG_DIR = USER_DATA_DIR;
process.env.PV_CRACKVISION_PACKAGED = '1';
process.env.NODE_ENV = 'production';
process.env.HOST = '127.0.0.1';

function log(message, error) {
  const line = `[${new Date().toISOString()}] ${message}${error ? `\n${error.stack || error}` : ''}\n`;
  try { if (logFile) fs.appendFileSync(logFile, line, 'utf8'); } catch { /* logging must not stop startup */ }
  process.stdout.write(line);
}

function readJson(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch { return {}; }
}

function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.tmp`;
  fs.writeFileSync(temporary, JSON.stringify(value, null, 2), 'utf8');
  fs.renameSync(temporary, file);
}

async function chooseDataRoot(userDataDir, configPath) {
  const config = readJson(configPath);
  const result = resolveDataRoot({
    envDataRoot: process.env.PV_CRACKVISION_DATA_DIR,
    configuredDataRoot: config.dataRoot,
    startDirectories: [
      process.env.PORTABLE_EXECUTABLE_DIR,
      process.env.PORTABLE_EXECUTABLE_FILE ? path.dirname(process.env.PORTABLE_EXECUTABLE_FILE) : null,
      path.dirname(process.execPath),
      app.getAppPath(),
      process.cwd(),
      process.resourcesPath,
    ].filter(Boolean),
    userDataDirectory: userDataDir,
  });

  if (result.source === 'working-folder') {
    // Persist auto-discovery so a portable EXE keeps using the same dataset even if it is moved later.
    writeJson(configPath, { ...config, dataRoot: result.dataRoot, updatedAt: new Date().toISOString() });
    return result.dataRoot;
  }
  if (result.source !== 'new-profile') return result.dataRoot;

  const choice = await dialog.showMessageBox({
    type: 'question',
    title: `${PRODUCT_NAME} — Training Data`,
    message: 'Where should PV CrackVision keep its models and training data?',
    detail: 'To continue using your existing YOLO model, crack classifier, annotations and training images, choose your existing Working Folder\\data directory. Choose Start fresh only when you do not need existing training data.',
    buttons: ['Choose existing data folder', 'Start fresh'],
    defaultId: 0,
    cancelId: 1,
    noLink: true,
  });

  if (choice.response === 0) {
    const selected = await dialog.showOpenDialog({
      title: 'Select your existing PV CrackVision data folder',
      buttonLabel: 'Use this data folder',
      properties: ['openDirectory', 'createDirectory'],
    });
    if (!selected.canceled && selected.filePaths[0]) {
      const selectedPath = path.resolve(selected.filePaths[0]);
      const looksLikeData = /data$/i.test(path.basename(selectedPath))
        || ['training', 'models', 'runtime'].some((part) => {
          try { return fs.statSync(path.join(selectedPath, part)).isDirectory(); } catch { return false; }
        });
      if (!looksLikeData) {
        await dialog.showMessageBox({ type: 'warning', title: 'Select the data folder', message: 'The selected folder does not look like a PV CrackVision data folder.', detail: 'Select the existing folder named data, containing training and/or models subfolders. No files were changed.' });
        return chooseDataRoot(userDataDir, configPath);
      }
      writeJson(configPath, { ...config, dataRoot: selectedPath, updatedAt: new Date().toISOString() });
      return selectedPath;
    }
  }

  return result.dataRoot;
}

async function getAvailablePort() {
  return new Promise((resolve, reject) => {
    const probe = net.createServer();
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const address = probe.address();
      const port = typeof address === 'object' && address ? address.port : 0;
      probe.close((error) => error ? reject(error) : resolve(port));
    });
  });
}

async function waitForHealth(url, timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${url}/api/health`, { signal: AbortSignal.timeout(2500) });
      if (response.ok) {
        const body = await response.json();
        if (body && body.ok) return body;
      }
    } catch (error) { lastError = error; }
    await new Promise((resolve) => setTimeout(resolve, 350));
  }
  throw new Error(`The local PV CrackVision server did not become ready within ${timeoutMs / 1000} seconds.${lastError ? ` Last error: ${lastError.message}` : ''}`);
}

async function createMainWindow(url) {
  mainWindow = new BrowserWindow({
    width: 1540,
    height: 980,
    minWidth: 1120,
    minHeight: 720,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#090d14',
    title: PRODUCT_NAME,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({ url: target }) => {
    if (/^https?:\/\//i.test(target)) shell.openExternal(target).catch((error) => log('Could not open external URL.', error));
    return { action: 'deny' };
  });
  mainWindow.webContents.on('will-navigate', (event, target) => {
    if (!target.startsWith(url)) event.preventDefault();
  });
  mainWindow.once('ready-to-show', () => mainWindow && mainWindow.show());
  mainWindow.on('closed', () => { mainWindow = null; });
  await mainWindow.loadURL(url);
}

async function startDesktop() {
  const userDataDir = USER_DATA_DIR;
  fs.mkdirSync(userDataDir, { recursive: true });
  logFile = path.join(userDataDir, 'logs', 'desktop.log');
  fs.mkdirSync(path.dirname(logFile), { recursive: true });

  const dataRoot = await chooseDataRoot(userDataDir, path.join(userDataDir, 'config.json'));
  process.env.PV_CRACKVISION_DATA_DIR = dataRoot;
  fs.mkdirSync(dataRoot, { recursive: true });
  serverPort = await getAvailablePort();
  process.env.PORT = String(serverPort);

  log(`Starting desktop application v${app.getVersion()}.`);
  log(`Persistent data folder: ${dataRoot}`);
  log(`Application bundle: ${app.getAppPath()}`);

  // Run the existing Express/TypeScript backend in Electron's bundled Node runtime.
  // This removes the need for a separately installed Node.js or a batch launcher.
  const { register } = await import('tsx/esm/api');
  register();
  const serverEntry = path.join(app.getAppPath(), 'server.ts');
  await import(pathToFileURL(serverEntry).href);

  const baseUrl = `http://127.0.0.1:${serverPort}`;
  const health = await waitForHealth(baseUrl);
  log(`Local server ready. Inspection engine: ${health.engine || 'local'}.`);
  if (process.argv.includes('--pv-smoke-test') || app.commandLine.hasSwitch('pv-smoke-test')) {
    const unpackedTraining = path.join(process.resourcesPath, 'app.asar.unpacked', 'training');
    for (const worker of ['train_cell_yolo.py', 'install_package_dependencies.py']) {
      const workerPath = path.join(unpackedTraining, worker);
      if (!fs.existsSync(workerPath)) throw new Error(`Packaged Python worker is missing: ${workerPath}`);
    }
    const sharpModule = await import('sharp');
    const sharp = sharpModule.default || sharpModule;
    const sharpProbe = await sharp({ create: { width: 1, height: 1, channels: 4, background: '#000000' } }).png().toBuffer();
    if (sharpProbe[0] !== 137 || sharpProbe[1] !== 80) throw new Error('Packaged Sharp failed its PNG smoke test.');
    const ort = await import('onnxruntime-node');
    if (typeof ort.InferenceSession?.create !== 'function') throw new Error('Packaged ONNX Runtime did not expose InferenceSession.create.');
    log('PACKAGED_DESKTOP_SMOKE_TEST_OK: backend health, Sharp native PNG, ONNX Runtime and external Python workers verified.');
    app.exit(0);
    return;
  }
  await createMainWindow(baseUrl);
}

app.on('window-all-closed', () => app.quit());
app.on('second-instance', () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  }
});

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.whenReady().then(() => {
    Menu.setApplicationMenu(null);
    return startDesktop();
  }).catch((error) => {
    log('Desktop startup failed.', error);
    dialog.showErrorBox(`${PRODUCT_NAME} could not start`, `${error.message || error}\n\nDiagnostic log: ${logFile || 'not created'}`);
    app.quit();
  });
}
