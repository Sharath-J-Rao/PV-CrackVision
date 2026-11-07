import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const mode = process.argv[2];
if (mode === 'production') {
  process.env.NODE_ENV = 'production';
} else if (mode === 'offline') {
  process.env.CV_ENGINE = 'classical';
} else if (mode !== 'dev') {
  console.error('Usage: node scripts/run-server.mjs <dev|production|offline>');
  process.exit(2);
}

const tsxCli = path.join(root, 'node_modules', 'tsx', 'dist', 'cli.mjs');
if (!fs.existsSync(tsxCli)) {
  console.error('The local tsx runtime is missing. Run npm install before starting PV CrackVision.');
  process.exit(1);
}

// Exit code 75 = "restart requested" (e.g. the user changed the data folder in the app). The server is started again
// so the new folder takes effect without closing this window.
const RESTART_EXIT_CODE = 75;
let child = null;
let stopping = false;
function launch() {
  child = spawn(process.execPath, [tsxCli, 'server.ts'], { cwd: root, env: process.env, stdio: 'inherit', windowsHide: true });
  child.on('error', (error) => {
    console.error(`Could not start PV CrackVision: ${error.message}`);
    process.exitCode = 1;
  });
  child.on('exit', (code) => {
    if (code === RESTART_EXIT_CODE && !stopping) { console.log('Restarting PV CrackVision to apply new settings...'); launch(); return; }
    process.exitCode = code ?? 1;
  });
}
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => { stopping = true; if (child && !child.killed) child.kill(signal); });
}
launch();
