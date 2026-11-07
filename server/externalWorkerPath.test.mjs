import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { resolveExternalWorkerPath } from './externalWorkerPath.ts';

test('external Python workers resolve to the physical app.asar.unpacked script path', (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'pv-cv-asar-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const resources = path.join(root, 'resources');
  const appRoot = path.join(resources, 'app.asar');
  const script = path.join(resources, 'app.asar.unpacked', 'training', 'train_cell_yolo.py');
  fs.mkdirSync(path.dirname(script), { recursive: true });
  fs.writeFileSync(script, 'print("ok")');
  assert.equal(resolveExternalWorkerPath('training/train_cell_yolo.py', appRoot, resources), script);
});

test('source installs continue to resolve workers from the project root', () => {
  assert.equal(resolveExternalWorkerPath('training/install_package_dependencies.py', '/opt/pv-crackvision'), path.join('/opt/pv-crackvision', 'training', 'install_package_dependencies.py'));
});
