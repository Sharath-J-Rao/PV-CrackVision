import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import { findWorkingFolder, resolveSharedDataRoot, hasCellModelInReleaseFolders } from './data-paths.mjs';

test('shared data root is the Working Folder data directory even when release ZIPs have wrapper folders', () => {
  const appRoot = path.join('D:', 'Python_Projects', 'PV-Crack Vision', 'Working Folder', 'PV_CrackVision_v1.4.5.7_Selectable_Cell_Annotation_Fix', 'PV_CrackVision_v1.4.5.7');
  const expected = path.win32.join('D:', 'Python_Projects', 'PV-Crack Vision', 'Working Folder');
  assert.equal(findWorkingFolder(appRoot), expected);
  assert.equal(resolveSharedDataRoot(appRoot, {}), path.win32.join(expected, 'data'));
});

test('PV_CRACKVISION_DATA_DIR explicitly overrides the default shared location', () => {
  const appRoot = path.resolve('some-release');
  assert.equal(resolveSharedDataRoot(appRoot, { PV_CRACKVISION_DATA_DIR: 'E:/PV-CrackVision-Shared' }), path.win32.resolve('E:/PV-CrackVision-Shared'));
});

test('launcher detects an ONNX model in a nested older release before the data migration runs', () => {
  const working = path.join(process.cwd(), `data-path-launcher-fixture-${process.pid}-${Date.now()}`);
  const wrapper = path.join(working, 'PV_CrackVision_Old_Wrapper');
  const release = path.join(wrapper, 'PV_CrackVision_v1.4.5.4');
  try {
    fs.mkdirSync(path.join(release, 'server', 'models'), { recursive: true });
    fs.mkdirSync(path.join(release, 'training'), { recursive: true });
    fs.writeFileSync(path.join(release, 'server.ts'), '');
    fs.writeFileSync(path.join(release, 'server', 'models', 'cell.onnx'), 'onnx');
    assert.equal(hasCellModelInReleaseFolders(working, path.join(working, 'PV_CrackVision_New')), true);
  } finally { fs.rmSync(working, { recursive: true, force: true }); }
});
