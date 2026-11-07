'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { findWorkingFolderData, resolveDataRoot } = require('./data-root.cjs');

test('finds the existing shared data folder above a nested desktop package', (t) => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'pv-cv-data-'));
  t.after(() => fs.rmSync(temp, { recursive: true, force: true }));
  const working = path.join(temp, 'Working Folder');
  const packageDir = path.join(working, 'New folder', 'PV CrackVision Desktop', 'resources');
  const data = path.join(working, 'data');
  fs.mkdirSync(packageDir, { recursive: true });
  fs.mkdirSync(path.join(data, 'training'), { recursive: true });
  assert.equal(findWorkingFolderData([packageDir]), data);
  assert.deepEqual(resolveDataRoot({ startDirectories: [packageDir], userDataDirectory: temp }), {
    dataRoot: data,
    source: 'working-folder',
  });
});

test('respects an explicit environment data root before auto-discovery', (t) => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'pv-cv-env-'));
  t.after(() => fs.rmSync(temp, { recursive: true, force: true }));
  const configured = path.join(temp, 'my-data');
  assert.deepEqual(resolveDataRoot({ envDataRoot: configured, configuredDataRoot: path.join(temp, 'old'), userDataDirectory: temp }), {
    dataRoot: configured,
    source: 'environment',
  });
});

test('uses a previously saved data location before searching for a sibling Working Folder', (t) => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'pv-cv-config-'));
  t.after(() => fs.rmSync(temp, { recursive: true, force: true }));
  const configured = path.join(temp, 'existing-data');
  assert.equal(resolveDataRoot({ configuredDataRoot: configured, startDirectories: [temp], userDataDirectory: temp }).source, 'saved-config');
  assert.equal(resolveDataRoot({ userDataDirectory: temp }).dataRoot, path.join(temp, 'data'));
});
