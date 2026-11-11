import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { mergeMigrationJsonArray, migrateLegacyTrainingData, discoverLegacyReleaseRoots, legacyTrainingPythonCandidates, DATA_ROOT, WORKING_FOLDER, CELL_ANNOTATIONS_DIR, FEEDBACK_DIR, CELL_YOLO_DIR, MODEL_DATA_DIR, VENV_DIR, JOB_STATE_FILE, TRAINING_MODELS_DIR } from './dataPaths.ts';

test('all mutable training and model stores resolve beneath one shared data directory', () => {
  for (const candidate of [CELL_ANNOTATIONS_DIR, FEEDBACK_DIR, CELL_YOLO_DIR, MODEL_DATA_DIR, VENV_DIR]) {
    assert.equal(candidate.startsWith(DATA_ROOT), true, `${candidate} should be inside ${DATA_ROOT}`);
  }
  assert.match(DATA_ROOT, /data$/i);
});

test('migration merges cell annotation records by image and unions distinct boxes', () => {
  const destination = [{ id: 'old', imageFile: 'image.img', fileName: 'a.jpg', createdAt: '2026-01-01T00:00:00Z', boxes: [{ left: 1, top: 2, width: 3, height: 4 }] }];
  const source = [{ id: 'new', imageFile: 'image.img', fileName: 'a.jpg', createdAt: '2026-02-01T00:00:00Z', boxes: [{ left: 1, top: 2, width: 3, height: 4 }, { left: 10, top: 20, width: 30, height: 40 }] }];
  const merged = JSON.parse(mergeMigrationJsonArray(JSON.stringify(source), JSON.stringify(destination)));
  assert.equal(merged.length, 1);
  assert.equal(merged[0].boxes.length, 2);
  assert.equal(merged[0].createdAt, '2026-02-01T00:00:00Z');
});

test('migration merges feedback annotations by stable id without losing labels', () => {
  const destination = [{ id: 'a', imageFile: 'one.img', label: 1, createdAt: '2026-01-01T00:00:00Z' }];
  const source = [{ id: 'b', imageFile: 'two.img', label: 0, createdAt: '2026-02-01T00:00:00Z' }];
  const merged = JSON.parse(mergeMigrationJsonArray(JSON.stringify(source), JSON.stringify(destination)));
  assert.deepEqual(merged.map((item) => item.id).sort(), ['a', 'b']);
});

test('first-launch migration imports legacy run files and rebases durable job-state paths', () => {
  const fixture = path.join(WORKING_FOLDER, `PV_CrackVision_MigrationFixture_${process.pid}_${Date.now()}`);
  const logRelative = path.join('training', 'cell_yolo', 'logs', 'migration-fixture.log');
  const legacyLog = path.join(fixture, logRelative);
  const legacyState = path.join(fixture, 'training', '.cell-yolo-job-state.json');
  const unique = `migration-fixture-${process.pid}-${Date.now()}`;
  const previousState = fs.existsSync(JOB_STATE_FILE) ? fs.readFileSync(JOB_STATE_FILE) : null;
  const targetLog = path.join(CELL_YOLO_DIR, 'logs', path.basename(legacyLog));
  try {
    fs.mkdirSync(path.dirname(legacyLog), { recursive: true });
    fs.mkdirSync(path.dirname(legacyState), { recursive: true });
    fs.mkdirSync(path.join(fixture, 'server'), { recursive: true });
    fs.writeFileSync(path.join(fixture, 'server.ts'), '');
    fs.writeFileSync(legacyLog, unique);
    fs.writeFileSync(legacyState, JSON.stringify({ status: 'completed', updatedAt: new Date().toISOString(), logFile: path.join(fixture, 'training', 'cell_yolo', 'logs', 'migration-fixture.log'), bestCheckpoint: path.join(fixture, 'training', 'models', 'cell-best.pt'), runDirectory: path.join(fixture, 'training', 'cell_yolo', 'runs', 'run-1') }));
    migrateLegacyTrainingData();
    assert.equal(fs.readFileSync(targetLog, 'utf8'), unique);
    const migratedState = JSON.parse(fs.readFileSync(JOB_STATE_FILE, 'utf8'));
    assert.equal(migratedState.logFile, targetLog);
    assert.equal(migratedState.bestCheckpoint, path.join(TRAINING_MODELS_DIR, 'cell-best.pt'));
    assert.equal(migratedState.runDirectory, path.join(CELL_YOLO_DIR, 'runs', 'run-1'));
  } finally {
    fs.rmSync(fixture, { recursive: true, force: true });
    fs.rmSync(targetLog, { force: true });
    if (previousState) fs.writeFileSync(JOB_STATE_FILE, previousState);
    else fs.rmSync(JOB_STATE_FILE, { force: true });
  }
});

test('release discovery handles ZIP wrapper folders nested under Working Folder', () => {
  const wrapper = path.join(WORKING_FOLDER, `PV_CrackVision_Wrapper_${process.pid}_${Date.now()}`);
  const project = path.join(wrapper, 'PV_CrackVision_v1.4.5.4');
  try {
    fs.mkdirSync(path.join(project, 'server'), { recursive: true });
    fs.mkdirSync(path.join(project, 'training'), { recursive: true });
    fs.writeFileSync(path.join(project, 'server.ts'), '');
    assert.ok(discoverLegacyReleaseRoots().includes(project));
  } finally { fs.rmSync(wrapper, { recursive: true, force: true }); }
});

test('legacy release Python runtime remains available as a fallback if a copied venv cannot launch', () => {
  const fixture = path.join(WORKING_FOLDER, `PV_CrackVision_RuntimeFixture_${process.pid}_${Date.now()}`);
  const executable = path.join(fixture, 'training', '.cell-yolo-venv', 'Scripts', 'python.exe');
  try {
    fs.mkdirSync(path.dirname(executable), { recursive: true });
    fs.mkdirSync(path.join(fixture, 'server'), { recursive: true });
    fs.writeFileSync(path.join(fixture, 'server.ts'), '');
    fs.writeFileSync(executable, 'fixture-python');
    assert.ok(legacyTrainingPythonCandidates('win32').includes(executable));
  } finally { fs.rmSync(fixture, { recursive: true, force: true }); }
});
