import test from 'node:test';
import assert from 'node:assert/strict';
import {
  PYPI_INDEX_URL,
  PYTORCH_CPU_INDEX_URL,
  buildPackageDependencyInstallArgs,
  buildPyPiWheelInstallArgs,
  buildTorchCpuWheelInstallArgs,
  buildTorchWheelInstallArgs,
  chooseTorchVariant,
} from './pythonInstallConfig.ts';

const valueAfter = (args, option) => args[args.indexOf(option) + 1];

test('torch and torchvision are installed as binary wheels from the official CPU index only', () => {
  const args = buildTorchCpuWheelInstallArgs();
  assert.equal(valueAfter(args, '--index-url'), PYTORCH_CPU_INDEX_URL);
  assert.ok(args.includes('--no-deps'));
  assert.ok(args.includes('--only-binary=:all:'));
  assert.ok(args.includes('torch'));
  assert.ok(args.includes('torchvision'));
  assert.ok(!args.includes('--extra-index-url'));
});

test('PyPI wheel install does not resolve or replace dependencies', () => {
  const args = buildPyPiWheelInstallArgs('ultralytics');
  assert.equal(valueAfter(args, '--index-url'), PYPI_INDEX_URL);
  assert.ok(args.includes('--no-deps'));
  assert.ok(args.includes('--only-binary=:all:'));
  assert.ok(args.includes('ultralytics'));
  assert.ok(!args.includes('--extra-index-url'));
  assert.throws(() => buildPyPiWheelInstallArgs(), /At least one/);
  assert.throws(() => buildPyPiWheelInstallArgs('  '), /At least one/);
});

test('metadata-driven dependency installer pins the existing CPU runtime', () => {
  const args = buildPackageDependencyInstallArgs('training/install_package_dependencies.py', ['torch', 'torchvision']);
  assert.deepEqual(args, [
    'training/install_package_dependencies.py', 'torch', 'torchvision',
    '--preserve', 'torch', 'torchvision',
  ]);
  assert.throws(() => buildPackageDependencyInstallArgs('', ['torch']), /helper path/);
  assert.throws(() => buildPackageDependencyInstallArgs('helper.py', []), /helper path/);
});

test('GPU wheel variant follows the NVIDIA driver CUDA version, CPU otherwise', () => {
  assert.equal(chooseTorchVariant(undefined, undefined), 'cpu');
  assert.equal(chooseTorchVariant(undefined, '| NVIDIA-SMI 551 Driver Version: 551.61 CUDA Version: 12.4 |'), 'cu124');
  assert.equal(chooseTorchVariant('auto', 'CUDA Version: 12.9'), 'cu128');
  assert.equal(chooseTorchVariant(undefined, 'CUDA Version: 11.8'), 'cu118');
  assert.equal(chooseTorchVariant(undefined, 'CUDA Version: 10.2'), 'cpu');
  assert.equal(chooseTorchVariant('cpu', 'CUDA Version: 12.8'), 'cpu');
  assert.throws(() => chooseTorchVariant('cu999', undefined), /PV_TORCH_VARIANT/);
  const args = buildTorchWheelInstallArgs('cu124', true);
  assert.equal(args[args.indexOf('--index-url') + 1], 'https://download.pytorch.org/whl/cu124');
  assert.ok(args.includes('--force-reinstall'));
});
