'use strict';

const fs = require('node:fs');
const path = require('node:path');

// v1.4.11: an optional build/icon.ico replaces the default Electron icon. Code signing activates automatically
// when CI provides CSC_LINK / CSC_KEY_PASSWORD (protected secrets); unsigned builds still work without them.
const icon = path.join(__dirname, 'build', 'icon.ico');

module.exports = {
  appId: 'com.sharathjrao.pvcrackvision',
  productName: 'PV CrackVision',
  electronVersion: '45.2.3',
  directories: {
    output: 'desktop-dist',
  },
  files: [
    'dist/**/*',
    'electron/**/*',
    'server.ts',
    'server/**/*',
    'src/types/**/*',
    'training/train_cell_yolo.py',
    'training/install_package_dependencies.py',
    'package.json',
    'tsconfig.json',
    '!**/*.test.*',
    '!**/__tests__/**',
    '!**/*.map',
    '!**/__pycache__/**',
    '!node_modules/onnxruntime-node/bin/napi-v*/darwin/**', // Windows-only app: drop macOS/Linux onnxruntime binaries (~158 MB)
    '!node_modules/onnxruntime-node/bin/napi-v*/linux/**',
    '!**/*.pyc',
  ],
  // v1.4.13.1: AI models live outside the asar (read by path by onnxruntime) -> <resources>/models, offline and read-only.
  extraResources: [{ from: 'resources/models', to: 'models', filter: ['cell.onnx', 'cellCrackModel.user.json', 'cellClassifier.onnx', 'cellClassifier.json', 'manifest.json'] }],
  asar: true,
  asarUnpack: [
    '**/*.node',
    'training/*.py',
    'node_modules/sharp/**/*',
    'node_modules/@img/**/*',
    'node_modules/onnxruntime-node/**/*',
    'node_modules/@esbuild/**/*', // v1.4.13.1: the esbuild executable must live outside the asar to be spawned
  ],
  npmRebuild: true,
  nodeGypRebuild: false,
  win: {
    target: [
      { target: 'nsis', arch: ['x64'] },
      { target: 'portable', arch: ['x64'] },
    ],
    artifactName: 'PV-CrackVision-${version}-${arch}.${ext}',
    ...(fs.existsSync(icon) ? { icon } : {}),
  },
  nsis: {
    oneClick: false,
    perMachine: false,
    allowToChangeInstallationDirectory: true,
    createDesktopShortcut: true,
    createStartMenuShortcut: true,
    deleteAppDataOnUninstall: false,
    runAfterFinish: true,
  },
  portable: {
    artifactName: 'PV-CrackVision-${version}-Portable.${ext}',
  },
};
