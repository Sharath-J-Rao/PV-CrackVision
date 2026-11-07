'use strict';

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
    '!**/*.pyc',
  ],
  asar: true,
  asarUnpack: [
    '**/*.node',
    'training/*.py',
    'node_modules/sharp/**/*',
    'node_modules/@img/**/*',
    'node_modules/onnxruntime-node/**/*',
  ],
  npmRebuild: true,
  nodeGypRebuild: false,
  win: {
    target: [
      { target: 'nsis', arch: ['x64'] },
      { target: 'portable', arch: ['x64'] },
    ],
    artifactName: 'PV-CrackVision-${version}-${arch}.${ext}',
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
