'use strict';

const fsDefault = require('node:fs');
const path = require('node:path');

function findWorkingFolderData(startDirectories, fsImpl = fsDefault) {
  for (const start of startDirectories || []) {
    if (!start || typeof start !== 'string') continue;
    let current = path.resolve(start);
    while (true) {
      if (path.basename(current).toLowerCase() === 'working folder') {
        const candidate = path.join(current, 'data');
        try {
          if (fsImpl.statSync(candidate).isDirectory()) return candidate;
        } catch { /* this ancestor has no shared data folder */ }
      }
      const parent = path.dirname(current);
      if (parent === current) break;
      current = parent;
    }
  }
  return null;
}

function resolveDataRoot({
  envDataRoot,
  configuredDataRoot,
  startDirectories = [],
  userDataDirectory,
  fsImpl = fsDefault,
}) {
  if (typeof envDataRoot === 'string' && envDataRoot.trim()) {
    return { dataRoot: path.resolve(envDataRoot.trim()), source: 'environment' };
  }
  if (typeof configuredDataRoot === 'string' && configuredDataRoot.trim() && path.isAbsolute(configuredDataRoot.trim())) {
    return { dataRoot: path.resolve(configuredDataRoot.trim()), source: 'saved-config' };
  }
  const discovered = findWorkingFolderData(startDirectories, fsImpl);
  if (discovered) return { dataRoot: discovered, source: 'working-folder' };
  if (!userDataDirectory) throw new Error('userDataDirectory is required when no existing data folder is configured.');
  return { dataRoot: path.join(path.resolve(userDataDirectory), 'data'), source: 'new-profile' };
}

module.exports = { findWorkingFolderData, resolveDataRoot };
