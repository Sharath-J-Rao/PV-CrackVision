/** v1.4.11: single runtime source for the release number (package.json), so model labels cannot drift from the app version. */
import fs from 'node:fs';
import path from 'node:path';
import { APP_ROOT } from './dataPaths.ts';

export const APP_VERSION: string = (() => {
  try { return String(JSON.parse(fs.readFileSync(path.join(APP_ROOT, 'package.json'), 'utf8')).version ?? 'unknown'); }
  catch { return 'unknown'; } // package.json is always bundled, but never let a label lookup stop training
})();
