/**
 * v1.4.12.6: persistent per-image review state.
 * Reviewed cell boxes, crack edits (deleted / added cracks) and the "review confirmed" flag used to live only in the
 * browser session, so every restart meant redoing the same images. Each image is stored as one small JSON file named
 * by the SHA-1 of its bytes, so the same image is recognised again whatever its file name or folder.
 */
import fs from 'node:fs';
import path from 'node:path';
import { REVIEW_STATE_DIR } from './dataPaths.ts';

const SCHEMA_VERSION = 1;
const MAX_RECORD_BYTES = 40 * 1024 * 1024; // sanity cap; the HTTP layer already limits request bodies
const HASH_PATTERN = /^[0-9a-f]{40}$/;

export interface ReviewRecord {
  version: number;
  imageHash: string;
  fileName: string;
  savedAt: string;
  /** Cell stage: reviewed / drafted cell boxes and the detection metadata needed to place them. */
  cell?: Record<string, unknown>;
  /** Crack stage: the crack result (including operator edits), deleted false detections and the confirmation flag. */
  crack?: Record<string, unknown>;
}

/** The hash is used as a file name, so only a plain lowercase SHA-1 hex string is accepted (blocks path traversal). */
export const isValidReviewHash = (hash: unknown): hash is string => typeof hash === 'string' && HASH_PATTERN.test(hash);
const fileFor = (hash: string) => path.join(REVIEW_STATE_DIR, `${hash}.json`);
const isPlainObject = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object' && !Array.isArray(value);

/** Returns the saved review for an image, or null when none exists or the file is unreadable/corrupt. */
export function loadReview(hash: string): ReviewRecord | null {
  if (!isValidReviewHash(hash)) return null;
  try {
    const record = JSON.parse(fs.readFileSync(fileFor(hash), 'utf8')) as ReviewRecord;
    return isPlainObject(record) && record.version === SCHEMA_VERSION && record.imageHash === hash ? record : null;
  } catch { return null; }
}

/** Validates and atomically writes one image's review state. Throws a readable Error for bad input. */
export function saveReview(hash: string, input: unknown): ReviewRecord {
  if (!isValidReviewHash(hash)) throw new Error('Invalid image hash.');
  if (!isPlainObject(input)) throw new Error('Review payload must be an object.');
  const cell = input.cell, crack = input.crack;
  if (cell !== undefined && (!isPlainObject(cell) || !Array.isArray(cell.boxes))) throw new Error('Review cell data must contain a boxes array.');
  if (crack !== undefined && (!isPlainObject(crack) || !isPlainObject(crack.result))) throw new Error('Review crack data must contain a result object.');
  if (cell === undefined && crack === undefined) throw new Error('Nothing to save: no cell or crack review data.');
  const record: ReviewRecord = {
    version: SCHEMA_VERSION, imageHash: hash,
    fileName: typeof input.fileName === 'string' ? input.fileName.split(/[\\/]/).pop() || 'image' : 'image', // base name only, whichever separator the client used
    savedAt: new Date().toISOString(),
    ...(cell !== undefined ? { cell: cell as Record<string, unknown> } : {}),
    ...(crack !== undefined ? { crack: crack as Record<string, unknown> } : {}),
  };
  const text = JSON.stringify(record);
  if (Buffer.byteLength(text) > MAX_RECORD_BYTES) throw new Error('Review data for this image is too large to save.');
  fs.mkdirSync(REVIEW_STATE_DIR, { recursive: true });
  const target = fileFor(hash), temp = `${target}.${process.pid}.tmp`;
  try {
    fs.writeFileSync(temp, text);
    try { fs.renameSync(temp, target); }
    catch { fs.rmSync(target, { force: true }); fs.renameSync(temp, target); } // Windows: replace may fail while another process reads
  } finally { fs.rmSync(temp, { force: true }); } // never leave a stray temp file behind after a failure
  return record;
}

/** Removes one image's saved review. Returns whether a file existed. */
export function deleteReview(hash: string): boolean {
  if (!isValidReviewHash(hash)) throw new Error('Invalid image hash.');
  const existed = fs.existsSync(fileFor(hash));
  fs.rmSync(fileFor(hash), { force: true });
  return existed;
}


/** Remove historical cell-stage results after a cell-detector reset, while keeping crack edits as explicitly stale history. */
export function clearSavedCellReviewData(): number {
  let changed = 0;
  let entries: fs.Dirent[];
  try { entries = fs.readdirSync(REVIEW_STATE_DIR, { withFileTypes: true }); }
  catch { return 0; }
  for (const entry of entries) {
    if (!entry.isFile() || !/^[0-9a-f]{40}\.json$/i.test(entry.name)) continue;
    const file = path.join(REVIEW_STATE_DIR, entry.name);
    try {
      const record = JSON.parse(fs.readFileSync(file, 'utf8')) as ReviewRecord;
      if (!record.cell) continue;
      delete record.cell;
      if (record.crack) {
        record.crack = { ...record.crack, crackReviewed: false, crackStale: true };
        fs.writeFileSync(file, JSON.stringify(record));
      } else fs.rmSync(file, { force: true });
      changed++;
    } catch { /* leave corrupt files untouched; review recovery is best-effort */ }
  }
  return changed;
}
