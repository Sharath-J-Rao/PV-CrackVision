/** Persistent cell-level training annotations. Legacy window feedback is migrated out by the v1.3.1 schema migration. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { FEEDBACK_DIR } from './dataPaths.ts';
import { getFeedbackReadiness, TRAINING_REQUIREMENTS } from './trainingValidation.ts';

export { FEEDBACK_DIR };
const IMG_DIR = path.join(FEEDBACK_DIR, 'images');
const INDEX = path.join(FEEDBACK_DIR, 'feedback.json');
const SCHEMA_MARKER = path.join(FEEDBACK_DIR, '.schema-version');
const UNDO_INDEX = path.join(FEEDBACK_DIR, '.feedback-undo.json');
const CELL_SCHEMA_VERSION = 'cell-level-v1';
const UUID_V4_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export interface FeedbackItem {
  id: string;
  fileName: string;
  /** SHA-1 image identifier; original inspection image is stored once in training/feedback/images. */
  imageFile: string;
  /** Exact, user-reviewed solar-cell rectangle in EXIF-oriented image pixel coordinates. */
  left: number;
  top: number;
  width: number;
  height: number;
  /** 1 = cracked cell; 0 = clean cell. */
  label: 0 | 1;
  category: string;
  /** Module ROI retained only for diversity accounting and operator traceability. */
  moduleLeft?: number;
  moduleTop?: number;
  moduleWidth?: number;
  moduleHeight?: number;
  /** v1.4.10: 'rejected-detection' = a model detection the operator deleted as false (hard negative, trained with extra weight). */
  /** v1.4.11.6: 'missed-crack' = a cracked cell the operator added that the model had not detected (trained with extra weight). */
  /** v1.4.17.1: 'auto-negative' = clean only because its module holds a cracked cell; never individually reviewed (hard-negative weighting does not apply). */
  source?: 'rejected-detection' | 'missed-crack' | 'bulk-negative' | 'auto-negative';
  /** Model confidence of a detection explicitly moved into the clean class. */
  modelScore?: number;
  createdAt: string;
}

/**
 * One-time v1.3.1 schema migration. Old v1.3.0 annotations were window-derived and the old model
 * was trained from legacy labels. They are not compatible with the cell-level feature schema.
 * This clears those old runtime artifacts once, then preserves new cell labels across restarts.
 */
export function initializeCellFeedbackStore(): boolean {
  let current = '';
  try { current = fs.readFileSync(SCHEMA_MARKER, 'utf8').trim(); } catch { /* first run or legacy dataset */ }
  if (current === CELL_SCHEMA_VERSION) {
    fs.mkdirSync(IMG_DIR, { recursive: true, mode: 0o700 });
    try { fs.chmodSync(FEEDBACK_DIR, 0o700); fs.chmodSync(IMG_DIR, 0o700); } catch { /* Windows ACLs are inherited from the user profile. */ }
    return false;
  }
  fs.mkdirSync(path.dirname(FEEDBACK_DIR), { recursive: true, mode: 0o700 });
  if (fs.existsSync(FEEDBACK_DIR)) {
    // Never destroy an unrecognized or legacy label store during an app upgrade.
    // Keep it beside the new schema store so an operator can recover it later.
    const backup = `${FEEDBACK_DIR}.legacy-${Date.now()}`;
    fs.renameSync(FEEDBACK_DIR, backup);
  }
  fs.mkdirSync(IMG_DIR, { recursive: true, mode: 0o700 });
  fs.writeFileSync(SCHEMA_MARKER, CELL_SCHEMA_VERSION, { encoding: 'utf8', mode: 0o600 });
  return true;
}

function validFeedbackItem(value: unknown): value is FeedbackItem {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const item = value as Record<string, unknown>;
  const finitePositive = (n: unknown) => typeof n === 'number' && Number.isFinite(n) && n > 0;
  const finiteNonNegative = (n: unknown) => typeof n === 'number' && Number.isFinite(n) && n >= 0;
  if (typeof item.id !== 'string' || !UUID_V4_RE.test(item.id)) return false;
  if (typeof item.fileName !== 'string' || !item.fileName || item.fileName.length > 260) return false;
  if (typeof item.imageFile !== 'string' || !/^[0-9a-f]{20}\.img$/.test(item.imageFile)) return false;
  if (!finiteNonNegative(item.left) || !finiteNonNegative(item.top) || !finitePositive(item.width) || !finitePositive(item.height)) return false;
  if (item.label !== 0 && item.label !== 1) return false;
  if (typeof item.category !== 'string' || !item.category || item.category.length > 128) return false;
  if (typeof item.createdAt !== 'string' || !Number.isFinite(Date.parse(item.createdAt))) return false;
  if (item.source !== undefined && !['rejected-detection', 'missed-crack', 'bulk-negative', 'auto-negative'].includes(String(item.source))) return false;
  if ((item.source === 'rejected-detection' || item.source === 'bulk-negative' || item.source === 'auto-negative') && item.label !== 0) return false;
  if (item.source === 'missed-crack' && item.label !== 1) return false;
  if (item.modelScore !== undefined && (typeof item.modelScore !== 'number' || !Number.isFinite(item.modelScore) || item.modelScore < 0 || item.modelScore > 1)) return false;
  for (const [name, n] of Object.entries(item)) {
    if (/^module(Left|Top)$/.test(name) && !finiteNonNegative(n)) return false;
    if (/^module(Width|Height)$/.test(name) && !finitePositive(n)) return false;
  }
  return true;
}

function atomicWriteJson(filePath: string, value: unknown): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true, mode: 0o700 });
  try { fs.chmodSync(path.dirname(filePath), 0o700); } catch { /* Windows ACLs are inherited from the user profile. */ }
  const tmp = `${filePath}.${process.pid}.${crypto.randomUUID()}.tmp`;
  const fd = fs.openSync(tmp, 'wx', 0o600);
  try {
    try {
      fs.writeFileSync(fd, JSON.stringify(value), 'utf8');
      fs.fsyncSync(fd);
    } finally {
      fs.closeSync(fd);
    }
  } catch (error) {
    fs.rmSync(tmp, { force: true });
    throw error;
  }
  try {
    fs.renameSync(tmp, filePath);
    try { fs.chmodSync(filePath, 0o600); } catch { /* Windows ACLs are inherited from the user profile. */ }
    if (process.platform !== 'win32') {
      try {
        const dirFd = fs.openSync(path.dirname(filePath), 'r');
        try { fs.fsyncSync(dirFd); } finally { fs.closeSync(dirFd); }
      } catch { /* Directory fsync is not supported on every filesystem. */ }
    }
  } catch (error) {
    fs.rmSync(tmp, { force: true });
    throw error;
  }
}

function read(): FeedbackItem[] {
  let raw: string;
  try { raw = fs.readFileSync(INDEX, 'utf8'); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return [];
    throw error;
  }
  let value: unknown;
  try { value = JSON.parse(raw); }
  catch { throw new Error('The crack feedback index is corrupt or unreadable. It was not overwritten; restore it from backup before continuing.'); }
  if (!Array.isArray(value) || !value.every(validFeedbackItem)) {
    throw new Error('The crack feedback index contains invalid records. It was not overwritten; restore or export the data before continuing.');
  }
  return value;
}

function write(items: FeedbackItem[]): void {
  if (!Array.isArray(items) || !items.every(validFeedbackItem)) throw new Error('Refusing to write an invalid crack feedback index.');
  atomicWriteJson(INDEX, items);
}

interface FeedbackBatchUndo {
  changes: Array<{ id: string; before: FeedbackItem | null; after: FeedbackItem | null }>;
  createdAt: number;
}
type PersistedUndoEntry = { token: string; createdAt: number; changes: FeedbackBatchUndo['changes'] };
const MAX_FEEDBACK_BATCH_UNDOS = 32;
const FEEDBACK_UNDO_TTL_MS = 24 * 60 * 60 * 1000;
const MAX_FEEDBACK_UNDO_BYTES = 64 * 1024 * 1024;

function readUndoEntries(): PersistedUndoEntry[] {
  let raw: string;
  try {
    const stat = fs.statSync(UNDO_INDEX);
    if (!stat.isFile() || stat.size > MAX_FEEDBACK_UNDO_BYTES) {
      throw new Error('The feedback undo journal exceeds the configured safe size limit. It was not loaded or overwritten; back it up and contact support.');
    }
    raw = fs.readFileSync(UNDO_INDEX, 'utf8');
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return [];
    throw error;
  }
  let value: unknown;
  try { value = JSON.parse(raw); }
  catch { throw new Error('The feedback undo journal is corrupt. New reversible batches are disabled until it is recovered.'); }
  if (!Array.isArray(value)) throw new Error('The feedback undo journal has an invalid format.');
  const valid = value.every((entry) => {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return false;
    const row = entry as Record<string, unknown>;
    if (typeof row.token !== 'string' || !UUID_V4_RE.test(row.token) || typeof row.createdAt !== 'number' || !Number.isFinite(row.createdAt) || row.createdAt < 0 || row.createdAt > Date.now() + 300_000 || !Array.isArray(row.changes) || row.changes.length < 1 || row.changes.length > 10000) return false;
    const ids = new Set<string>();
    return row.changes.every((change) => {
      if (!change || typeof change !== 'object' || Array.isArray(change)) return false;
      const c = change as Record<string, unknown>;
      if (typeof c.id !== 'string' || !UUID_V4_RE.test(c.id) || ids.has(c.id) || (c.before === null && c.after === null)) return false;
      ids.add(c.id);
      const beforeOk = c.before === null || (validFeedbackItem(c.before) && c.before.id === c.id);
      const afterOk = c.after === null || (validFeedbackItem(c.after) && c.after.id === c.id);
      return beforeOk && afterOk;
    });
  });
  if (!valid) throw new Error('The feedback undo journal contains invalid records. It was not overwritten; restore it from backup before continuing.');
  return value as PersistedUndoEntry[];
}

function writeUndoEntries(entries: PersistedUndoEntry[]): void {
  const bounded = entries.slice(-MAX_FEEDBACK_BATCH_UNDOS);
  const serialized = JSON.stringify(bounded);
  if (Buffer.byteLength(serialized, 'utf8') > MAX_FEEDBACK_UNDO_BYTES) {
    throw new Error('This reversible batch would exceed the safe undo-journal size limit. No feedback labels were committed. Split the operation into smaller batches.');
  }
  atomicWriteJson(UNDO_INDEX, bounded);
}

function pruneUndoEntries(entries: PersistedUndoEntry[], now = Date.now()): PersistedUndoEntry[] {
  return entries.filter((entry) => now - entry.createdAt <= FEEDBACK_UNDO_TTL_MS).slice(-MAX_FEEDBACK_BATCH_UNDOS);
}

/**
 * One-shot, durable and reversible batch write. Each touched row stores both its before and
 * after image. Undo verifies that no touched row has changed since the batch; a stale undo
 * is refused instead of overwriting a newer operator edit.
 */
export function addFeedbackManyTransactional(
  image: Buffer,
  metas: Array<Omit<FeedbackItem, 'id' | 'imageFile' | 'createdAt'>>,
  options: { replace?: boolean } = {}
): { saved: FeedbackItem[]; replaced: { positives: number; negatives: number }; undoToken: string } {
  if (!Buffer.isBuffer(image) || image.length === 0) throw new Error('A non-empty source image is required.');
  if (!Array.isArray(metas) || metas.length < 1 || metas.length > 5000) throw new Error('A reversible feedback batch must contain 1 to 5,000 annotations.');

  const before = read();
  const prepared = prepareFeedbackMany(image, metas, before, options);
  const beforeById = new Map<string, FeedbackItem | null>();
  if (options.replace) {
    for (const item of before.filter((row) => row.imageFile === prepared.imageFile && row.category === 'spider_web_crack')) {
      beforeById.set(item.id, item);
    }
  }
  for (const item of prepared.saved) {
    if (!beforeById.has(item.id)) beforeById.set(item.id, before.find((prior) => prior.id === item.id) ?? null);
  }
  const changes = [...beforeById].map(([id, prior]) => ({
    id,
    before: prior,
    after: prepared.items.find((item) => item.id === id) ?? null,
  }));
  if (changes.length < 1 || changes.length > 10000) {
    throw new Error('This reversible batch affects more than 10,000 stored labels. No labels were committed; split the operation into smaller batches.');
  }
  const undoToken = crypto.randomUUID();
  const journal = pruneUndoEntries(readUndoEntries());
  journal.push({ token: undoToken, createdAt: Date.now(), changes });

  // Write-ahead journal first. If the process dies after the index commit, the before/after
  // record is already durable. If it dies before the index commit, getLastFeedbackUndo()
  // ignores this entry because its after-state does not match the current feedback index.
  writeUndoEntries(journal);
  try {
    storeFeedbackImage(image, prepared.imageFile);
    write(prepared.items);
  } catch (error) {
    try { writeUndoEntries(readUndoEntries().filter((entry) => entry.token !== undoToken)); } catch { /* orphaned journal entries are harmless and are ignored by after-state verification */ }
    const referenced = new Set(read().map((item) => item.imageFile));
    if (!referenced.has(prepared.imageFile)) fs.rmSync(path.join(IMG_DIR, prepared.imageFile), { force: true });
    throw new Error(`The feedback batch was not committed: ${error instanceof Error ? error.message : String(error)}`);
  }
  return { saved: prepared.saved, replaced: prepared.replaced, undoToken };
}

/** Restore one batch only when every touched row is still exactly as this batch left it. */
export function undoFeedbackBatch(token: string): boolean {
  if (typeof token !== 'string' || !UUID_V4_RE.test(token)) throw new Error('Invalid feedback undo token.');
  const now = Date.now();
  const entries = pruneUndoEntries(readUndoEntries(), now);
  const index = entries.findIndex((entry) => entry.token === token);
  if (index < 0) {
    writeUndoEntries(entries);
    return false;
  }
  const undo = entries[index];
  const items = read();
  const currentById = new Map<string, FeedbackItem>(items.map((item) => [item.id, item] as const));
  const stale = undo.changes.some((change) => !isDeepStrictEqual(currentById.get(change.id) ?? null, change.after));
  if (stale) {
    // Conflict is terminal: retaining a stale token only invites repeated attempts to overwrite user edits.
    entries.splice(index, 1);
    writeUndoEntries(entries);
    return false;
  }
  const changes = new Map(undo.changes.map((change) => [change.id, change.before]));
  const restored = items.filter((item) => !changes.has(item.id));
  for (const prior of changes.values()) if (prior) restored.push(prior);
  write(restored);
  entries.splice(index, 1);
  try { writeUndoEntries(entries); } catch { /* Restored feedback is committed; a stale token will fail its after-state check. */ }
  const referencedImages = new Set(restored.map((item) => item.imageFile));
  for (const change of undo.changes) {
    const current = currentById.get(change.id);
    const imageFile = change.before?.imageFile ?? current?.imageFile ?? change.after?.imageFile;
    if (imageFile && !referencedImages.has(imageFile)) fs.rmSync(path.join(IMG_DIR, imageFile), { force: true });
  }
  return true;
}

/** Return metadata for the newest durable batch; cell snapshots remain server-side. */
export function getLastFeedbackUndo(): { token: string; createdAt: number; count: number } | null {
  const entries = pruneUndoEntries(readUndoEntries());
  const currentById = new Map<string, FeedbackItem>(read().map((item) => [item.id, item] as const));
  // Skip stale entries without deleting them: undoing a newer overlapping batch can make an
  // earlier batch valid again. Only return a token whose after-state currently matches.
  const available = entries.filter((entry) => entry.changes.every((change) =>
    isDeepStrictEqual(currentById.get(change.id) ?? null, change.after)));
  const latest = available.reduce<PersistedUndoEntry | null>((best, entry) => !best || entry.createdAt > best.createdAt ? entry : best, null);
  if (!latest) return null;
  return { token: latest.token, createdAt: latest.createdAt, count: latest.changes.length };
}

export const listFeedback = (): FeedbackItem[] => read();

export const feedbackImagePath = (item: Pick<FeedbackItem, 'imageFile'>): string => {
  if (!/^[0-9a-f]{20}\.img$/.test(item.imageFile)) throw new Error('Invalid feedback image identifier.');
  return path.join(IMG_DIR, item.imageFile);
};

export function feedbackCounts() {
  const items = read().filter((item) => item.category === 'spider_web_crack');
  const readiness = getFeedbackReadiness(items);
  return {
    total: readiness.total,
    positives: readiness.positives,
    negatives: readiness.negatives,
    images: readiness.images,
    positiveImages: readiness.positiveImages,
    negativeImages: readiness.negativeImages,
    annotatedRegions: readiness.annotatedRegions,
    trainingReady: readiness.ready,
    trainingMissing: readiness.missing,
    trainingRequirements: TRAINING_REQUIREMENTS,
  };
}

/** Re-clicking a cell updates its label instead of adding a contradictory duplicate. */
export function addFeedback(
  image: Buffer,
  meta: Omit<FeedbackItem, 'id' | 'imageFile' | 'createdAt'>
): FeedbackItem {
  return addFeedbackMany(image, [meta]).saved[0];
}

/** Save the original source image once, with restrictive file permissions and atomic creation. */
function storeFeedbackImage(image: Buffer, imageFile: string): void {
  fs.mkdirSync(IMG_DIR, { recursive: true, mode: 0o700 });
  try { fs.chmodSync(IMG_DIR, 0o700); } catch { /* Windows ACLs are inherited from the user profile. */ }
  const imagePath = path.join(IMG_DIR, imageFile);
  if (fs.existsSync(imagePath)) return;
  const tmp = `${imagePath}.${process.pid}.${crypto.randomUUID()}.tmp`;
  const fd = fs.openSync(tmp, 'wx', 0o600);
  try {
    fs.writeFileSync(fd, image);
    fs.fsyncSync(fd);
  } catch (error) {
    fs.rmSync(tmp, { force: true });
    throw error;
  } finally {
    fs.closeSync(fd);
  }
  try {
    fs.renameSync(tmp, imagePath);
    try { fs.chmodSync(imagePath, 0o600); } catch { /* Windows ACLs are inherited from the user profile. */ }
  } catch (error) {
    fs.rmSync(tmp, { force: true });
    if (!fs.existsSync(imagePath)) throw error;
  }
}

function prepareFeedbackMany(
  image: Buffer,
  metas: Array<Omit<FeedbackItem, 'id' | 'imageFile' | 'createdAt'>>,
  existing: FeedbackItem[],
  options: { replace?: boolean } = {}
): { items: FeedbackItem[]; saved: FeedbackItem[]; replaced: { positives: number; negatives: number }; imageFile: string } {
  const imageFile = crypto.createHash('sha1').update(image).digest('hex').slice(0, 20) + '.img';
  let items = [...existing];
  const replaced = { positives: 0, negatives: 0 };
  if (options.replace) {
    for (const item of items) if (item.imageFile === imageFile && item.category === 'spider_web_crack') {
      if (item.label === 1) replaced.positives++; else replaced.negatives++;
    }
    items = items.filter((item) => !(item.imageFile === imageFile && item.category === 'spider_web_crack'));
  }

  const out: FeedbackItem[] = [];
  for (const meta of metas) {
    if (!meta || (meta.label !== 0 && meta.label !== 1) || typeof meta.fileName !== 'string' || typeof meta.category !== 'string') {
      throw new Error('A feedback annotation contains an invalid label, file name or category.');
    }
    const cx = meta.left + meta.width / 2, cy = meta.top + meta.height / 2;
    const dupIndex = items.findIndex((item) =>
      item.imageFile === imageFile && item.category === meta.category &&
      Math.hypot(item.left + item.width / 2 - cx, item.top + item.height / 2 - cy) < Math.max(item.width, meta.width) * 0.35
    );
    if (dupIndex >= 0) {
      const updated = { ...items[dupIndex], ...meta, createdAt: new Date().toISOString() };
      if (!validFeedbackItem(updated)) throw new Error('A feedback annotation failed validation.');
      items[dupIndex] = updated;
      out.push(updated);
    } else {
      const item: FeedbackItem = { ...meta, id: crypto.randomUUID(), imageFile, createdAt: new Date().toISOString() };
      if (!validFeedbackItem(item)) throw new Error('A feedback annotation failed validation.');
      items.push(item);
      out.push(item);
    }
  }
  return { items, saved: out, replaced, imageFile };
}

/** v1.4.8.3: one read + one atomic write for a whole batch (was a full rewrite per cell). */
export function addFeedbackMany(
  image: Buffer,
  metas: Array<Omit<FeedbackItem, 'id' | 'imageFile' | 'createdAt'>>,
  options: { replace?: boolean } = {}
): { saved: FeedbackItem[]; replaced: { positives: number; negatives: number } } {
  if (!Buffer.isBuffer(image) || image.length === 0) throw new Error('A non-empty source image is required.');
  if (!Array.isArray(metas) || metas.length < 1 || metas.length > 5000) throw new Error('A feedback batch must contain 1 to 5,000 annotations.');
  const prepared = prepareFeedbackMany(image, metas, read(), options);
  storeFeedbackImage(image, prepared.imageFile);
  write(prepared.items);
  return { saved: prepared.saved, replaced: prepared.replaced };
}

/** How many crack labels the training set already holds for this exact image (imageFile = sha1 id). */
export function existingFeedbackForImage(imageFile: string): { positives: number; negatives: number } {
  const mine = read().filter((item) => item.imageFile === imageFile && item.category === 'spider_web_crack');
  return { positives: mine.filter((item) => item.label === 1).length, negatives: mine.filter((item) => item.label === 0).length };
}

export function removeFeedback(id: string): boolean {
  const items = read();
  const hit = items.find((item) => item.id === id);
  if (!hit) return false;
  const remaining = items.filter((item) => item.id !== id);
  write(remaining);
  if (!remaining.some((item) => item.imageFile === hit.imageFile)) fs.rmSync(path.join(IMG_DIR, hit.imageFile), { force: true });
  return true;
}

/** Explicitly destroy all reviewed labels and stored source images for a true fresh start. */
export function clearFeedbackDataset(): void {
  fs.rmSync(FEEDBACK_DIR, { recursive: true, force: true });
  fs.mkdirSync(IMG_DIR, { recursive: true, mode: 0o700 });
  fs.writeFileSync(SCHEMA_MARKER, CELL_SCHEMA_VERSION, { encoding: 'utf8', mode: 0o600 });
}
