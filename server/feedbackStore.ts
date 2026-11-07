/** Persistent cell-level training annotations. Legacy window feedback is migrated out by the v1.3.1 schema migration. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { FEEDBACK_DIR } from './dataPaths.ts';
import { getFeedbackReadiness, TRAINING_REQUIREMENTS } from './trainingValidation.ts';

export { FEEDBACK_DIR };
const IMG_DIR = path.join(FEEDBACK_DIR, 'images');
const INDEX = path.join(FEEDBACK_DIR, 'feedback.json');
const SCHEMA_MARKER = path.join(FEEDBACK_DIR, '.schema-version');
const CELL_SCHEMA_VERSION = 'cell-level-v1';

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
  if (current === CELL_SCHEMA_VERSION) return false;
  fs.mkdirSync(path.dirname(FEEDBACK_DIR), { recursive: true });
  if (fs.existsSync(FEEDBACK_DIR)) {
    // Never destroy an unrecognized or legacy label store during an app upgrade.
    // Keep it beside the new schema store so an operator can recover it later.
    const backup = `${FEEDBACK_DIR}.legacy-${Date.now()}`;
    fs.renameSync(FEEDBACK_DIR, backup);
  }
  fs.mkdirSync(IMG_DIR, { recursive: true });
  fs.writeFileSync(SCHEMA_MARKER, CELL_SCHEMA_VERSION, 'utf8');
  return true;
}

function read(): FeedbackItem[] {
  try {
    const value = JSON.parse(fs.readFileSync(INDEX, 'utf8'));
    return Array.isArray(value) ? value : [];
  } catch { return []; }
}

function write(items: FeedbackItem[]): void {
  fs.mkdirSync(FEEDBACK_DIR, { recursive: true });
  const tmp = INDEX + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(items, null, 1));
  fs.renameSync(tmp, INDEX);
}

export const listFeedback = (): FeedbackItem[] => read();
export const feedbackImagePath = (item: Pick<FeedbackItem, 'imageFile'>): string => path.join(IMG_DIR, item.imageFile);

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

/** v1.4.8.3: one read + one atomic write for a whole batch (was a full rewrite per cell). */
export function addFeedbackMany(
  image: Buffer,
  metas: Array<Omit<FeedbackItem, 'id' | 'imageFile' | 'createdAt'>>,
  options: { replace?: boolean } = {}
): { saved: FeedbackItem[]; replaced: { positives: number; negatives: number } } {
  let items = read();
  const imageFile = crypto.createHash('sha1').update(image).digest('hex').slice(0, 20) + '.img';
  // v1.4.8.3+: replace = the incoming set is the complete reviewed crack-label set for this image.
  const replaced = { positives: 0, negatives: 0 };
  if (options.replace) {
    for (const item of items) if (item.imageFile === imageFile && item.category === 'spider_web_crack') { if (item.label === 1) replaced.positives++; else replaced.negatives++; }
    items = items.filter((item) => !(item.imageFile === imageFile && item.category === 'spider_web_crack'));
  }
  fs.mkdirSync(IMG_DIR, { recursive: true });
  const imagePath = path.join(IMG_DIR, imageFile);
  if (!fs.existsSync(imagePath)) { const tmp = `${imagePath}.${process.pid}.tmp`; fs.writeFileSync(tmp, image); fs.renameSync(tmp, imagePath); } // atomic: no truncated image on crash
  const out: FeedbackItem[] = [];
  for (const meta of metas) {
    const cx = meta.left + meta.width / 2, cy = meta.top + meta.height / 2;
    const dupIndex = items.findIndex((item) =>
      item.imageFile === imageFile && item.category === meta.category &&
      Math.hypot(item.left + item.width / 2 - cx, item.top + item.height / 2 - cy) < Math.max(item.width, meta.width) * 0.35
    );
    if (dupIndex >= 0) {
      items[dupIndex] = { ...items[dupIndex], ...meta, createdAt: new Date().toISOString() };
      out.push(items[dupIndex]);
    } else {
      const item: FeedbackItem = { ...meta, id: crypto.randomUUID(), imageFile, createdAt: new Date().toISOString() };
      items.push(item);
      out.push(item);
    }
  }
  write(items);
  return { saved: out, replaced };
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
  fs.mkdirSync(IMG_DIR, { recursive: true });
  fs.writeFileSync(SCHEMA_MARKER, CELL_SCHEMA_VERSION, 'utf8');
}
