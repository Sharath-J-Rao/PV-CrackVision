/** Persistent user feedback (added issues / rejected false detections) used to fine-tune the crack model. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { getFeedbackReadiness, TRAINING_REQUIREMENTS } from './trainingValidation.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const FEEDBACK_DIR = path.join(ROOT, 'training', 'feedback');
const IMG_DIR = path.join(FEEDBACK_DIR, 'images');
const INDEX = path.join(FEEDBACK_DIR, 'feedback.json');

export interface FeedbackItem {
  id: string;
  fileName: string;
  /** file name inside training/feedback/images (sha1 of the image bytes) */
  imageFile: string;
  /** box in the analysed (EXIF-rotated) image's pixel space */
  left: number;
  top: number;
  width: number;
  height: number;
  /** 1 = real issue the user added, 0 = false detection the user deleted */
  label: 0 | 1;
  category: string;
  /** Optional user-selected whole-module ROI, needed when automatic module localization misses it. */
  moduleLeft?: number;
  moduleTop?: number;
  moduleWidth?: number;
  moduleHeight?: number;
  createdAt: string;
}

function read(): FeedbackItem[] {
  try {
    const v = JSON.parse(fs.readFileSync(INDEX, 'utf8'));
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

function write(items: FeedbackItem[]): void {
  fs.mkdirSync(FEEDBACK_DIR, { recursive: true });
  const tmp = INDEX + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(items, null, 1));
  fs.renameSync(tmp, INDEX); // atomic: a crash can't leave a half-written index
}

export const listFeedback = (): FeedbackItem[] => read();
export const feedbackImagePath = (it: FeedbackItem): string => path.join(IMG_DIR, it.imageFile);

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

/** Same image + same label + centres within half the box size => same issue (re-clicking must not duplicate it). */
export function addFeedback(
  image: Buffer,
  meta: Omit<FeedbackItem, 'id' | 'imageFile' | 'createdAt'>
): FeedbackItem {
  const items = read();
  const imageFile = crypto.createHash('sha1').update(image).digest('hex').slice(0, 20) + '.img';
  const cx = meta.left + meta.width / 2, cy = meta.top + meta.height / 2;
  // Treat a re-label of the same cell as an update, not a second contradictory sample.
  const dupIndex = items.findIndex(
    (i) =>
      i.imageFile === imageFile &&
      Math.hypot(i.left + i.width / 2 - cx, i.top + i.height / 2 - cy) < Math.max(i.width, meta.width) * 0.5
  );
  if (dupIndex >= 0) {
    const updated: FeedbackItem = { ...items[dupIndex], ...meta, createdAt: new Date().toISOString() };
    items[dupIndex] = updated;
    write(items);
    return updated;
  }
  fs.mkdirSync(IMG_DIR, { recursive: true });
  const imgPath = path.join(IMG_DIR, imageFile);
  if (!fs.existsSync(imgPath)) fs.writeFileSync(imgPath, image);
  const item: FeedbackItem = {
    ...meta,
    id: crypto.randomUUID(),
    imageFile,
    createdAt: new Date().toISOString(),
  };
  items.push(item);
  write(items);
  return item;
}

export function removeFeedback(id: string): boolean {
  const items = read();
  const hit = items.find((i) => i.id === id);
  if (!hit) return false;
  const rest = items.filter((i) => i.id !== id);
  write(rest);
  if (!rest.some((i) => i.imageFile === hit.imageFile)) fs.rmSync(path.join(IMG_DIR, hit.imageFile), { force: true });
  return true;
}
