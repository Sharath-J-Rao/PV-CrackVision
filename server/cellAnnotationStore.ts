/** Persistent operator-drawn individual PV cell boxes used to train the direct YOLO cell detector. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { CELL_ANNOTATIONS_DIR } from './dataPaths.ts';
import type { PixelBox } from '../src/types/inspection.ts';

const DIR = CELL_ANNOTATIONS_DIR;
const IMAGE_DIR = path.join(DIR, 'images');
const INDEX = path.join(DIR, 'annotations.json');
export interface CellAnnotationRecord { id: string; fileName: string; imageFile: string; sourceImageId?: string; boxes: PixelBox[]; /** v1.4.11.5: reviewed boxes the model did not detect (operator-added). */ missedBoxes?: PixelBox[]; sampleType?: 'positive' | 'negative'; /** v1.4.12.8: only some cells were marked; training greys out the rest instead of treating it as background. */ partial?: boolean; createdAt: string }
function read(): CellAnnotationRecord[] {
  try { const data = JSON.parse(fs.readFileSync(INDEX, 'utf8')); return Array.isArray(data) ? data : []; } catch { return []; }
}
function write(records: CellAnnotationRecord[]) {
  fs.mkdirSync(DIR, { recursive: true });
  const temp = `${INDEX}.tmp`;
  fs.writeFileSync(temp, JSON.stringify(records, null, 2));
  fs.renameSync(temp, INDEX);
}
function mergeCellBoxes(existing: PixelBox[], incoming: PixelBox[]): PixelBox[] {
  const merged = existing.map((box) => ({ ...box }));
  for (const box of incoming) {
    const cx = box.left + box.width / 2, cy = box.top + box.height / 2;
    const duplicate = merged.findIndex((prior) => {
      const px = prior.left + prior.width / 2, py = prior.top + prior.height / 2;
      const scale = Math.max(4, Math.min(prior.width, prior.height, box.width, box.height) * 0.18);
      const overlapX = Math.max(0, Math.min(prior.left + prior.width, box.left + box.width) - Math.max(prior.left, box.left));
      const overlapY = Math.max(0, Math.min(prior.top + prior.height, box.top + box.height) - Math.max(prior.top, box.top));
      const intersection = overlapX * overlapY;
      const union = prior.width * prior.height + box.width * box.height - intersection;
      return Math.hypot(px - cx, py - cy) <= scale || (union > 0 && intersection / union >= 0.82);
    });
    if (duplicate >= 0) merged[duplicate] = { ...box };
    else merged.push({ ...box });
  }
  return merged;
}

export function saveCellAnnotations(image: Buffer, fileName: string, boxes: PixelBox[], sourceImageId?: string, missedBoxes: PixelBox[] = [], partial?: boolean): CellAnnotationRecord {
  fs.mkdirSync(IMAGE_DIR, { recursive: true });
  const imageFile = `${crypto.createHash('sha1').update(image).digest('hex').slice(0, 20)}.img`;
  const imagePath = path.join(IMAGE_DIR, imageFile);
  if (!fs.existsSync(imagePath)) { const tmp = `${imagePath}.${process.pid}.tmp`; fs.writeFileSync(tmp, image); fs.renameSync(tmp, imagePath); } // atomic: no truncated image on crash
  const records = read();
  const existing = records.findIndex((entry) => entry.imageFile === imageFile);
  // This API receives the operator's complete reviewed box set for this image.
  // Replace it wholesale: merging would resurrect false detections the operator deleted.
  const reviewedBoxes = mergeCellBoxes([], boxes);
  const record: CellAnnotationRecord = existing >= 0
    ? { ...records[existing], fileName, ...(sourceImageId ? { sourceImageId } : {}), boxes: reviewedBoxes, missedBoxes, sampleType: 'positive', partial: partial ?? records[existing].partial === true, createdAt: new Date().toISOString() } // undefined = keep the earlier choice (e.g. a save from the crack tab)
    : { id: crypto.randomUUID(), fileName, imageFile, ...(sourceImageId ? { sourceImageId } : {}), boxes: reviewedBoxes, ...(missedBoxes.length ? { missedBoxes } : {}), sampleType: 'positive', partial: partial === true, createdAt: new Date().toISOString() };
  if (existing >= 0) records[existing] = record;
  else records.push(record);
  write(records);
  return record;
}
export function saveNegativeCellSample(image: Buffer, fileName: string): CellAnnotationRecord {
  fs.mkdirSync(IMAGE_DIR, { recursive: true });
  const imageFile = `${crypto.createHash('sha1').update(image).digest('hex').slice(0, 20)}.img`;
  const imagePath = path.join(IMAGE_DIR, imageFile);
  const records = read();
  const existing = records.findIndex((entry) => entry.imageFile === imageFile);
  if (existing >= 0 && (records[existing].boxes?.length ?? 0) > 0) {
    throw new Error('This image already has positive cell boxes. A negative image must contain no target PV cells; choose a different image.');
  }
  if (!fs.existsSync(imagePath)) { const tmp = `${imagePath}.${process.pid}.tmp`; fs.writeFileSync(tmp, image); fs.renameSync(tmp, imagePath); } // atomic: no truncated image on crash
  const record: CellAnnotationRecord = existing >= 0
    ? { ...records[existing], fileName, boxes: [], sampleType: 'negative', createdAt: new Date().toISOString() }
    : { id: crypto.randomUUID(), fileName, imageFile, boxes: [], sampleType: 'negative', createdAt: new Date().toISOString() };
  if (existing >= 0) records[existing] = record; else records.push(record);
  write(records);
  return record;
}

export const listCellAnnotations = (): CellAnnotationRecord[] => read();
export const cellAnnotationImagePath = (record: CellAnnotationRecord): string => path.join(IMAGE_DIR, record.imageFile);

/** Add legacy/imported records without overwriting newer local reviews for the same source image. */
export function mergeImportedCellAnnotations(incoming: CellAnnotationRecord[]): { imported: number; alreadyPresent: number } {
  const current = read();
  const byImage = new Set(current.map((record) => record.imageFile));
  let imported = 0, alreadyPresent = 0;
  for (const record of incoming) {
    if (byImage.has(record.imageFile)) { alreadyPresent++; continue; }
    current.push(record);
    byImage.add(record.imageFile);
    imported++;
  }
  if (imported) write(current);
  return { imported, alreadyPresent };
}

/** Clear only the cell-localisation annotation store; crack-classifier feedback is a separate dataset. */
export function clearCellAnnotations(): void {
  fs.rmSync(DIR, { recursive: true, force: true });
}
