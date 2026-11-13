/** Portable, versioned backup/restore for direct cell-detector annotations. */
import crypto from 'node:crypto';
import JSZip from 'jszip';
import type { CellAnnotationRecord } from './cellAnnotationStore.ts';

const FORMAT = 'pv-crackvision-cell-training-data';
const FORMAT_VERSION = 1;
const MAX_ARCHIVE_BYTES = 600 * 1024 * 1024;
const MAX_RECORDS = 50000;
const MAX_BOXES_PER_IMAGE = 3000;
const SAFE_IMAGE_NAME = /^[a-f0-9]{20}\.img$/i;

type BundleManifest = {
  format: typeof FORMAT;
  formatVersion: number;
  appVersion: string;
  createdAt: string;
  annotationImages: number;
  cellBoxes: number;
  negativeImages: number;
  checkpoint?: { name: 'cell-best.pt'; size: number; sha256: string };
};

function validBox(value: any): value is { left: number; top: number; width: number; height: number } {
  return value && ['left', 'top', 'width', 'height'].every((key) => Number.isFinite(value[key]) && Math.abs(value[key]) <= 100000)
    && value.width > 0 && value.height > 0;
}

function validateRecord(raw: any): CellAnnotationRecord {
  if (!raw || typeof raw !== 'object' || !SAFE_IMAGE_NAME.test(String(raw.imageFile ?? ''))) throw new Error('The cell training backup contains an invalid image reference.');
  if (!Array.isArray(raw.boxes) || raw.boxes.length > MAX_BOXES_PER_IMAGE || !raw.boxes.every(validBox)) throw new Error(`Invalid cell-box annotations for ${String(raw.fileName ?? raw.imageFile)}.`);
  const missedBoxes = raw.missedBoxes === undefined ? [] : raw.missedBoxes;
  if (!Array.isArray(missedBoxes) || missedBoxes.length > MAX_BOXES_PER_IMAGE || !missedBoxes.every(validBox)) throw new Error(`Invalid missed-cell annotations for ${String(raw.fileName ?? raw.imageFile)}.`);
  if (raw.sampleType === 'negative' && raw.boxes.length > 0) throw new Error(`A negative sample cannot contain cell boxes: ${String(raw.fileName ?? raw.imageFile)}.`);
  const sampleType = raw.sampleType === 'negative' || (!raw.sampleType && raw.boxes.length === 0) ? 'negative' : 'positive';
  if (sampleType === 'positive' && raw.boxes.length === 0) throw new Error(`A positive annotation has no cell boxes: ${String(raw.fileName ?? raw.imageFile)}.`);
  return {
    id: typeof raw.id === 'string' && raw.id.length <= 128 ? raw.id : crypto.randomUUID(),
    fileName: typeof raw.fileName === 'string' ? raw.fileName.slice(0, 512) : String(raw.imageFile),
    imageFile: raw.imageFile,
    ...(typeof raw.sourceImageId === 'string' ? { sourceImageId: raw.sourceImageId.slice(0, 256) } : {}),
    boxes: raw.boxes.map((box: any) => ({ left: box.left, top: box.top, width: box.width, height: box.height })),
    ...(missedBoxes.length ? { missedBoxes: missedBoxes.map((box: any) => ({ left: box.left, top: box.top, width: box.width, height: box.height })) } : {}),
    sampleType,
    ...(typeof raw.partial === 'boolean' ? { partial: raw.partial } : {}),
    createdAt: typeof raw.createdAt === 'string' && !Number.isNaN(Date.parse(raw.createdAt)) ? raw.createdAt : new Date(0).toISOString(),
  };
}

export async function createCellTrainingDataBundle(
  records: CellAnnotationRecord[],
  readImage: (record: CellAnnotationRecord) => Buffer,
  appVersion: string,
  readCheckpoint?: () => Buffer | null,
): Promise<Buffer> {
  if (records.length > MAX_RECORDS) throw new Error(`Too many annotation records to export (${records.length}).`);
  const zip = new JSZip();
  const safeRecords: CellAnnotationRecord[] = [];
  const seen = new Set<string>();
  for (const raw of records) {
    const record = validateRecord(raw);
    if (seen.has(record.imageFile)) continue;
    seen.add(record.imageFile);
    const image = readImage(record);
    if (!image.length || image.length > 100 * 1024 * 1024) throw new Error(`The source image is missing or too large: ${record.fileName}.`);
    const digest = crypto.createHash('sha1').update(image).digest('hex').slice(0, 20);
    if (digest.toLowerCase() !== record.imageFile.toLowerCase()) throw new Error(`The source image hash does not match its annotation: ${record.fileName}.`);
    zip.file(`annotations/images/${record.imageFile}`, image);
    safeRecords.push(record);
  }
  const checkpoint = readCheckpoint?.() ?? null;
  if (checkpoint && (checkpoint.length < 1024 || checkpoint.length > MAX_ARCHIVE_BYTES || checkpoint[0] !== 0x50 || checkpoint[1] !== 0x4b)) throw new Error('The reusable cell checkpoint is invalid or too large to export.');
  if (checkpoint) zip.file('models/cell-best.pt', checkpoint);
  const manifest: BundleManifest = {
    format: FORMAT,
    formatVersion: FORMAT_VERSION,
    appVersion,
    createdAt: new Date().toISOString(),
    annotationImages: safeRecords.length,
    cellBoxes: safeRecords.reduce((sum, record) => sum + record.boxes.length, 0),
    negativeImages: safeRecords.filter((record) => record.sampleType === 'negative').length,
    ...(checkpoint ? { checkpoint: { name: 'cell-best.pt' as const, size: checkpoint.length, sha256: crypto.createHash('sha256').update(checkpoint).digest('hex') } } : {}),
  };
  zip.file('manifest.json', JSON.stringify(manifest, null, 2));
  zip.file('annotations/annotations.json', JSON.stringify(safeRecords, null, 2));
  const result = await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE', compressionOptions: { level: 4 } });
  if (result.length > MAX_ARCHIVE_BYTES) throw new Error('The exported training backup exceeds the 600 MB safety limit.');
  return result;
}

export async function parseCellTrainingDataBundle(buffer: Buffer): Promise<{ manifest: BundleManifest; records: CellAnnotationRecord[]; images: Map<string, Buffer>; checkpoint: Buffer | null }> {
  if (!buffer.length || buffer.length > MAX_ARCHIVE_BYTES) throw new Error('The backup is empty or exceeds the 600 MB upload limit.');
  let zip: JSZip;
  try { zip = await JSZip.loadAsync(buffer, { checkCRC32: false }); }
  catch { throw new Error('The selected file is not a valid cell training ZIP backup.'); }
  let declaredExpandedBytes = 0;
  for (const [entryName, entry] of Object.entries(zip.files)) {
    if (entry.dir) continue;
    if (!['manifest.json', 'annotations/annotations.json', 'models/cell-best.pt'].includes(entryName) && !entryName.startsWith('annotations/images/')) throw new Error(`The backup contains an unsupported file: ${entryName.slice(0, 120)}.`);
    const size = Number((entry as any)._data?.uncompressedSize ?? 0);
    if (Number.isFinite(size) && size > 0) declaredExpandedBytes += size;
    if (declaredExpandedBytes > MAX_ARCHIVE_BYTES) throw new Error('The expanded backup exceeds the 600 MB safety limit.');
  }
  const manifestFile = zip.file('manifest.json');
  const annotationsFile = zip.file('annotations/annotations.json');
  if (!manifestFile || !annotationsFile) throw new Error('This ZIP does not contain a PV CrackVision cell training backup.');
  let manifest: BundleManifest;
  let rawRecords: any;
  try {
    manifest = JSON.parse(await manifestFile.async('string'));
    rawRecords = JSON.parse(await annotationsFile.async('string'));
  } catch { throw new Error('The cell training backup has an unreadable manifest or annotation index.'); }
  if (!manifest || typeof manifest !== 'object' || manifest.format !== FORMAT || manifest.formatVersion !== FORMAT_VERSION) throw new Error(`Unsupported cell training backup format (${String(manifest?.format)} v${String(manifest?.formatVersion)}).`);
  if (!Array.isArray(rawRecords) || rawRecords.length > MAX_RECORDS) throw new Error('The backup contains an invalid or excessive annotation index.');
  if (Number((manifest as any).annotationImages) !== rawRecords.length) throw new Error('The backup manifest does not match its annotation index.');
  const records = rawRecords.map(validateRecord);
  const images = new Map<string, Buffer>();
  let uncompressedBytes = 0;
  for (const [entryName, entry] of Object.entries(zip.files)) {
    if (entry.dir || !entryName.startsWith('annotations/images/')) continue;
    const imageFile = entryName.slice('annotations/images/'.length);
    if (!SAFE_IMAGE_NAME.test(imageFile)) throw new Error('The backup contains an unsafe image path.');
    const bytes = await entry.async('nodebuffer');
    uncompressedBytes += bytes.length;
    if (bytes.length > 100 * 1024 * 1024 || uncompressedBytes > MAX_ARCHIVE_BYTES) throw new Error('The expanded backup exceeds the safety limit.');
    const digest = crypto.createHash('sha1').update(bytes).digest('hex').slice(0, 20);
    if (digest.toLowerCase() !== imageFile.toLowerCase()) throw new Error(`Image integrity check failed for ${imageFile}.`);
    images.set(imageFile, bytes);
  }
  const required = new Set(records.map((record) => record.imageFile));
  for (const imageFile of required) if (!images.has(imageFile)) throw new Error(`The backup is missing source image ${imageFile}.`);
  if (images.size !== required.size) throw new Error('The backup contains unreferenced source images.');
  const checkpointEntry = zip.file('models/cell-best.pt');
  let checkpoint: Buffer | null = null;
  if (checkpointEntry) {
    checkpoint = await checkpointEntry.async('nodebuffer');
    const declared = manifest.checkpoint;
    if (!declared || declared.name !== 'cell-best.pt' || declared.size !== checkpoint.length || declared.sha256 !== crypto.createHash('sha256').update(checkpoint).digest('hex') || checkpoint.length < 1024 || checkpoint[0] !== 0x50 || checkpoint[1] !== 0x4b) throw new Error('The reusable cell checkpoint failed its integrity check.');
  } else if (manifest.checkpoint) throw new Error('The backup manifest lists a cell checkpoint that is missing.');
  return { manifest, records, images, checkpoint };
}
