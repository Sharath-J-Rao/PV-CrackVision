/**
 * v1.4.12.8: "partially annotated image" support for YOLO cell training.
 *
 * Why: YOLO treats every pixel outside the labelled boxes as background, so an image where only some panels were marked
 * teaches the model that the unmarked (real) cells are NOT cells. Ultralytics has no "ignore region" label, so for images
 * the operator flags as partially annotated we neutralise everything farther than a small margin from a marked cell by
 * painting it the same neutral grey (114) the letterbox padding already uses. The model then learns nothing about the
 * greyed area, while gridlines between marked cells and the immediate surroundings stay intact.
 * Pure helpers (no sharp) so they are unit-testable; the sharp orchestration is in maskUnannotatedAreas().
 */
export interface MaskBox { left: number; top: number; width: number; height: number }

const GREY = 114;

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** 1 = keep (inside a marked cell grown by the margin), 0 = grey out. Malformed boxes are ignored, never thrown on. */
export function buildAnnotatedMask(width: number, height: number, boxes: readonly MaskBox[], marginFraction: number): Uint8Array {
  if (!(width > 0 && height > 0)) throw new Error('Mask dimensions must be positive.');
  const valid = boxes.filter((b) => [b.left, b.top, b.width, b.height].every(Number.isFinite) && b.width > 0 && b.height > 0);
  const margin = Math.max(0, Math.round(Math.max(0, Math.min(1, marginFraction)) * median(valid.map((b) => Math.min(b.width, b.height)))));
  const mask = new Uint8Array(width * height);
  for (const b of valid) {
    const x0 = Math.max(0, Math.floor(b.left - margin)), x1 = Math.min(width, Math.ceil(b.left + b.width + margin));
    const y0 = Math.max(0, Math.floor(b.top - margin)), y1 = Math.min(height, Math.ceil(b.top + b.height + margin));
    if (x1 <= x0) continue;
    for (let y = y0; y < y1; y++) mask.fill(1, y * width + x0, y * width + x1);
  }
  return mask;
}

/** Returns a copy of a 3-channel RGB buffer with every masked-out pixel painted neutral grey. */
export function greyOutsideMask(rgb: Buffer, width: number, height: number, mask: Uint8Array): Buffer {
  if (rgb.length !== width * height * 3 || mask.length !== width * height) throw new Error('Image and mask sizes do not match.');
  const out = Buffer.from(rgb);
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) { const o = i * 3; out[o] = GREY; out[o + 1] = GREY; out[o + 2] = GREY; }
  }
  return out;
}

/** Encode a partially annotated training image: marked areas intact, everything else neutral grey (JPEG q95 like other training images). */
export async function maskUnannotatedAreas(imageData: Buffer, boxes: readonly MaskBox[], marginFraction: number): Promise<Buffer> {
  const { default: sharp } = await import('sharp'); // loaded lazily so the pure helpers above stay importable without native deps
  const { data, info } = await sharp(imageData).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const mask = buildAnnotatedMask(info.width, info.height, boxes, marginFraction);
  const masked = greyOutsideMask(data, info.width, info.height, mask);
  return sharp(masked, { raw: { width: info.width, height: info.height, channels: 3 } }).jpeg({ quality: 95 }).toBuffer();
}
