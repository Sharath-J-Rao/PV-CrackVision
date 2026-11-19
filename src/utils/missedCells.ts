import type { PixelBox } from '../types/inspection';

const iou = (a: PixelBox, b: PixelBox): number => {
  const w = Math.min(a.left + a.width, b.left + b.width) - Math.max(a.left, b.left);
  const h = Math.min(a.top + a.height, b.top + b.height) - Math.max(a.top, b.top);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  return inter / (a.width * a.height + b.width * b.height - inter);
};

/**
 * v1.4.11.5: reviewed cell boxes that no model detection overlaps (IoU < threshold) = cells the model MISSED.
 * `detected` must be the model's own untouched output; without it (undefined, e.g. classical fallback) nothing can be
 * attributed to the model, so no box is flagged.
 */
export function findMissedCells(reviewed: readonly PixelBox[], detected: readonly PixelBox[] | undefined, iouThreshold = 0.3): PixelBox[] {
  if (!detected) return [];
  return reviewed.filter((box) => !detected.some((model) => iou(box, model) >= iouThreshold));
}

const inside = (box: PixelBox, x: number, y: number) => x >= box.left && x <= box.left + box.width && y >= box.top && y <= box.top + box.height;

/**
 * v1.4.11.6: a cell labelled "cracked" by the operator is a crack the model MISSED when no model crack detection
 * (kept or deleted-as-false) has its point or box centre inside that cell. Callers must only use this after a crack scan.
 */
export function isMissedCrack(cell: PixelBox, modelDetections: ReadonlyArray<{ pointPx: { x: number; y: number }; pixelBox: PixelBox }>): boolean {
  return !modelDetections.some((d) => inside(cell, d.pointPx.x, d.pointPx.y) || inside(cell, d.pixelBox.left + d.pixelBox.width / 2, d.pixelBox.top + d.pixelBox.height / 2));
}
