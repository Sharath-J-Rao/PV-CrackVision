import type { PixelBox } from '../types/inspection';

export interface ImageBounds {
  width: number;
  height: number;
}

/** Move a group by one shared offset, preserving relative spacing and keeping it in-image. */
export function translateCellBoxes(
  boxes: PixelBox[],
  delta: { x: number; y: number },
  bounds: ImageBounds,
): PixelBox[] {
  if (!boxes.length) return [];
  const minLeft = Math.min(...boxes.map((box) => box.left));
  const minTop = Math.min(...boxes.map((box) => box.top));
  const maxRight = Math.max(...boxes.map((box) => box.left + box.width));
  const maxBottom = Math.max(...boxes.map((box) => box.top + box.height));
  const dx = Math.max(-minLeft, Math.min(bounds.width - maxRight, delta.x));
  const dy = Math.max(-minTop, Math.min(bounds.height - maxBottom, delta.y));
  return boxes.map((box) => ({
    ...box,
    left: Math.round(box.left + dx),
    top: Math.round(box.top + dy),
  }));
}

/** Place a copied group around the clicked image point, retaining exact dimensions and spacing. */
export function placeCellBoxesAt(
  boxes: PixelBox[],
  point: { x: number; y: number },
  bounds: ImageBounds,
): PixelBox[] {
  if (!boxes.length) return [];
  const minLeft = Math.min(...boxes.map((box) => box.left));
  const minTop = Math.min(...boxes.map((box) => box.top));
  const maxRight = Math.max(...boxes.map((box) => box.left + box.width));
  const maxBottom = Math.max(...boxes.map((box) => box.top + box.height));
  const centerX = (minLeft + maxRight) / 2;
  const centerY = (minTop + maxBottom) / 2;
  return translateCellBoxes(boxes, { x: point.x - centerX, y: point.y - centerY }, bounds);
}

/** Space/Spacebar commits the active copy preview without triggering page scroll. */
export function isCellBoxPlacementCommitKey(key: string, code: string): boolean {
  return code === 'Space' || key === ' ';
}

// ---- v1.4.7.1: fast duplication (Alt/Ctrl+drag and Duplicate buttons) ----

export interface CellGaps { x: number; y: number }

function medianOf(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Typical seam between neighbouring cells (median gap to the nearest neighbour on the same row / column).
 * Lets a duplicated group land exactly one cell pitch away instead of touching or overlapping.
 */
export function estimateCellGaps(boxes: PixelBox[]): CellGaps {
  const gapsX: number[] = [];
  const gapsY: number[] = [];
  for (const box of boxes) {
    let bestX = Infinity;
    let bestY = Infinity;
    for (const other of boxes) {
      if (other === box) continue;
      const overlapY = Math.min(box.top + box.height, other.top + other.height) - Math.max(box.top, other.top);
      const overlapX = Math.min(box.left + box.width, other.left + other.width) - Math.max(box.left, other.left);
      const gx = other.left - (box.left + box.width);
      const gy = other.top - (box.top + box.height);
      if (overlapY > box.height * 0.5 && gx >= -box.width * 0.2 && gx < bestX) bestX = gx;
      if (overlapX > box.width * 0.5 && gy >= -box.height * 0.2 && gy < bestY) bestY = gy;
    }
    // Negative / huge gaps are detector noise, not seams.
    if (Number.isFinite(bestX) && bestX < box.width) gapsX.push(Math.max(0, bestX));
    if (Number.isFinite(bestY) && bestY < box.height) gapsY.push(Math.max(0, bestY));
  }
  return { x: medianOf(gapsX), y: medianOf(gapsY) };
}

function groupSize(boxes: PixelBox[]): { width: number; height: number } {
  const minLeft = Math.min(...boxes.map((box) => box.left));
  const minTop = Math.min(...boxes.map((box) => box.top));
  const maxRight = Math.max(...boxes.map((box) => box.left + box.width));
  const maxBottom = Math.max(...boxes.map((box) => box.top + box.height));
  return { width: maxRight - minLeft, height: maxBottom - minTop };
}

/**
 * While Alt-dragging a copy, snap the offset to one group step (group size + seam) per axis, and lock an axis
 * to 0 when the pointer is close to it, so rows and columns tile exactly without precise aiming.
 */
export function snapDuplicateOffset(
  delta: { x: number; y: number },
  boxes: PixelBox[],
  gaps: CellGaps,
  tolerance = 0.22,
): { x: number; y: number } {
  if (!boxes.length) return delta;
  const size = groupSize(boxes);
  const snapAxis = (value: number, step: number) => {
    if (step <= 0) return value;
    const tol = step * tolerance;
    if (Math.abs(value) <= tol) return 0;
    const nearest = Math.round(value / step) * step;
    return Math.abs(value - nearest) <= tol ? nearest : value;
  };
  return { x: snapAxis(delta.x, size.width + gaps.x), y: snapAxis(delta.y, size.height + gaps.y) };
}

/** Copy of the group one step away in a direction. Returns null when there is no room inside the image. */
export function duplicateStep(
  boxes: PixelBox[],
  direction: 'left' | 'right' | 'up' | 'down',
  gaps: CellGaps,
  bounds: ImageBounds,
): PixelBox[] | null {
  if (!boxes.length) return null;
  const size = groupSize(boxes);
  const stepX = size.width + gaps.x;
  const stepY = size.height + gaps.y;
  const delta = direction === 'right' ? { x: stepX, y: 0 } : direction === 'left' ? { x: -stepX, y: 0 } : direction === 'down' ? { x: 0, y: stepY } : { x: 0, y: -stepY };
  const moved = translateCellBoxes(boxes, delta, bounds);
  // translateCellBoxes clamps; a clamped result would overlap the source, so treat it as "no room".
  if (Math.abs((moved[0].left - boxes[0].left) - delta.x) > 1.5 || Math.abs((moved[0].top - boxes[0].top) - delta.y) > 1.5) return null;
  return moved;
}

/**
 * v1.4.7.2: repeat the group in one direction, one step at a time, until `count` copies exist or the next copy would
 * leave the image (count 0 = fill to the edge). Copies that would sit on an existing box (IoU >= 0.5) are skipped so
 * running it over a partly annotated row never creates duplicates, but the walk continues past them.
 */
export function duplicateRepeat(
  boxes: PixelBox[],
  direction: 'left' | 'right' | 'up' | 'down',
  gaps: CellGaps,
  bounds: ImageBounds,
  count: number,
  existing: PixelBox[] = [],
): PixelBox[] {
  const iou = (a: PixelBox, b: PixelBox) => {
    const w = Math.min(a.left + a.width, b.left + b.width) - Math.max(a.left, b.left);
    const h = Math.min(a.top + a.height, b.top + b.height) - Math.max(a.top, b.top);
    if (w <= 0 || h <= 0) return 0;
    const inter = w * h;
    return inter / (a.width * a.height + b.width * b.height - inter);
  };
  const out: PixelBox[] = [];
  let current = boxes;
  const limit = count > 0 ? Math.min(count, 500) : 500; // hard cap: a corrupt step size must not loop forever
  for (let i = 0; i < limit; i++) {
    const next = duplicateStep(current, direction, gaps, bounds);
    if (!next) break;
    current = next;
    const occupied = next.every((box) => [...existing, ...out].some((other) => iou(box, other) >= 0.5));
    if (!occupied) out.push(...next);
  }
  return out;
}
