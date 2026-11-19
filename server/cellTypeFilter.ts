/**
 * v1.4.7.0 — cell "type" filter. Only YOLO detections whose shape/size matches an enabled ACCEPT profile pass;
 * detections matching an enabled REJECT profile (a negative type) are dropped. Pure + dependency-free so it is unit-testable.
 * Aspect is long/short side, so it is independent of box orientation; sizes are relative to the image's own median cell
 * (scale-independent across GSDs) plus optional absolute pixel bounds per profile.
 */
import type { CellTypeFilterSettings, CellTypeProfile } from '../src/types/trainingSettings.ts';

export interface FilterBox { left: number; top: number; width: number; height: number; confidence?: number }

const longSide = (b: FilterBox) => Math.max(b.width, b.height);
const shortSide = (b: FilterBox) => Math.max(1e-6, Math.min(b.width, b.height));

export function matchesProfile(box: FilterBox, p: CellTypeProfile): boolean {
  const long = longSide(box);
  const aspect = long / shortSide(box);
  if (Math.abs(aspect - p.aspect) / p.aspect > p.aspectTolerance) return false;
  if (p.minLongPx > 0 && long < p.minLongPx) return false;
  if (p.maxLongPx > 0 && long > p.maxLongPx) return false;
  return true;
}

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Reference "typical cell" for the undersize gates. Fragments must not drag the reference down, so when scores exist only the
 * more confident half of the boxes defines it (median long side, short side and area).
 */
export function referenceCell(boxes: FilterBox[]): { long: number; short: number; area: number } | null {
  if (boxes.length < 3) return null;
  const scored = boxes.filter((b) => typeof b.confidence === 'number');
  let pool = boxes;
  if (scored.length === boxes.length) {
    const cut = median(scored.map((b) => b.confidence as number));
    const upper = scored.filter((b) => (b.confidence as number) >= cut);
    if (upper.length >= 3) pool = upper;
  }
  // v1.4.12.7: the cells of one image share a size, but many small fragments can dominate even the confident half and drag a
  // plain median down until the fragments look "typical". Anchor on the densest cluster of box areas instead.
  // The confident half can itself consist of fragments, so the cluster is searched among ALL boxes (it is count based).
  const cluster = boxes.length >= 8 ? densestAreaCluster(boxes) : pool;
  return { long: median(cluster.map(longSide)), short: median(cluster.map(shortSide)), area: median(cluster.map((b) => b.width * b.height)) };
}

/** Boxes in the most populated log-area bin (and its two neighbours); ties prefer the larger size. Needs >= 8 boxes. */
function densestAreaCluster<T extends FilterBox>(boxes: T[]): T[] {
  if (boxes.length < 8) return boxes;
  const BIN = 0.25; // log2 units: ~19 % area per bin
  const keyOf = (b: T) => Math.floor(Math.log2(Math.max(1, b.width * b.height)) / BIN);
  const counts = new Map<number, number>();
  for (const box of boxes) counts.set(keyOf(box), (counts.get(keyOf(box)) ?? 0) + 1);
  let bestKey = 0, bestScore = -1;
  for (const key of counts.keys()) {
    const score = (counts.get(key - 1) ?? 0) + (counts.get(key) ?? 0) + (counts.get(key + 1) ?? 0);
    if (score > bestScore || (score === bestScore && key > bestKey)) { bestKey = key; bestScore = score; }
  }
  const cluster = boxes.filter((box) => Math.abs(keyOf(box) - bestKey) <= 1);
  return cluster.length >= 3 ? cluster : boxes;
}

/** Overlap of two boxes as a share of the SMALLER box (1 = the smaller box lies completely inside the other). */
function overlapOfSmaller(a: FilterBox, b: FilterBox): number {
  const w = Math.min(a.left + a.width, b.left + b.width) - Math.max(a.left, b.left);
  const h = Math.min(a.top + a.height, b.top + b.height) - Math.max(a.top, b.top);
  if (w <= 0 || h <= 0) return 0;
  return (w * h) / Math.max(1e-6, Math.min(a.width * a.height, b.width * b.height));
}

/**
 * v1.4.12.7: neighbouring PV cells never overlap, so two boxes sharing >= minOverlap of the smaller box cannot both be cells.
 * Boxes are visited from the most typical size outwards and a box is dropped when it overlaps one already kept: this removes small
 * fragments inside a cell, oversized merges and shifted duplicates, whatever their confidence. The input order of the kept
 * boxes (confidence order) is preserved.
 */
export function suppressOverlappingFragments<T extends FilterBox>(boxes: T[], minOverlap: number): { kept: T[]; rejected: T[] } {
  if (!(minOverlap > 0) || boxes.length < 2) return { kept: [...boxes], rejected: [] };
  const refArea = referenceCell(boxes)?.area ?? median(boxes.map((b) => b.width * b.height));
  const deviation = (b: FilterBox) => Math.abs(Math.log(Math.max(1e-6, b.width * b.height) / Math.max(1e-6, refArea)));
  const byTypicality = [...boxes].sort((a, b) => deviation(a) - deviation(b) || (b.confidence ?? 0) - (a.confidence ?? 0));
  const keptSet = new Set<T>();
  const keptList: T[] = [];
  for (const candidate of byTypicality) {
    if (keptList.some((prior) => overlapOfSmaller(prior, candidate) >= minOverlap)) continue;
    keptSet.add(candidate); keptList.push(candidate);
  }
  return { kept: boxes.filter((box) => keptSet.has(box)), rejected: boxes.filter((box) => !keptSet.has(box)) };
}

export function filterCellsByType<T extends FilterBox>(boxes: T[], settings: CellTypeFilterSettings): { kept: T[]; rejected: T[] } {
  if (!settings.enabled || !boxes.length) return { kept: [...boxes], rejected: [] };
  const accept = settings.profiles.filter((p) => p.enabled && p.mode === 'accept');
  const reject = settings.profiles.filter((p) => p.enabled && p.mode === 'reject');
  const rejected: T[] = [];
  // Stage 0 (v1.4.8.7): drop low-confidence and absolutely undersized detections before anything is measured against them.
  const minConfidence = settings.minConfidence ?? 0, minShortPx = settings.minShortPx ?? 0;
  let kept = boxes.filter((box) => {
    const ok = (minConfidence <= 0 || typeof box.confidence !== 'number' || box.confidence >= minConfidence) &&
      (minShortPx <= 0 || shortSide(box) >= minShortPx);
    if (!ok) rejected.push(box);
    return ok;
  });
  // Stage 0a (v1.4.12.7): drop fragments inside a cell and shifted duplicates before measuring the typical cell.
  const fragmentOverlap = settings.fragmentOverlap ?? 0;
  if (fragmentOverlap > 0) {
    const fragments = suppressOverlappingFragments(kept, fragmentOverlap);
    kept = fragments.kept;
    rejected.push(...fragments.rejected);
  }
  // Stage 0b: undersized relative to the typical cell of this image (area and short side) — fragments, half-visible cells, noise.
  // An enabled elongated accept type (e.g. half-cut 2:1) is legitimately half the typical area/short side, so relax the gates.
  const elongated = accept.some((p) => p.aspect >= 1.5) ? 0.5 : 1;
  const minArea = (settings.minAreaRelative ?? 0) * elongated, minShortRel = (settings.minShortRelative ?? 0) * elongated;
  if ((minArea > 0 || minShortRel > 0) && kept.length >= 3) {
    const ref = referenceCell(kept);
    if (ref) {
      kept = kept.filter((box) => {
        const ok = (minArea <= 0 || box.width * box.height >= ref.area * minArea) && (minShortRel <= 0 || shortSide(box) >= ref.short * minShortRel);
        if (!ok) rejected.push(box);
        return ok;
      });
    }
  }
  // Stage 1: negative types first, then (if any accept types are enabled) require a match to at least one.
  kept = kept.filter((box) => {
    const ok = !reject.some((p) => matchesProfile(box, p)) && (accept.length === 0 || accept.some((p) => matchesProfile(box, p)));
    if (!ok) rejected.push(box);
    return ok;
  });
  // Stage 2: relative long-side window against this image's own median cell (needs a few cells to be meaningful).
  if (kept.length >= 3 && (settings.relativeMin > 0 || settings.relativeMax > 0)) {
    const med = median(kept.map(longSide));
    kept = kept.filter((box) => {
      const long = longSide(box);
      const ok = (settings.relativeMin <= 0 || long >= med * settings.relativeMin) && (settings.relativeMax <= 0 || long <= med * settings.relativeMax);
      if (!ok) rejected.push(box);
      return ok;
    });
  }
  return { kept, rejected };
}

/** Measure a profile from reviewed boxes: median long/short aspect, tolerance from the spread (clamped). */
export function learnProfileFromBoxes(boxes: FilterBox[]): { aspect: number; aspectTolerance: number; samples: number } | null {
  const valid = boxes.filter((b) => b.width > 1 && b.height > 1);
  if (valid.length < 4) return null;
  const aspects = valid.map((b) => longSide(b) / shortSide(b));
  const aspect = median(aspects);
  const spread = median(aspects.map((a) => Math.abs(a - aspect))) / aspect; // median absolute deviation, relative
  return { aspect: Math.round(aspect * 1000) / 1000, aspectTolerance: Math.round(Math.max(0.12, Math.min(0.6, spread * 4 + 0.1)) * 1000) / 1000, samples: valid.length };
}
