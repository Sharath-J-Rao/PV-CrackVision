import type { ModuleRegion, PixelBox, SolarCellBox, SolarCellGrid } from '../src/types/inspection.ts';

export interface ScoredCellBox extends PixelBox { confidence: number }

/**
 * v1.4.9: cells the operator detected/reviewed in the Cell Annotation stage. When present, the crack pass uses exactly
 * these boxes as its cells and does not fall back to the classical edge-periodicity grid (unless explicitly allowed).
 */
export interface ReviewedCellInput {
  boxes: PixelBox[];
  /** Frame size the boxes were drawn in (cell pass caps at 8000 px); boxes are rescaled to this pass's frame. */
  coordinateWidth?: number;
  coordinateHeight?: number;
  /** Allow the classical grid for an image that has fewer reviewed cells than needed (default false). */
  allowClassicalFallback?: boolean;
}

/** Rescale reviewed boxes into this pass's frame and drop anything malformed (never throws on user data). */
export function normalizeReviewedCellBoxes(input: ReviewedCellInput, imgWidth: number, imgHeight: number): { boxes: Array<PixelBox & { confidence: number }>; dropped: number } {
  const cw = Number(input.coordinateWidth), ch = Number(input.coordinateHeight);
  const kx = Number.isFinite(cw) && cw > 0 ? imgWidth / cw : 1;
  const ky = Number.isFinite(ch) && ch > 0 ? imgHeight / ch : 1;
  const boxes: Array<PixelBox & { confidence: number }> = [];
  let dropped = 0;
  for (const raw of Array.isArray(input.boxes) ? input.boxes.slice(0, 5000) : []) {
    const left = Number(raw?.left) * kx, top = Number(raw?.top) * ky, width = Number(raw?.width) * kx, height = Number(raw?.height) * ky;
    if (![left, top, width, height].every(Number.isFinite) || width < 4 || height < 4) { dropped++; continue; }
    // Clip to the frame so a box nudged past the edge still crops safely.
    const l = Math.max(0, left), t = Math.max(0, top);
    const w = Math.min(imgWidth, left + width) - l, h = Math.min(imgHeight, top + height) - t;
    if (w < 4 || h < 4) { dropped++; continue; }
    boxes.push({ left: l, top: t, width: w, height: h, confidence: 1 }); // operator-reviewed cells are authoritative
  }
  return { boxes, dropped };
}


function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function clusterAxis(values: number[], tolerance: number): number[] {
  const sorted = [...values].sort((a, b) => a - b);
  const groups: number[][] = [];
  for (const value of sorted) {
    const last = groups[groups.length - 1];
    if (last && Math.abs(value - last.reduce((sum, item) => sum + item, 0) / last.length) <= tolerance) last.push(value);
    else groups.push([value]);
  }
  return groups.map((group) => group.reduce((sum, item) => sum + item, 0) / group.length);
}

function nearestIndex(value: number, centers: number[]): number {
  let best = 0;
  for (let index = 1; index < centers.length; index++) {
    if (Math.abs(centers[index] - value) < Math.abs(centers[best] - value)) best = index;
  }
  return best;
}

export function gridFromYoloCellBoxes(boxes: ScoredCellBox[], minCells = 4): SolarCellGrid | undefined {
  // v1.4.9: reviewed cell sets may legitimately be small (the operator kept only a few cells), so the minimum is a parameter.
  if (boxes.length < Math.max(1, minCells)) return undefined;
  const centersX = boxes.map((box) => box.left + box.width / 2);
  const centersY = boxes.map((box) => box.top + box.height / 2);
  const typicalWidth = median(boxes.map((box) => box.width));
  const typicalHeight = median(boxes.map((box) => box.height));
  const columns = clusterAxis(centersX, Math.max(3, typicalWidth * 0.55));
  const rows = clusterAxis(centersY, Math.max(3, typicalHeight * 0.55));
  if (!rows.length || !columns.length) return undefined;

  const cells: SolarCellBox[] = boxes.map((box) => ({
    row: nearestIndex(box.top + box.height / 2, rows) + 1,
    column: nearestIndex(box.left + box.width / 2, columns) + 1,
    pixelBox: { left: box.left, top: box.top, width: box.width, height: box.height },
    confidence: Math.max(0, Math.min(1, box.confidence)),
    partial: false,
  }));
  const left = Math.min(...boxes.map((box) => box.left));
  const top = Math.min(...boxes.map((box) => box.top));
  const right = Math.max(...boxes.map((box) => box.left + box.width));
  const bottom = Math.max(...boxes.map((box) => box.top + box.height));
  const pitchX = median(columns.slice(1).map((value, index) => value - columns[index])) || typicalWidth;
  const pitchY = median(rows.slice(1).map((value, index) => value - rows[index])) || typicalHeight;
  return {
    rows: rows.length,
    columns: columns.length,
    pixelBox: { left, top, width: right - left, height: bottom - top },
    cells,
    pitchX,
    pitchY,
    confidence: median(boxes.map((box) => box.confidence)),
    method: 'yolo-cell-detection',
  };
}

/**
 * v1.4.12.8: weighted box fusion for cells proposed by several passes (full frame + tiles).
 * Plain NMS keeps one pass's box and drops the rest, so a cell clipped by a tile edge can win and stay misaligned.
 * Here each greedy cluster (leader = highest score, members must overlap the LEADER so clusters cannot chain across
 * neighbouring cells) becomes one box whose edges are the score-weighted mean of its members; the score stays the
 * cluster maximum so recall/threshold behaviour is unchanged. Never throws on malformed boxes (they pass through).
 */
export function fuseOverlappingCellBoxes<T extends ScoredCellBox>(detections: T[], iouThreshold = 0.5): T[] {
  const valid = detections.filter((b) => [b.left, b.top, b.width, b.height, b.confidence].every(Number.isFinite) && b.width > 0 && b.height > 0);
  const passthrough = detections.filter((b) => !valid.includes(b));
  const iou = (a: ScoredCellBox, b: ScoredCellBox) => {
    const w = Math.min(a.left + a.width, b.left + b.width) - Math.max(a.left, b.left);
    const h = Math.min(a.top + a.height, b.top + b.height) - Math.max(a.top, b.top);
    if (w <= 0 || h <= 0) return 0;
    const inter = w * h;
    return inter / (a.width * a.height + b.width * b.height - inter);
  };
  const sorted = [...valid].sort((a, b) => b.confidence - a.confidence);
  const used = new Array<boolean>(sorted.length).fill(false);
  const fused: T[] = [];
  for (let i = 0; i < sorted.length; i++) {
    if (used[i]) continue;
    used[i] = true;
    const members = [sorted[i]];
    for (let j = i + 1; j < sorted.length; j++) {
      if (!used[j] && iou(sorted[i], sorted[j]) >= iouThreshold) { used[j] = true; members.push(sorted[j]); }
    }
    if (members.length === 1) { fused.push(sorted[i]); continue; }
    const weight = members.reduce((sum, m) => sum + Math.max(1e-6, m.confidence), 0);
    const mean = (pick: (m: T) => number) => members.reduce((sum, m) => sum + pick(m) * Math.max(1e-6, m.confidence), 0) / weight;
    const left = mean((m) => m.left), top = mean((m) => m.top);
    const right = mean((m) => m.left + m.width), bottom = mean((m) => m.top + m.height);
    fused.push({ ...sorted[i], left, top, width: right - left, height: bottom - top });
  }
  return [...fused, ...passthrough];
}

export interface ApplyCellsOptions {
  /**
   * v1.4.12.8: grow a region's box so it always contains the cells assigned to it. The classical panel locator can stop
   * short of the array (cells outside its box looked "unowned" on screen); the box only ever grows, never shrinks.
   * Needs imageWidth/imageHeight to keep the normalised box in step.
   */
  expandRegionToCells?: boolean,
  /** Minimum boxes needed to accept a region / the whole set (default 4 = unreviewed YOLO behaviour). */
  minCells?: number;
  /**
   * Reviewed mode: boxes that fall outside every located PV region (e.g. a cell the operator drew by hand where the
   * classical stage found no panel) are kept in an extra region instead of being silently dropped.
   */
  adoptOrphans?: boolean;
  /**
   * Reviewed mode: a region that received no reviewed cells loses its classical cell grid, so the crack classifier
   * only ever sees cells the operator kept in Cell Annotation (never cells the operator deleted).
   */
  clearUnassigned?: boolean;
  /** Needed with adoptOrphans to build the normalised box of the extra region. */
  imageWidth?: number;
  imageHeight?: number;
}

/**
 * Apply direct YOLO cell boxes to the existing automatically located PV regions.
 * The regions are only spatial groupings; there is no learned module detector.
 * If the classical stage only produced a whole-image fallback, all direct cell
 * detections are kept in that region so the cell detector can still drive cell crops.
 */
export function applyYoloCellsToRegions(regions: ModuleRegion[], detections: ScoredCellBox[], options: ApplyCellsOptions = {}): { regions: ModuleRegion[]; applied: number; orphans: number } {
  const minCells = Math.max(1, options.minCells ?? 4);
  if (detections.length < minCells || regions.length === 0) return { regions, applied: 0, orphans: 0 };
  const regionBoxes = regions.map((region) => ({
    region,
    area: region.pixelBox.width * region.pixelBox.height,
  })).sort((a, b) => a.area - b.area);
  const assigned = new Map<string, ScoredCellBox[]>();
  const orphanBoxes: ScoredCellBox[] = [];
  for (const detection of detections) {
    const cx = detection.left + detection.width / 2;
    const cy = detection.top + detection.height / 2;
    const match = regionBoxes.find(({ region }) => {
      const box = region.pixelBox;
      const marginX = box.width * 0.04;
      const marginY = box.height * 0.04;
      return cx >= box.left - marginX && cx <= box.left + box.width + marginX &&
        cy >= box.top - marginY && cy <= box.top + box.height + marginY;
    }) ?? (regions.length === 1 && regions[0].isFallback ? regionBoxes[0] : undefined);
    if (!match) { orphanBoxes.push(detection); continue; }
    const list = assigned.get(match.region.id) ?? [];
    list.push(detection);
    assigned.set(match.region.id, list);
  }

  let applied = 0;
  const next = regions.map((region) => {
    const boxes = assigned.get(region.id) ?? [];
    // Reviewed mode keeps even a region with a single reviewed cell; unreviewed YOLO keeps the 4-cell rule.
    const grid = gridFromYoloCellBoxes(boxes, options.clearUnassigned ? 1 : minCells);
    if (!grid) return options.clearUnassigned ? { ...region, cellGrid: undefined } : region;
    applied += grid.cells.length;
    if (options.expandRegionToCells && options.imageWidth && options.imageHeight && !region.isFallback) {
      const left = Math.max(0, Math.min(region.pixelBox.left, grid.pixelBox.left));
      const top = Math.max(0, Math.min(region.pixelBox.top, grid.pixelBox.top));
      const right = Math.min(options.imageWidth, Math.max(region.pixelBox.left + region.pixelBox.width, grid.pixelBox.left + grid.pixelBox.width));
      const bottom = Math.min(options.imageHeight, Math.max(region.pixelBox.top + region.pixelBox.height, grid.pixelBox.top + grid.pixelBox.height));
      return {
        ...region, cellGrid: grid,
        pixelBox: { left, top, width: Math.max(1, right - left), height: Math.max(1, bottom - top) },
        box: { xmin: left / options.imageWidth, ymin: top / options.imageHeight, xmax: right / options.imageWidth, ymax: bottom / options.imageHeight },
      };
    }
    return { ...region, cellGrid: grid };
  });

  // Reviewed cells outside every located region get their own region so they still reach the crack classifier.
  let orphans = 0;
  if (options.adoptOrphans && orphanBoxes.length && options.imageWidth && options.imageHeight) {
    const grid = gridFromYoloCellBoxes(orphanBoxes, 1);
    if (grid) {
      const pad = Math.max(grid.pitchX, grid.pitchY) * 0.1;
      const left = Math.max(0, grid.pixelBox.left - pad), top = Math.max(0, grid.pixelBox.top - pad);
      const right = Math.min(options.imageWidth, grid.pixelBox.left + grid.pixelBox.width + pad);
      const bottom = Math.min(options.imageHeight, grid.pixelBox.top + grid.pixelBox.height + pad);
      next.push({
        id: 'REVIEWED-CELLS',
        label: 'Reviewed cells (outside located panels)',
        pixelBox: { left, top, width: Math.max(1, right - left), height: Math.max(1, bottom - top) },
        box: { xmin: left / options.imageWidth, ymin: top / options.imageHeight, xmax: right / options.imageWidth, ymax: bottom / options.imageHeight },
        cellGrid: grid,
      });
      applied += grid.cells.length;
      orphans = grid.cells.length;
    }
  }
  return { regions: next, applied, orphans };
}

/**
 * Suppress oversized detections that swallow one or more repeating individual
 * PV-cell boxes. At least two similarly-sized, spatially distinct small boxes
 * must repeat within the larger box; one nested detection alone is insufficient.
 */
export function suppressOversizedPatternOverlaps<T extends ScoredCellBox>(detections: T[], nmsIou = 0.45, enclosingSuppression = true): T[] {
  if (detections.length < 2) return [...detections];
  const area = (box: ScoredCellBox) => Math.max(1, box.width * box.height);
  const intersectionArea = (a: ScoredCellBox, b: ScoredCellBox) => {
    const width = Math.max(0, Math.min(a.left + a.width, b.left + b.width) - Math.max(a.left, b.left));
    const height = Math.max(0, Math.min(a.top + a.height, b.top + b.height) - Math.max(a.top, b.top));
    return width * height;
  };
  const containsCenter = (outer: ScoredCellBox, inner: ScoredCellBox) => {
    const cx = inner.left + inner.width / 2, cy = inner.top + inner.height / 2;
    return cx >= outer.left && cx <= outer.left + outer.width && cy >= outer.top && cy <= outer.top + outer.height;
  };
  const isClearlyLarger = (large: ScoredCellBox, small: ScoredCellBox) =>
    large.width >= small.width * 1.35 && large.height >= small.height * 1.35 && area(large) >= area(small) * 1.8;

  const widths = detections.map((item) => item.width).sort((a, b) => a - b);
  const heights = detections.map((item) => item.height).sort((a, b) => a - b);
  const medianWidth = widths[Math.floor(widths.length / 2)];
  const medianHeight = heights[Math.floor(heights.length / 2)];
  const repeatingCandidates = detections.filter((item) =>
    item.width >= medianWidth * 0.55 && item.width <= medianWidth * 1.55 &&
    item.height >= medianHeight * 0.55 && item.height <= medianHeight * 1.55,
  );

  const rejected = new Set<T>();
  for (const large of detections) {
    if (!enclosingSuppression || large.width < medianWidth * 1.65 || large.height < medianHeight * 1.65) continue; // v1.4.7.0: toggle
    const containedPattern = repeatingCandidates.filter((small) =>
      small !== large && isClearlyLarger(large, small) && containsCenter(large, small),
    );
    // Require at least two distinct, similarly-sized cell candidates. One
    // isolated nested detection is not enough evidence to discard a larger box.
    const hasRepeatedCells = containedPattern.some((first, index) => {
      const firstX = first.left + first.width / 2;
      const firstY = first.top + first.height / 2;
      return containedPattern.slice(index + 1).some((second) => {
        const secondX = second.left + second.width / 2;
        const secondY = second.top + second.height / 2;
        const separation = Math.hypot(secondX - firstX, secondY - firstY);
        const cellScale = Math.min(first.width, first.height, second.width, second.height);
        return separation >= cellScale * 0.65;
      });
    });
    if (hasRepeatedCells) rejected.add(large);
  }

  const ordered = detections.filter((item) => !rejected.has(item)).sort((a, b) => b.confidence - a.confidence);
  const kept: T[] = [];
  for (const candidate of ordered) {
    const duplicate = kept.some((prior) => {
      const intersection = intersectionArea(prior, candidate);
      const union = area(prior) + area(candidate) - intersection;
      return union > 0 && intersection / union >= nmsIou;
    });
    if (!duplicate) kept.push(candidate);
  }
  return kept;
}
