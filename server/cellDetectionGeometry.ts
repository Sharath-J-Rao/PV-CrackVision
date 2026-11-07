import type { ModuleRegion, PixelBox, SolarCellBox, SolarCellGrid } from '../src/types/inspection.ts';

export interface ScoredCellBox extends PixelBox { confidence: number }

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

export function gridFromYoloCellBoxes(boxes: ScoredCellBox[]): SolarCellGrid | undefined {
  if (boxes.length < 4) return undefined;
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
 * Apply direct YOLO cell boxes to the existing automatically located PV regions.
 * The regions are only spatial groupings; there is no learned module detector.
 * If the classical stage only produced a whole-image fallback, all direct cell
 * detections are kept in that region so the cell detector can still drive cell crops.
 */
export function applyYoloCellsToRegions(regions: ModuleRegion[], detections: ScoredCellBox[]): { regions: ModuleRegion[]; applied: number } {
  if (detections.length < 4 || regions.length === 0) return { regions, applied: 0 };
  const regionBoxes = regions.map((region) => ({
    region,
    area: region.pixelBox.width * region.pixelBox.height,
  })).sort((a, b) => a.area - b.area);
  const assigned = new Map<string, ScoredCellBox[]>();
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
    if (!match) continue;
    const list = assigned.get(match.region.id) ?? [];
    list.push(detection);
    assigned.set(match.region.id, list);
  }

  let applied = 0;
  const next = regions.map((region) => {
    const boxes = assigned.get(region.id) ?? [];
    const grid = gridFromYoloCellBoxes(boxes);
    if (!grid) return region;
    applied += grid.cells.length;
    return { ...region, cellGrid: grid };
  });
  return { regions: next, applied };
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
