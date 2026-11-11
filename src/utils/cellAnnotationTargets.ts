import type { ModuleRegion, PixelBox, SolarCellBox, SolarCellGrid } from '../types/inspection';

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function clusterCenters(values: number[], tolerance: number): number[] {
  const sorted = [...values].sort((a, b) => a - b);
  const clusters: number[][] = [];
  for (const value of sorted) {
    const current = clusters[clusters.length - 1];
    const center = current?.reduce((sum, item) => sum + item, 0) / (current?.length || 1);
    if (current && Math.abs(value - center) <= tolerance) current.push(value);
    else clusters.push([value]);
  }
  return clusters.map((cluster) => cluster.reduce((sum, value) => sum + value, 0) / cluster.length);
}

function nearestCenterIndex(value: number, centers: number[]): number {
  let best = 0;
  for (let index = 1; index < centers.length; index++) {
    if (Math.abs(centers[index] - value) < Math.abs(centers[best] - value)) best = index;
  }
  return best;
}

function gridFromDetectedBoxes(boxes: PixelBox[]): SolarCellGrid | undefined {
  if (!boxes.length) return undefined;
  const centersX = boxes.map((box) => box.left + box.width / 2);
  const centersY = boxes.map((box) => box.top + box.height / 2);
  const typicalWidth = median(boxes.map((box) => box.width));
  const typicalHeight = median(boxes.map((box) => box.height));
  const columns = clusterCenters(centersX, Math.max(3, typicalWidth * 0.55));
  const rows = clusterCenters(centersY, Math.max(3, typicalHeight * 0.55));
  const cells: SolarCellBox[] = boxes.map((box) => ({
    row: nearestCenterIndex(box.top + box.height / 2, rows) + 1,
    column: nearestCenterIndex(box.left + box.width / 2, columns) + 1,
    pixelBox: { ...box },
    // The API currently exposes geometry but not per-box confidence. Do not invent a
    // model score; this neutral value is used only for the annotation UI.
    confidence: 0.5,
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
    pixelBox: { left, top, width: Math.max(1, right - left), height: Math.max(1, bottom - top) },
    cells,
    pitchX,
    pitchY,
    confidence: 0.5,
    method: 'yolo-cell-detection',
  };
}

function centerDistanceToBox(point: { x: number; y: number }, box: PixelBox): number {
  const dx = point.x < box.left ? box.left - point.x : point.x > box.left + box.width ? point.x - box.left - box.width : 0;
  const dy = point.y < box.top ? box.top - point.y : point.y > box.top + box.height ? point.y - box.top - box.height : 0;
  return Math.hypot(dx, dy);
}

/**
 * Turn the exact boxes drawn by the direct YOLO overlay into annotation hit targets.
 * This deliberately accepts one or more boxes: the old training/inference path required
 * four boxes before it created a cell grid, which left visible detections impossible to
 * click-label in the crack-classifier workflow.
 */
export function buildSelectableCellRegions(
  regions: readonly ModuleRegion[],
  detectionBoxes: readonly PixelBox[],
  imageWidth: number,
  imageHeight: number,
): ModuleRegion[] {
  if (!detectionBoxes.length) return [...regions];

  const candidates = [...regions].sort((a, b) =>
    (a.pixelBox.width * a.pixelBox.height) - (b.pixelBox.width * b.pixelBox.height),
  );
  const assigned = new Map<string, PixelBox[]>();
  const unassigned: PixelBox[] = [];

  for (const box of detectionBoxes) {
    if (![box.left, box.top, box.width, box.height].every(Number.isFinite) || box.width <= 0 || box.height <= 0) continue;
    const center = { x: box.left + box.width / 2, y: box.top + box.height / 2 };
    const match = candidates.find(({ pixelBox }) => {
      const marginX = pixelBox.width * 0.04;
      const marginY = pixelBox.height * 0.04;
      return center.x >= pixelBox.left - marginX && center.x <= pixelBox.left + pixelBox.width + marginX &&
        center.y >= pixelBox.top - marginY && center.y <= pixelBox.top + pixelBox.height + marginY;
    });
    if (match) {
      const list = assigned.get(match.id) ?? [];
      list.push(box);
      assigned.set(match.id, list);
    } else {
      unassigned.push(box);
    }
  }

  const enriched = regions.map((region) => {
    const boxes = assigned.get(region.id) ?? [];
    const cellGrid = gridFromDetectedBoxes(boxes);
    return cellGrid ? { ...region, cellGrid } : region;
  });

  // Do not silently hide detections that fall just outside an automatically segmented
  // PV region. Keep them selectable in a review region so the operator can mark them as
  // positive or negative rather than losing the visible detector output.
  if (unassigned.length) {
    const cellGrid = gridFromDetectedBoxes(unassigned);
    if (cellGrid) {
      const left = Math.max(0, Math.min(...unassigned.map((box) => box.left)));
      const top = Math.max(0, Math.min(...unassigned.map((box) => box.top)));
      const right = Math.min(imageWidth, Math.max(...unassigned.map((box) => box.left + box.width)));
      const bottom = Math.min(imageHeight, Math.max(...unassigned.map((box) => box.top + box.height)));
      const pixelBox = { left, top, width: Math.max(1, right - left), height: Math.max(1, bottom - top) };
      const nearRegion = [...regions].sort((a, b) =>
        centerDistanceToBox({ x: left + pixelBox.width / 2, y: top + pixelBox.height / 2 }, a.pixelBox) -
        centerDistanceToBox({ x: left + pixelBox.width / 2, y: top + pixelBox.height / 2 }, b.pixelBox),
      )[0];
      const label = nearRegion ? `Unassigned YOLO detections near ${nearRegion.label}` : 'Unassigned YOLO cell detections';
      enriched.push({
        id: 'yolo-unassigned-cells',
        label,
        isFallback: true,
        pixelBox,
        box: {
          xmin: pixelBox.left / Math.max(1, imageWidth),
          ymin: pixelBox.top / Math.max(1, imageHeight),
          xmax: (pixelBox.left + pixelBox.width) / Math.max(1, imageWidth),
          ymax: (pixelBox.top + pixelBox.height) / Math.max(1, imageHeight),
        },
        cellGrid,
      });
    }
  }

  return enriched;
}
