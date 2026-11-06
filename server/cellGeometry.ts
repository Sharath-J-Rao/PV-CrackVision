/** Pure cell-rectangle validation shared by the detector and its regression tests. */
export interface CellPixelBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface ValidatedSolarCell {
  row: number;
  column: number;
  pixelBox: CellPixelBox;
  confidence: number;
  partial: boolean;
}

export interface ValidatedCellGeometry {
  cells: ValidatedSolarCell[];
  confidence: number;
}

export interface CellGeometryInput {
  gridX0: number;
  gridY0: number;
  pitchX: number;
  pitchY: number;
  rows: number;
  columns: number;
  width: number;
  height: number;
  imageWidth: number;
  imageHeight: number;
  pvSiliconMask: Uint8Array;
  /** Directional maps used to verify that each proposed cell has real boundary edges. */
  verticalEdgeMap?: Float32Array;
  horizontalEdgeMap?: Float32Array;
  periodicityConfidence: number;
  /** Visible PV-bearing band; cell proposals are clipped to this area. */
  clipBounds: { left: number; top: number; right: number; bottom: number };
}

/**
 * Convert a proposed regular lattice into individually validated solar-cell rectangles.
 * Slots without PV-silicon occupancy are omitted: a gap or adjacent non-PV surface is never
 * filled simply because it falls inside a broad region's bounding rectangle.
 */
export function buildValidatedCellGeometry(input: CellGeometryInput): ValidatedCellGeometry {
  const {
    gridX0, gridY0, pitchX, pitchY, rows, columns, width, height,
    imageWidth, imageHeight, pvSiliconMask, clipBounds,
    verticalEdgeMap, horizontalEdgeMap,
  } = input;
  const periodicityConfidence = clamp01(input.periodicityConfidence);
  if (
    !Number.isFinite(pitchX) || !Number.isFinite(pitchY) || pitchX <= 0 || pitchY <= 0 ||
    width < 1 || height < 1 || pvSiliconMask.length !== width * height ||
    (verticalEdgeMap && verticalEdgeMap.length !== width * height) ||
    (horizontalEdgeMap && horizontalEdgeMap.length !== width * height) ||
    ![clipBounds.left, clipBounds.top, clipBounds.right, clipBounds.bottom].every(Number.isFinite) ||
    clipBounds.right <= clipBounds.left || clipBounds.bottom <= clipBounds.top
  ) return { cells: [], confidence: 0 };

  const cells: ValidatedSolarCell[] = [];
  let occupancyTotal = 0;

  // Estimate what a genuine grid seam looks like in this local band. Using a local
  // median makes the rule adaptive to image exposure and resolution.
  const edgeStrengthAtX = (x: number): number => {
    if (!verticalEdgeMap || x < 0 || x >= width) return 0;
    const yStart = Math.max(0, Math.floor(clipBounds.top + pitchY * 0.12));
    const yEnd = Math.min(height, Math.ceil(clipBounds.bottom - pitchY * 0.12));
    let sum = 0, n = 0;
    for (let y = yStart; y < yEnd; y += 2) {
      let best = 0;
      for (let dx = -2; dx <= 2; dx++) {
        const xx = x + dx;
        if (xx >= 0 && xx < width) best = Math.max(best, verticalEdgeMap[y * width + xx]);
      }
      sum += best; n++;
    }
    return n ? sum / n : 0;
  };
  const edgeStrengthAtY = (y: number): number => {
    if (!horizontalEdgeMap || y < 0 || y >= height) return 0;
    const xStart = Math.max(0, Math.floor(clipBounds.left + pitchX * 0.12));
    const xEnd = Math.min(width, Math.ceil(clipBounds.right - pitchX * 0.12));
    let sum = 0, n = 0;
    for (let x = xStart; x < xEnd; x += 2) {
      let best = 0;
      for (let dy = -2; dy <= 2; dy++) {
        const yy = y + dy;
        if (yy >= 0 && yy < height) best = Math.max(best, horizontalEdgeMap[yy * width + x]);
      }
      sum += best; n++;
    }
    return n ? sum / n : 0;
  };
  const median = (values: number[]): number => {
    const sorted = values.filter((v) => Number.isFinite(v) && v > 0).sort((a, b) => a - b);
    return sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
  };
  const verticalSupports = Array.from({ length: columns + 1 }, (_, i) =>
    edgeStrengthAtX(Math.round(gridX0 + i * pitchX))
  );
  const horizontalSupports = Array.from({ length: rows + 1 }, (_, i) =>
    edgeStrengthAtY(Math.round(gridY0 + i * pitchY))
  );
  const verticalThreshold = verticalEdgeMap ? Math.max(60, median(verticalSupports) * 0.42) : 0;
  const horizontalThreshold = horizontalEdgeMap ? Math.max(60, median(horizontalSupports) * 0.42) : 0;
  for (let row = 0; row < rows; row++) {
    const cellTop = gridY0 + row * pitchY;
    const cellBottom = cellTop + pitchY;
    for (let column = 0; column < columns; column++) {
      const cellLeft = gridX0 + column * pitchX;
      const cellRight = cellLeft + pitchX;

      // A region only proposes where to look; it must never truncate a cell box.
      // This avoids the old behavior where module/region ROI edges created short,
      // overlapping rectangles. The final rectangle is clipped only by the image frame.
      const overlapW = Math.max(0, Math.min(cellRight, clipBounds.right) - Math.max(cellLeft, clipBounds.left));
      const overlapH = Math.max(0, Math.min(cellBottom, clipBounds.bottom) - Math.max(cellTop, clipBounds.top));
      const proposalOverlap = (overlapW * overlapH) / (pitchX * pitchY);
      if (proposalOverlap < 0.28) continue;

      const x0 = Math.max(0, Math.floor(cellLeft));
      const x1 = Math.min(width, Math.ceil(cellRight));
      const y0 = Math.max(0, Math.floor(cellTop));
      const y1 = Math.min(height, Math.ceil(cellBottom));
      const visibleW = x1 - x0;
      const visibleH = y1 - y0;
      if (visibleW < Math.max(5, pitchX * 0.28) || visibleH < Math.max(5, pitchY * 0.28)) continue;

      // Ignore the cell rim, which contains the gridlines used for lattice alignment.
      const insetX = Math.min(Math.max(1, Math.round(pitchX * 0.12)), Math.floor(visibleW * 0.22));
      const insetY = Math.min(Math.max(1, Math.round(pitchY * 0.12)), Math.floor(visibleH * 0.22));
      const ix0 = Math.min(x1 - 1, x0 + insetX);
      const ix1 = Math.max(ix0 + 1, x1 - insetX);
      const iy0 = Math.min(y1 - 1, y0 + insetY);
      const iy1 = Math.max(iy0 + 1, y1 - insetY);
      let pvPixels = 0;
      let testedPixels = 0;
      for (let y = iy0; y < iy1; y++) {
        const offset = y * width;
        for (let x = ix0; x < ix1; x++) {
          testedPixels++;
          pvPixels += pvSiliconMask[offset + x] ? 1 : 0;
        }
      }
      const occupancy = testedPixels ? pvPixels / testedPixels : 0;
      // Only the real image boundary makes a cell partial. A region edge must not
      // lower the PV-occupancy threshold and admit dark ground as a clipped cell.
      const partialLeft = cellLeft <= 0;
      const partialTop = cellTop <= 0;
      const partialRight = cellRight >= width;
      const partialBottom = cellBottom >= height;
      const partial = partialLeft || partialTop || partialRight || partialBottom;
      if (occupancy < (partial ? 0.34 : 0.52)) continue;

      // A dark shadow can pass a color-only PV mask. Require all visible cell sides
      // to coincide with directional edge-map seams; cropped sides at the real image
      // frame are exempt, so half-visible edge cells can still be detected.
      if (verticalEdgeMap && horizontalEdgeMap) {
        const leftSupport = edgeStrengthAtX(Math.round(cellLeft));
        const rightSupport = edgeStrengthAtX(Math.round(cellRight));
        const topSupport = edgeStrengthAtY(Math.round(cellTop));
        const bottomSupport = edgeStrengthAtY(Math.round(cellBottom));
        if ((!partialLeft && leftSupport < verticalThreshold) ||
            (!partialRight && rightSupport < verticalThreshold) ||
            (!partialTop && topSupport < horizontalThreshold) ||
            (!partialBottom && bottomSupport < horizontalThreshold)) continue;
      }

      const left = Math.round((x0 / width) * imageWidth);
      const top = Math.round((y0 / height) * imageHeight);
      const right = Math.round((x1 / width) * imageWidth);
      const bottom = Math.round((y1 / height) * imageHeight);
      cells.push({
        row: row + 1,
        column: column + 1,
        pixelBox: {
          left,
          top,
          width: Math.max(1, right - left),
          height: Math.max(1, bottom - top),
        },
        confidence: Number(clamp01(0.55 * periodicityConfidence + 0.45 * occupancy).toFixed(3)),
        partial,
      });
      occupancyTotal += occupancy;
    }
  }

  if (cells.length < 4) return { cells: [], confidence: 0 };
  const meanOccupancy = occupancyTotal / cells.length;
  return {
    cells,
    confidence: Number(clamp01(0.58 * periodicityConfidence + 0.42 * meanOccupancy).toFixed(3)),
  };
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}
