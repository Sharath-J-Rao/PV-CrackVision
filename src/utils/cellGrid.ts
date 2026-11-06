import type { PixelBox } from '../types/inspection';

export interface CellGridEstimate {
  rows: number;
  columns: number;
  confidence: number;
  method: 'reference-cell';
  /** Bounding box of the contiguous PV-cell array inferred from the reference cell. */
  pixelBox: PixelBox;
}

/**
 * Detect a regular PV-cell array from one user-marked reference cell.
 *
 * Instead of guessing a grid count from the module's divider-line profile, this uses the
 * selected cell's width and height as the pitch, then walks in four directions while the
 * neighbouring cell centres retain the colour/brightness signature of PV silicon. This
 * is intentionally suited to the user's fixed-cell-size workflow; it is not a perspective
 * rectifier, so strongly tilted or perspective-distorted arrays still need manual review.
 */
export async function detectCellGrid(
  imageSrc: string,
  referenceCell: PixelBox,
  imageBounds: { width: number; height: number }
): Promise<CellGridEstimate> {
  const cellWidth = referenceCell.width;
  const cellHeight = referenceCell.height;
  if (
    cellWidth < 8 || cellHeight < 8 ||
    cellWidth > imageBounds.width * 0.4 || cellHeight > imageBounds.height * 0.6
  ) {
    throw new Error('The reference box does not look like one solar cell. Mark the four cell corners more tightly.');
  }

  const image = await loadImage(imageSrc);
  const maxSide = 2200;
  const scale = Math.min(1, maxSide / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Could not prepare the image for solar-cell detection.');
  context.drawImage(image, 0, 0, width, height);
  const rgba = context.getImageData(0, 0, width, height).data;

  const ref = {
    left: referenceCell.left * scale,
    top: referenceCell.top * scale,
    width: cellWidth * scale,
    height: cellHeight * scale,
  };
  const imageW = width;
  const imageH = height;
  const sampleOffsets = [-0.24, -0.12, 0, 0.12, 0.24];

  // A valid PV cell is predominantly dark/blue-neutral silicon. Warm soil and dry grass
  // are rejected by the blue-vs-red test; frames and gaps are rejected by brightness.
  const pvScore = (cx: number, cy: number): number => {
    let matches = 0;
    let samples = 0;
    for (const oy of sampleOffsets) {
      for (const ox of sampleOffsets) {
        const x = Math.round(cx + ox * ref.width);
        const y = Math.round(cy + oy * ref.height);
        if (x < 0 || y < 0 || x >= imageW || y >= imageH) continue;
        const i = (y * imageW + x) * 4;
        const r = rgba[i], g = rgba[i + 1], b = rgba[i + 2];
        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        samples++;
        if (
          b >= r - 8 &&
          lum > 12 && lum < 208 &&
          r < 190 &&
          !(r > g + 25 && r > b + 25)
        ) matches++;
      }
    }
    return samples ? matches / samples : 0;
  };

  const refCx = ref.left + ref.width / 2;
  const refCy = ref.top + ref.height / 2;
  const scan = (dx: number, dy: number): { count: number; confidence: number } => {
    let lastValid = 0;
    let misses = 0;
    let totalScore = 0;
    let validSamples = 0;
    const maxSteps = Math.ceil(Math.max(imageW / ref.width, imageH / ref.height)) + 1;

    for (let step = 1; step <= maxSteps; step++) {
      const cx = refCx + dx * ref.width * step;
      const cy = refCy + dy * ref.height * step;
      const boxLeft = cx - ref.width / 2;
      const boxTop = cy - ref.height / 2;
      // Do not count partial cells at the image boundary.
      if (
        boxLeft < 0 || boxTop < 0 ||
        boxLeft + ref.width > imageW || boxTop + ref.height > imageH
      ) break;

      const score = pvScore(cx, cy);
      if (score >= 0.64) {
        lastValid = step;
        misses = 0;
        totalScore += score;
        validSamples++;
      } else {
        misses++;
        // One weak cell can occur under glare; two consecutive misses normally mark a frame/gap.
        if (misses >= 2) break;
      }
    }
    return {
      count: lastValid,
      confidence: validSamples ? totalScore / validSamples : 0,
    };
  };

  const left = scan(-1, 0);
  const right = scan(1, 0);
  const up = scan(0, -1);
  const down = scan(0, 1);
  const columns = left.count + 1 + right.count;
  const rows = up.count + 1 + down.count;
  if (columns < 2 || rows < 2) {
    throw new Error('Could not find enough neighbouring solar cells. Zoom in and mark a complete, typical cell inside the panel.');
  }

  const pixelBox: PixelBox = {
    left: Math.round(referenceCell.left - left.count * cellWidth),
    top: Math.round(referenceCell.top - up.count * cellHeight),
    width: Math.round(columns * cellWidth),
    height: Math.round(rows * cellHeight),
  };
  const confidence = (left.confidence + right.confidence + up.confidence + down.confidence) / 4;
  return {
    rows,
    columns,
    confidence: Math.max(0, Math.min(1, confidence)),
    method: 'reference-cell',
    pixelBox,
  };
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Could not load source image for solar-cell detection.'));
    image.src = src;
  });
}
