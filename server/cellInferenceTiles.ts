/** Window planner for high-recall cell inference on images whose scale differs from training samples. */
export interface CellInferenceWindow {
  left: number;
  top: number;
  width: number;
  height: number;
  kind: 'full' | 'tile';
}

function axisOffsets(length: number, tile: number, overlap: number): number[] {
  if (length <= tile) return [0];
  const stride = Math.max(1, Math.floor(tile * (1 - overlap)));
  const offsets: number[] = [];
  for (let offset = 0; offset < length - tile; offset += stride) offsets.push(offset);
  offsets.push(length - tile);
  return [...new Set(offsets)];
}

/**
 * Keep the whole-image pass for context, then add overlapping crops at 1.5x the
 * ONNX input dimensions. The crop pass restores small-cell detail in large field
 * images instead of shrinking the entire array into one 960px square.
 */
export function buildCellInferenceTiles(
  imageWidth: number,
  imageHeight: number,
  inputWidth: number,
  inputHeight: number,
  options: { overlap?: number; tileScale?: number; maxTiles?: number } = {},
): CellInferenceWindow[] {
  if (![imageWidth, imageHeight, inputWidth, inputHeight].every((value) => Number.isFinite(value) && value > 0)) {
    throw new Error('Cell inference image and model dimensions must be positive finite numbers.');
  }
  const overlap = Math.max(0.1, Math.min(0.45, options.overlap ?? 0.22));
  const maxTiles = Math.max(1, Math.floor(options.maxTiles ?? 36));
  let scale = Math.max(1, options.tileScale ?? 1.5);
  let tileWidth = Math.min(imageWidth, Math.max(1, Math.round(inputWidth * scale)));
  let tileHeight = Math.min(imageHeight, Math.max(1, Math.round(inputHeight * scale)));
  let xs = axisOffsets(imageWidth, tileWidth, overlap);
  let ys = axisOffsets(imageHeight, tileHeight, overlap);

  // Bound CPU work on very large drone mosaics while still covering every pixel.
  while (xs.length * ys.length > maxTiles && scale < 12) {
    scale *= 1.2;
    tileWidth = Math.min(imageWidth, Math.max(1, Math.round(inputWidth * scale)));
    tileHeight = Math.min(imageHeight, Math.max(1, Math.round(inputHeight * scale)));
    xs = axisOffsets(imageWidth, tileWidth, overlap);
    ys = axisOffsets(imageHeight, tileHeight, overlap);
  }

  if (tileWidth >= imageWidth && tileHeight >= imageHeight) return [];
  const tiles: CellInferenceWindow[] = [];
  for (const top of ys) {
    for (const left of xs) {
      const width = Math.min(tileWidth, imageWidth - left);
      const height = Math.min(tileHeight, imageHeight - top);
      if (width <= 0 || height <= 0) continue;
      tiles.push({ left, top, width, height, kind: 'tile' });
    }
  }
  return tiles;
}

export function offsetCellBoxes<T extends { left: number; top: number; width: number; height: number }>(
  boxes: readonly T[],
  window: Pick<CellInferenceWindow, 'left' | 'top' | 'width' | 'height'>,
  imageWidth: number,
  imageHeight: number,
): T[] {
  return boxes.flatMap((box) => {
    const left = Math.max(0, Math.min(imageWidth, box.left + window.left));
    const top = Math.max(0, Math.min(imageHeight, box.top + window.top));
    const right = Math.max(left, Math.min(imageWidth, box.left + box.width + window.left));
    const bottom = Math.max(top, Math.min(imageHeight, box.top + box.height + window.top));
    if (right - left < 2 || bottom - top < 2) return [];
    return [{ ...box, left, top, width: right - left, height: bottom - top }];
  });
}
