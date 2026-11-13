/** Cell-crop geometry shared by the crack classifier (v1.4.15.1: the legacy boosted-tree/CNN model and its feature extraction were removed). */

/** Default minimal cell buffer (fraction of the short side); about one pixel on a 35 px cell, three on a 100 px cell. */
export const DEFAULT_CELL_BUFFER = 0.03;

/**
 * Crop rectangle for one detected cell: the box grown by a minimal buffer on every side (so the cell boundary is
 * included even when the detection is slightly tight), clamped to the image. buffer = 0 reproduces the exact box.
 */
export function bufferedCellCrop(
  imageWidth: number,
  imageHeight: number,
  box: { left: number; top: number; width: number; height: number },
  buffer = 0,
): { left: number; top: number; width: number; height: number } {
  const b = buffer > 0 ? Math.max(1, Math.round(Math.min(box.width, box.height) * buffer)) : 0;
  const left = Math.max(0, Math.floor(box.left - b));
  const top = Math.max(0, Math.floor(box.top - b));
  const right = Math.min(imageWidth, Math.ceil(box.left + box.width + b));
  const bottom = Math.min(imageHeight, Math.ceil(box.top + box.height + b));
  return { left, top, width: right - left, height: bottom - top };
}
