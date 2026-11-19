import type { PixelBox } from '../src/types/inspection.ts';

/** Convert coordinates from the inspected (possibly 8k-capped) frame to the full normalized source image. */
export function scaleCellBoxesToSourceImage(
  boxes: readonly PixelBox[],
  sourceWidth: number,
  sourceHeight: number,
  coordinateWidth?: number,
  coordinateHeight?: number,
): PixelBox[] {
  if (![sourceWidth, sourceHeight].every((value) => Number.isFinite(value) && value > 0)) {
    throw new Error('Normalized source-image dimensions must be positive finite numbers.');
  }
  const hasCoordinateSpace = Number.isFinite(coordinateWidth) && Number(coordinateWidth) > 0 &&
    Number.isFinite(coordinateHeight) && Number(coordinateHeight) > 0;
  const scaleX = hasCoordinateSpace ? sourceWidth / Number(coordinateWidth) : 1;
  const scaleY = hasCoordinateSpace ? sourceHeight / Number(coordinateHeight) : 1;
  if (hasCoordinateSpace && Math.abs(scaleX - scaleY) / Math.max(scaleX, scaleY) > 0.01) {
    throw new Error(`The inspected frame (${coordinateWidth}×${coordinateHeight}) and normalized source image (${sourceWidth}×${sourceHeight}) have different aspect ratios. Re-run inspection against the original image before saving.`);
  }
  return boxes.map((box, index) => {
    if (![box.left, box.top, box.width, box.height].every(Number.isFinite)) {
      throw new Error(`Cell box ${index + 1} has non-finite coordinates.`);
    }
    const scaled = {
      left: box.left * scaleX,
      top: box.top * scaleY,
      width: box.width * scaleX,
      height: box.height * scaleY,
    };
    if (scaled.left < 0 || scaled.top < 0 || scaled.width < 8 || scaled.height < 8 ||
        scaled.left + scaled.width > sourceWidth + 1 || scaled.top + scaled.height > sourceHeight + 1) {
      throw new Error(`Cell box ${index + 1} is outside the source image or too small after coordinate conversion.`);
    }
    const ratio = scaled.width / scaled.height;
    if (ratio < 0.35 || ratio > 3.2) {
      throw new Error(`Cell box ${index + 1} has an implausible shape. Draw tightly around one individual PV cell.`);
    }
    return {
      left: Math.round(scaled.left),
      top: Math.round(scaled.top),
      width: Math.round(scaled.width),
      height: Math.round(scaled.height),
    };
  });
}
