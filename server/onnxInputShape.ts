/** Resolve fixed ONNX NCHW dimensions so image preprocessing always matches the exported model. */
export const DEFAULT_ONNX_IMAGE_SIZE = 960;

export type OnnxInputMetadataLike = { name?: string; isTensor?: boolean; shape?: ReadonlyArray<number | string>; dimensions?: ReadonlyArray<number | string>; dims?: ReadonlyArray<number | string> };

export function resolveOnnxInputShape(
  metadata: OnnxInputMetadataLike | undefined,
  fallbackSize = DEFAULT_ONNX_IMAGE_SIZE,
): { channels: 3; height: number; width: number } {
  // onnxruntime-node exposes `dimensions` in some versions and wrappers expose
  // `shape`; accept both so the running server never guesses a stale input size.
  const shape = metadata?.shape ?? metadata?.dimensions ?? metadata?.dims;
  if (!Array.isArray(shape) || shape.length !== 4) {
    return { channels: 3, height: fallbackSize, width: fallbackSize };
  }
  const channelDimension = shape[1];
  if (typeof channelDimension === 'number' && channelDimension > 0 && channelDimension !== 3) {
    throw new Error(`Unsupported ONNX input channel count ${channelDimension}; expected 3-channel RGB NCHW input.`);
  }
  const height = typeof shape[2] === 'number' && Number.isInteger(shape[2]) && shape[2] > 0 ? shape[2] : fallbackSize;
  const width = typeof shape[3] === 'number' && Number.isInteger(shape[3]) && shape[3] > 0 ? shape[3] : fallbackSize;
  return { channels: 3, height, width };
}

export function calculateLetterbox(
  imageWidth: number,
  imageHeight: number,
  targetWidth: number,
  targetHeight: number,
): { scale: number; resizedWidth: number; resizedHeight: number; padX: number; padY: number } {
  if (![imageWidth, imageHeight, targetWidth, targetHeight].every((value) => Number.isFinite(value) && value > 0)) {
    throw new Error('Image and ONNX input dimensions must be positive finite numbers.');
  }
  const scale = Math.min(targetWidth / imageWidth, targetHeight / imageHeight);
  const resizedWidth = Math.max(1, Math.min(targetWidth, Math.round(imageWidth * scale)));
  const resizedHeight = Math.max(1, Math.min(targetHeight, Math.round(imageHeight * scale)));
  return {
    scale,
    resizedWidth,
    resizedHeight,
    padX: Math.floor((targetWidth - resizedWidth) / 2),
    padY: Math.floor((targetHeight - resizedHeight) / 2),
  };
}
