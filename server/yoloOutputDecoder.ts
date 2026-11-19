import type { PixelBox } from '../src/types/inspection.ts';
import { suppressOversizedPatternOverlaps, type ScoredCellBox } from './cellDetectionGeometry.ts';

export interface YoloOutputTensorLike {
  dims?: number[];
  data: ArrayLike<number>;
}

export interface YoloDecodeDiagnostics {
  outputShape: string;
  candidates: number;
  candidatesAboveThreshold: number;
  validBoxesBeforeSuppression: number;
  maximumScore: number;
  confidenceThreshold: number;
  candidatesLimited: number;
}

export interface YoloDecodeResult {
  boxes: Array<PixelBox & { confidence: number }>;
  diagnostics: YoloDecodeDiagnostics;
}

/**
 * Decode Ultralytics single-class raw detection output [1, 5, N] or [1, N, 5].
 * The five attributes are cx, cy, width, height, and the single class score.
 */
export function decodeYoloCellOutput(
  output: YoloOutputTensorLike,
  imageWidth: number,
  imageHeight: number,
  scale: number,
  padX: number,
  padY: number,
  confidenceThreshold = 0.005,
  maxCandidatesBeforeNms = 700,
  // v1.4.7.0: previously hard-coded (0.65 / 1200 / enclosing-suppression on); defaults preserve old behaviour.
  options: { nmsIou?: number; maxBoxes?: number; enclosingSuppression?: boolean } = {},
): YoloDecodeResult {
  const dims = output.dims ?? [];
  if (dims.length !== 3 || dims[0] !== 1) {
    throw new Error(`Unsupported YOLO ONNX output shape: ${dims.join('x')}`);
  }
  // Do not infer the layout from which axis is smaller: tiny test tensors such
  // as [1,5,1] are valid channel-first outputs, and [1,N,5] is channel-last.
  const channelFirst = dims[1] === 5;
  const channelLast = dims[2] === 5;
  if (channelFirst === channelLast) {
    throw new Error(`Expected exactly one 5-value attribute axis for a single-class YOLO output; received shape ${dims.join('x')}.`);
  }
  const channelsFirst = channelFirst;
  const attributes = 5;
  const candidates = channelsFirst ? dims[2] : dims[1];
  const read = (attribute: number, candidate: number) => Number(
    output.data[channelsFirst ? attribute * candidates + candidate : candidate * attributes + attribute],
  );
  const found: ScoredCellBox[] = [];
  let maximumScore = 0;
  let candidatesAboveThreshold = 0;

  for (let index = 0; index < candidates; index++) {
    const cx = read(0, index);
    const cy = read(1, index);
    const width = read(2, index);
    const height = read(3, index);
    const score = read(4, index);
    if (Number.isFinite(score)) maximumScore = Math.max(maximumScore, score);
    if (!Number.isFinite(score) || score < confidenceThreshold) continue;
    candidatesAboveThreshold++;
    if (![cx, cy, width, height].every(Number.isFinite) || width <= 1 || height <= 1) continue;

    const left = Math.max(0, Math.min(imageWidth, (cx - width / 2 - padX) / scale));
    const top = Math.max(0, Math.min(imageHeight, (cy - height / 2 - padY) / scale));
    const right = Math.max(left, Math.min(imageWidth, (cx + width / 2 - padX) / scale));
    const bottom = Math.max(top, Math.min(imageHeight, (cy + height / 2 - padY) / scale));
    if (right - left < 2 || bottom - top < 2) continue;
    found.push({ left, top, width: right - left, height: bottom - top, confidence: score });
  }

  // Prefer recall for a cell-localisation model. Operators can remove extra boxes;
  // a high threshold that silently hides all low-confidence cells cannot be corrected.
  // Bound NMS cost for large images / tiled inference while preserving the
  // strongest candidates. A 960px tile can contain hundreds of real cells.
  const bounded = found.sort((a, b) => b.confidence - a.confidence).slice(0, Math.max(1, Math.floor(maxCandidatesBeforeNms)));
  const filtered = suppressOversizedPatternOverlaps(bounded, options.nmsIou ?? 0.65, options.enclosingSuppression ?? true).slice(0, Math.max(1, Math.floor(options.maxBoxes ?? 1200)));
  return {
    boxes: filtered,
    diagnostics: {
      outputShape: dims.join('x'),
      candidates,
      candidatesAboveThreshold,
      validBoxesBeforeSuppression: found.length,
      maximumScore,
      confidenceThreshold,
      candidatesLimited: Math.max(0, found.length - bounded.length),
    },
  };
}
