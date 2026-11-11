import type { CrackClassifierTrainingSettings } from '../types/trainingSettings';
import type { DetectedAnomaly, InspectionResult, PixelBox } from '../types/inspection';

export interface FeedbackSummary {
  total: number;
  positives: number;
  negatives: number;
  images: number;
  positiveImages: number;
  negativeImages: number;
  annotatedRegions: number;
  trainingReady: boolean;
  trainingMissing: string[];
  trainingRequirements: {
    positives: number;
    negatives: number;
    images: number;
    positiveImages: number;
    negativeImages: number;
    annotatedRegions: number;
  };
  modelVersion: string | null;
  userTuned: boolean;
  modelAlgorithm?: 'ensemble' | 'gbm' | 'cnn' | 'logistic' | 'unknown' | null;
  modelHasCnn?: boolean;
  modelHasGbm?: boolean;
  modelTrainedAt?: string | null;
  modelValidation?: { precision?: number; recall?: number; specificity?: number; balancedAccuracy?: number; logLoss?: number; f1?: number; accuracy?: number; tp?: number; fp?: number; tn?: number; fn?: number; threshold?: number; positives?: number; negatives?: number } | null;
  componentValidation?: Record<string, { precision?: number; recall?: number; specificity?: number; balancedAccuracy?: number; logLoss?: number; f1?: number; accuracy?: number; tp?: number; fp?: number; tn?: number; fn?: number; threshold?: number; positives?: number; negatives?: number }> | null;
  modelCellBuffer?: number | null;
}

export interface TrainResult extends FeedbackSummary {
  accepted: boolean;
  reason?: string;
  /** v1.4.11: which image set produced the reported metrics (mirrors server TrainReport). */
  evaluationSet?: 'independent-test' | 'model-selection-validation' | 'held-out-validation';
  evaluationNote?: string;
  missedCrackSamples?: number;
  classWeights?: { enabled: boolean; positives: number; negatives: number; imbalanceRatio: number | null; positiveWeight: number; negativeWeight: number };
  samples: number;
  validationImages: number;
  validationSamples: number;
  addedRecallBefore: number;
  addedRecallAfter: number;
  rejectedClearedBefore: number;
  rejectedClearedAfter: number;
  validationBefore: { precision: number; recall: number; specificity: number; balancedAccuracy: number; logLoss: number; f1?: number; accuracy?: number; tp?: number; fp?: number; tn?: number; fn?: number; threshold?: number; positives?: number; negatives?: number };
  validationAfter: { precision: number; recall: number; specificity: number; balancedAccuracy: number; logLoss: number; f1?: number; accuracy?: number; tp?: number; fp?: number; tn?: number; fn?: number; threshold?: number; positives?: number; negatives?: number };
  /** v1.4.8.7 */
  algorithm?: 'ensemble' | 'gbm' | 'cnn' | 'logistic';
  components?: { gbm?: { balancedAccuracy: number; f1?: number }; cnn?: { balancedAccuracy: number; f1?: number } };
}

async function json<T>(res: Response): Promise<T> {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((data as { error?: string }).error || `Request failed (${res.status})`);
  return data as T;
}

export const getFeedbackSummary = (): Promise<FeedbackSummary> => fetch('/api/feedback').then((r) => json<FeedbackSummary>(r));

export const deleteFeedback = (id: string): Promise<unknown> =>
  fetch(`/api/feedback/${encodeURIComponent(id)}`, { method: 'DELETE' }).then((r) => json(r));

export interface CrackTrainingJobStatus {
  status: 'idle' | 'preparing' | 'training' | 'validating' | 'paused' | 'completed' | 'failed';
  jobId?: string;
  progress: number;
  stage: string;
  message: string;
  algorithm?: string;
  epoch?: number;
  epochs?: number;
  tree?: number;
  trees?: number;
  samples?: number;
  trainingSamples?: number;
  validationSamples?: number;
  metrics: Array<{ epoch: number; trainLoss: number; validationLoss: number; precision: number; recall: number; specificity: number; balancedAccuracy: number; f1: number; component: 'CNN' }>;
  log: string[];
  startedAt?: string;
  finishedAt?: string;
  updatedAt?: string;
  error?: string;
  errorTraceback?: string;
  report?: TrainResult;
}
export const trainModel = (settings?: Partial<CrackClassifierTrainingSettings>): Promise<CrackTrainingJobStatus> => fetch('/api/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settings ?? {}) }).then((r) => json<CrackTrainingJobStatus>(r));
export const getCrackTrainingStatus = (): Promise<CrackTrainingJobStatus> => fetch('/api/train/status', { cache: 'no-store' }).then((r) => json<CrackTrainingJobStatus>(r));

/** v1.4.10: pause / resume / abort the running crack-classifier training job. */
export const crackTrainingAction = (action: 'pause' | 'resume' | 'abort'): Promise<CrackTrainingJobStatus> => fetch(`/api/train/${action}`, { method: 'POST' }).then((r) => json<CrackTrainingJobStatus>(r));

export const resetModel = (): Promise<FeedbackSummary> => fetch('/api/train/reset', { method: 'DELETE' }).then((r) => json<FeedbackSummary>(r));

/** Keep summary counters consistent after the user edits the anomaly list. */
export function withAnomalies(result: InspectionResult, anomalies: DetectedAnomaly[]): InspectionResult {
  const n = (f: (a: DetectedAnomaly) => boolean) => anomalies.filter(f).length;
  return {
    ...result,
    anomalies,
    pipelineSummary: {
      ...result.pipelineSummary,
      spiderWebCracksCount: n((a) => a.category === 'spider_web_crack'),
      edgeSpiderWebCracksCount: n((a) => a.category === 'spider_web_crack' && a.isEdgeCrack),
      impactPointsCount: n((a) => a.category === 'impact_point'),
      linearFracturesCount: n((a) => a.category === 'linear_fracture'),
      surfaceResidueCount: n((a) => a.category === 'surface_residue'),
    },
  };
}

/** Save all selected cell labels with one image upload instead of repeating the image per cell. */
export async function sendFeedbackBatch(
  image: ArrayBuffer,
  fileName: string,
  annotations: Array<{ box: PixelBox; label: 0 | 1; category: string; moduleBox?: PixelBox; source?: 'rejected-detection' | 'missed-crack' }>,
  coordinateSpace?: { width: number; height: number },
  replace = false
): Promise<{ items: Array<{ index: number; id: string }>; replaced?: { positives: number; negatives: number }; total: number; positives: number; negatives: number }> {
  if (!annotations.length) throw new Error('Select at least one cell before saving.');
  const metadata = {
    fileName,
    ...(replace ? { replace: true } : {}),
    ...(coordinateSpace ? { coordinateWidth: coordinateSpace.width, coordinateHeight: coordinateSpace.height } : {}),
    annotations: annotations.map((item) => ({
      left: Math.round(item.box.left),
      top: Math.round(item.box.top),
      width: Math.round(item.box.width),
      height: Math.round(item.box.height),
      label: item.label,
      category: item.category,
      ...(item.source ? { source: item.source } : {}),
      moduleBox: item.moduleBox ? {
        left: Math.round(item.moduleBox.left), top: Math.round(item.moduleBox.top),
        width: Math.round(item.moduleBox.width), height: Math.round(item.moduleBox.height),
      } : undefined,
    })),
  };
  // Prefix one JSON metadata line, then append the original image bytes; the image is uploaded once per module batch.
  const body = new Blob([new TextEncoder().encode(JSON.stringify(metadata)), new Uint8Array([10]), new Uint8Array(image)]);
  const res = await fetch('/api/feedback/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream', 'x-feedback-format': 'metadata-line-v1' },
    body,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Batch feedback save failed (${res.status}).`);
  return data as { items: Array<{ index: number; id: string }>; replaced?: { positives: number; negatives: number }; total: number; positives: number; negatives: number };
}

/** Crack labels already stored for this exact image (same SHA-1 id the server uses). Null if it cannot be checked here. */
export async function getExistingImageFeedback(image: ArrayBuffer): Promise<{ positives: number; negatives: number } | null> {
  try {
    if (!globalThis.crypto?.subtle) return null;
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-1', image));
    const id = Array.from(digest).map((b) => b.toString(16).padStart(2, '0')).join('').slice(0, 20) + '.img';
    const res = await fetch(`/api/feedback/existing/${id}`);
    if (!res.ok) return null;
    return (await res.json()) as { positives: number; negatives: number };
  } catch { return null; }
}
