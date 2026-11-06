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
}

export interface TrainResult extends FeedbackSummary {
  accepted: boolean;
  reason?: string;
  samples: number;
  validationImages: number;
  validationSamples: number;
  addedRecallBefore: number;
  addedRecallAfter: number;
  rejectedClearedBefore: number;
  rejectedClearedAfter: number;
  validationBefore: { precision: number; recall: number; specificity: number; balancedAccuracy: number; logLoss: number };
  validationAfter: { precision: number; recall: number; specificity: number; balancedAccuracy: number; logLoss: number };
}

async function json<T>(res: Response): Promise<T> {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((data as { error?: string }).error || `Request failed (${res.status})`);
  return data as T;
}

export const getFeedbackSummary = (): Promise<FeedbackSummary> => fetch('/api/feedback').then((r) => json<FeedbackSummary>(r));

/** label 1 = real issue the user added; 0 = false detection the user deleted. Returns the stored feedback id. */
export async function sendFeedback(
  image: ArrayBuffer,
  fileName: string,
  label: 0 | 1,
  box: PixelBox,
  category: string
): Promise<string> {
  const res = await fetch('/api/feedback', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/octet-stream',
      'x-file-name': encodeURIComponent(fileName),
      'x-label': String(label),
      'x-category': encodeURIComponent(category),
      'x-box': [box.left, box.top, box.width, box.height].map((n) => Math.round(n)).join(','),
    },
    body: image,
  });
  return (await json<{ id: string }>(res)).id;
}

export const deleteFeedback = (id: string): Promise<unknown> =>
  fetch(`/api/feedback/${encodeURIComponent(id)}`, { method: 'DELETE' }).then((r) => json(r));

export const trainModel = (): Promise<TrainResult> => fetch('/api/train', { method: 'POST' }).then((r) => json<TrainResult>(r));

export const resetModel = (): Promise<FeedbackSummary> => fetch('/api/train/model', { method: 'DELETE' }).then((r) => json<FeedbackSummary>(r));

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
  annotations: Array<{ box: PixelBox; label: 0 | 1; category: string; moduleBox?: PixelBox }>
): Promise<{ items: Array<{ index: number; id: string }>; total: number; positives: number; negatives: number }> {
  if (!annotations.length) throw new Error('Select at least one cell before saving.');
  const metadata = {
    fileName,
    annotations: annotations.map((item) => ({
      left: Math.round(item.box.left),
      top: Math.round(item.box.top),
      width: Math.round(item.box.width),
      height: Math.round(item.box.height),
      label: item.label,
      category: item.category,
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
  return data as { items: Array<{ index: number; id: string }>; total: number; positives: number; negatives: number };
}
