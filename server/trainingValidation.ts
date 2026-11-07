/** Shared readiness checks for reviewed cell labels and image-level holdout validation. */
export const TRAINING_REQUIREMENTS = {
  positives: 8,
  negatives: 8,
  images: 4,
  positiveImages: 2,
  negativeImages: 2,
  annotatedRegions: 2,
} as const;

export interface FeedbackReadinessItem {
  imageFile: string;
  label: 0 | 1;
  moduleLeft?: number;
  moduleTop?: number;
  moduleWidth?: number;
  moduleHeight?: number;
}

export interface FeedbackReadiness {
  total: number;
  positives: number;
  negatives: number;
  images: number;
  positiveImages: number;
  negativeImages: number;
  annotatedRegions: number;
  ready: boolean;
  missing: string[];
}

export function getFeedbackReadiness(items: FeedbackReadinessItem[]): FeedbackReadiness {
  const positives = items.filter((item) => item.label === 1);
  const negatives = items.filter((item) => item.label === 0);
  const images = new Set(items.map((item) => item.imageFile));
  const positiveImages = new Set(positives.map((item) => item.imageFile));
  const negativeImages = new Set(negatives.map((item) => item.imageFile));
  const regions = new Set<string>();
  for (const item of items) {
    const values = [item.moduleLeft, item.moduleTop, item.moduleWidth, item.moduleHeight];
    if (values.every((value) => Number.isFinite(value))) {
      // Quantisation tolerates small floating-point differences from the canvas transform.
      const key = `${item.imageFile}:${values.map((value) => Math.round((value as number) / 8)).join(':')}`;
      regions.add(key);
    }
  }

  const missing: string[] = [];
  if (positives.length < TRAINING_REQUIREMENTS.positives) missing.push(`${TRAINING_REQUIREMENTS.positives - positives.length} more positive cell labels`);
  if (negatives.length < TRAINING_REQUIREMENTS.negatives) missing.push(`${TRAINING_REQUIREMENTS.negatives - negatives.length} more negative cell labels`);
  if (images.size < TRAINING_REQUIREMENTS.images) missing.push(`${TRAINING_REQUIREMENTS.images - images.size} more distinct images`);
  if (positiveImages.size < TRAINING_REQUIREMENTS.positiveImages) missing.push(`positive labels across ${TRAINING_REQUIREMENTS.positiveImages - positiveImages.size} more images`);
  if (negativeImages.size < TRAINING_REQUIREMENTS.negativeImages) missing.push(`negative labels across ${TRAINING_REQUIREMENTS.negativeImages - negativeImages.size} more images`);
  if (regions.size < TRAINING_REQUIREMENTS.annotatedRegions) missing.push(`${TRAINING_REQUIREMENTS.annotatedRegions - regions.size} more annotated PV regions`);

  return {
    total: items.length,
    positives: positives.length,
    negatives: negatives.length,
    images: images.size,
    positiveImages: positiveImages.size,
    negativeImages: negativeImages.size,
    annotatedRegions: regions.size,
    ready: missing.length === 0,
    missing,
  };
}

export interface ImageSampleGroup<T extends { y: 0 | 1 }> {
  imageFile: string;
  samples: T[];
}

function stableHash(value: string): number {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

/**
 * Deterministic holdout split by whole image, never by individual cells. It preserves at
 * least one training image for each class and puts at least one image of each class in holdout.
 */
export function splitByImage<T extends { y: 0 | 1 }>(
  groups: ImageSampleGroup<T>[],
  validationFraction = 0.25
): { train: T[]; validation: T[]; trainImages: string[]; validationImages: string[] } {
  const ordered = [...groups].sort((a, b) => stableHash(a.imageFile) - stableHash(b.imageFile));
  if (ordered.length < 4) throw new Error('At least four labelled images are required for a separate image-level validation set; cell crops from one image are not independent validation samples.');
  const target = Math.min(ordered.length - 2, Math.max(2, Math.ceil(ordered.length * validationFraction)));
  const validationIds = new Set<string>();
  for (const label of [1, 0] as const) {
    const candidates = ordered.filter((group) => group.samples.some((sample) => sample.y === label));
    if (!candidates.length) throw new Error(`No ${label === 1 ? 'positive' : 'negative'} samples are available for image-level validation.`);
    if (!candidates.some((group) => validationIds.has(group.imageFile))) validationIds.add(candidates[0].imageFile);
  }
  for (const group of ordered) {
    if (validationIds.size >= target) break;
    const remaining = ordered.filter((candidate) => !validationIds.has(candidate.imageFile));
    const keepsTrainingPositive = remaining.some((candidate) => candidate.samples.some((sample) => sample.y === 1) && candidate.imageFile !== group.imageFile);
    const keepsTrainingNegative = remaining.some((candidate) => candidate.samples.some((sample) => sample.y === 0) && candidate.imageFile !== group.imageFile);
    if (keepsTrainingPositive && keepsTrainingNegative) validationIds.add(group.imageFile);
  }
  const trainGroups = groups.filter((group) => !validationIds.has(group.imageFile));
  const validationGroups = groups.filter((group) => validationIds.has(group.imageFile));
  const flatten = (selected: ImageSampleGroup<T>[]) => selected.flatMap((group) => group.samples);
  const train = flatten(trainGroups);
  const validation = flatten(validationGroups);
  if (!train.some((sample) => sample.y === 1) || !train.some((sample) => sample.y === 0) ||
      !validation.some((sample) => sample.y === 1) || !validation.some((sample) => sample.y === 0)) {
    throw new Error('The image-level split could not preserve both classes in training and validation. Add labels from more distinct images.');
  }
  return {
    train,
    validation,
    trainImages: trainGroups.map((group) => group.imageFile),
    validationImages: validationGroups.map((group) => group.imageFile),
  };
}
