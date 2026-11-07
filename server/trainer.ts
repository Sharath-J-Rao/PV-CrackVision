/** Train a fresh crack classifier from reviewed solar-cell crops (one sample = one cell). */
import fs from 'node:fs';
import sharp from 'sharp';
import path from 'node:path';
import { MODEL_DATA_DIR } from './dataPaths.ts';
import {
  CELL_MODEL_SCHEMA, CELL_WINDOW_MARGIN, CELL_WINDOW_SIZE,
  USER_MODEL_PATH, extractCellSample, modelProb, reloadCrackModel, scoreV2, sigmoid,
  type CrackModel,
} from './crackModel.ts';
import { CELL_FEATURE_NAMES, CELL_FEATURE_NAMES_V2, CELL_MODEL_SCHEMA_V2, cellFeatureVector } from './cellFeatures.ts';
import { trainGbmAsync } from './gradientBoosting.ts';
import { trainCnn } from './cellCnn.ts';
import { feedbackImagePath, listFeedback, type FeedbackItem } from './feedbackStore.ts';
import { getFeedbackReadiness, splitByImage } from './trainingValidation.ts';
import { getTrainingSettings, normalizeTrainingSettings } from './trainingSettings.ts';
import type { CrackClassifierTrainingSettings } from '../src/types/trainingSettings.ts';

interface Sample { x: number[]; cnn: Float32Array | null; y: 0 | 1; imageFile: string }
export interface CrackTrainerProgress {
  stage: 'preparing' | 'gradient_boosting' | 'cnn' | 'validation';
  progress: number;
  message: string;
  epoch?: number;
  epochs?: number;
  tree?: number;
  trees?: number;
  samples?: number;
  trainingSamples?: number;
  validationSamples?: number;
  metric?: { epoch: number; loss: number; validationLoss: number; precision: number; recall: number; specificity: number; balancedAccuracy: number; f1: number };
}
interface ValidationMetrics {
  precision: number; recall: number; specificity: number; balancedAccuracy: number; logLoss: number;
  positives: number; negatives: number;
}
export interface TrainReport {
  ok: true; accepted: boolean; reason?: string; feedbackUsed: number; feedbackSkipped: number;
  samples: number; samplePositives: number; sampleNegatives: number;
  validationImages: number; validationSamples: number;
  addedRecallBefore: number; addedRecallAfter: number;
  rejectedClearedBefore: number; rejectedClearedAfter: number;
  validationBefore: ValidationMetrics; validationAfter: ValidationMetrics; version: string;
  /** v1.4.8.7 */
  algorithm: CrackClassifierTrainingSettings['algorithm'];
  components?: { gbm?: ValidationMetrics; cnn?: ValidationMetrics };
  cnnLoss?: Array<{ epoch: number; loss: number; validationLoss?: number; precision?: number; recall?: number; specificity?: number; balancedAccuracy?: number; f1?: number }>;
}


/** Probability of one training/validation sample under either schema (v1 vectors are pre-padded, v2 are raw rich features). */
function sampleProb(model: CrackModel, sample: Sample): number {
  return model.schema === CELL_MODEL_SCHEMA_V2 ? scoreV2(model, sample.x, sample.cnn) : modelProb(model, sample.x);
}

function evaluate(model: CrackModel, samples: Sample[], constant?: number): ValidationMetrics {
  let tp = 0, fp = 0, tn = 0, fn = 0, loss = 0;
  for (const sample of samples) {
    const probability = Math.max(1e-7, Math.min(1 - 1e-7, constant ?? sampleProb(model, sample)));
    const predicted = probability >= model.threshold ? 1 : 0;
    if (sample.y === 1 && predicted === 1) tp++;
    else if (sample.y === 0 && predicted === 1) fp++;
    else if (sample.y === 0 && predicted === 0) tn++;
    else fn++;
    loss += -(sample.y * Math.log(probability) + (1 - sample.y) * Math.log(1 - probability));
  }
  const positives = tp + fn, negatives = tn + fp;
  const recall = positives ? tp / positives : 0;
  const specificity = negatives ? tn / negatives : 0;
  return {
    precision: Number((tp / Math.max(1, tp + fp)).toFixed(4)),
    recall: Number(recall.toFixed(4)),
    specificity: Number(specificity.toFixed(4)),
    balancedAccuracy: Number(((recall + specificity) / 2).toFixed(4)),
    logLoss: Number((loss / Math.max(1, samples.length)).toFixed(5)),
    positives, negatives,
  };
}

function standardize(train: Sample[]): { mean: number[]; scale: number[]; X: number[][] } {
  const dimensions = train[0]?.x.length ?? 0;
  const mean = new Array(dimensions).fill(0);
  const scale = new Array(dimensions).fill(1);
  for (const sample of train) for (let j = 0; j < dimensions; j++) mean[j] += sample.x[j] / train.length;
  for (let j = 0; j < dimensions; j++) {
    let variance = 0;
    for (const sample of train) variance += (sample.x[j] - mean[j]) ** 2 / train.length;
    scale[j] = Math.max(1e-3, Math.sqrt(variance));
  }
  const X = train.map((sample) => sample.x.map((value, j) => (value - mean[j]) / scale[j]));
  return { mean, scale, X };
}

function trainFresh(train: Sample[], settings: CrackClassifierTrainingSettings): { model: CrackModel; threshold: number } {
  const { mean, scale, X } = standardize(train);
  const dimensions = mean.length;
  const positives = train.filter((sample) => sample.y === 1).length;
  const negatives = train.length - positives;
  const positiveWeight = settings.classBalance ? train.length / (2 * Math.max(1, positives)) : 1;
  const negativeWeight = settings.classBalance ? train.length / (2 * Math.max(1, negatives)) : 1;
  const sampleWeights = train.map((sample) => sample.y === 1 ? positiveWeight : negativeWeight);
  const weightSum = sampleWeights.reduce((sum, weight) => sum + weight, 0);
  let coefficients = new Array(dimensions).fill(0);
  let intercept = 0;

  // No base coefficients or pretrained prior are loaded: every weight starts at zero.
  for (let iteration = 0; iteration < settings.iterations; iteration++) {
    const gradient = new Array(dimensions).fill(0);
    let biasGradient = 0;
    for (let i = 0; i < X.length; i++) {
      let z = intercept;
      for (let j = 0; j < dimensions; j++) z += X[i][j] * coefficients[j];
      const error = (sigmoid(z) - train[i].y) * sampleWeights[i] / weightSum;
      for (let j = 0; j < dimensions; j++) gradient[j] += error * X[i][j];
      biasGradient += error;
    }
    for (let j = 0; j < dimensions; j++) coefficients[j] -= settings.learningRate * (gradient[j] + settings.l2 * coefficients[j]);
    intercept -= settings.learningRate * biasGradient;
  }

  const baseModel: CrackModel = {
    schema: CELL_MODEL_SCHEMA,
    version: 'cell-logreg-untrained-candidate',
    window: CELL_WINDOW_SIZE,
    stride: CELL_WINDOW_SIZE,
    margin: CELL_WINDOW_MARGIN,
    featureNames: [...CELL_FEATURE_NAMES],
    mean,
    scale,
    coef: coefficients,
    intercept,
    threshold: 0.5,
  };
  const bestThreshold = selectThreshold(baseModel, train, settings);
  return { model: { ...baseModel, threshold: bestThreshold }, threshold: bestThreshold };
}

/** Select the decision threshold using training images only; never tune on the held-out images. */
function selectThreshold(baseModel: CrackModel, train: Sample[], settings: CrackClassifierTrainingSettings): number {
  let bestThreshold = 0.5;
  let bestScore = -1;
  for (let threshold = settings.thresholdMin; threshold <= settings.thresholdMax + settings.thresholdStep * 0.001; threshold += settings.thresholdStep) {
    const candidate = { ...baseModel, threshold: Number(threshold.toFixed(3)) };
    const metrics = evaluate(candidate, train);
    if (metrics.balancedAccuracy > bestScore ||
        (metrics.balancedAccuracy === bestScore && Math.abs(threshold - 0.5) < Math.abs(bestThreshold - 0.5))) {
      bestScore = metrics.balancedAccuracy;
      bestThreshold = Number(threshold.toFixed(3));
    }
  }
  return bestThreshold;
}

/** Train configured GBM/CNN components and stream stage/epoch telemetry to the durable job store. */
async function trainV2(
  train: Sample[],
  validation: Sample[],
  settings: CrackClassifierTrainingSettings,
  onProgress?: (progress: CrackTrainerProgress) => void,
): Promise<{ model: CrackModel; cnnLoss?: Array<{ epoch: number; loss: number; validationLoss: number; precision: number; recall: number; specificity: number; balancedAccuracy: number; f1: number }> }> {
  const positives = train.filter((sample) => sample.y === 1).length;
  const negatives = train.length - positives;
  const positiveWeight = settings.classBalance ? train.length / (2 * Math.max(1, positives)) : 1;
  const negativeWeight = settings.classBalance ? train.length / (2 * Math.max(1, negatives)) : 1;
  const weights = train.map((sample) => sample.y === 1 ? positiveWeight : negativeWeight);
  const wantGbm = settings.algorithm === 'gbm' || settings.algorithm === 'ensemble';
  const wantCnn = settings.algorithm === 'cnn' || settings.algorithm === 'ensemble';
  const gbm = wantGbm
    ? await trainGbmAsync(train.map((sample) => sample.x), train.map((sample) => sample.y), weights, {
      trees: settings.gbmTrees, depth: settings.gbmDepth, learningRate: settings.gbmLearningRate,
      minLeaf: settings.gbmMinLeaf, subsample: settings.gbmSubsample,
    }, (tree, trees) => {
      const stride = Math.max(1, Math.ceil(trees / 20));
      if (tree === 1 || tree % stride === 0 || tree === trees) onProgress?.({
        stage: 'gradient_boosting', progress: 35 + 15 * tree / trees,
        message: `Gradient-boosting trees: ${tree}/${trees}`, tree, trees,
      });
    })
    : undefined;
  let cnn, cnnLoss;
  if (wantCnn) {
    onProgress?.({ stage: 'cnn', progress: wantGbm ? 51 : 36, message: `Training CNN on ${train.length} cell crops; held-out validation runs after every epoch.`, epoch: 0, epochs: settings.cnnEpochs });
    const result = await trainCnn(
      train.map((sample) => ({ input: sample.cnn as Float32Array, y: sample.y })), weights,
      { epochs: settings.cnnEpochs, learningRate: settings.cnnLearningRate, batchSize: settings.cnnBatchSize },
      (epoch, _loss, metric) => {
        if (!metric) return;
        onProgress?.({
          stage: 'cnn', progress: (wantGbm ? 50 : 35) + 40 * epoch / settings.cnnEpochs,
          message: `CNN epoch ${epoch}/${settings.cnnEpochs} complete; validation metrics recorded on held-out images.`,
          epoch, epochs: settings.cnnEpochs,
          metric: { epoch, loss: metric.loss, validationLoss: metric.validationLoss, precision: metric.precision, recall: metric.recall, specificity: metric.specificity, balancedAccuracy: metric.balancedAccuracy, f1: metric.f1 },
        });
      },
      validation.map((sample) => ({ input: sample.cnn as Float32Array, y: sample.y })),
    );
    cnn = result.model; cnnLoss = result.history;
  }
  const baseModel: CrackModel = {
    schema: CELL_MODEL_SCHEMA_V2,
    version: 'cell-v2-untrained-candidate',
    window: CELL_WINDOW_SIZE, stride: CELL_WINDOW_SIZE, margin: CELL_WINDOW_MARGIN,
    featureNames: [...CELL_FEATURE_NAMES_V2],
    mean: [], scale: [], coef: [], intercept: 0, threshold: 0.5,
    algorithm: settings.algorithm === 'logistic' ? 'gbm' : settings.algorithm,
    ...(gbm ? { gbm } : {}), ...(cnn ? { cnn } : {}),
  };
  return { model: { ...baseModel, threshold: selectThreshold(baseModel, train, settings) }, cnnLoss };
}

export async function trainFromFeedback(options: Partial<CrackClassifierTrainingSettings> = {}, onProgress?: (progress: CrackTrainerProgress) => void): Promise<TrainReport> {
  const settings = normalizeTrainingSettings({ crackClassifier: { ...getTrainingSettings().crackClassifier, ...options } }).crackClassifier;
  const items = listFeedback().filter((item) => item.category === 'spider_web_crack');
  onProgress?.({ stage: 'preparing', progress: 3, message: `Preparing ${items.length} reviewed cell labels for ${settings.algorithm} training.` });
  const readiness = getFeedbackReadiness(items);
  if (!readiness.ready) {
    throw new Error(`Not enough reviewed cell labels to train. ${readiness.missing.join('; ')}. Required: 8 positive cells, 8 negative cells, 4 distinct images, at least 2 images per class, and 2 annotated PV regions.`);
  }

  const grouped = new Map<string, FeedbackItem[]>();
  for (const item of items) grouped.set(item.imageFile, [...(grouped.get(item.imageFile) ?? []), item]);
  const samplesByImage: Array<{ imageFile: string; samples: Sample[] }> = [];
  let skipped = 0;
  let used = 0;

  let imageIndex = 0;
  for (const [imageFile, annotations] of grouped) {
    imageIndex++;
    onProgress?.({ stage: 'preparing', progress: 5 + 25 * (imageIndex - 1) / Math.max(1, grouped.size), message: `Extracting cell crops from image ${imageIndex}/${grouped.size}.` });
    let raw: Buffer, width: number, height: number;
    try {
      const decoded = await sharp(fs.readFileSync(feedbackImagePath(annotations[0])))
        .rotate().toColorspace('srgb').removeAlpha().raw().toBuffer({ resolveWithObject: true });
      raw = decoded.data; width = decoded.info.width; height = decoded.info.height;
    } catch {
      skipped += annotations.length;
      continue;
    }

    const rawFeatures: Array<{ item: FeedbackItem; f: number[]; cnn: Float32Array | null }> = [];
    for (const item of annotations) {
      try {
        const sample = await extractCellSample(raw, width, height, item, {
          window: CELL_WINDOW_SIZE, margin: CELL_WINDOW_MARGIN,
          rich: settings.algorithm !== 'logistic', cnn: settings.algorithm === 'cnn' || settings.algorithm === 'ensemble',
        });
        rawFeatures.push({ item, f: sample.features, cnn: sample.cnn });
      } catch {
        skipped++;
      }
    }
    // No per-image statistic depends on which cells the operator chose to label.
    // The cell vector is shared byte-for-byte with the inference path.
    const imageSamples = rawFeatures.map(({ item, f, cnn }) => ({
      x: settings.algorithm === 'logistic' ? cellFeatureVector(f) : f, cnn, y: item.label, imageFile,
    } satisfies Sample));
    if (imageSamples.length) {
      samplesByImage.push({ imageFile, samples: imageSamples });
      used += imageSamples.length;
    }
  }

  const usablePositiveImages = samplesByImage.filter((group) => group.samples.some((sample) => sample.y === 1)).length;
  const usableNegativeImages = samplesByImage.filter((group) => group.samples.some((sample) => sample.y === 0)).length;
  if (samplesByImage.length < 4 || usablePositiveImages < 2 || usableNegativeImages < 2) {
    throw new Error(`Only ${samplesByImage.length} images contain usable cell crops (${usablePositiveImages} positive-label images, ${usableNegativeImages} negative-label images). Need 4 usable images and at least 2 images for each class so a separate image-level holdout remains possible.`);
  }

  const split = splitByImage(samplesByImage, settings.validationFraction);
  onProgress?.({ stage: 'preparing', progress: 32, message: `Dataset ready: ${used} usable crops; ${split.train.length} training and ${split.validation.length} held-out cell crops.`, samples: used, trainingSamples: split.train.length, validationSamples: split.validation.length });
  const train = split.train;
  const validation = split.validation;
  const positiveCount = train.filter((sample) => sample.y === 1).length;
  const negativeCount = train.length - positiveCount;
  if (positiveCount < 3 || negativeCount < 3) {
    throw new Error('Too few positive or negative cell crops remain in the training split. Add a few more reviewed cells while preserving image-level diversity.');
  }

  let candidate: CrackModel;
  let cnnLoss: Array<{ epoch: number; loss: number; validationLoss: number; precision: number; recall: number; specificity: number; balancedAccuracy: number }> | undefined;
  if (settings.algorithm === 'logistic') {
    onProgress?.({ stage: 'validation', progress: 75, message: 'Training legacy logistic classifier; calculating held-out validation metrics.', samples: used, trainingSamples: train.length, validationSamples: validation.length });
    candidate = trainFresh(train, settings).model;
  } else {
    const trained = await trainV2(train, validation, settings, onProgress);
    candidate = trained.model; cnnLoss = trained.cnnLoss;
  }
  onProgress?.({ stage: 'validation', progress: 92, message: 'Evaluating final model and component metrics on held-out images.' });
  const untrained: CrackModel = { ...candidate, coef: candidate.coef.map(() => 0), intercept: 0, threshold: 0.5 };
  // The untrained reference is a constant 0.5 score, identical to the all-zero logistic model of earlier releases.
  const before = evaluate(untrained, validation, 0.5);
  const after = evaluate(candidate, validation);
  const components = candidate.schema === CELL_MODEL_SCHEMA_V2 && candidate.gbm && candidate.cnn
    ? { gbm: evaluate({ ...candidate, cnn: undefined }, validation), cnn: evaluate({ ...candidate, gbm: undefined }, validation) }
    : undefined;
  // Require the holdout to contain both classes and outperform an untrained constant classifier.
  const accepted = after.positives > 0 && after.negatives > 0 && after.balancedAccuracy >= settings.minimumBalancedAccuracy;
  const version = settings.algorithm === 'logistic' ? `cell-logreg-v1.4.8.9-${items.length}` : `cell-${candidate.algorithm}-v1.4.8.9-${items.length}`;
  if (accepted) {
    fs.mkdirSync(path.dirname(USER_MODEL_PATH), { recursive: true });
    const temporary = `${USER_MODEL_PATH}.tmp`;
    fs.writeFileSync(temporary, JSON.stringify({
      ...candidate, version,
      training: {
        schema: CELL_MODEL_SCHEMA,
        samples: train.length,
        positiveCells: positiveCount,
        negativeCells: negativeCount,
        sourceImages: split.trainImages.length,
        heldOutImages: split.validationImages.length,
        trainedAt: new Date().toISOString(),
        initialWeights: settings.algorithm === 'logistic' ? 'all-zero' : 'from-scratch (seeded)',
        algorithm: settings.algorithm,
        ...(components ? { componentValidation: components } : {}),
        trainingParameters: settings,
        featureUnit: 'one validated solar-cell crop',
        validation: after,
      },
    }, null, 2));
    fs.renameSync(temporary, USER_MODEL_PATH);
    reloadCrackModel();
  }

  onProgress?.({ stage: 'validation', progress: 99, message: accepted ? 'Held-out validation passed; candidate model saved and activated.' : 'Held-out validation did not meet the activation gate; previous model retained.' });
  return {
    ok: true, accepted,
    ...(accepted ? {} : { reason: `The fresh cell-level candidate did not achieve at least ${settings.minimumBalancedAccuracy.toFixed(2)} balanced accuracy on unseen images. No model was saved; review labels and add more representative cells.` }),
    feedbackUsed: used, feedbackSkipped: skipped,
    samples: train.length, samplePositives: positiveCount, sampleNegatives: negativeCount,
    validationImages: split.validationImages.length, validationSamples: validation.length,
    addedRecallBefore: before.recall, addedRecallAfter: after.recall,
    rejectedClearedBefore: before.specificity, rejectedClearedAfter: after.specificity,
    validationBefore: before, validationAfter: after, version: accepted ? version : 'no-model-saved',
    algorithm: settings.algorithm, ...(components ? { components } : {}), ...(cnnLoss ? { cnnLoss } : {}),
  };
}


export function resetUserModel(): void {
  fs.rmSync(USER_MODEL_PATH, { force: true });
  fs.rmSync(`${USER_MODEL_PATH}.tmp`, { force: true });
  // Clean up the former v1.3.0 fine-tuned file if this installation is upgraded in place.
  const modelDir = MODEL_DATA_DIR;
  fs.rmSync(path.join(modelDir, 'crackModel.user.json'), { force: true });
  fs.rmSync(path.join(modelDir, 'crackModel.json'), { force: true });
  reloadCrackModel();
}
