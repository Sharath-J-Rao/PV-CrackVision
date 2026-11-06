/** Fine-tunes the crack classifier from reviewed cell labels with an image-level holdout. */
import fs from 'node:fs';
import sharp from 'sharp';
import { runClassicalStage1 } from './cvPipeline.ts';
import {
  USER_MODEL_PATH, baseCrackModel, collectWindows, cropCanonical, loadCrackModel, modelProb, relStats,
  reloadCrackModel, sigmoid, windowFeaturesAt, windowVector,
  type Canon, type CrackModel,
} from './crackModel.ts';
import { feedbackImagePath, listFeedback, type FeedbackItem } from './feedbackStore.ts';
import { getFeedbackReadiness, selectLabelCandidates, shouldAcceptCandidate, splitByImage } from './trainingValidation.ts';

interface Sample { x: number[]; y: 0 | 1; w: number; imageFile: string }
interface ValidationMetrics {
  precision: number;
  recall: number;
  specificity: number;
  balancedAccuracy: number;
  logLoss: number;
  positives: number;
  negatives: number;
}

export interface TrainReport {
  ok: true;
  accepted: boolean;
  reason?: string;
  feedbackUsed: number;
  feedbackSkipped: number;
  samples: number;
  samplePositives: number;
  sampleNegatives: number;
  validationImages: number;
  validationSamples: number;
  addedRecallBefore: number;
  addedRecallAfter: number;
  rejectedClearedBefore: number;
  rejectedClearedAfter: number;
  validationBefore: ValidationMetrics;
  validationAfter: ValidationMetrics;
  version: string;
}

const USER_W = 3;
const JITTER = [[0, 0], [-0.5, 0], [0.5, 0], [0, -0.5], [0, 0.5]];
const LAMBDA = 0.5;
const ITERS = 600;
const LR = 0.1;

function evaluate(model: CrackModel, samples: Sample[]): ValidationMetrics {
  let tp = 0, fp = 0, tn = 0, fn = 0, loss = 0;
  for (const sample of samples) {
    const probability = Math.max(1e-7, Math.min(1 - 1e-7, modelProb(model, sample.x)));
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
    positives,
    negatives,
  };
}

export async function trainFromFeedback(): Promise<TrainReport> {
  const base = baseCrackModel();
  if (!base) throw new Error('Base model (server/models/crackModel.json) is missing; nothing to fine-tune.');
  const cur = loadCrackModel() ?? base;
  const D = base.window, nF = base.featureNames.length;
  const items = listFeedback().filter((item) => item.category === 'spider_web_crack');
  const readiness = getFeedbackReadiness(items);
  if (!readiness.ready) {
    throw new Error(`Not enough diverse cell labels to train safely. ${readiness.missing.join('; ')}. Required: 30 positive cells, 50 negative cells, 10 images, at least 3 positive-label images, 3 negative-label images, and 5 annotated PV regions.`);
  }

  const byImage = new Map<string, FeedbackItem[]>();
  for (const item of items) byImage.set(item.imageFile, [...(byImage.get(item.imageFile) ?? []), item]);

  const samplesByImage: Array<{ imageFile: string; samples: Sample[] }> = [];
  let skipped = 0;
  let usedItems = 0;
  for (const [imageFile, group] of byImage) {
    let data: Buffer, info: { width: number; height: number };
    try {
      ({ data, info } = await sharp(fs.readFileSync(feedbackImagePath(group[0]))).rotate().toColorspace('srgb').removeAlpha().raw().toBuffer({ resolveWithObject: true }));
    } catch {
      skipped += group.length;
      continue;
    }

    const stage1 = await runClassicalStage1(data, info.width, info.height);
    const autoMods = stage1.moduleRegions
      .filter((region) => !region.isFallback && region.pixelBox.width >= 180 && region.pixelBox.height >= 110)
      .map((region) => ({ id: region.id, pixelBox: region.pixelBox }));
    const manualMods = group
      .filter((item) => [item.moduleLeft, item.moduleTop, item.moduleWidth, item.moduleHeight].every(Number.isFinite))
      .map((item, index) => ({
        id: `manual-${index}`,
        pixelBox: { left: item.moduleLeft!, top: item.moduleTop!, width: item.moduleWidth!, height: item.moduleHeight! },
      }));
    const uniqueManual = [...new Map(manualMods.map((region) => [
      `${region.pixelBox.left}:${region.pixelBox.top}:${region.pixelBox.width}:${region.pixelBox.height}`, region,
    ])).values()];
    // A manual PV-region annotation is authoritative; automatic regions are only a fallback.
    const regions = [...uniqueManual, ...autoMods].filter((region) => region.pixelBox.width >= 120 && region.pixelBox.height >= 80);
    const windows = await collectWindows(data, info.width, info.height, regions.map((region) => region.pixelBox), base);
    const relativeStats = relStats(windows, nF);
    const crops = new Map<string, Canon | null>();
    const imageSamples: Sample[] = [];
    let imageUsed = 0;

    for (const item of group) {
      const cx = item.left + item.width / 2, cy = item.top + item.height / 2;
      const region = regions.find((candidate) =>
        cx >= candidate.pixelBox.left && cx <= candidate.pixelBox.left + candidate.pixelBox.width &&
        cy >= candidate.pixelBox.top && cy <= candidate.pixelBox.top + candidate.pixelBox.height
      );
      if (!region) { skipped++; continue; }
      if (!crops.has(region.id)) crops.set(region.id, await cropCanonical(data, info.width, info.height, region.pixelBox));
      const crop = crops.get(region.id);
      if (!crop) { skipped++; continue; }

      // A marked cell supplies a 3x3 candidate set. For BOTH labels, mine the highest-scoring
      // windows: positive marks add missed-crack examples, while negative marks target false alarms.
      const bx0 = Math.max(0, Math.min(crop.cw - 1, (item.left - crop.left) * crop.scale));
      const by0 = Math.max(0, Math.min(crop.ch - 1, (item.top - crop.top) * crop.scale));
      const bx1 = Math.max(bx0 + 1, Math.min(crop.cw, (item.left + item.width - crop.left) * crop.scale));
      const by1 = Math.max(by0 + 1, Math.min(crop.ch, (item.top + item.height - crop.top) * crop.scale));
      const candidates = new Map<string, { x: number; y: number; vec: number[]; score: number }>();
      for (const fy of [0.2, 0.5, 0.8]) {
        for (const fx of [0.2, 0.5, 0.8]) {
          const px = bx0 + (bx1 - bx0) * fx;
          const py = by0 + (by1 - by0) * fy;
          const x = Math.max(0, Math.min(crop.cw - D, Math.round(px - D / 2)));
          const y = Math.max(0, Math.min(crop.ch - D, Math.round(py - D / 2)));
          const key = `${x}:${y}`;
          if (candidates.has(key)) continue;
          const features = windowFeaturesAt(crop, x, y, base);
          const vector = windowVector(features, relativeStats);
          const score = modelProb(cur, vector) + 0.015 * features[6] + 0.0002 * features[11];
          candidates.set(key, { x, y, vec: vector, score });
        }
      }
      const selected = selectLabelCandidates([...candidates.values()], 2);
      if (!selected.length) { skipped++; continue; }
      for (const candidate of selected) {
        for (const [jx, jy] of JITTER) {
          const x = Math.max(0, Math.min(crop.cw - D, candidate.x + Math.round(jx * base.stride)));
          const y = Math.max(0, Math.min(crop.ch - D, candidate.y + Math.round(jy * base.stride)));
          const vector = windowVector(windowFeaturesAt(crop, x, y, base), relativeStats);
          imageSamples.push({ x: vector, y: item.label, w: USER_W, imageFile });
        }
      }
      imageUsed++;
    }
    if (imageSamples.length) samplesByImage.push({ imageFile, samples: imageSamples });
    usedItems += imageUsed;
  }

  const allSamples = samplesByImage.flatMap((group) => group.samples);
  const usablePositiveImages = samplesByImage.filter((group) => group.samples.some((sample) => sample.y === 1)).length;
  const usableNegativeImages = samplesByImage.filter((group) => group.samples.some((sample) => sample.y === 0)).length;
  if (samplesByImage.length < 10 || usablePositiveImages < 3 || usableNegativeImages < 3 ||
      !allSamples.some((sample) => sample.y === 1) || !allSamples.some((sample) => sample.y === 0)) {
    throw new Error(`Only ${samplesByImage.length} images mapped to valid PV-cell windows (${usablePositiveImages} positive-label images, ${usableNegativeImages} negative-label images). At least 10 usable images and 3 images per class are required; correct the cell/PV-region labels or add more images.`);
  }
  const split = splitByImage(samplesByImage, 0.25);
  const trainSamples = split.train;
  const validationSamples = split.validation;
  const pos = trainSamples.filter((sample) => sample.y === 1).length;
  const neg = trainSamples.filter((sample) => sample.y === 0).length;
  if (pos < 10 || neg < 10) throw new Error('Too few mapped training windows remain after the image-level holdout. Add more reviewed cells from additional images.');

  // Logistic regression with balanced class weights and an L2 prior to the current model.
  const X = trainSamples.map((sample) => sample.x.map((value, index) => (value - base.mean[index]) / base.scale[index]));
  const wPos = pos ? trainSamples.length / (2 * pos) : 1;
  const wNeg = neg ? trainSamples.length / (2 * neg) : 1;
  const weights = trainSamples.map((sample) => sample.w * (sample.y === 1 ? wPos : wNeg));
  const totalWeight = weights.reduce((sum, value) => sum + value, 0);
  const w0 = cur.coef.slice(), b0 = cur.intercept;
  let coefficients = w0.slice(), intercept = b0;
  for (let iteration = 0; iteration < ITERS; iteration++) {
    const gradient = new Array(coefficients.length).fill(0);
    let biasGradient = 0;
    for (let index = 0; index < X.length; index++) {
      let z = intercept;
      for (let feature = 0; feature < coefficients.length; feature++) z += X[index][feature] * coefficients[feature];
      const error = ((sigmoid(z) - trainSamples[index].y) * weights[index]) / totalWeight;
      for (let feature = 0; feature < coefficients.length; feature++) gradient[feature] += error * X[index][feature];
      biasGradient += error;
    }
    for (let feature = 0; feature < coefficients.length; feature++) {
      coefficients[feature] -= LR * (gradient[feature] + LAMBDA * (coefficients[feature] - w0[feature]) / coefficients.length);
    }
    intercept -= LR * biasGradient;
  }

  const tuned: CrackModel = { ...cur, coef: coefficients, intercept };
  const before = evaluate(cur, validationSamples);
  const after = evaluate(tuned, validationSamples);
  // Reject a candidate that materially harms either class on unseen images. Require a real
  // improvement in balanced accuracy or log loss before persisting any user model.
  const accepted = shouldAcceptCandidate(before, after);
  const version = accepted ? `${base.version}+feedback-${items.length}` : cur.version;
  if (accepted) {
    const temporaryModelPath = `${USER_MODEL_PATH}.tmp`;
    fs.writeFileSync(temporaryModelPath, JSON.stringify({
      ...tuned,
      version,
      userTuned: {
        feedbackItems: items.length,
        samples: trainSamples.length,
        tunedAt: new Date().toISOString(),
        basedOn: base.version,
        validation: after,
        validationBefore: before,
        validationImages: split.validationImages.length,
        trainingImages: split.trainImages.length,
      },
    }, null, 1));
    fs.renameSync(temporaryModelPath, USER_MODEL_PATH);
    reloadCrackModel();
  }

  return {
    ok: true,
    accepted,
    ...(accepted ? {} : { reason: 'The candidate did not improve held-out image performance without materially reducing positive recall or negative specificity. No model file was changed.' }),
    feedbackUsed: usedItems,
    feedbackSkipped: skipped,
    samples: trainSamples.length,
    samplePositives: pos,
    sampleNegatives: neg,
    validationImages: split.validationImages.length,
    validationSamples: validationSamples.length,
    addedRecallBefore: before.recall,
    addedRecallAfter: after.recall,
    rejectedClearedBefore: before.specificity,
    rejectedClearedAfter: after.specificity,
    validationBefore: before,
    validationAfter: after,
    version,
  };
}

export function resetUserModel(): void {
  fs.rmSync(USER_MODEL_PATH, { force: true });
  reloadCrackModel();
}
