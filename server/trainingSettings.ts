import fs from 'node:fs';
import path from 'node:path';
import { DATA_TRAINING_DIR } from './dataPaths.ts';
import { CELL_IMAGE_SIZES } from '../src/types/trainingSettings.ts';
import type { TrainingSettings, CellDetectorTrainingSettings, CrackClassifierTrainingSettings, CellInferenceSettings, CellTypeFilterSettings, CellTypeProfile, CrackDetectionSettings } from '../src/types/trainingSettings.ts';

export const TRAINING_SETTINGS_FILE = path.join(DATA_TRAINING_DIR, 'training-settings.json');

export const DEFAULT_TRAINING_SETTINGS: TrainingSettings = {
  cellDetector: {
    epochs: 200, patience: 30, imageSize: 1280, batch: 4, freeze: 0, optimizer: 'AdamW',
    lr0: 0.001, lrf: 0.01, momentum: 0.9, weightDecay: 0.0005,
    warmupEpochs: 3, warmupMomentum: 0.8, warmupBiasLr: 0.1, cosLr: true,
    closeMosaic: 10, mosaic: 1, mixup: 0, degrees: 0, translate: 0.1, scale: 0.5,
    fliplr: 0.5, flipud: 0.5, hsvH: 0.015, hsvS: 0.7, hsvV: 0.4,
    boxLoss: 7.5, clsLoss: 0.5, dflLoss: 1.5, multiScale: 0,
    seed: 0, deterministic: true, cache: true, pretrained: true, savePeriod: 10, minimumMap50: 0.05, onnxGateTolerance: 0.02, missedCellRepeat: 2, partialMaskMargin: 0.15, // bootstrap-friendly: a rough first model must be activatable so it can pre-label images for correction
  },
  crackClassifier: {
    iterations: 1400, learningRate: 0.055, l2: 0.015, validationFraction: 0.25,
    thresholdMin: 0.2, thresholdMax: 0.8, thresholdStep: 0.025,
    minimumBalancedAccuracy: 0.55, classBalance: true, cellBuffer: 0.03,
    algorithm: 'ensemble', gbmTrees: 160, gbmDepth: 3, gbmLearningRate: 0.08, gbmMinLeaf: 3, gbmSubsample: 0.8,
    cnnEpochs: 20, cnnLearningRate: 0.004, cnnBatchSize: 24,
    cnnEarlyStopPatience: 8, edgeThresholdDrop: 0.1, edgeRequireAgreement: true, edgeSampleWeight: 2, hardNegativeWeight: 2, missedCrackWeight: 3,
  },
  // Defaults equal the values that were hard-coded before v1.4.7.0, so behaviour is unchanged until edited.
  cellInference: {
    // v1.4.12.7: the candidate/box caps were sized for CPU speed. At a 0.005 floor, duplicates of strong cells filled the old
    // 700-candidate cap before NMS, so genuine low-score cells in dense arrays were cut off.
    confidenceThreshold: 0.005, decoderNmsIou: 0.65, mergeNmsIou: 0.5, maxCandidatesBeforeNms: 2500,
    maxDecoderBoxes: 3000, maxFinalBoxes: 4000, enclosingSuppression: true,
    useTiles: true, fuseTileBoxes: true, tileScale: 1.5, tileOverlap: 0.22, maxTiles: 12,
  },
  // Default: accept only near-square full cells of the standard module type; everything else is rejected.
  // Use "Learn from my annotations" (Training Settings) to replace the default with the measured aspect of your boxes.
  cellTypeFilter: {
    enabled: true, relativeMin: 0.5, relativeMax: 1.8,
    minShortPx: 12, minAreaRelative: 0.65, minShortRelative: 0.7, minConfidence: 0.1, fragmentOverlap: 0.5,
    profiles: [
      { id: 'standard', name: 'Standard cell (default)', mode: 'accept', enabled: true, aspect: 1.0, aspectTolerance: 0.3, minLongPx: 0, maxLongPx: 0 },
      { id: 'half-cut', name: 'Half-cut cell (2:1)', mode: 'accept', enabled: false, aspect: 2.0, aspectTolerance: 0.25, minLongPx: 0, maxLongPx: 0 },
    ],
  },
  crackDetection: { groundMaxSoilFraction: 0.3, groundMinPvFraction: 0.5, groundMinPvFractionFallback: 0.25, decisionThreshold: 0 },
};

function number(value: unknown, fallback: number, min: number, max: number, integer = false): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  const bounded = Math.max(min, Math.min(max, parsed));
  return integer ? Math.round(bounded) : bounded;
}
function bool(value: unknown, fallback: boolean): boolean { return typeof value === 'boolean' ? value : fallback; }

function normalizeProfiles(input: unknown, fallback: CellTypeProfile[]): CellTypeProfile[] {
  if (!Array.isArray(input)) return structuredClone(fallback);
  const seen = new Set<string>();
  const profiles: CellTypeProfile[] = [];
  for (const raw of input.slice(0, 20)) { // bounded: a corrupted file must not create unbounded UI/filters
    if (!raw || typeof raw !== 'object') continue;
    const p = raw as Record<string, any>;
    let id = typeof p.id === 'string' && p.id.trim() ? p.id.trim().slice(0, 40) : `type-${profiles.length + 1}`;
    while (seen.has(id)) id = `${id}-x`;
    seen.add(id);
    profiles.push({
      id, name: typeof p.name === 'string' && p.name.trim() ? p.name.trim().slice(0, 60) : id,
      mode: p.mode === 'reject' ? 'reject' : 'accept', enabled: bool(p.enabled, true),
      aspect: number(p.aspect, 1, 1, 20), aspectTolerance: number(p.aspectTolerance, 0.3, 0.02, 3),
      minLongPx: number(p.minLongPx, 0, 0, 100000), maxLongPx: number(p.maxLongPx, 0, 0, 100000),
    });
  }
  return profiles;
}

export function normalizeTrainingSettings(input: unknown): TrainingSettings {
  const source = input && typeof input === 'object' ? input as Record<string, any> : {};
  const cell = source.cellDetector && typeof source.cellDetector === 'object' ? source.cellDetector : {};
  const crack = source.crackClassifier && typeof source.crackClassifier === 'object' ? source.crackClassifier : {};
  const d = DEFAULT_TRAINING_SETTINGS;
  const inf = source.cellInference && typeof source.cellInference === 'object' ? source.cellInference : {};
  const typ = source.cellTypeFilter && typeof source.cellTypeFilter === 'object' ? source.cellTypeFilter : {};
  const grd = source.crackDetection && typeof source.crackDetection === 'object' ? source.crackDetection : {};
  const cellInference: CellInferenceSettings = {
    confidenceThreshold: number(inf.confidenceThreshold, d.cellInference.confidenceThreshold, 0.0001, 0.95),
    decoderNmsIou: number(inf.decoderNmsIou, d.cellInference.decoderNmsIou, 0.05, 0.95),
    mergeNmsIou: number(inf.mergeNmsIou, d.cellInference.mergeNmsIou, 0.05, 0.95),
    maxCandidatesBeforeNms: number(inf.maxCandidatesBeforeNms, d.cellInference.maxCandidatesBeforeNms, 10, 5000, true),
    maxDecoderBoxes: number(inf.maxDecoderBoxes, d.cellInference.maxDecoderBoxes, 10, 5000, true),
    maxFinalBoxes: number(inf.maxFinalBoxes, d.cellInference.maxFinalBoxes, 10, 10000, true),
    enclosingSuppression: bool(inf.enclosingSuppression, d.cellInference.enclosingSuppression),
    useTiles: bool(inf.useTiles, d.cellInference.useTiles),
    fuseTileBoxes: bool(inf.fuseTileBoxes, d.cellInference.fuseTileBoxes),
    tileScale: number(inf.tileScale, d.cellInference.tileScale, 1, 6),
    tileOverlap: number(inf.tileOverlap, d.cellInference.tileOverlap, 0.1, 0.45),
    maxTiles: number(inf.maxTiles, d.cellInference.maxTiles, 1, 64, true),
  };
  const cellTypeFilter: CellTypeFilterSettings = {
    enabled: bool(typ.enabled, d.cellTypeFilter.enabled),
    relativeMin: number(typ.relativeMin, d.cellTypeFilter.relativeMin, 0, 1),
    relativeMax: number(typ.relativeMax, d.cellTypeFilter.relativeMax, 0, 20),
    minShortPx: number(typ.minShortPx, d.cellTypeFilter.minShortPx, 0, 100000),
    minAreaRelative: number(typ.minAreaRelative, d.cellTypeFilter.minAreaRelative, 0, 1),
    minShortRelative: number(typ.minShortRelative, d.cellTypeFilter.minShortRelative, 0, 1),
    minConfidence: number(typ.minConfidence, d.cellTypeFilter.minConfidence, 0, 0.95),
    fragmentOverlap: number(typ.fragmentOverlap, d.cellTypeFilter.fragmentOverlap, 0, 0.95),
    profiles: normalizeProfiles(typ.profiles, d.cellTypeFilter.profiles),
  };
  const crackDetection: CrackDetectionSettings = {
    groundMaxSoilFraction: number(grd.groundMaxSoilFraction, d.crackDetection.groundMaxSoilFraction, 0, 1),
    groundMinPvFraction: number(grd.groundMinPvFraction, d.crackDetection.groundMinPvFraction, 0, 1),
    groundMinPvFractionFallback: number(grd.groundMinPvFractionFallback, d.crackDetection.groundMinPvFractionFallback, 0, 1),
    decisionThreshold: number(grd.decisionThreshold, d.crackDetection.decisionThreshold, 0, 0.95), // 0 = trained model's own threshold
  };
  const optimizer = ['auto', 'SGD', 'Adam', 'AdamW', 'RMSProp'].includes(cell.optimizer) ? cell.optimizer : d.cellDetector.optimizer;
  const imageSize = (CELL_IMAGE_SIZES as readonly number[]).includes(Number(cell.imageSize)) ? Number(cell.imageSize) : d.cellDetector.imageSize;
  const thresholdMin = number(crack.thresholdMin, d.crackClassifier.thresholdMin, 0.05, 0.8);
  const thresholdMax = Math.max(thresholdMin + 0.025, number(crack.thresholdMax, d.crackClassifier.thresholdMax, 0.2, 0.95));
  return {
    cellDetector: {
      epochs: number(cell.epochs, d.cellDetector.epochs, 1, 500, true),
      patience: number(cell.patience, d.cellDetector.patience, 0, 500, true),
      imageSize, batch: number(cell.batch, d.cellDetector.batch, 1, 16, true),
      freeze: number(cell.freeze, d.cellDetector.freeze, 0, 30, true), optimizer,
      lr0: number(cell.lr0, d.cellDetector.lr0, 0.000001, 1), lrf: number(cell.lrf, d.cellDetector.lrf, 0.000001, 1),
      momentum: number(cell.momentum, d.cellDetector.momentum, 0, 0.999), weightDecay: number(cell.weightDecay, d.cellDetector.weightDecay, 0, 1),
      warmupEpochs: number(cell.warmupEpochs, d.cellDetector.warmupEpochs, 0, 50),
      warmupMomentum: number(cell.warmupMomentum, d.cellDetector.warmupMomentum, 0, 0.999),
      warmupBiasLr: number(cell.warmupBiasLr, d.cellDetector.warmupBiasLr, 0, 1),
      cosLr: bool(cell.cosLr, d.cellDetector.cosLr), closeMosaic: number(cell.closeMosaic, d.cellDetector.closeMosaic, 0, 500, true),
      mosaic: number(cell.mosaic, d.cellDetector.mosaic, 0, 1), mixup: number(cell.mixup, d.cellDetector.mixup, 0, 1),
      degrees: number(cell.degrees, d.cellDetector.degrees, 0, 180), translate: number(cell.translate, d.cellDetector.translate, 0, 1),
      scale: number(cell.scale, d.cellDetector.scale, 0, 1), fliplr: number(cell.fliplr, d.cellDetector.fliplr, 0, 1), flipud: number(cell.flipud, d.cellDetector.flipud, 0, 1),
      hsvH: number(cell.hsvH, d.cellDetector.hsvH, 0, 0.5), hsvS: number(cell.hsvS, d.cellDetector.hsvS, 0, 1), hsvV: number(cell.hsvV, d.cellDetector.hsvV, 0, 1),
      boxLoss: number(cell.boxLoss, d.cellDetector.boxLoss, 0.1, 50), clsLoss: number(cell.clsLoss, d.cellDetector.clsLoss, 0.1, 50), dflLoss: number(cell.dflLoss, d.cellDetector.dflLoss, 0.1, 50),
      multiScale: number(cell.multiScale, d.cellDetector.multiScale, 0, 1), seed: number(cell.seed, d.cellDetector.seed, 0, 2147483647, true),
      deterministic: bool(cell.deterministic, d.cellDetector.deterministic), cache: bool(cell.cache, d.cellDetector.cache),
      pretrained: bool(cell.pretrained, d.cellDetector.pretrained), savePeriod: number(cell.savePeriod, d.cellDetector.savePeriod, -1, 500, true),
      minimumMap50: number(cell.minimumMap50, d.cellDetector.minimumMap50, 0, 0.95),
      onnxGateTolerance: number(cell.onnxGateTolerance, d.cellDetector.onnxGateTolerance, 0, 0.3),
      missedCellRepeat: number(cell.missedCellRepeat, d.cellDetector.missedCellRepeat, 0, 10, true),
      partialMaskMargin: number(cell.partialMaskMargin, d.cellDetector.partialMaskMargin, 0, 1),
    },
    crackClassifier: {
      iterations: number(crack.iterations, d.crackClassifier.iterations, 100, 50000, true),
      learningRate: number(crack.learningRate, d.crackClassifier.learningRate, 0.000001, 1),
      l2: number(crack.l2, d.crackClassifier.l2, 0, 10),
      validationFraction: number(crack.validationFraction, d.crackClassifier.validationFraction, 0.1, 0.4),
      thresholdMin, thresholdMax,
      thresholdStep: number(crack.thresholdStep, d.crackClassifier.thresholdStep, 0.005, 0.2),
      minimumBalancedAccuracy: number(crack.minimumBalancedAccuracy, d.crackClassifier.minimumBalancedAccuracy, 0.5, 0.95),
      classBalance: bool(crack.classBalance, d.crackClassifier.classBalance),
      cellBuffer: number(crack.cellBuffer, d.crackClassifier.cellBuffer, 0, 0.1),
      algorithm: ['ensemble', 'gbm', 'cnn', 'logistic'].includes(crack.algorithm) ? crack.algorithm : d.crackClassifier.algorithm,
      gbmTrees: number(crack.gbmTrees, d.crackClassifier.gbmTrees, 10, 1000, true),
      gbmDepth: number(crack.gbmDepth, d.crackClassifier.gbmDepth, 1, 8, true),
      gbmLearningRate: number(crack.gbmLearningRate, d.crackClassifier.gbmLearningRate, 0.005, 0.5),
      gbmMinLeaf: number(crack.gbmMinLeaf, d.crackClassifier.gbmMinLeaf, 1, 50, true),
      gbmSubsample: number(crack.gbmSubsample, d.crackClassifier.gbmSubsample, 0.3, 1),
      cnnEpochs: number(crack.cnnEpochs, d.crackClassifier.cnnEpochs, 1, 200, true),
      cnnLearningRate: number(crack.cnnLearningRate, d.crackClassifier.cnnLearningRate, 0.0001, 0.05),
      cnnBatchSize: number(crack.cnnBatchSize, d.crackClassifier.cnnBatchSize, 4, 128, true),
      cnnEarlyStopPatience: number(crack.cnnEarlyStopPatience, d.crackClassifier.cnnEarlyStopPatience, 0, 100, true),
      edgeThresholdDrop: number(crack.edgeThresholdDrop, d.crackClassifier.edgeThresholdDrop, 0, 0.4),
      edgeRequireAgreement: bool(crack.edgeRequireAgreement, d.crackClassifier.edgeRequireAgreement), // absent in older saved settings -> default (on)
      edgeSampleWeight: number(crack.edgeSampleWeight, d.crackClassifier.edgeSampleWeight, 1, 10),
      hardNegativeWeight: number(crack.hardNegativeWeight, d.crackClassifier.hardNegativeWeight, 1, 10),
      missedCrackWeight: number(crack.missedCrackWeight, d.crackClassifier.missedCrackWeight, 1, 10),
    },
    cellInference, cellTypeFilter, crackDetection,
    defaultsVersion: DEFAULTS_VERSION,
  };
}

// v1.4.7.0: getTrainingSettings() is now called per detection (ground gate) and per image (cell inference), so cache
// the parsed file keyed by its mtime instead of re-reading + re-normalising JSON every call. External edits still apply.
let cache: { mtimeMs: number; value: TrainingSettings } | null = null;

/**
 * v1.4.12.7: settings files persist every value, so changing a built-in default never reached existing installs. Files written
 * before this version are upgraded once: a value that still equals the OLD default (i.e. was never changed by the operator) is
 * replaced by the new default; anything the operator edited is kept.
 */
export const DEFAULTS_VERSION = 3; // v1.4.12.8: 3 = trainer defaults (batch/cosLr/flipud/cache) reach installs that never changed them
const LEGACY_DEFAULTS: Array<['cellInference' | 'cellTypeFilter' | 'cellDetector', string, number | boolean]> = [
  ['cellInference', 'maxCandidatesBeforeNms', 700], ['cellInference', 'maxDecoderBoxes', 1200], ['cellInference', 'maxFinalBoxes', 1600],
  ['cellTypeFilter', 'minAreaRelative', 0.55], ['cellTypeFilter', 'minShortRelative', 0.6], ['cellTypeFilter', 'minConfidence', 0.05],
  ['cellDetector', 'batch', 2], ['cellDetector', 'cosLr', false], ['cellDetector', 'flipud', 0], ['cellDetector', 'cache', false],
];
export function upgradeLegacyDefaults(raw: unknown): unknown {
  if (!raw || typeof raw !== 'object') return raw;
  const source = raw as Record<string, any>;
  if (Number(source.defaultsVersion) >= DEFAULTS_VERSION) return raw;
  const upgraded = structuredClone(source);
  for (const [section, key, legacy] of LEGACY_DEFAULTS) {
    const block = upgraded[section];
    if (block && typeof block === 'object' && block[key] === legacy) block[key] = (DEFAULT_TRAINING_SETTINGS[section] as unknown as Record<string, number | boolean>)[key]; // strict equality: also covers booleans
  }
  return upgraded;
}

export function getTrainingSettings(): TrainingSettings {
  try {
    const mtimeMs = fs.statSync(TRAINING_SETTINGS_FILE).mtimeMs;
    if (!cache || cache.mtimeMs !== mtimeMs) {
      const raw = JSON.parse(fs.readFileSync(TRAINING_SETTINGS_FILE, 'utf8'));
      const upgraded = Number(raw?.defaultsVersion) < DEFAULTS_VERSION || raw?.defaultsVersion === undefined;
      const value = normalizeTrainingSettings(upgradeLegacyDefaults(raw));
      if (upgraded) { try { saveTrainingSettings(value); } catch { /* read-only data dir: keep serving the upgraded values from memory */ } }
      cache = { mtimeMs: (upgraded ? safeMtime() : mtimeMs), value };
    }
    return structuredClone(cache.value);
  } catch {
    const defaults = structuredClone(DEFAULT_TRAINING_SETTINGS);
    try { saveTrainingSettings(defaults); } catch { /* read-only data dir: still return defaults rather than failing detection */ }
    return defaults;
  }
}

const safeMtime = () => { try { return fs.statSync(TRAINING_SETTINGS_FILE).mtimeMs; } catch { return 0; } };

export function saveTrainingSettings(input: unknown): TrainingSettings {
  const normalized = normalizeTrainingSettings(input);
  fs.mkdirSync(DATA_TRAINING_DIR, { recursive: true });
  const temporary = `${TRAINING_SETTINGS_FILE}.tmp`;
  fs.writeFileSync(temporary, JSON.stringify(normalized, null, 2), 'utf8');
  fs.renameSync(temporary, TRAINING_SETTINGS_FILE);
  cache = null; // force re-read so the next getTrainingSettings() sees exactly what was persisted
  return normalized;
}
