/** v1.4.12.6: single source of truth for the selectable YOLO cell-detector training sizes (UI list + server validation). */
export const CELL_IMAGE_SIZES = [416, 512, 640, 768, 960, 1280, 1536, 1920] as const;

export type CellDetectorTrainingSettings = {
  epochs: number;
  patience: number;
  imageSize: number;
  batch: number;
  freeze: number;
  optimizer: 'auto' | 'SGD' | 'Adam' | 'AdamW' | 'RMSProp';
  lr0: number;
  lrf: number;
  momentum: number;
  weightDecay: number;
  warmupEpochs: number;
  warmupMomentum: number;
  warmupBiasLr: number;
  cosLr: boolean;
  closeMosaic: number;
  mosaic: number;
  mixup: number;
  degrees: number;
  translate: number;
  scale: number;
  fliplr: number;
  flipud: number;
  hsvH: number;
  hsvS: number;
  hsvV: number;
  boxLoss: number;
  clsLoss: number;
  dflLoss: number;
  multiScale: number;
  seed: number;
  deterministic: boolean;
  cache: boolean;
  pretrained: boolean;
  savePeriod: number;
  minimumMap50: number;
  /** v1.4.12.7 hotfix: how far the exported ONNX mAP50 may fall below the activation gate (export rounding); 0 = strict. */
  onnxGateTolerance: number;
  /** v1.4.11.5: extra train-split copies of images that contain cells the model missed (0 = off). */
  missedCellRepeat: number;
  partialMaskMargin: number;      // v1.4.12.8: how far (as a fraction of a cell) around marked cells stays visible in partially annotated images
};

export type CrackClassifierTrainingSettings = {
  iterations: number;
  learningRate: number;
  l2: number;
  validationFraction: number;
  thresholdMin: number;
  thresholdMax: number;
  thresholdStep: number;
  minimumBalancedAccuracy: number;
  classBalance: boolean;
  /** v1.4.9: minimal buffer (fraction of the cell short side, 0 = off) added around every detected cell box when its crop is taken, so the true cell boundary is not clipped by a slightly tight detection. */
  cellBuffer: number;
  /** v1.4.8.7: which classifier is trained on the cell crops. 'logistic' = the legacy v1.4.8.6 model. */
  algorithm: 'ensemble' | 'gbm' | 'cnn' | 'logistic';
  gbmTrees: number;
  gbmDepth: number;
  gbmLearningRate: number;
  gbmMinLeaf: number;
  gbmSubsample: number;
  cnnEpochs: number;
  cnnLearningRate: number;
  cnnBatchSize: number;
  /** v1.4.10: stop CNN training after this many epochs without a better held-out balanced accuracy and restore the best epoch's weights (0 = off; best weights are still restored after the last epoch). */
  cnnEarlyStopPatience: number;
  /** v1.4.10: module-perimeter cells use (decision threshold - this); 0 = no priority. */
  edgeThresholdDrop: number;
  /** v1.4.12.9: outer-ring cells are only reported when EVERY classifier component (boosting and CNN) individually clears the perimeter threshold, not just their average. */
  edgeRequireAgreement: boolean;
  /** v1.4.10: training weight multiplier for module-perimeter cells (1 = no priority). */
  edgeSampleWeight: number;
  /** v1.4.10: training weight multiplier for cells the operator deleted as false detections in review. */
  hardNegativeWeight: number;
  /** v1.4.11.6: training weight multiplier for cracked cells the operator added after the model missed them. */
  missedCrackWeight: number;
};

/** v1.4.7.0: inference-time cell detector parameters (previously hard-coded in cellDetector/yoloOutputDecoder). */
export type CellInferenceSettings = {
  confidenceThreshold: number;   // minimum YOLO score kept (low = high recall)
  decoderNmsIou: number;         // per-pass NMS IoU in the decoder
  mergeNmsIou: number;           // cross-pass (full image + tiles) NMS IoU
  maxCandidatesBeforeNms: number;
  maxDecoderBoxes: number;       // cap per inference pass after NMS
  maxFinalBoxes: number;         // cap per image after merging
  enclosingSuppression: boolean; // reject a big box that encloses >=2 similar small cells
  useTiles: boolean;             // overlapping detail tiles in addition to the whole-image pass
  fuseTileBoxes: boolean;        // average (score-weighted) the same cell seen by several passes instead of keeping only the top box
  tileScale: number;             // tile size as a multiple of the model input
  tileOverlap: number;
  maxTiles: number;
};

/** A cell "type" (shape/size signature). accept = only matching detections pass; reject = matching detections are dropped (negative type). */
export type CellTypeProfile = {
  id: string;
  name: string;
  mode: 'accept' | 'reject';
  enabled: boolean;
  aspect: number;          // long side / short side (orientation-independent)
  aspectTolerance: number; // allowed relative deviation, e.g. 0.25 = ±25 %
  minLongPx: number;       // absolute long-side bounds in image px; 0 = no bound
  maxLongPx: number;
};

export type CellTypeFilterSettings = {
  enabled: boolean;
  relativeMin: number; // long side must be >= relativeMin x median long side of the passing cells (0 = off)
  relativeMax: number; // ... and <= relativeMax x median (0 = off)
  // v1.4.8.7: undersized / low-confidence detections are rejected too.
  minShortPx: number;       // absolute: short side must be >= this many image px (0 = off)
  minAreaRelative: number;  // box area must be >= this x the reference cell area (0 = off)
  minShortRelative: number; // short side must be >= this x the reference cell short side (0 = off)
  minConfidence: number;    // YOLO score must be >= this (0 = off); the decoder floor stays low for recall
  // v1.4.12.7: two boxes overlapping by at least this share of the SMALLER box cannot both be cells; only the one closest
  // to the typical cell size is kept (removes fragments inside a cell and shifted duplicates). 0 = off.
  fragmentOverlap: number;
  profiles: CellTypeProfile[];
};

/** v1.4.7.0: crack-stage ground-gate fractions (previously constants in cvPipeline). */
export type CrackDetectionSettings = {
  groundMaxSoilFraction: number;
  groundMinPvFraction: number;
  groundMinPvFractionFallback: number;
  /** v1.4.12.7 hotfix: crack decision threshold (probability) applied at detection time. 0 = use the trained model's own tuned threshold. */
  decisionThreshold: number;
  /** v1.4.14: detections whose confidence is below this PERCENT are drawn yellow (still reported). Display only; does not remove detections. */
  lowConfidenceThreshold: number;
};

/** v1.4.14: pretrained crack models. A = crop classifier (decides), B = tiled module detector (shadow comparison only). */
export type CrackNetSettings = {
  clsBase: 'yolo11n-cls.pt' | 'yolo11s-cls.pt';
  clsEpochs: number; clsImgsz: number; clsBatch: number; clsPatience: number;
  /** Decision threshold is picked on held-out images as the highest value that still reaches this recall. */
  targetRecall: number;
  minBalancedAccuracy: number;
  valFraction: number;
  detBase: 'yolo11n.pt' | 'yolo11s.pt';
  detEpochs: number; detImgsz: number; detBatch: number; detPatience: number; detTileOverlap: number;
  /** Run model B beside A and tag each detection with whether B agrees. Needs a trained B; never changes what is reported. */
  shadowEnabled: boolean;
  /** v1.4.14.5: re-score would-be detections on flipped views and require agreement (fewer false positives, tiny cost). */
  candidateTta: boolean;
  seed: number;
};

export type TrainingSettings = {
  cellDetector: CellDetectorTrainingSettings;
  crackClassifier: CrackClassifierTrainingSettings;
  cellInference: CellInferenceSettings;
  cellTypeFilter: CellTypeFilterSettings;
  crackDetection: CrackDetectionSettings;
  crackNet: CrackNetSettings;
  /** v1.4.12.7: version of the built-in defaults this file was last upgraded to (see server/trainingSettings.ts). */
  defaultsVersion?: number;
};
