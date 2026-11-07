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
  profiles: CellTypeProfile[];
};

/** v1.4.7.0: crack-stage ground-gate fractions (previously constants in cvPipeline). */
export type CrackDetectionSettings = {
  groundMaxSoilFraction: number;
  groundMinPvFraction: number;
  groundMinPvFractionFallback: number;
};

export type TrainingSettings = {
  cellDetector: CellDetectorTrainingSettings;
  crackClassifier: CrackClassifierTrainingSettings;
  cellInference: CellInferenceSettings;
  cellTypeFilter: CellTypeFilterSettings;
  crackDetection: CrackDetectionSettings;
};
