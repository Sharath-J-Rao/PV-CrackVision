/**
 * v1.4.8.7 — small CNN on single solar-cell crops (pure TypeScript; no native/ONNX dependency, trains inside the app).
 * Input: 2 channels x SxS built from ONE cell crop — (0) standardised luminance (gridlines flattened to the cell mean),
 * (1) max-pooled top-hat ridge response, which keeps hairline cracks alive when a 210 px crop is shrunk to SxS.
 * Net: conv3x3(2->8)+ReLU+pool -> conv3x3(8->12)+ReLU+pool -> conv3x3(12->16)+ReLU -> [global avg | global max] -> dense -> sigmoid.
 * Training: Adam, class-balanced weights, random flips/transposes (a cell crack is orientation-free), L2 weight decay.
 */
import type { CellCropCapture } from './cellFeatures.ts';

export const CNN_INPUT_SIZE = 48;
export const CNN_CHANNELS = [2, 8, 12, 16] as const;

export interface CnnModel {
  size: number;
  channels: number[];
  w1: number[]; b1: number[]; w2: number[]; b2: number[]; w3: number[]; b3: number[]; wd: number[]; bd: number;
}
export interface CnnTrainParams { epochs: number; learningRate: number; batchSize: number; weightDecay: number; seed: number; earlyStopPatience: number }
export const DEFAULT_CNN_PARAMS: CnnTrainParams = { epochs: 20, learningRate: 0.004, batchSize: 24, weightDecay: 0.0005, seed: 11, earlyStopPatience: 0 };
/** v1.4.10: cooperative pause hook awaited at every batch/tree boundary; it rejects when the operator aborts. */
export interface TrainControl { waitIfPaused?: () => Promise<void> }
export interface CnnSample { input: Float32Array; y: 0 | 1 }

const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Build the 2xSxS CNN input from the maps captured by cellCropFeatures. Used identically in training and inference. */
export function buildCnnInput(capture: CellCropCapture, size = CNN_INPUT_SIZE): Float32Array {
  const { cw, gray, ridge, mask } = capture;
  let mean = 0, count = 0;
  for (let i = 0; i < cw * cw; i++) if (mask[i]) { mean += gray[i]; count++; }
  mean = count ? mean / count : 0;
  let variance = 0;
  for (let i = 0; i < cw * cw; i++) if (mask[i]) variance += (gray[i] - mean) ** 2;
  const std = Math.max(4, Math.sqrt(count ? variance / count : 0));
  const out = new Float32Array(2 * size * size);
  for (let v = 0; v < size; v++) {
    const y0 = Math.floor((v * cw) / size), y1 = Math.max(y0 + 1, Math.floor(((v + 1) * cw) / size));
    for (let u = 0; u < size; u++) {
      const x0 = Math.floor((u * cw) / size), x1 = Math.max(x0 + 1, Math.floor(((u + 1) * cw) / size));
      let sum = 0, n = 0, peak = 0;
      for (let y = y0; y < y1 && y < cw; y++) {
        for (let x = x0; x < x1 && x < cw; x++) {
          const i = y * cw + x;
          const value = mask[i] ? gray[i] : mean; // flatten busbars/fingers
          sum += value; n++;
          if (ridge[i] > peak) peak = ridge[i];
        }
      }
      const avg = n ? sum / n : mean;
      out[v * size + u] = Math.max(-3, Math.min(3, (avg - mean) / std));
      out[size * size + v * size + u] = Math.min(3, peak / 30);
    }
  }
  return out;
}

// ---------------- layers ----------------
function convForward(inp: Float32Array, cin: number, size: number, w: Float32Array, b: Float32Array, cout: number, out: Float32Array): void {
  const plane = size * size;
  for (let o = 0; o < cout; o++) {
    const base = o * plane;
    for (let p = 0; p < plane; p++) out[base + p] = b[o];
    for (let i = 0; i < cin; i++) {
      const ip = i * plane, wb = (o * cin + i) * 9;
      for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
          const wv = w[wb + ky * 3 + kx];
          if (wv === 0) continue;
          const dy = ky - 1, dx = kx - 1;
          for (let y = Math.max(0, -dy); y < Math.min(size, size - dy); y++) {
            const orow = base + y * size, irow = ip + (y + dy) * size + dx;
            for (let x = Math.max(0, -dx); x < Math.min(size, size - dx); x++) out[orow + x] += wv * inp[irow + x];
          }
        }
      }
    }
  }
}

function convBackward(inp: Float32Array, cin: number, size: number, w: Float32Array, cout: number, dout: Float32Array, dw: Float32Array, db: Float32Array, dinp: Float32Array | null): void {
  const plane = size * size;
  for (let o = 0; o < cout; o++) {
    const base = o * plane;
    let bsum = 0;
    for (let p = 0; p < plane; p++) bsum += dout[base + p];
    db[o] += bsum;
    for (let i = 0; i < cin; i++) {
      const ip = i * plane, wb = (o * cin + i) * 9;
      for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
          const dy = ky - 1, dx = kx - 1;
          const wv = w[wb + ky * 3 + kx];
          let acc = 0;
          for (let y = Math.max(0, -dy); y < Math.min(size, size - dy); y++) {
            const orow = base + y * size, irow = ip + (y + dy) * size + dx;
            for (let x = Math.max(0, -dx); x < Math.min(size, size - dx); x++) {
              const d = dout[orow + x];
              acc += d * inp[irow + x];
              if (dinp) dinp[irow + x] += wv * d;
            }
          }
          dw[wb + ky * 3 + kx] += acc;
        }
      }
    }
  }
}

/** Work buffers for one sample (re-used across samples to avoid allocation churn). */
class Workspace {
  a1: Float32Array; p1: Float32Array; i1: Int32Array; a2: Float32Array; p2: Float32Array; i2: Int32Array; a3: Float32Array;
  gmpIdx: Int32Array; feat: Float32Array;
  da1: Float32Array; dp1: Float32Array; da2: Float32Array; dp2: Float32Array; da3: Float32Array;
  S: number; C: readonly number[];
  constructor(S: number, C: readonly number[]) {
    this.S = S; this.C = C;
    const s1 = S, s2 = S / 2, s3 = S / 4;
    this.a1 = new Float32Array(C[1] * s1 * s1); this.p1 = new Float32Array(C[1] * s2 * s2); this.i1 = new Int32Array(C[1] * s2 * s2);
    this.a2 = new Float32Array(C[2] * s2 * s2); this.p2 = new Float32Array(C[2] * s3 * s3); this.i2 = new Int32Array(C[2] * s3 * s3);
    this.a3 = new Float32Array(C[3] * s3 * s3);
    this.gmpIdx = new Int32Array(C[3]); this.feat = new Float32Array(2 * C[3]);
    this.da1 = new Float32Array(this.a1.length); this.dp1 = new Float32Array(this.p1.length);
    this.da2 = new Float32Array(this.a2.length); this.dp2 = new Float32Array(this.p2.length); this.da3 = new Float32Array(this.a3.length);
  }
}

function relu(a: Float32Array): void { for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0; }

function maxPool(a: Float32Array, channels: number, size: number, out: Float32Array, idx: Int32Array): void {
  const half = size / 2;
  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < half; y++) {
      for (let x = 0; x < half; x++) {
        const p0 = c * size * size + 2 * y * size + 2 * x;
        let best = p0, bv = a[p0];
        for (const q of [p0 + 1, p0 + size, p0 + size + 1]) if (a[q] > bv) { bv = a[q]; best = q; }
        const o = c * half * half + y * half + x;
        out[o] = bv; idx[o] = best;
      }
    }
  }
}

interface Params { w1: Float32Array; b1: Float32Array; w2: Float32Array; b2: Float32Array; w3: Float32Array; b3: Float32Array; wd: Float32Array; bd: Float32Array }

function forward(P: Params, input: Float32Array, ws: Workspace): number {
  const { S, C } = ws;
  convForward(input, C[0], S, P.w1, P.b1, C[1], ws.a1); relu(ws.a1);
  maxPool(ws.a1, C[1], S, ws.p1, ws.i1);
  convForward(ws.p1, C[1], S / 2, P.w2, P.b2, C[2], ws.a2); relu(ws.a2);
  maxPool(ws.a2, C[2], S / 2, ws.p2, ws.i2);
  convForward(ws.p2, C[2], S / 4, P.w3, P.b3, C[3], ws.a3); relu(ws.a3);
  const plane = (S / 4) * (S / 4);
  for (let c = 0; c < C[3]; c++) {
    let sum = 0, best = c * plane, bv = ws.a3[best];
    for (let p = 0; p < plane; p++) { const v = ws.a3[c * plane + p]; sum += v; if (v > bv) { bv = v; best = c * plane + p; } }
    ws.feat[c] = sum / plane; ws.feat[C[3] + c] = bv; ws.gmpIdx[c] = best;
  }
  let z = P.bd[0];
  for (let k = 0; k < ws.feat.length; k++) z += ws.feat[k] * P.wd[k];
  return z;
}

function backward(P: Params, G: Params, input: Float32Array, ws: Workspace, dz: number): void {
  const { S, C } = ws;
  const plane = (S / 4) * (S / 4);
  G.bd[0] += dz;
  ws.da3.fill(0);
  for (let k = 0; k < ws.feat.length; k++) G.wd[k] += dz * ws.feat[k];
  for (let c = 0; c < C[3]; c++) {
    const avgGrad = (dz * P.wd[c]) / plane;
    for (let p = 0; p < plane; p++) ws.da3[c * plane + p] += avgGrad;
    ws.da3[ws.gmpIdx[c]] += dz * P.wd[C[3] + c];
  }
  for (let i = 0; i < ws.da3.length; i++) if (ws.a3[i] <= 0) ws.da3[i] = 0;
  ws.dp2.fill(0);
  convBackward(ws.p2, C[2], S / 4, P.w3, C[3], ws.da3, G.w3, G.b3, ws.dp2);
  ws.da2.fill(0);
  for (let i = 0; i < ws.dp2.length; i++) ws.da2[ws.i2[i]] += ws.dp2[i];
  for (let i = 0; i < ws.da2.length; i++) if (ws.a2[i] <= 0) ws.da2[i] = 0;
  ws.dp1.fill(0);
  convBackward(ws.p1, C[1], S / 2, P.w2, C[2], ws.da2, G.w2, G.b2, ws.dp1);
  ws.da1.fill(0);
  for (let i = 0; i < ws.dp1.length; i++) ws.da1[ws.i1[i]] += ws.dp1[i];
  for (let i = 0; i < ws.da1.length; i++) if (ws.a1[i] <= 0) ws.da1[i] = 0;
  convBackward(input, C[0], S, P.w1, C[1], ws.da1, G.w1, G.b1, null);
}

function allocParams(C: readonly number[], random: (() => number) | null): Params {
  const make = (n: number, fanIn: number) => {
    const a = new Float32Array(n);
    if (random) for (let i = 0; i < n; i++) {
      const u = Math.max(1e-9, random()), v = random();
      a[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * Math.sqrt(2 / fanIn); // He init
    }
    return a;
  };
  return {
    w1: make(C[1] * C[0] * 9, C[0] * 9), b1: new Float32Array(C[1]),
    w2: make(C[2] * C[1] * 9, C[1] * 9), b2: new Float32Array(C[2]),
    w3: make(C[3] * C[2] * 9, C[2] * 9), b3: new Float32Array(C[3]),
    wd: make(2 * C[3], 2 * C[3]), bd: new Float32Array(1),
  };
}

function paramList(P: Params): Float32Array[] { return [P.w1, P.b1, P.w2, P.b2, P.w3, P.b3, P.wd, P.bd]; }

/** Dihedral augmentation (8 symmetries) applied to both channels. */
export function dihedral(input: Float32Array, size: number, k: number): Float32Array {
  if (k === 0) return input;
  const out = new Float32Array(input.length);
  const plane = size * size;
  for (let c = 0; c < input.length / plane; c++) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let sx = x, sy = y;
        if (k & 4) { const t = sx; sx = sy; sy = t; }
        if (k & 1) sx = size - 1 - sx;
        if (k & 2) sy = size - 1 - sy;
        out[c * plane + y * size + x] = input[c * plane + sy * size + sx];
      }
    }
  }
  return out;
}

export function exportCnn(P: Params, size: number): CnnModel {
  return {
    size, channels: [...CNN_CHANNELS],
    w1: Array.from(P.w1), b1: Array.from(P.b1), w2: Array.from(P.w2), b2: Array.from(P.b2),
    w3: Array.from(P.w3), b3: Array.from(P.b3), wd: Array.from(P.wd), bd: P.bd[0],
  };
}
function importCnn(model: CnnModel): Params {
  const P = allocParams(model.channels, null);
  P.w1.set(model.w1); P.b1.set(model.b1); P.w2.set(model.w2); P.b2.set(model.b2);
  P.w3.set(model.w3); P.b3.set(model.b3); P.wd.set(model.wd); P.bd[0] = model.bd;
  return P;
}

export function isValidCnn(model: unknown): model is CnnModel {
  const m = model as CnnModel | undefined;
  if (!m || m.size !== CNN_INPUT_SIZE || !Array.isArray(m.channels) || m.channels.join() !== CNN_CHANNELS.join()) return false;
  const C = m.channels;
  const finite = (a: unknown, n: number) => Array.isArray(a) && a.length === n && a.every((v) => Number.isFinite(v));
  return finite(m.w1, C[1] * C[0] * 9) && finite(m.b1, C[1]) && finite(m.w2, C[2] * C[1] * 9) && finite(m.b2, C[2]) &&
    finite(m.w3, C[3] * C[2] * 9) && finite(m.b3, C[3]) && finite(m.wd, 2 * C[3]) && Number.isFinite(m.bd);
}

const importCache = new WeakMap<CnnModel, { P: Params; ws: Workspace }>();
export function cnnProbability(model: CnnModel, input: Float32Array): number {
  let cached = importCache.get(model);
  if (!cached) { cached = { P: importCnn(model), ws: new Workspace(model.size, model.channels) }; importCache.set(model, cached); }
  return sigmoid(forward(cached.P, input, cached.ws));
}

export interface CnnEpochMetrics {
  epoch: number;
  loss: number;
  validationLoss: number;
  precision: number;
  recall: number;
  specificity: number;
  balancedAccuracy: number;
  f1: number;
}
export interface CnnTrainResult { model: CnnModel; history: CnnEpochMetrics[]; bestEpoch: number; stoppedEarly: boolean }

/** Train from scratch. Yields to the event loop regularly so the server stays responsive during long runs. */
export async function trainCnn(
  samples: CnnSample[],
  weights: number[],
  overrides: Partial<CnnTrainParams> = {},
  onEpoch?: (epoch: number, loss: number, metrics?: CnnEpochMetrics, bestEpoch?: number) => void,
  validationSamples: CnnSample[] = [],
  control?: TrainControl,
): Promise<CnnTrainResult> {
  if (!samples.length) throw new Error('The CNN needs at least one training crop.');
  const params = { ...DEFAULT_CNN_PARAMS, ...overrides };
  const random = mulberry32(params.seed);
  const S = CNN_INPUT_SIZE;
  const P = allocParams(CNN_CHANNELS, random);
  const G = allocParams(CNN_CHANNELS, null);
  const m = paramList(P).map((a) => new Float32Array(a.length));
  const v = paramList(P).map((a) => new Float32Array(a.length));
  const ws = new Workspace(S, CNN_CHANNELS);
  const order = Array.from({ length: samples.length }, (_, i) => i);
  const history: CnnEpochMetrics[] = [];
  const [beta1, beta2, eps] = [0.9, 0.999, 1e-8];
  const weightMean = weights.reduce((a, b) => a + b, 0) / Math.max(1, weights.length);
  let step = 0;
  // v1.4.10: best-epoch tracking (highest held-out balanced accuracy, lower validation loss breaks ties).
  let bestEpoch = 0, bestBa = -1, bestLoss = Infinity, stale = 0, stoppedEarly = false;
  let snapshot: Float32Array[] | null = null;

  for (let epoch = 1; epoch <= params.epochs; epoch++) {
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    const lr = params.learningRate * (0.5 * (1 + Math.cos((Math.PI * (epoch - 1)) / params.epochs)) * 0.9 + 0.1); // cosine decay to 10%
    let lossSum = 0;
    for (let start = 0; start < order.length; start += params.batchSize) {
      const batch = order.slice(start, start + params.batchSize);
      for (const g of paramList(G)) g.fill(0);
      for (const idx of batch) {
        const sample = samples[idx];
        const input = dihedral(sample.input, S, Math.floor(random() * 8));
        const z = forward(P, input, ws);
        const p = Math.max(1e-7, Math.min(1 - 1e-7, sigmoid(z)));
        const w = weights[idx] / weightMean;
        lossSum += -w * (sample.y * Math.log(p) + (1 - sample.y) * Math.log(1 - p));
        backward(P, G, input, ws, ((p - sample.y) * w) / batch.length);
      }
      step++;
      const bias1 = 1 - Math.pow(beta1, step), bias2 = 1 - Math.pow(beta2, step);
      const ps = paramList(P), gs = paramList(G);
      for (let t = 0; t < ps.length; t++) {
        const decay = t % 2 === 0 ? params.weightDecay : 0; // decay weights, not biases
        for (let k = 0; k < ps[t].length; k++) {
          const grad = gs[t][k] + decay * ps[t][k];
          m[t][k] = beta1 * m[t][k] + (1 - beta1) * grad;
          v[t][k] = beta2 * v[t][k] + (1 - beta2) * grad * grad;
          ps[t][k] -= (lr * (m[t][k] / bias1)) / (Math.sqrt(v[t][k] / bias2) + eps);
        }
      }
      await new Promise<void>((resolve) => setImmediate(resolve));
      await control?.waitIfPaused?.(); // pause/abort point
    }
    const loss = lossSum / samples.length;
    let validationLoss = 0, tp = 0, fp = 0, tn = 0, fn = 0;
    for (const sample of validationSamples) {
      const probability = Math.max(1e-7, Math.min(1 - 1e-7, sigmoid(forward(P, sample.input, ws))));
      validationLoss += -(sample.y * Math.log(probability) + (1 - sample.y) * Math.log(1 - probability));
      const predicted = probability >= 0.5 ? 1 : 0;
      if (sample.y === 1 && predicted === 1) tp++;
      else if (sample.y === 0 && predicted === 1) fp++;
      else if (sample.y === 0 && predicted === 0) tn++;
      else fn++;
    }
    const positives = tp + fn, negatives = tn + fp;
    const recall = positives ? tp / positives : 0;
    const specificity = negatives ? tn / negatives : 0;
    const row: CnnEpochMetrics = {
      epoch,
      loss: Number(loss.toFixed(5)),
      validationLoss: validationSamples.length ? Number((validationLoss / validationSamples.length).toFixed(5)) : 0,
      precision: Number((tp / Math.max(1, tp + fp)).toFixed(4)),
      recall: Number(recall.toFixed(4)),
      specificity: Number(specificity.toFixed(4)),
      balancedAccuracy: Number(((recall + specificity) / 2).toFixed(4)),
      f1: Number(((2 * (tp / Math.max(1, tp + fp)) * recall) / Math.max(1e-12, (tp / Math.max(1, tp + fp)) + recall)).toFixed(4)),
    };
    history.push(row);
    if (validationSamples.length) {
      const better = row.balancedAccuracy > bestBa + 1e-9 || (Math.abs(row.balancedAccuracy - bestBa) <= 1e-9 && row.validationLoss < bestLoss);
      if (better) { bestBa = row.balancedAccuracy; bestLoss = row.validationLoss; bestEpoch = epoch; snapshot = paramList(P).map((a) => a.slice()); stale = 0; } else stale++;
    }
    onEpoch?.(epoch, loss, row, bestEpoch);
    if (params.earlyStopPatience > 0 && validationSamples.length && stale >= params.earlyStopPatience && epoch < params.epochs) { stoppedEarly = true; break; }
  }
  if (snapshot) { const ps = paramList(P); snapshot.forEach((s, i) => ps[i].set(s)); } // restore the best epoch's weights
  return { model: exportCnn(P, S), history, bestEpoch, stoppedEarly };
}
