/**
 * Learned edge spider-web window classifier.
 * Feature extraction mirrors training/feat.py (same definitions, same order).
 * Model weights: server/models/crackModel.json (logistic regression on raw + per-image-relative features).
 * If the JSON is missing/invalid, loadCrackModel() returns null and the pipeline falls back to the classical detector only.
 */
import fs from 'fs';
import sharp from 'sharp';
import path from 'path';
import { fileURLToPath } from 'url';

export interface CrackModel {
  version: string;
  window: number;
  stride: number;
  margin: number;
  featureNames: string[];
  mean: number[];
  scale: number[];
  coef: number[];
  intercept: number;
  threshold: number;
}

export interface WindowHit {
  x: number;
  y: number;
  /** window edge length in ORIGINAL image pixels (canonical window / module zoom) */
  size: number;
  prob: number;
}

let cached: CrackModel | null | undefined;

const MODEL_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'models');
export const BASE_MODEL_PATH = path.join(MODEL_DIR, 'crackModel.json');
/** Written by server/trainer.ts from user feedback; takes precedence over the base model while present. */
export const USER_MODEL_PATH = path.join(MODEL_DIR, 'crackModel.user.json');

function readModel(file: string): CrackModel | null {
  try {
    const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
    const n = raw.featureNames?.length ?? 0;
    return n && raw.coef?.length === 2 * n && raw.mean?.length === 2 * n && raw.scale?.length === 2 * n ? (raw as CrackModel) : null;
  } catch {
    return null;
  }
}

export function loadCrackModel(): CrackModel | null {
  if (cached === undefined) cached = readModel(USER_MODEL_PATH) ?? readModel(BASE_MODEL_PATH);
  return cached;
}

/** Drop the in-memory model so the next loadCrackModel() re-reads the user/base file (after retrain or reset). */
export function reloadCrackModel(): CrackModel | null {
  cached = undefined;
  return loadCrackModel();
}

export const baseCrackModel = (): CrackModel | null => readModel(BASE_MODEL_PATH);
export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));
export function modelProb(model: CrackModel, vec: number[]): number {
  let z = model.intercept;
  for (let j = 0; j < vec.length; j++) z += ((vec[j] - model.mean[j]) / model.scale[j]) * model.coef[j];
  return sigmoid(z);
}

// ---------- separable filters (clamped borders == scipy mode='nearest') ----------
function filt1(src: Float32Array, w: number, h: number, size: number, axis: 0 | 1, mode: 'min' | 'max' | 'mean'): Float32Array {
  const out = new Float32Array(w * h);
  const r = size >> 1;
  const lo = -r;
  const hi = size - r - 1; // scipy centered window: [-r, size-r-1]
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = mode === 'min' ? Infinity : mode === 'max' ? -Infinity : 0;
      for (let k = lo; k <= hi; k++) {
        let yy = y;
        let xx = x;
        if (axis === 0) yy = Math.min(h - 1, Math.max(0, y + k));
        else xx = Math.min(w - 1, Math.max(0, x + k));
        const v = src[yy * w + xx];
        if (mode === 'min') { if (v < acc) acc = v; }
        else if (mode === 'max') { if (v > acc) acc = v; }
        else acc += v;
      }
      out[y * w + x] = mode === 'mean' ? acc / size : acc;
    }
  }
  return out;
}

/** Extract the 16 training features from a D x D RGB window (row-major, 3 bytes/px). */
export function windowFeatures(rgb: Uint8Array | Buffer, D: number, M: number): number[] {
  const n = D * D;
  const g = new Float32Array(n);
  for (let i = 0; i < n; i++) g[i] = 0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2];

  const bright = new Float32Array(n);
  for (let i = 0; i < n; i++) bright[i] = g[i] > 175 ? 1 : 0;
  const grid = filt1(filt1(bright, D, D, 9, 0, 'max'), D, D, 9, 1, 'max');

  const gb = filt1(g, D, D, 9, 0, 'mean');
  const eroded = filt1(filt1(gb, D, D, 7, 0, 'min'), D, D, 7, 1, 'min');
  const opened = filt1(filt1(eroded, D, D, 7, 0, 'max'), D, D, 7, 1, 'max');

  const cw = D - 2 * M;
  const th = new Float32Array(cw * cw);
  const gm = new Uint8Array(cw * cw);
  const gc: number[] = [];
  const v: number[] = [];
  for (let y = 0; y < cw; y++) {
    for (let x = 0; x < cw; x++) {
      const si = (y + M) * D + (x + M);
      const i = y * cw + x;
      const nogrid = grid[si] === 0;
      let t = gb[si] - opened[si];
      if (t < 0) t = 0;
      if (!nogrid) t = 0;
      th[i] = t;
      gm[i] = nogrid ? 1 : 0;
      if (nogrid) { gc.push(g[si]); v.push(t); }
    }
  }
  const cnt = v.length;
  let lumMean = 0, lumStd = 0, vMean = 0, vStd = 0;
  if (cnt) {
    for (let i = 0; i < cnt; i++) { lumMean += gc[i]; vMean += v[i]; }
    lumMean /= cnt; vMean /= cnt;
    for (let i = 0; i < cnt; i++) { lumStd += (gc[i] - lumMean) ** 2; vStd += (v[i] - vMean) ** 2; }
    lumStd = Math.sqrt(lumStd / cnt); vStd = Math.sqrt(vStd / cnt);
  }
  const sorted = Float64Array.from(cnt ? v : [0]).sort().reverse();
  const k2 = Math.max(1, Math.floor(0.02 * sorted.length));
  const k5 = Math.max(1, Math.floor(0.05 * sorted.length));
  let top2 = 0, top5 = 0;
  for (let i = 0; i < k5; i++) { if (i < k2) top2 += sorted[i]; top5 += sorted[i]; }
  top2 /= k2; top5 /= k5;
  const frac = (t: number) => { let c = 0; for (let i = 0; i < cnt; i++) if (v[i] > t) c++; return cnt ? c / cnt : 0; };

  // 8-connected components of th > 14
  const labels = new Int32Array(cw * cw);
  let nComp = 0, nBig = 0, maxArea = 0, sumArea = 0, maxDiag = 0, nVert = 0, vertPix = 0, elong = 0;
  const stack: number[] = [];
  for (let s = 0; s < cw * cw; s++) {
    if (th[s] <= 14 || labels[s]) continue;
    nComp++;
    labels[s] = nComp;
    stack.length = 0; stack.push(s);
    let area = 0, minX = cw, maxX = 0, minY = cw, maxY = 0;
    while (stack.length) {
      const p = stack.pop()!;
      const px = p % cw, py = (p / cw) | 0;
      area++;
      if (px < minX) minX = px; if (px > maxX) maxX = px;
      if (py < minY) minY = py; if (py > maxY) maxY = py;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = px + dx, ny = py + dy;
          if (nx < 0 || ny < 0 || nx >= cw || ny >= cw) continue;
          const q = ny * cw + nx;
          if (th[q] > 14 && !labels[q]) { labels[q] = nComp; stack.push(q); }
        }
      }
    }
    if (area < 12) continue;
    const h = maxY - minY + 1, w = maxX - minX + 1;
    const d = Math.hypot(h, w);
    if (area > maxArea) maxArea = area;
    sumArea += area;
    if (d > maxDiag) maxDiag = d;
    elong = Math.max(elong, d / Math.sqrt(area));
    nBig++;
    if (area >= 15 && h >= 0.6 * w) { nVert++; vertPix += area; }
  }

  return [
    lumMean, lumStd, vMean, top2, top5,
    frac(8), frac(14), frac(22),
    nBig, maxArea, sumArea, maxDiag, nVert,
    sumArea ? vertPix / sumArea : 0,
    top2 / (vStd + 1e-3),
    elong,
  ];
}

/** Cheap panel gate used to choose windows (white grid lines present, bluish, no red cable, dark). */
export function isPanelWindow(rgb: Uint8Array | Buffer, D: number): boolean {
  const n = D * D;
  let white = 0, red = 0, br = 0, lum = 0, warm = 0;
  for (let i = 0; i < n; i++) {
    const r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
    const l = 0.299 * r + 0.587 * g + 0.114 * b;
    lum += l; br += b - r;
    if (l > 150 && Math.abs(b - r) < 25) white++;
    if (r > 140 && g < 100 && b < 100) red++;
    if (r > b + 8 && l > 68) warm++; // soil / dry grass (same test as Stage-1 soilMask)
  }
  const wf = white / n;
  // warm <= 10%: clean module windows measured <= 7.5%, windows overlapping soil 12-17% (ground must not be scanned)
  return wf >= 0.05 && wf <= 0.3 && red / n <= 0.01 && warm / n <= 0.1 && br / n >= 5 && lum / n < 140;
}

/**
 * Cell-texture gate (WHY: held-out DJI review showed ground/hardware windows passing the colour tests; PV cells have
 * strong horizontal busbar/finger structure so mean |dy| / mean |dx| (white gridlines excluded) was >= 2.69 on module
 * windows but <= 2.52 on every ground/hardware window sampled). Cheap central-difference estimate on luminance.
 */
const MIN_CELL_ANISOTROPY = 2.55;
export function hasCellTexture(rgb: Uint8Array | Buffer, D: number): boolean {
  let sx = 0, sy = 0, cnt = 0;
  const lum = (i: number) => 0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2];
  for (let y = 1; y < D - 1; y++) {
    for (let x = 1; x < D - 1; x++) {
      const i = y * D + x;
      if (lum(i) > 150) continue; // skip bright gridline pixels, as in calibration
      sx += Math.abs(lum(i + 1) - lum(i - 1));
      sy += Math.abs(lum(i + D) - lum(i - D));
      cnt++;
    }
  }
  return cnt > 500 && sy / (sx + 1e-3) >= MIN_CELL_ANISOTROPY;
}

function percentile(sortedAsc: number[], q: number): number {
  const pos = (sortedAsc.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sortedAsc[lo] + (sortedAsc[hi] - sortedAsc[lo]) * (pos - lo);
}

export interface ScanBox { left: number; top: number; width: number; height: number }

export interface Canon { buf: Buffer; cw: number; ch: number; scale: number; left: number; top: number }
export interface RawWindow { x: number; y: number; scale: number; left: number; top: number; f: number[] }

/** Crop one module (clamped to the image) and resize to the canonical width; null when too small to scan. */
export async function cropCanonical(rawRgb: Buffer, imgW: number, imgH: number, b: ScanBox): Promise<Canon | null> {
  const left = Math.max(0, Math.floor(b.left)), top = Math.max(0, Math.floor(b.top));
  const width = Math.min(imgW - left, Math.floor(b.width)), height = Math.min(imgH - top, Math.floor(b.height));
  if (width < 180 || height < 110) return null;
  const scale = CANON_W / width;
  const cw = CANON_W, ch = Math.max(210, Math.round(height * scale));
  const buf = await sharp(rawRgb, { raw: { width: imgW, height: imgH, channels: 3 } })
    .extract({ left, top, width, height })
    .resize(cw, ch, { fit: 'fill', kernel: 'lanczos3' })
    .raw()
    .toBuffer();
  return { buf, cw, ch, scale, left, top };
}

/** 16 raw features of the D x D canonical window whose top-left corner is (x, y) in a canonical crop. */
export function windowFeaturesAt(c: Canon, x: number, y: number, model: CrackModel): number[] {
  const D = model.window;
  const win = Buffer.allocUnsafe(D * D * 3);
  for (let r = 0; r < D; r++) c.buf.copy(win, r * D * 3, ((y + r) * c.cw + x) * 3, ((y + r) * c.cw + x + D) * 3);
  return windowFeatures(win, D, model.margin);
}

/** Every edge-ring window (raw features) of the given module boxes; shared by inference and training. */
export async function collectWindows(rawRgb: Buffer, imgW: number, imgH: number, boxes: ScanBox[], model: CrackModel, maxWindows = 1500): Promise<RawWindow[]> {
  const D = model.window;
  const feats: RawWindow[] = [];
  for (const b of boxes) {
    const c = await cropCanonical(rawRgb, imgW, imgH, b);
    if (!c) continue;
    for (const { x, y } of edgeRingWindows(c.buf, c.cw, c.ch, D, model.stride)) {
      if (feats.length >= maxWindows) break;
      feats.push({ x, y, scale: c.scale, left: c.left, top: c.top, f: windowFeaturesAt(c, x, y, model) });
    }
  }
  return feats;
}

/** Per-image median / IQR of every feature over ALL scanned windows; < 30 windows => relative part is 0 (as in training). */
export function relStats(feats: { f: number[] }[], nF: number): { med: number[]; iqr: number[]; useRel: boolean } {
  const med = new Array(nF).fill(0), iqr = new Array(nF).fill(1);
  if (feats.length < 30) return { med, iqr, useRel: false };
  for (let j = 0; j < nF; j++) {
    const col = feats.map((w) => w.f[j]).sort((a, c) => a - c);
    med[j] = percentile(col, 0.5);
    iqr[j] = percentile(col, 0.75) - percentile(col, 0.25);
  }
  return { med, iqr, useRel: true };
}

/** raw + per-image-relative model input vector for one window's raw features. */
export function windowVector(f: number[], st: { med: number[]; iqr: number[]; useRel: boolean }): number[] {
  const nF = f.length;
  const vec = new Array(2 * nF);
  for (let j = 0; j < nF; j++) {
    vec[j] = f[j];
    vec[nF + j] = st.useRel ? (f[j] - st.med[j]) / (st.iqr[j] + 1e-3) : 0;
  }
  return vec;
}

/**
 * Score edge-ring windows of every module box. Each module is cropped and resized to CANON_W so windows span the same
 * physical area whatever the drone zoom (this is exactly how training/tools/extract.mts built the training data).
 * Hits are returned in ORIGINAL image pixels (x, y, size).
 */
export async function scanForEdgeWebs(
  rawRgb: Buffer,
  imgW: number,
  imgH: number,
  boxes: ScanBox[],
  model: CrackModel,
  maxWindows = 1500
): Promise<{ hits: WindowHit[]; scanned: number }> {
  const D = model.window;
  const feats = await collectWindows(rawRgb, imgW, imgH, boxes, model, maxWindows);
  const st = relStats(feats, model.featureNames.length);
  const hits: WindowHit[] = [];
  for (const w of feats) {
    const prob = modelProb(model, windowVector(w.f, st));
    if (prob >= model.threshold) {
      hits.push({ x: Math.round(w.left + w.x / w.scale), y: Math.round(w.top + w.y / w.scale), size: Math.round(D / w.scale), prob });
    }
  }
  return { hits, scanned: feats.length };
}

/** Greedy non-max suppression of overlapping windows (keep highest prob); distance is relative to window size. */
export function nmsWindows(hits: WindowHit[], minDist = 0.6): WindowHit[] {
  const sorted = [...hits].sort((a, b) => b.prob - a.prob);
  const keep: WindowHit[] = [];
  for (const h of sorted) {
    if (keep.every((k) => Math.hypot(k.x - h.x, k.y - h.y) >= Math.max(k.size, h.size) * minDist)) keep.push(h);
  }
  return keep;
}

// ---------- canonical-scale edge-ring windows (shared by training/tools/extract.mts and the pipeline) ----------
/** Module crops are resized to this width so a window always spans ~2.3 cells whatever the drone zoom / GSD. */
export const CANON_W = 2200;
export const RING = 210; // window must start within RING canonical px of the (inner) module border to be an edge window

export interface InnerBox { x0: number; x1: number; y0: number; y1: number }

/**
 * Inner limits of a canonical module crop: windows must lie inside it.
 * Why: the Stage-1 module box can overshoot the real module by a few tens of px (shadowed grass / soil next to the
 * aluminium frame), and ground must never be scanned. A real frame is a THICK (>= 12px), long (>= 60% of the side)
 * bright bar within 140px of the box border; when one exists, everything outside its inner edge is ground.
 * No bar => the module continues beyond the image (or has no visible frame): no trimming.
 */
export function panelInnerBox(rgb: Uint8Array | Buffer, cw: number, ch: number): InnerBox {
  const SEARCH = 140, MIN_THICK = 12, FRAC = 0.6;
  const bright = (i: number): boolean => {
    const r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
    return 0.299 * r + 0.587 * g + 0.114 * b > 165 && Math.abs(b - r) < 30;
  };
  const colFrac = (x: number): number => { let c = 0; for (let y = 0; y < ch; y++) if (bright(y * cw + x)) c++; return c / ch; };
  const rowFrac = (y: number): number => { let c = 0; for (let x = 0; x < cw; x++) if (bright(y * cw + x)) c++; return c / cw; };
  // Outermost bar scanning inward from the border; returns the first index AFTER the bar, or 0 if none
  const innerEdge = (len: number, frac: (k: number) => number, fromEnd: boolean): number => {
    let run = 0;
    for (let k = 0; k < Math.min(SEARCH, len >> 1); k++) {
      const idx = fromEnd ? len - 1 - k : k;
      if (frac(idx) >= FRAC) run++;
      else if (run >= MIN_THICK) return k; // bar ended at k-1 -> inner edge at k
      else run = 0;
    }
    return run >= MIN_THICK ? Math.min(SEARCH, len >> 1) : 0;
  };
  const l = innerEdge(cw, colFrac, false), r = innerEdge(cw, colFrac, true);
  const t = innerEdge(ch, rowFrac, false), b = innerEdge(ch, rowFrac, true);
  return { x0: l, x1: cw - r, y0: t, y1: ch - b };
}

/** Top-left corners (canonical px) of the edge-ring windows of one module crop that pass the ground / panel gates. */
export function edgeRingWindows(rgb: Uint8Array | Buffer, cw: number, ch: number, D: number, stride: number): { x: number; y: number }[] {
  const box = panelInnerBox(rgb, cw, ch);
  const out: { x: number; y: number }[] = [];
  const win = Buffer.allocUnsafe(D * D * 3);
  for (let y = box.y0; y + D <= box.y1; y += stride) {
    for (let x = box.x0; x + D <= box.x1; x += stride) {
      if (Math.min(x - box.x0, box.x1 - D - x, y - box.y0, box.y1 - D - y) > RING) continue;
      for (let r = 0; r < D; r++) {
        win.set(rgb.subarray(((y + r) * cw + x) * 3, ((y + r) * cw + x + D) * 3), r * D * 3);
      }
      if (isPanelWindow(win, D) && hasCellTexture(win, D)) out.push({ x, y });
    }
  }
  return out;
}
