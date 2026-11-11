/** Pure handcrafted feature extractor for one normalized solar-cell crop. Shared by training and inference. */

export const CELL_MODEL_SCHEMA = 'cell-level-v1' as const;
export const CELL_WINDOW_SIZE = 210;
export const CELL_WINDOW_MARGIN = 8;
/** v1.4.8.7 schema: richer handcrafted features (edge orientation + continuity) feeding gradient boosting, plus a small CNN. */
export const CELL_MODEL_SCHEMA_V2 = 'cell-level-v2' as const;
export const CELL_FEATURE_NAMES = [
  'lum_mean', 'lum_std', 'th_mean', 'th_top2', 'th_top5', 'fr_th8', 'fr_th14', 'fr_th22',
  'n_comp', 'max_area', 'sum_area', 'max_diag', 'n_vert', 'vert_frac', 'th_ratio', 'elong_max',
] as const;
/** Extra v2 features appended after the 16 base features. */
export const CELL_RICH_FEATURE_NAMES = [
  'grad_mean', 'grad_p95', 'edge_frac',
  'ori_0', 'ori_1', 'ori_2', 'ori_3', 'ori_4', 'ori_5', 'ori_6', 'ori_7',
  'ori_coh', 'ori_dom', 'ori_diag', 'ori_axis',
  'strong_coh', 'cont_ratio', 'cont_span', 'n_long', 'line_frac', 'max_run_frac', 'diag_comp',
] as const;
export const CELL_FEATURE_NAMES_V2 = [...CELL_FEATURE_NAMES, ...CELL_RICH_FEATURE_NAMES] as const;

/** Optional capture of the intermediate maps the small CNN consumes (interior gray, ridge response, no-grid mask). */
export interface CellCropCapture { cw: number; gray: Float32Array; ridge: Float32Array; mask: Uint8Array }

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

/** Extract the 16 handcrafted texture features from one normalized solar-cell crop. */
export function cellCropFeatures(rgb: Uint8Array | Buffer, D: number, M: number, rich = false, capture?: CellCropCapture): number[] {
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
  const grayInterior = new Float32Array(cw * cw);
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
      grayInterior[i] = g[si];
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
  let bigH = 0, bigW = 0, nLong = 0, linePix = 0, goodPix = 0;
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
    if (area > maxArea) { maxArea = area; bigH = h; bigW = w; }
    sumArea += area;
    goodPix += area;
    if (d >= 0.25 * cw) nLong++;
    if (d / Math.sqrt(area) > 2.5) linePix += area;
    if (d > maxDiag) maxDiag = d;
    elong = Math.max(elong, d / Math.sqrt(area));
    nBig++;
    if (area >= 15 && h >= 0.6 * w) { nVert++; vertPix += area; }
  }

  const base = [
    lumMean, lumStd, vMean, top2, top5,
    frac(8), frac(14), frac(22),
    nBig, maxArea, sumArea, maxDiag, nVert,
    sumArea ? vertPix / sumArea : 0,
    top2 / (vStd + 1e-3),
    elong,
  ];
  if (capture) { capture.cw = cw; capture.gray = grayInterior; capture.ridge = th; capture.mask = gm; }
  if (!rich) return base;

  // ---- v1.4.8.7 richer features: gradient orientation histogram + crack continuity -------------------------------
  // Sobel gradients of the vertically-smoothed luminance; gridline-adjacent pixels (grid dilation) are excluded.
  const mag = new Float32Array(cw * cw);
  let cx2 = 0, sx2 = 0, magSum = 0;
  let scx2 = 0, ssx2 = 0, strongSum = 0;
  const hist = new Array(8).fill(0);
  const valid: number[] = [];
  const at = (xx: number, yy: number) => gb[(yy + M) * D + (xx + M)];
  for (let y = 1; y < cw - 1; y++) {
    for (let x = 1; x < cw - 1; x++) {
      const i = y * cw + x;
      if (!gm[i]) continue;
      const gx = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
      const gy = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
      const m = Math.hypot(gx, gy) / 4;
      mag[i] = m;
      valid.push(m);
      if (m < 1.5) continue; // ignore flat texture when building the orientation histogram
      let theta = Math.atan2(gy, gx);
      if (theta < 0) theta += Math.PI; // orientation is axial: 0..pi
      const b = Math.round(theta / (Math.PI / 8)) % 8;
      hist[b] += m; magSum += m;
      cx2 += m * Math.cos(2 * theta); sx2 += m * Math.sin(2 * theta);
      if (th[i] > 14) { scx2 += m * Math.cos(2 * theta); ssx2 += m * Math.sin(2 * theta); strongSum += m; }
    }
  }
  valid.sort((a, b) => a - b);
  const gradMean = valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : 0;
  const gradP95 = valid.length ? valid[Math.min(valid.length - 1, Math.floor(0.95 * valid.length))] : 0;
  let edgeCount = 0;
  for (const m of valid) if (m >= 6) edgeCount++;
  const histNorm = hist.map((v) => (magSum ? v / magSum : 0));
  const coherence = magSum ? Math.hypot(cx2, sx2) / magSum : 0;
  const strongCoherence = strongSum ? Math.hypot(scx2, ssx2) / strongSum : 0;
  const domBin = Math.max(...histNorm);
  const diagShare = histNorm[2] + histNorm[6];
  const axisShare = histNorm[0] + histNorm[4];

  // Directional continuity: longest straight-ish run of strong ridge pixels along the 4 principal directions.
  let maxRun = 0;
  const dirs: Array<[number, number]> = [[1, 0], [0, 1], [1, 1], [1, -1]];
  for (const [dx, dy] of dirs) {
    for (let y = 0; y < cw; y++) {
      for (let x = 0; x < cw; x++) {
        if (th[y * cw + x] <= 14) continue;
        const px = x - dx, py = y - dy;
        if (px >= 0 && py >= 0 && px < cw && py < cw && th[py * cw + px] > 14) continue; // not a run start
        let run = 0, gap = 0, cxr = x, cyr = y, best = 0;
        while (cxr >= 0 && cyr >= 0 && cxr < cw && cyr < cw && gap <= 1) {
          if (th[cyr * cw + cxr] > 14) { run += 1 + gap; gap = 0; best = run; } else gap++;
          cxr += dx; cyr += dy;
        }
        if (best > maxRun) maxRun = best;
      }
    }
  }
  const contRatio = goodPix ? maxArea / goodPix : 0;
  const contSpan = maxArea ? Math.max(bigH, bigW) / cw : 0;
  const diagComp = maxArea ? Math.hypot(bigH, bigW) / (cw * Math.SQRT2) : 0;
  return [
    ...base,
    gradMean, gradP95, valid.length ? edgeCount / valid.length : 0,
    ...histNorm,
    coherence, domBin, diagShare, axisShare,
    strongCoherence, contRatio, contSpan, nLong, goodPix ? linePix / goodPix : 0, maxRun / cw, diagComp,
  ];
}

/** v2 vectors are used as-is (no zero padding); the classifier consumes every richer feature. */
export function cellFeatureVectorV2(features: number[]): number[] { return [...features]; }

/** Raw cell features are used identically during training and inference; no label-dependent context is injected. */
export function cellFeatureVector(features: number[]): number[] {
  return [...features, ...new Array(features.length).fill(0)];
}
