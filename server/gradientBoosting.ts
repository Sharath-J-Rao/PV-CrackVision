/**
 * v1.4.8.7 — dependency-free gradient-boosted decision trees for binary classification (logistic loss, Newton leaves).
 * Exact greedy splits, depth-limited trees, shrinkage, row subsampling and L2 leaf regularisation. Fast enough to retrain
 * in seconds on a few thousand cell crops, which keeps the quick-training workflow.
 */

/** Leaf when f < 0 (value in v). Otherwise: x[f] <= t goes to node l, else node r. */
export interface GbmNode { f: number; t: number; l: number; r: number; v: number }
export interface GbmModel { featureCount: number; baseScore: number; learningRate: number; trees: GbmNode[][] }
export interface GbmParams { trees: number; depth: number; learningRate: number; minLeaf: number; subsample: number; lambda: number; seed: number }

export const DEFAULT_GBM_PARAMS: GbmParams = { trees: 160, depth: 3, learningRate: 0.08, minLeaf: 3, subsample: 0.8, lambda: 1, seed: 7 };

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

function buildTree(X: number[][], g: Float64Array, h: Float64Array, rows: number[], params: GbmParams): GbmNode[] {
  const nodes: GbmNode[] = [];
  const featureCount = X[0].length;
  const leafValue = (G: number, H: number) => -G / (H + params.lambda);

  const grow = (idx: number[], depth: number): number => {
    let G = 0, H = 0;
    for (const i of idx) { G += g[i]; H += h[i]; }
    const nodeIndex = nodes.length;
    nodes.push({ f: -1, t: 0, l: -1, r: -1, v: leafValue(G, H) });
    if (depth >= params.depth || idx.length < 2 * params.minLeaf) return nodeIndex;

    const parentScore = (G * G) / (H + params.lambda);
    let bestGain = 1e-9, bestF = -1, bestT = 0;
    for (let f = 0; f < featureCount; f++) {
      const sorted = [...idx].sort((a, b) => X[a][f] - X[b][f]);
      let GL = 0, HL = 0;
      for (let k = 0; k < sorted.length - 1; k++) {
        const i = sorted[k];
        GL += g[i]; HL += h[i];
        const v = X[i][f], vNext = X[sorted[k + 1]][f];
        if (v === vNext) continue;
        if (k + 1 < params.minLeaf || sorted.length - (k + 1) < params.minLeaf) continue;
        const GR = G - GL, HR = H - HL;
        if (HL < 1e-6 || HR < 1e-6) continue;
        const gain = (GL * GL) / (HL + params.lambda) + (GR * GR) / (HR + params.lambda) - parentScore;
        if (gain > bestGain) { bestGain = gain; bestF = f; bestT = (v + vNext) / 2; }
      }
    }
    if (bestF < 0) return nodeIndex;
    const left: number[] = [], right: number[] = [];
    for (const i of idx) (X[i][bestF] <= bestT ? left : right).push(i);
    if (!left.length || !right.length) return nodeIndex;
    nodes[nodeIndex].f = bestF;
    nodes[nodeIndex].t = bestT;
    nodes[nodeIndex].l = grow(left, depth + 1);
    nodes[nodeIndex].r = grow(right, depth + 1);
    return nodeIndex;
  };
  grow(rows, 0);
  return nodes;
}

function treeValue(tree: GbmNode[], x: ArrayLike<number>): number {
  let n = 0;
  for (let guard = 0; guard < 64; guard++) {
    const node = tree[n];
    if (node.f < 0) return node.v;
    n = x[node.f] <= node.t ? node.l : node.r;
  }
  return tree[n].v;
}

export function gbmScore(model: GbmModel, x: ArrayLike<number>): number {
  let z = model.baseScore;
  for (const tree of model.trees) z += model.learningRate * treeValue(tree, x);
  return z;
}
export function gbmProbability(model: GbmModel, x: ArrayLike<number>): number { return sigmoid(gbmScore(model, x)); }

/** Train a boosted-tree classifier. `weights` are per-sample weights (class balancing). All inputs must be finite. */
export function trainGbm(X: number[][], y: Array<0 | 1>, weights: number[], overrides: Partial<GbmParams> = {}, onTree?: (tree: number, totalTrees: number) => void): GbmModel {
  if (!X.length) throw new Error('Gradient boosting needs at least one training sample.');
  const params: GbmParams = { ...DEFAULT_GBM_PARAMS, ...overrides };
  const n = X.length;
  let wPos = 0, wAll = 0;
  for (let i = 0; i < n; i++) { wAll += weights[i]; if (y[i] === 1) wPos += weights[i]; }
  const prior = Math.min(1 - 1e-6, Math.max(1e-6, wPos / Math.max(1e-9, wAll)));
  const baseScore = Math.log(prior / (1 - prior));
  const scores = new Float64Array(n).fill(baseScore);
  const g = new Float64Array(n), h = new Float64Array(n);
  const random = mulberry32(params.seed);
  const trees: GbmNode[][] = [];
  for (let t = 0; t < params.trees; t++) {
    for (let i = 0; i < n; i++) {
      const p = sigmoid(scores[i]);
      g[i] = (p - y[i]) * weights[i];
      h[i] = Math.max(1e-6, p * (1 - p)) * weights[i];
    }
    let rows: number[] = [];
    for (let i = 0; i < n; i++) if (params.subsample >= 1 || random() < params.subsample) rows.push(i);
    if (rows.length < 2 * params.minLeaf) rows = Array.from({ length: n }, (_, i) => i);
    const tree = buildTree(X, g, h, rows, params);
    trees.push(tree);
    for (let i = 0; i < n; i++) scores[i] += params.learningRate * treeValue(tree, X[i]);
    onTree?.(t + 1, params.trees);
  }
  return { featureCount: X[0].length, baseScore, learningRate: params.learningRate, trees };
}

/** Async training variant used by the UI worker so each tree yields to status polling and log rendering. */
export async function trainGbmAsync(
  X: number[][],
  y: Array<0 | 1>,
  weights: number[],
  overrides: Partial<GbmParams> = {},
  onTree?: (tree: number, totalTrees: number) => void,
): Promise<GbmModel> {
  if (!X.length) throw new Error('Gradient boosting needs at least one training sample.');
  const params: GbmParams = { ...DEFAULT_GBM_PARAMS, ...overrides };
  const n = X.length;
  let wPos = 0, wAll = 0;
  for (let i = 0; i < n; i++) { wAll += weights[i]; if (y[i] === 1) wPos += weights[i]; }
  const prior = Math.min(1 - 1e-6, Math.max(1e-6, wPos / Math.max(1e-9, wAll)));
  const baseScore = Math.log(prior / (1 - prior));
  const scores = new Float64Array(n).fill(baseScore);
  const g = new Float64Array(n), h = new Float64Array(n);
  const random = mulberry32(params.seed);
  const trees: GbmNode[][] = [];
  for (let t = 0; t < params.trees; t++) {
    for (let i = 0; i < n; i++) {
      const probability = sigmoid(scores[i]);
      g[i] = (probability - y[i]) * weights[i];
      h[i] = Math.max(1e-6, probability * (1 - probability)) * weights[i];
    }
    let rows: number[] = [];
    for (let i = 0; i < n; i++) if (params.subsample >= 1 || random() < params.subsample) rows.push(i);
    if (rows.length < 2 * params.minLeaf) rows = Array.from({ length: n }, (_, i) => i);
    const tree = buildTree(X, g, h, rows, params);
    trees.push(tree);
    for (let i = 0; i < n; i++) scores[i] += params.learningRate * treeValue(tree, X[i]);
    onTree?.(t + 1, params.trees);
    // Allow Express to serve the status poll between completed trees.
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
  return { featureCount: X[0].length, baseScore, learningRate: params.learningRate, trees };
}

/** Structural validation for models loaded from disk. */
export function isValidGbm(model: unknown, featureCount: number): model is GbmModel {
  const m = model as GbmModel | undefined;
  if (!m || m.featureCount !== featureCount || !Number.isFinite(m.baseScore) || !Number.isFinite(m.learningRate) || !Array.isArray(m.trees) || !m.trees.length) return false;
  return m.trees.every((tree) => Array.isArray(tree) && tree.length > 0 && tree.every((node) =>
    Number.isFinite(node.v) && Number.isFinite(node.t) && Number.isInteger(node.f) && node.f < featureCount &&
    (node.f < 0 || (Number.isInteger(node.l) && Number.isInteger(node.r) && node.l > 0 && node.r > 0 && node.l < tree.length && node.r < tree.length))));
}
