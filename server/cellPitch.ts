/** Pure periodicity candidate generation for the PV cell seam lattice. */
export interface CellAxisCandidate {
  pitch: number;
  confidence: number;
}

/** Return several real periodicity peaks so the two axes can be resolved together. */
export function estimateCellAxisPitchCandidates(profile: Float32Array): CellAxisCandidate[] {
  const n = profile.length;
  if (n < 36) return [];

  const smooth = new Float32Array(n);
  let mean = 0;
  let maxValue = 0;
  for (let i = 0; i < n; i++) {
    const value = (
      profile[Math.max(0, i - 1)] +
      profile[i] +
      profile[Math.min(n - 1, i + 1)]
    ) / 3;
    smooth[i] = value;
    mean += value;
    if (value > maxValue) maxValue = value;
  }
  mean /= n;
  if (maxValue - mean < 0.025) return [];

  const centered = new Float32Array(n);
  for (let i = 0; i < n; i++) centered[i] = smooth[i] - mean;

  // Busbar/finger repeats are commonly 4–8 px in the working imagery. They are not
  // cell pitch. Keep a conservative minimum of 10 px / 2% of the local module span;
  // larger whole-cell seams (e.g. ~35 px) remain candidates for the joint X/Y fit.
  const minLag = Math.max(10, Math.round(n * 0.02));
  const maxLag = Math.min(Math.floor(n / 3), Math.round(n * 0.36));
  const correlations: number[] = new Array(maxLag + 1).fill(-1);
  let best = -1;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let dot = 0;
    let aa = 0;
    let bb = 0;
    for (let i = 0; i < n - lag; i++) {
      const a = centered[i];
      const b = centered[i + lag];
      dot += a * b;
      aa += a * a;
      bb += b * b;
    }
    const correlation = dot / (Math.sqrt(aa * bb) + 1e-8);
    correlations[lag] = correlation;
    if (correlation > best) best = correlation;
  }
  if (best < 0.12) return [];

  const peaks: CellAxisCandidate[] = [];
  for (let lag = minLag + 1; lag < maxLag; lag++) {
    const score = correlations[lag];
    if (
      score >= best * 0.52 &&
      score >= correlations[lag - 1] &&
      score >= correlations[lag + 1]
    ) {
      peaks.push({ pitch: lag, confidence: Math.max(0, Math.min(1, score)) });
    }
  }

  // Keep distinct local maxima; a true cell pitch and its harmonic (2x, 3x) remain
  // available for joint selection, while adjacent one-pixel maxima are collapsed.
  peaks.sort((a, b) => b.confidence - a.confidence);
  const selected: CellAxisCandidate[] = [];
  for (const candidate of peaks) {
    if (selected.every((existing) => Math.abs(existing.pitch - candidate.pitch) >= 3)) {
      selected.push(candidate);
    }
    if (selected.length >= 12) break;
  }
  return selected;
}
