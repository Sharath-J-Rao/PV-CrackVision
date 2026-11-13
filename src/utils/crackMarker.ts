/**
 * v1.4.14: marker colour rules for model-detected cracks (pure, so server tests and the canvas share one definition).
 *   outer ring of the module's cell grid -> red     (checked first: it is the usual false-positive zone and needs a second look)
 *   confidence below the exposed threshold -> yellow
 *   everything else                        -> magenta (confident interior crack)
 * Cracks the operator added by hand keep their own green style and are never recoloured.
 */
export type CrackMarkerKind = 'user' | 'outer' | 'low' | 'ok';

export const CRACK_MARKER_STYLE: Record<Exclude<CrackMarkerKind, 'user'>, { stroke: string; fill: string; tag: string }> = {
  outer: { stroke: '#ef4444', fill: 'rgba(239, 68, 68, 0.16)', tag: 'OUTER' },
  low: { stroke: '#facc15', fill: 'rgba(250, 204, 21, 0.14)', tag: 'LOW' },
  ok: { stroke: '#e879f9', fill: 'rgba(232, 121, 249, 0.12)', tag: '' },
};

/** `lowConfidencePercent` is 0-100 (default 90). A non-finite value disables the yellow state instead of colouring everything. */
export function crackMarkerKind(
  anomaly: { source?: 'model' | 'user'; outerRing?: boolean; confidence: number },
  lowConfidencePercent = 90,
): CrackMarkerKind {
  if (anomaly.source === 'user') return 'user';
  if (anomaly.outerRing) return 'outer';
  if (Number.isFinite(lowConfidencePercent) && anomaly.confidence * 100 < lowConfidencePercent) return 'low';
  return 'ok';
}
