/**
 * v1.4.11: frames that block the crack stage. A single-image run (explicitTargets) is blocked only by
 * its own targets; the all-images action is blocked by any uploaded frame without finished cell detection.
 */
export function framesBlockingCrackStage<T extends { id: string; imageSrc?: string; cellDetection?: { status?: string } | null }>(
  frames: T[],
  explicitTargets?: Array<{ id: string }>
): T[] {
  const scope = explicitTargets ? frames.filter((frame) => explicitTargets.some((target) => target.id === frame.id)) : frames;
  return scope.filter((frame) => frame.imageSrc && frame.cellDetection?.status !== 'completed');
}
