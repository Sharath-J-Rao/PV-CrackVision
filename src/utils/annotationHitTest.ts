import type { ModuleRegion, PixelBox, SolarCellBox } from '../types/inspection';

export interface ValidatedCellHit {
  region: ModuleRegion;
  cell: SolarCellBox;
}

export interface CalibratedCellRegion {
  id: string;
  label: string;
  /** Full detected module bounds, or the fallback grid bounds when no module was detected. */
  moduleBox: PixelBox;
  /** Bounds occupied by the calibrated cell lattice. */
  pixelBox: PixelBox;
  rows: number;
  columns: number;
  confidence: number;
  /** The exact pitch selected by the four-corner reference cell. */
  pitchWidth: number;
  pitchHeight: number;
  moduleId?: string;
  cells: SolarCellBox[];
}

/** Find the validated automatic cell under a native-image point. */
export function findValidatedCellAtPoint(
  regions: readonly ModuleRegion[],
  point: Pick<PixelBox, 'left' | 'top'>,
): ValidatedCellHit | null {
  for (const region of regions) {
    const cell = region.cellGrid?.cells.find(({ pixelBox }) =>
      point.left >= pixelBox.left && point.left < pixelBox.left + pixelBox.width &&
      point.top >= pixelBox.top && point.top < pixelBox.top + pixelBox.height
    );
    if (cell) return { region, cell };
  }
  return null;
}

/**
 * Find a cell from an explicit reference-cell calibration.
 * Newer calibrations are tested first and take precedence over the automatic grid in
 * their own footprint. This makes the manually drawn reference width/height the exact
 * selection pitch after calibration, while leaving automatic detection primary elsewhere.
 */
export function findCalibratedCellAtPoint(
  regions: readonly CalibratedCellRegion[],
  point: Pick<PixelBox, 'left' | 'top'>,
): { region: CalibratedCellRegion; cell: SolarCellBox } | null {
  for (let index = regions.length - 1; index >= 0; index--) {
    const region = regions[index];
    const cell = region.cells.find(({ pixelBox }) =>
      point.left >= pixelBox.left && point.left < pixelBox.left + pixelBox.width &&
      point.top >= pixelBox.top && point.top < pixelBox.top + pixelBox.height
    );
    if (cell) return { region, cell };
  }
  return null;
}

export type AnnotationCellHit =
  | { source: 'calibrated'; region: CalibratedCellRegion; cell: SolarCellBox }
  | { source: 'automatic'; region: ModuleRegion; cell: SolarCellBox };

/**
 * Resolve annotation clicks with explicit calibration precedence. A calibrated module
 * must not fall back to its old automatic rectangles outside the calibrated lattice: its
 * old overlay is hidden and its hit targets must match the exact manually drawn pitch.
 * All uncalibrated modules continue to use the automatic detector.
 */
export function findAnnotationCellAtPoint(
  automaticRegions: readonly ModuleRegion[],
  calibratedRegions: readonly CalibratedCellRegion[],
  point: Pick<PixelBox, 'left' | 'top'>,
): AnnotationCellHit | null {
  const calibrated = findCalibratedCellAtPoint(calibratedRegions, point);
  if (calibrated) return { source: 'calibrated', ...calibrated };

  const calibratedModuleIds = new Set(
    calibratedRegions.map((region) => region.moduleId).filter((id): id is string => Boolean(id)),
  );
  const remainingAutomatic = automaticRegions.filter((region) => !calibratedModuleIds.has(region.id));
  const automatic = findValidatedCellAtPoint(remainingAutomatic, point);
  return automatic ? { source: 'automatic', ...automatic } : null;
}
