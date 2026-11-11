/** v1.4.10: module-perimeter ("four sides") cell priority helpers shared by training and inference. */
import type { PixelBox } from '../src/types/inspection.ts';

/** True when the cell lies in the outermost ring of its module (its edge is within ~0.9 cell of the module outline). */
export function isPerimeterCell(cell: PixelBox, module: PixelBox | undefined | null): boolean {
  if (!module || !(module.width > 0) || !(module.height > 0)) return false;
  const mx = cell.width * 0.9, my = cell.height * 0.9; // ring-2 cells sit >= 1 full cell inside the outline
  return cell.left - module.left < mx || (module.left + module.width) - (cell.left + cell.width) < mx
    || cell.top - module.top < my || (module.top + module.height) - (cell.top + cell.height) < my;
}

/**
 * v1.4.12.9: bounding box of a set of detected cell boxes (the outer rectangle of the real cell grid).
 * Why: the module REGION box is looser than the cell grid (it can include the frame and the gap to the next module row),
 * so a region-based ring put second-row cells and gap/frame artefacts into the "perimeter". Null when there are no valid boxes.
 */
export function cellGridBounds(cells: readonly PixelBox[]): PixelBox | null {
  let left = Infinity, top = Infinity, right = -Infinity, bottom = -Infinity;
  for (const c of cells) {
    if (![c.left, c.top, c.width, c.height].every(Number.isFinite) || !(c.width > 0) || !(c.height > 0)) continue; // malformed boxes are skipped, never thrown on
    left = Math.min(left, c.left); top = Math.min(top, c.top);
    right = Math.max(right, c.left + c.width); bottom = Math.max(bottom, c.top + c.height);
  }
  return right > left && bottom > top ? { left, top, width: right - left, height: bottom - top } : null;
}

/**
 * v1.4.12.9: true only for the outermost ring of the detected cell grid (cell outer side within half a cell of the grid
 * outline). The second ring sits about one full cell inside, so the 0.5 tolerance cleanly separates the two rings.
 */
export function isGridEdgeCell(cell: PixelBox, grid: PixelBox | undefined | null): boolean {
  if (!grid || !(grid.width > 0) || !(grid.height > 0)) return false;
  const mx = cell.width * 0.5, my = cell.height * 0.5;
  return cell.left - grid.left < mx || (grid.left + grid.width) - (cell.left + cell.width) < mx
    || cell.top - grid.top < my || (grid.top + grid.height) - (cell.top + cell.height) < my;
}
