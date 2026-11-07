import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Crosshair, Maximize2, Minus, Plus, Save, MousePointer2 } from 'lucide-react';
import { detectCellGrid } from '../utils/cellGrid';
import { findAnnotationCellAtPoint, type CalibratedCellRegion } from '../utils/annotationHitTest';
import { buildSelectableCellRegions } from '../utils/cellAnnotationTargets';
import { duplicateRepeat, estimateCellGaps, isCellBoxPlacementCommitKey, placeCellBoxesAt, snapDuplicateOffset, translateCellBoxes } from '../utils/cellBoxPlacement';
import type {
  DefectCategory,
  DetectedAnomaly,
  InspectionResult,
  PixelBox,
  TileDiagnostic,
  SolarCellBox,
} from '../types/inspection';

export interface CellBoxSaveContext { imageId: string; fileName: string; coordinateWidth: number; coordinateHeight: number }

export interface CellAnnotationInput {
  box: PixelBox;
  label: 0 | 1;
  moduleLabel: string;
  cellLocation: string;
  moduleBox: PixelBox;
}

interface InspectionCanvasProps {
  imageSrc: string;
  result: InspectionResult | null;
  isAnalyzing: boolean;
  selectedAnomalyId: string | null;
  onSelectAnomaly: (id: string | null) => void;
  focusedTile: TileDiagnostic | null;
  visibleCategories: Set<DefectCategory>;
  onSaveCellAnnotations: (items: CellAnnotationInput[]) => Promise<void>;
  onSaveCellBoxes: (boxes: PixelBox[], source: CellBoxSaveContext) => Promise<void>;
  advancedMode?: boolean;
  workflowMode?: 'cells' | 'cracks' | 'combined' | 'crackLabel';
}

const USER_STYLE = { stroke: '#22c55e', fill: 'rgba(34, 197, 94, 0.14)', glyph: '★' };

function getCategoryStyle(category: DefectCategory): {
  stroke: string;
  fill: string;
  glyph: string;
} {
  switch (category) {
    case 'spider_web_crack':
      return {
        stroke: '#ef4444',
        fill: 'rgba(239, 68, 68, 0.12)',
        glyph: '✖',
      };
    case 'impact_point':
      return {
        stroke: '#f59e0b',
        fill: 'rgba(245, 158, 11, 0.12)',
        glyph: '▲',
      };
    case 'linear_fracture':
      return {
        stroke: '#fb923c',
        fill: 'rgba(251, 146, 60, 0.12)',
        glyph: '◆',
      };
    case 'surface_residue':
      return {
        stroke: '#38bdf8',
        fill: 'rgba(56, 189, 248, 0.08)',
        glyph: '○',
      };
  }
}

export const InspectionCanvas: React.FC<InspectionCanvasProps> = ({
  imageSrc,
  result,
  isAnalyzing,
  selectedAnomalyId,
  onSelectAnomaly,
  focusedTile,
  visibleCategories,
  onSaveCellAnnotations,
  onSaveCellBoxes,
  advancedMode = false,
  workflowMode = 'combined',
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  // Distinguish a cell click from a canvas pan so a drag never labels a cell.
  const panMovedRef = useRef(false);
  const suppressNextCellBoxClickRef = useRef(false);
  // v1.4.7.1: duplicated = Alt/Ctrl+drag copy in progress; sourceIndices = the boxes it was copied from (restored if the drag goes nowhere).
  const movingCellBoxesRef = useRef<{ startX: number; startY: number; indices: number[]; boxes: PixelBox[]; moved: boolean; duplicated?: boolean; sourceIndices?: number[]; offset?: { x: number; y: number }; gaps?: { x: number; y: number } } | null>(null);
  const resizingCellBoxRef = useRef<{ index: number; anchorX: number; anchorY: number } | null>(null);

  const [naturalDim, setNaturalDim] = useState<{ w: number; h: number }>({
    w: 3664,
    h: 2748,
  });

  const [camera, setCamera] = useState<{ cx: number; cy: number; zoom: number }>({
    cx: 1832,
    cy: 1374,
    zoom: 1,
  });

  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isMovingCellBoxes, setIsMovingCellBoxes] = useState(false);
  const dragRef = useRef<{
    startClientX: number;
    startClientY: number;
    startCx: number;
    startCy: number;
  }>({
    startClientX: 0,
    startClientY: 0,
    startCx: 1832,
    startCy: 1374,
  });

  const [showGeoPoints, setShowGeoPoints] = useState<boolean>(true);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [showTileGrid, setShowTileGrid] = useState<boolean>(false);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);
  const [showCellEdgeMap, setShowCellEdgeMap] = useState<boolean>(false);
  const [showModules, setShowModules] = useState<boolean>(true);
  const [cursorCoords, setCursorCoords] = useState<{ x: number; y: number } | null>(null);
  const [imgError, setImgError] = useState<boolean>(false);
  const [draft, setDraft] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const [cellSelectMode, setCellSelectMode] = useState(false);
  const [cellBoxAnnotationMode, setCellBoxAnnotationMode] = useState(false);
  const [cellBoxReviewModeState, setCellBoxReviewMode] = useState(false);
  // v1.4.7.3 fix: box editing (a leftover from Cell Annotation / box selection) swallowed every click in the Crack
  // Annotation tab, so no cell could be labelled and Save stayed at (0). Box editing is never active while labelling cracks.
  const cellBoxReviewMode = cellBoxReviewModeState && workflowMode !== 'crackLabel';
  useEffect(() => {
    if (workflowMode === 'crackLabel') { setCellBoxReviewMode(false); setCellBoxAnnotationMode(false); setSelectedCellBoxIndices([]); setIsPlacingCellBoxes(false); }
  }, [workflowMode]);
  const [cellBoxDraft, setCellBoxDraft] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const [cellBoxes, setCellBoxes] = useState<PixelBox[]>([]);
  // Multi-select/copy buffer is scoped to the current image and stores image-pixel geometry.
  const [selectedCellBoxIndices, setSelectedCellBoxIndices] = useState<number[]>([]);
  const [copiedCellBoxes, setCopiedCellBoxes] = useState<PixelBox[]>([]);
  // v1.4.7.2: how many copies a Duplicate arrow makes; 0 = repeat until the image edge.
  const [duplicateCount, setDuplicateCount] = useState(0);
  const [isPlacingCellBoxes, setIsPlacingCellBoxes] = useState(false);
  const [pastePreviewPoint, setPastePreviewPoint] = useState<{ x: number; y: number } | null>(null);
  const [savingCellBoxes, setSavingCellBoxes] = useState(false);
  // v1.4.8.2: Alt+drag copy in the Crack Annotation tab (cell boxes copied here are only added once saved).
  const [crackCopyNote, setCrackCopyNote] = useState('');
  const baseCellCount = result?.cellDetectionBoxes?.length ?? 0;
  const [cellCorners, setCellCorners] = useState<Array<{ x: number; y: number }>>([]);
  const [selectedModule, setSelectedModule] = useState<{ id: string; label: string; pixelBox: PixelBox } | null>(null);
  const [grid, setGrid] = useState<{ rows: number; columns: number; confidence: number; pixelBox: PixelBox } | null>(null);
  const [selectedCells, setSelectedCells] = useState<SolarCellBox[]>([]);
  // Explicit reference-cell calibrations are local pitch overrides. The automatic
  // server grid remains primary until a particular region is calibrated.
  const [manualCellRegions, setManualCellRegions] = useState<CalibratedCellRegion[]>([]);
  const [activeLabel, setActiveLabel] = useState<0 | 1>(1);
  const [pendingCells, setPendingCells] = useState<Record<string, CellAnnotationInput>>({});
  const [savingCells, setSavingCells] = useState(false);
  // v1.4.7.0: when on, every cell of a module that has >=1 positive (cracked) cell and is not otherwise labelled is saved as negative (clean).
  const [autoNegativeRest, setAutoNegativeRest] = useState(true);
  const [gridError, setGridError] = useState('');
  const hasResult = Boolean(result);
  // YOLO review and crack labeling are available in the normal workspace as soon
  // as a YOLO inference result exists; Advanced Mode remains for calibration/diagnostics.
  const showCellAnnotationTools = workflowMode !== 'cracks' && (workflowMode === 'cells' || advancedMode || result?.cellDetectionSource === 'yolo11n');
  // Crack-label tab: cells are the fixed base; box drawing/editing belongs to the Cell Annotation tab.
  // A cell is "cracked" when any crack finding's origin lies inside it (colour-coded red).
  const crackedBox = useCallback((box: PixelBox) => (result?.anomalies ?? []).some((a) => a.pointPx.x >= box.left && a.pointPx.x <= box.left + box.width && a.pointPx.y >= box.top && a.pointPx.y <= box.top + box.height), [result?.anomalies]);
  const showBoxTools = showCellAnnotationTools && workflowMode !== 'crackLabel';
  const commitCopiedCellBoxes = useCallback((point: { x: number; y: number }) => {
    if (!copiedCellBoxes.length) return;
    const pasted = placeCellBoxesAt(copiedCellBoxes, point, { width: naturalDim.w, height: naturalDim.h });
    if (!pasted.length) return;
    const firstIndex = cellBoxes.length;
    setCellBoxes((previous) => [...previous, ...pasted]);
    setSelectedCellBoxIndices(pasted.map((_, index) => firstIndex + index));
    suppressNextCellBoxClickRef.current = false;
    setIsPlacingCellBoxes(false);
    setPastePreviewPoint(null);
  }, [copiedCellBoxes, naturalDim, cellBoxes.length]);
  // Direct YOLO boxes must also be annotation hit targets. Previously they were drawn
  // as a pointer-events:none visual overlay, while click selection only queried the
  // four-or-more-cell module grid. A visible YOLO detection could therefore be impossible
  // to select for positive/negative crack-classifier feedback, especially on sparse runs.
  const selectableRegions = useMemo(
    () => buildSelectableCellRegions(
      result?.cellDetectionSource === 'yolo11n' ? [] : (result?.moduleRegions ?? []),
      result?.cellDetectionSource === 'yolo11n' ? (result.cellDetectionBoxes ?? []) : [],
      result?.imageWidth ?? naturalDim.w,
      result?.imageHeight ?? naturalDim.h,
    ),
    [result?.moduleRegions, result?.cellDetectionBoxes, result?.cellDetectionSource, result?.imageWidth, result?.imageHeight, naturalDim.w, naturalDim.h]
  );
  const autoCellRegions = useMemo(
    () => selectableRegions.filter((region) =>
      (!region.isFallback || region.cellGrid?.method === 'yolo-cell-detection') &&
      Boolean(region.cellGrid && (region.cellGrid.cells.length >= 4 || (region.cellGrid.method === 'yolo-cell-detection' && region.cellGrid.cells.length > 0)))
    ),
    [selectableRegions]
  );

  useEffect(() => {
    if (!isPlacingCellBoxes) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        setIsPlacingCellBoxes(false);
        setPastePreviewPoint(null);
        return;
      }
      // Space commits the cyan preview at the current pointer location. Capture
      // the key before the browser handles it as page-scroll or button activation.
      if (isCellBoxPlacementCommitKey(event.key, event.code) && !event.repeat) {
        const target = event.target;
        if (target instanceof HTMLElement && target.closest('input, textarea, select, [contenteditable="true"]')) return;
        event.preventDefault();
        event.stopPropagation();
        if (pastePreviewPoint) commitCopiedCellBoxes(pastePreviewPoint);
      }
    };
    window.addEventListener('keydown', onKeyDown, true);
    return () => window.removeEventListener('keydown', onKeyDown, true);
  }, [isPlacingCellBoxes, pastePreviewPoint, commitCopiedCellBoxes]);

  const clampCamera = useCallback(
    (cx: number, cy: number, zoom: number, w: number, h: number) => {
      const z = Math.max(1, Math.min(14, Number(zoom.toFixed(2))));
      if (z <= 1.01) {
        return { cx: w / 2, cy: h / 2, zoom: 1 };
      }
      const halfW = w / (2 * z);
      const halfH = h / (2 * z);
      return {
        cx: Math.max(halfW, Math.min(w - halfW, cx)),
        cy: Math.max(halfH, Math.min(h - halfH, cy)),
        zoom: z,
      };
    },
    []
  );

  useEffect(() => {
    setImgError(false);
    setCamera({ cx: naturalDim.w / 2, cy: naturalDim.h / 2, zoom: 1 });
    setSelectedModule(null);
    setGrid(null);
    setSelectedCells([]);
    setManualCellRegions([]);
    setPendingCells({});
    setCellSelectMode(false);
    setCellBoxAnnotationMode(false);
    setCellBoxReviewMode(false);
    setCellBoxDraft(null);
    setCellBoxes([]);
    setSelectedCellBoxIndices([]);
    setCopiedCellBoxes([]);
    setIsPlacingCellBoxes(false);
    setPastePreviewPoint(null);
    movingCellBoxesRef.current = null;
    resizingCellBoxRef.current = null;
    setIsMovingCellBoxes(false);
    suppressNextCellBoxClickRef.current = false;
    setCellCorners([]);
    setGridError('');
  }, [imageSrc]);

  // In YOLO mode, its detections are the authoritative cell boxes. Copy them into the
  // editable annotation layer (not a baked image/SVG overlay) so operators can select,
  // move, delete, duplicate and save the exact detector geometry.
  useEffect(() => {
    if (workflowMode !== 'cells' && result?.cellDetectionSource !== 'yolo11n') return;
    const detections = (result?.cellDetectionBoxes ?? []).map((box) => ({ ...box }));
    setCellBoxes(detections);
    setSelectedCellBoxIndices([]);
    setCellBoxReviewMode(true);
    setCellBoxAnnotationMode(false);
    setGridError(detections.length
      ? `YOLO supplied ${detections.length} editable cell boxes. Select and drag a box to reposition it, or delete/redraw it before saving.`
      : 'YOLO ran and returned no cell boxes. Classical auto-detection is disabled for this image; adjust the model or annotate manually.');
  }, [imageSrc, result?.cellDetectionSource, result?.cellDetectionBoxes, workflowMode]);

  // v1.4.8.2: in crack labelling, keep an editable copy of the detected cells so Alt+drag can add missed neighbours.
  // Keyed by image + count (not array identity) so unrelated re-renders never wipe copies that are still pending.
  useEffect(() => {
    if (workflowMode !== 'crackLabel') return;
    setCellBoxes((result?.cellDetectionBoxes ?? []).map((box) => ({ ...box })));
    setSelectedCellBoxIndices([]);
    setCrackCopyNote('');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workflowMode, result?.imageId, baseCellCount]);

  const selectDetectedRegion = useCallback((region: NonNullable<InspectionResult['moduleRegions'][number]>) => {
    if (!region.cellGrid || region.cellGrid.cells.length < 1) return;
    // Automatic detection is the default. An explicit reference-cell calibration only
    // replaces cell pitch/boxes for the detected module in which it was drawn.
    const calibrated = manualCellRegions.find((item) => item.moduleId === region.id);
    const cells = calibrated?.cells ?? region.cellGrid.cells;
    const activeId = calibrated?.id ?? region.id;
    const activeLabel = calibrated?.label ?? region.label;
    setSelectedModule({ id: activeId, label: activeLabel, pixelBox: region.pixelBox });
    setSelectedCells(cells);
    setGrid({
      rows: calibrated?.rows ?? region.cellGrid.rows,
      columns: calibrated?.columns ?? region.cellGrid.columns,
      confidence: calibrated?.confidence ?? region.cellGrid.confidence,
      pixelBox: calibrated?.pixelBox ?? region.cellGrid.pixelBox,
    });
    setGridError(calibrated
      ? `${region.label}: calibrated ${cells.length} cells using the exact reference pitch (${calibrated.pitchWidth} × ${calibrated.pitchHeight} px). Automatic PV-region grouping remains active.`
      : `${region.cellGrid.method === 'yolo-cell-detection' ? 'YOLO directly detected' : 'Automatically validated'} ${cells.length} selectable cells in ${region.label} (${region.cellGrid.rows} × ${region.cellGrid.columns}). Click a detected cell to label it.`);
  }, [manualCellRegions]);

  // Primary path: automatically detected server-side PV surface regions and their validated
  // cell grids. The optional four-corner tool calibrates only a chosen region's cell pitch.
  useEffect(() => {
    if (autoCellRegions.length) {
      const selected = autoCellRegions.find((region) =>
        region.id === selectedModule?.id || manualCellRegions.some((item) => item.id === selectedModule?.id && item.moduleId === region.id)
      );
      if (selected) {
        selectDetectedRegion(selected);
      } else if (!selectedModule?.id.startsWith('manual-')) {
        selectDetectedRegion(autoCellRegions[0]);
      }
    } else if (hasResult && !selectedModule?.id.startsWith('manual-')) {
      setGridError('Automatic cell geometry was not validated for this image. Use the four-corner reference-cell calibration only for this region.');
    }
  }, [autoCellRegions, hasResult, imageSrc, selectedModule?.id, manualCellRegions, selectDetectedRegion]);

  const chooseReferenceCell = useCallback(async (referenceCell: PixelBox) => {
    setGridError('Calibrating the cell pitch from the four manually marked corners…');
    try {
      const estimate = await detectCellGrid(imageSrc, referenceCell, { width: naturalDim.w, height: naturalDim.h });
      const centerX = referenceCell.left + referenceCell.width / 2;
      const centerY = referenceCell.top + referenceCell.height / 2;
      const matchedAutoRegion = autoCellRegions.find((region) =>
        centerX >= region.pixelBox.left && centerX < region.pixelBox.left + region.pixelBox.width &&
        centerY >= region.pixelBox.top && centerY < region.pixelBox.top + region.pixelBox.height
      );
      const id = matchedAutoRegion
        ? `cal-${matchedAutoRegion.id}`
        : `manual-${Date.now()}-${Math.round(referenceCell.left)}-${Math.round(referenceCell.top)}`;
      const label = matchedAutoRegion?.label ?? `Manual fallback ${manualCellRegions.length + 1}`;
      const moduleBox = matchedAutoRegion?.pixelBox ?? estimate.pixelBox;
      const calibratedRegion: CalibratedCellRegion = {
        id,
        moduleId: matchedAutoRegion?.id,
        label,
        moduleBox,
        pixelBox: estimate.pixelBox,
        rows: estimate.rows,
        columns: estimate.columns,
        confidence: estimate.confidence,
        // These rounded dimensions are exactly the manually drawn reference-cell box.
        pitchWidth: Math.round(referenceCell.width),
        pitchHeight: Math.round(referenceCell.height),
        cells: estimate.cells,
      };

      // Recalibration replaces the previous grid for the same auto-detected module.
      // Old unsaved labels are discarded because their pixel rectangles may have changed.
      if (matchedAutoRegion) {
        // The prior automatic grid and calibrated grid can have different row/column
        // indexes and pixel rectangles. Drop unsaved labels from both identities so no
        // stale automatic labels remain hidden under the calibrated overlay.
        const stalePrefixes = [`${matchedAutoRegion.id}-`, `${id}-`];
        setPendingCells((previous) => Object.fromEntries(
          Object.entries(previous).filter(([key]) => !stalePrefixes.some((prefix) => key.startsWith(prefix)))
        ));
      }
      setManualCellRegions((previous) => [
        ...previous.filter((item) => matchedAutoRegion
          ? item.moduleId !== matchedAutoRegion.id
          : item.id !== id),
        calibratedRegion,
      ]);
      setSelectedModule({ id, label, pixelBox: moduleBox });
      setSelectedCells(estimate.cells);
      setGrid({
        rows: estimate.rows,
        columns: estimate.columns,
        confidence: estimate.confidence,
        pixelBox: estimate.pixelBox,
      });
      setGridError(`${label}: pitch locked to the exact drawn reference size (${calibratedRegion.pitchWidth} × ${calibratedRegion.pitchHeight} px per cell). ${estimate.cells.length} validated cells. These calibrated boxes take priority in this region; automatic detection remains primary everywhere else.`);
    } catch (error) {
      setGridError(error instanceof Error ? error.message : 'Reference-cell calibration could not establish a reliable cell grid.');
    }
  }, [imageSrc, naturalDim, autoCellRegions, manualCellRegions.length]);

  const toggleCell = (
    cell: SolarCellBox,
    module = selectedModule,
  ) => {
    if (!module) return;
    const key = `${module.id}-${cell.row}-${cell.column}`;
    setPendingCells((prev) => {
      if (prev[key]?.label === activeLabel) { const next = { ...prev }; delete next[key]; return next; }
      return {
        ...prev,
        [key]: {
          box: cell.pixelBox,
          label: activeLabel,
          moduleLabel: module.label,
          cellLocation: `R${cell.row}C${cell.column}`,
          moduleBox: module.pixelBox,
        },
      };
    });
  };

  const saveCells = async () => {
    const explicit = Object.values(pendingCells);
    if (!explicit.length || savingCells) return;
    // Auto-negatives: pending keys are `${moduleId}-${row}-${col}`; a module qualifies when it holds a positive label.
    const items: CellAnnotationInput[] = [...explicit];
    let autoNegatives = 0;
    if (autoNegativeRest) {
      const regions: Array<{ id: string; label: string; box: PixelBox; cells: SolarCellBox[] }> = [
        ...autoCellRegions.filter((region) => !manualCellRegions.some((calibration) => calibration.moduleId === region.id)).map((region) => ({ id: region.id, label: region.label, box: region.pixelBox, cells: region.cellGrid?.cells ?? [] })),
        ...manualCellRegions.map((region) => ({ id: region.id, label: region.label, box: region.moduleBox, cells: region.cells })),
      ];
      for (const region of regions) {
        if (!region.box) continue;
        const hasPositive = Object.entries(pendingCells).some(([key, item]) => key.startsWith(`${region.id}-`) && item.label === 1);
        if (!hasPositive) continue;
        for (const cell of region.cells) {
          if (pendingCells[`${region.id}-${cell.row}-${cell.column}`]) continue; // never override an explicit label
          items.push({ box: cell.pixelBox, label: 0, moduleLabel: region.label, cellLocation: `R${cell.row}C${cell.column}`, moduleBox: region.box });
          autoNegatives++;
        }
      }
    }
    setSavingCells(true);
    try {
      await onSaveCellAnnotations(items);
      setPendingCells({});
      setGridError(`Saved ${items.length} cell label(s) to the training feedback dataset${autoNegatives ? ` (${explicit.length} selected + ${autoNegatives} remaining cells auto-labelled negative)` : ''}.`);
    } catch (error) {
      setGridError(error instanceof Error ? error.message : 'Could not save the selected cells.');
    } finally {
      setSavingCells(false);
    }
  };

  useEffect(() => {
    if (result?.imageWidth && result?.imageHeight) {
      const w = result.imageWidth;
      const h = result.imageHeight;
      setNaturalDim({ w, h });
      setCamera((prev) =>
        prev.zoom === 1 ? { cx: w / 2, cy: h / 2, zoom: 1 } : clampCamera(prev.cx, prev.cy, prev.zoom, w, h)
      );
    }
  }, [result?.imageWidth, result?.imageHeight, clampCamera]);

  useEffect(() => {
    if (!selectedAnomalyId || !result) return;
    const target = result.anomalies.find((a) => a.id === selectedAnomalyId);
    if (!target) return;

    setCamera((prev) =>
      clampCamera(
        target.pointPx.x,
        target.pointPx.y,
        Math.max(prev.zoom, 3.6),
        result.imageWidth,
        result.imageHeight
      )
    );
  }, [selectedAnomalyId, result, clampCamera]);

  useEffect(() => {
    if (!focusedTile || !result) return;
    const targetCx = focusedTile.pixelBox.left + focusedTile.pixelBox.width / 2;
    const targetCy = focusedTile.pixelBox.top + focusedTile.pixelBox.height / 2;
    setShowTileGrid(true);
    setCamera(clampCamera(targetCx, targetCy, 2.5, result.imageWidth, result.imageHeight));
  }, [focusedTile, result, clampCamera]);

  const viewBoxRect = useMemo(() => {
    const vw = naturalDim.w / camera.zoom;
    const vh = naturalDim.h / camera.zoom;
    const vx = camera.cx - vw / 2;
    const vy = camera.cy - vh / 2;
    return { vx, vy, vw, vh };
  }, [naturalDim, camera]);

  const imageScroll = useMemo(() => {
    const viewW = naturalDim.w / camera.zoom;
    const viewH = naturalDim.h / camera.zoom;
    const maxLeft = Math.max(0, naturalDim.w - viewW);
    const maxTop = Math.max(0, naturalDim.h - viewH);
    return {
      x: maxLeft > 0 ? Math.round(((camera.cx - viewW / 2) / maxLeft) * 1000) : 0,
      y: maxTop > 0 ? Math.round(((camera.cy - viewH / 2) / maxTop) * 1000) : 0,
      enabled: camera.zoom > 1.01,
    };
  }, [naturalDim, camera]);

  const scrollImage = (axis: 'x' | 'y', value: number) => {
    setCamera((prev) => {
      const viewW = naturalDim.w / prev.zoom;
      const viewH = naturalDim.h / prev.zoom;
      const maxLeft = Math.max(0, naturalDim.w - viewW);
      const maxTop = Math.max(0, naturalDim.h - viewH);
      const nextCx = axis === 'x' ? (value / 1000) * maxLeft + viewW / 2 : prev.cx;
      const nextCy = axis === 'y' ? (value / 1000) * maxTop + viewH / 2 : prev.cy;
      return clampCamera(nextCx, nextCy, prev.zoom, naturalDim.w, naturalDim.h);
    });
  };

  const clientToSvgPoint = useCallback(
    (clientX: number, clientY: number): { x: number; y: number } | null => {
      const svg = svgRef.current;
      if (!svg) return null;
      const ctm = svg.getScreenCTM();
      if (!ctm) return null;
      const pt = svg.createSVGPoint();
      pt.x = clientX;
      pt.y = clientY;
      const transformed = pt.matrixTransform(ctm.inverse());
      return { x: transformed.x, y: transformed.y };
    },
    []
  );

  // Use a non-passive native wheel listener. React's delegated wheel handling can
  // otherwise allow the document/outer workspace to scroll while the image zooms.
  useEffect(() => {
    // v1.4.8.3: listen on the whole stage (not just the <svg>) so wheel events over overlays, labels or
    // the stage margin can never fall through and scroll the page while zooming.
    const stage = stageRef.current;
    if (!stage) return;
    const onWheel = (event: WheelEvent) => {
      if (event.cancelable) event.preventDefault();
      event.stopPropagation();
      const factor = event.deltaY < 0 ? 1.22 : 0.82;
      const svgPt = clientToSvgPoint(event.clientX, event.clientY);
      setCamera((prev) => {
        const nextZoom = Math.max(1, Math.min(14, prev.zoom * factor));
        if (!svgPt || nextZoom <= 1.01) {
          return { cx: naturalDim.w / 2, cy: naturalDim.h / 2, zoom: 1 };
        }
        const ratio = prev.zoom / nextZoom;
        const nextCx = svgPt.x + (prev.cx - svgPt.x) * ratio;
        const nextCy = svgPt.y + (prev.cy - svgPt.y) * ratio;
        return clampCamera(nextCx, nextCy, nextZoom, naturalDim.w, naturalDim.h);
      });
    };
    stage.addEventListener('wheel', onWheel, { passive: false });
    return () => stage.removeEventListener('wheel', onWheel);
  }, [clientToSvgPoint, clampCamera, naturalDim]);

  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    panMovedRef.current = false;
    if (isPlacingCellBoxes) return;

    // v1.4.8.2: Alt+drag copies a cell in Crack Annotation. Alt is REQUIRED, so plain clicks keep labelling cells exactly
    // as in v1.4.7.3 (general box-edit mode stays off in this tab; that was the v1.4.7.3 click-blocking bug).
    if (workflowMode === 'crackLabel' && e.altKey && cellBoxes.length) {
      const altPoint = clientToSvgPoint(e.clientX, e.clientY);
      const altHit = altPoint ? [...cellBoxes.keys()].reverse().find((index) => {
        const box = cellBoxes[index];
        return altPoint.x >= box.left && altPoint.x <= box.left + box.width && altPoint.y >= box.top && altPoint.y <= box.top + box.height;
      }) : undefined;
      if (altPoint && altHit !== undefined) {
        e.preventDefault();
        startCellBoxMove(selectedCellBoxIndices.includes(altHit) ? selectedCellBoxIndices : [altHit], altPoint, true);
        return;
      }
    }

    // In crack-label mode, a click on a detector target must not be captured by
    // canvas panning or by an overlapping manually drawn YOLO training rectangle.
    // Drawing/calibration modes remain explicit and take precedence when enabled.
    if (!cellBoxAnnotationMode && !cellSelectMode && e.target instanceof SVGElement &&
        e.target.closest('[data-cell-detection-hit="true"]')) return;

    // Direct manipulation: click/drag a box to move it. Shift-click toggles
    // boxes into a multi-selection; dragging any selected box moves the group.
    if (cellBoxReviewMode && cellBoxes.length) {
      const pt = clientToSvgPoint(e.clientX, e.clientY);
      const hitIndex = pt ? [...cellBoxes.keys()].reverse().find((index) => {
        const box = cellBoxes[index];
        return pt.x >= box.left && pt.x <= box.left + box.width && pt.y >= box.top && pt.y <= box.top + box.height;
      }) : undefined;
      if (hitIndex !== undefined) {
        if (e.shiftKey) {
          setSelectedCellBoxIndices((previous) => previous.includes(hitIndex)
            ? previous.filter((index) => index !== hitIndex)
            : [...previous, hitIndex]);
          suppressNextCellBoxClickRef.current = true;
          return;
        }
        const indices = selectedCellBoxIndices.includes(hitIndex) ? selectedCellBoxIndices : [hitIndex];
        startCellBoxMove(indices, pt!, e.altKey || e.ctrlKey || e.metaKey);
        return;
      }
    }

    if (cellBoxAnnotationMode) {
      const pt = clientToSvgPoint(e.clientX, e.clientY);
      if (!pt || pt.x < 0 || pt.y < 0 || pt.x > naturalDim.w || pt.y > naturalDim.h) return;
      setCellBoxDraft({ x0: pt.x, y0: pt.y, x1: pt.x, y1: pt.y });
      setIsDragging(false);
      return;
    }
    if (cellSelectMode) {
      const pt = clientToSvgPoint(e.clientX, e.clientY);
      if (!pt || pt.x < 0 || pt.y < 0 || pt.x > naturalDim.w || pt.y > naturalDim.h) return;
      const next = [...cellCorners, pt];
      if (next.length === 4) {
        const xs = next.map((point) => point.x);
        const ys = next.map((point) => point.y);
        const left = Math.max(0, Math.min(...xs));
        const top = Math.max(0, Math.min(...ys));
        const right = Math.min(naturalDim.w, Math.max(...xs));
        const bottom = Math.min(naturalDim.h, Math.max(...ys));
        const box = { left: Math.round(left), top: Math.round(top), width: Math.round(right - left), height: Math.round(bottom - top) };
        if (box.width >= 8 && box.height >= 8) void chooseReferenceCell(box);
        setCellCorners([]);
        setCellSelectMode(false);
      } else {
        setCellCorners(next);
      }
      return;
    }
    setIsDragging(true);
    dragRef.current = {
      startClientX: e.clientX,
      startClientY: e.clientY,
      startCx: camera.cx,
      startCy: camera.cy,
    };
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const pt = clientToSvgPoint(e.clientX, e.clientY);
    if (pt && pt.x >= 0 && pt.x <= naturalDim.w && pt.y >= 0 && pt.y <= naturalDim.h) {
      setCursorCoords({ x: Math.round(pt.x), y: Math.round(pt.y) });
    } else {
      setCursorCoords(null);
    }

    if (isPlacingCellBoxes) {
      setPastePreviewPoint(pt && pt.x >= 0 && pt.x <= naturalDim.w && pt.y >= 0 && pt.y <= naturalDim.h ? pt : null);
      return;
    }

    const resizing = resizingCellBoxRef.current;
    if (resizing && pt) {
      const left = Math.max(0, Math.min(resizing.anchorX, pt.x));
      const top = Math.max(0, Math.min(resizing.anchorY, pt.y));
      const right = Math.min(naturalDim.w, Math.max(resizing.anchorX, pt.x));
      const bottom = Math.min(naturalDim.h, Math.max(resizing.anchorY, pt.y));
      if (right - left >= 8 && bottom - top >= 8) {
        setCellBoxes((previous) => previous.map((box, index) => index === resizing.index
          ? { left: Math.round(left), top: Math.round(top), width: Math.round(right - left), height: Math.round(bottom - top) }
          : box));
      }
      return;
    }

    const moving = movingCellBoxesRef.current;
    if (moving && pt) {
      const dx = pt.x - moving.startX;
      const dy = pt.y - moving.startY;
      if (Math.hypot(dx, dy) > 2) moving.moved = true;
      if (moving.moved) {
        // Copies snap to one cell pitch per axis; plain moves stay free.
        const delta = moving.duplicated ? snapDuplicateOffset({ x: dx, y: dy }, moving.boxes, moving.gaps ?? { x: 0, y: 0 }) : { x: dx, y: dy };
        moving.offset = delta;
        const translated = translateCellBoxes(moving.boxes, delta, { width: naturalDim.w, height: naturalDim.h });
        setCellBoxes((previous) => {
          const next = [...previous];
          moving.indices.forEach((index, itemIndex) => { next[index] = translated[itemIndex]; });
          return next;
        });
      }
      return;
    }

    if (cellBoxDraft && pt) {
      setCellBoxDraft({ ...cellBoxDraft, x1: pt.x, y1: pt.y });
      return;
    }
    if (draft && pt) {
      setDraft({ ...draft, x1: pt.x, y1: pt.y });
      return;
    }
    if (!isDragging || !svgRef.current) return;
    // Require an actual pointer movement before treating the gesture as a pan.
    if (Math.hypot(e.clientX - dragRef.current.startClientX, e.clientY - dragRef.current.startClientY) > 4) {
      panMovedRef.current = true;
    }
    const rect = svgRef.current.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const scalePx = Math.max(viewBoxRect.vw / rect.width, viewBoxRect.vh / rect.height);
    const dx = (e.clientX - dragRef.current.startClientX) * scalePx;
    const dy = (e.clientY - dragRef.current.startClientY) * scalePx;

    setCamera((prev) =>
      clampCamera(
        dragRef.current.startCx - dx,
        dragRef.current.startCy - dy,
        prev.zoom,
        naturalDim.w,
        naturalDim.h
      )
    );
  };

  // Calibrated cell rectangles win only inside their explicit footprint, because their
  // pitch is the exact size drawn by the operator. The automatic grid remains the primary
  // hit-test for all other modules and uncalibrated cells.
  const applyAnnotationCellHit = (hit: ReturnType<typeof findAnnotationCellAtPoint>) => {
    if (!hit) return;
    if (hit.source === 'calibrated') {
      const { region, cell } = hit;
      const module = { id: region.id, label: region.label, pixelBox: region.moduleBox };
      setSelectedModule(module);
      setSelectedCells(region.cells);
      setGrid({ rows: region.rows, columns: region.columns, confidence: region.confidence, pixelBox: region.pixelBox });
      toggleCell(cell, module);
      return;
    }
    const { region, cell } = hit;
    const module = { id: region.id, label: region.label, pixelBox: region.pixelBox };
    selectDetectedRegion(region);
    toggleCell(cell, module);
  };

  // Direct hit target for each visible YOLO box. This bypasses reliance on SVG event
  // bubbling/canvas-wide coordinate hit-testing, so a detected cell can be selected
  // directly even when crack markers or other overlays intersect the same area.
  const handleDetectedCellClick = (event: React.MouseEvent<SVGRectElement>, box: PixelBox) => {
    if (cellSelectMode || cellBoxAnnotationMode || cellBoxDraft || draft || panMovedRef.current) return;
    // Match the clicked SVG element to the exact detector box first. This guarantees
    // the saved classifier crop is the same cell the user clicked, even if manual
    // pitch calibration or another overlay also covers this location.
    const exactHit = autoCellRegions.flatMap((region) =>
      (region.cellGrid?.cells ?? []).map((cell) => ({ region, cell }))
    ).find(({ cell }) =>
      Math.abs(cell.pixelBox.left - box.left) < 0.5 &&
      Math.abs(cell.pixelBox.top - box.top) < 0.5 &&
      Math.abs(cell.pixelBox.width - box.width) < 0.5 &&
      Math.abs(cell.pixelBox.height - box.height) < 0.5
    );
    if (exactHit) {
      event.preventDefault();
      event.stopPropagation();
      const { region, cell } = exactHit;
      const module = { id: region.id, label: region.label, pixelBox: region.pixelBox };
      setSelectedModule(module);
      setSelectedCells(region.cellGrid!.cells);
      setGrid({
        rows: region.cellGrid!.rows,
        columns: region.cellGrid!.columns,
        confidence: region.cellGrid!.confidence,
        pixelBox: region.cellGrid!.pixelBox,
      });
      toggleCell(cell, module);
      return;
    }
    const point = { left: box.left + box.width / 2, top: box.top + box.height / 2 };
    const hit = findAnnotationCellAtPoint(autoCellRegions, manualCellRegions, point);
    if (!hit) return;
    event.preventDefault();
    event.stopPropagation();
    applyAnnotationCellHit(hit);
  };

  const handleCanvasClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (cellBoxReviewMode) {
      suppressNextCellBoxClickRef.current = false;
      panMovedRef.current = false;
      return;
    }
    if (suppressNextCellBoxClickRef.current) {
      suppressNextCellBoxClickRef.current = false;
      panMovedRef.current = false;
      return;
    }
    if (isPlacingCellBoxes && copiedCellBoxes.length) {
      const pt = clientToSvgPoint(e.clientX, e.clientY);
      if (pt && pt.x >= 0 && pt.x <= naturalDim.w && pt.y >= 0 && pt.y <= naturalDim.h) commitCopiedCellBoxes(pt);
      panMovedRef.current = false;
      return;
    }
    if (cellSelectMode || cellBoxAnnotationMode || cellBoxDraft || draft || panMovedRef.current) {
      panMovedRef.current = false;
      return;
    }
    const pt = clientToSvgPoint(e.clientX, e.clientY);
    if (!pt) return;

    const hit = findAnnotationCellAtPoint(
      autoCellRegions,
      manualCellRegions,
      { left: pt.x, top: pt.y },
    );
    applyAnnotationCellHit(hit);
    panMovedRef.current = false;
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    if (resizingCellBoxRef.current) { resizingCellBoxRef.current = null; setIsMovingCellBoxes(false); return; }
    if (movingCellBoxesRef.current) {
      const finished = movingCellBoxesRef.current;
      movingCellBoxesRef.current = null;
      setIsMovingCellBoxes(false);
      // v1.4.8.2: never let an Alt-copy drag leave the click-suppress flag armed, or the next plain label click would be swallowed.
      if (workflowMode === 'crackLabel') window.setTimeout(() => { suppressNextCellBoxClickRef.current = false; }, 60);
      // An Alt/Ctrl copy that was never dragged (or snapped back onto the original) would leave exact duplicates: discard it.
      if (finished.duplicated && (!finished.moved || Math.hypot(finished.offset?.x ?? 0, finished.offset?.y ?? 0) < 1)) {
        const discard = new Set(finished.indices);
        setCellBoxes((previous) => previous.filter((_, index) => !discard.has(index)));
        setSelectedCellBoxIndices(finished.sourceIndices ?? []);
      }
      return;
    }
    if (cellBoxDraft) {
      const left = Math.min(cellBoxDraft.x0, cellBoxDraft.x1);
      const top = Math.min(cellBoxDraft.y0, cellBoxDraft.y1);
      const width = Math.abs(cellBoxDraft.x1 - cellBoxDraft.x0);
      const height = Math.abs(cellBoxDraft.y1 - cellBoxDraft.y0);
      if (width >= 8 && height >= 8) { setCellBoxes((previous) => [...previous, { left: Math.round(left), top: Math.round(top), width: Math.round(width), height: Math.round(height) }]); setCellBoxReviewMode(true); }
      setCellBoxDraft(null);
      return;
    }
    if (draft) setDraft(null);
  };


  // v1.4.7.1: single entry point for starting a box move. With duplicate=true (Alt/Ctrl+drag) the selection is copied
  // first and the COPIES are dragged, snapped to the cell pitch, so a column/row can be repeated in one gesture.
  const startCellBoxMove = (indices: number[], point: { x: number; y: number }, duplicate: boolean) => {
    const valid = indices.filter((item) => Boolean(cellBoxes[item]));
    if (!valid.length) return;
    if (duplicate) {
      const copies = valid.map((item) => ({ ...cellBoxes[item] }));
      const first = cellBoxes.length;
      const copyIndices = copies.map((_, offset) => first + offset);
      setCellBoxes((previous) => [...previous, ...copies.map((box) => ({ ...box }))]);
      setSelectedCellBoxIndices(copyIndices);
      movingCellBoxesRef.current = { startX: point.x, startY: point.y, indices: copyIndices, boxes: copies, moved: false, duplicated: true, sourceIndices: valid, offset: { x: 0, y: 0 }, gaps: estimateCellGaps(cellBoxes) };
    } else {
      setSelectedCellBoxIndices(valid);
      movingCellBoxesRef.current = { startX: point.x, startY: point.y, indices: valid, boxes: valid.map((item) => ({ ...cellBoxes[item] })), moved: false };
    }
    suppressNextCellBoxClickRef.current = true;
    setIsDragging(false);
    setIsMovingCellBoxes(true);
  };

  // One click repeats the selection in a direction (duplicateCount copies, or to the image edge when 0), skipping slots
  // that already hold a box. The LAST copy becomes the selection so another click continues from there.
  const duplicateSelectedStep = (direction: 'left' | 'right' | 'up' | 'down') => {
    const source = selectedCellBoxIndices.map((index) => cellBoxes[index]).filter(Boolean);
    if (!source.length) return;
    const copies = duplicateRepeat(source, direction, estimateCellGaps(cellBoxes), { width: naturalDim.w, height: naturalDim.h }, duplicateCount, cellBoxes);
    if (!copies.length) { setGridError('Nothing to add: no room in that direction, or those positions already have boxes.'); return; }
    const first = cellBoxes.length;
    setCellBoxes((previous) => [...previous, ...copies]);
    setSelectedCellBoxIndices(copies.slice(-source.length).map((_, offset) => first + copies.length - source.length + offset));
    setCellBoxReviewMode(true);
    setGridError(`Added ${copies.length} cell box(es).`);
  };

  const copySelectedCellBoxes = () => {
    const selected = selectedCellBoxIndices.map((index) => cellBoxes[index]).filter(Boolean);
    if (selected.length) setCopiedCellBoxes(selected.map((box) => ({ ...box })));
  };

  const pasteCellBoxes = () => {
    if (!copiedCellBoxes.length) return;
    suppressNextCellBoxClickRef.current = false;
    // Paste enters an explicit placement mode. The user positions the preview
    // over the destination cell and clicks to commit it (Escape cancels).
    setIsPlacingCellBoxes(true);
    setPastePreviewPoint(null);
    setCellBoxAnnotationMode(false);
    setCellSelectMode(false);
  };

  const deleteSelectedCellBoxes = () => {
    if (!selectedCellBoxIndices.length) return;
    const remove = new Set(selectedCellBoxIndices);
    const kept = cellBoxes.filter((_, index) => !remove.has(index));
    setCellBoxes(kept);
    setSelectedCellBoxIndices([]);
    setGridError(`Removed ${remove.size} selected detection(s) from this image. Save reviewed boxes to keep these deletions in the training dataset.`);
  };

  const beginCellBoxEditDrag = (index: number, event: React.MouseEvent<SVGRectElement>) => {
    if (!cellBoxReviewMode) return;
    event.preventDefault();
    event.stopPropagation();
    panMovedRef.current = false;
    const point = clientToSvgPoint(event.clientX, event.clientY);
    if (!point) return;
    if (event.shiftKey) {
      setSelectedCellBoxIndices((previous) => previous.includes(index)
        ? previous.filter((item) => item !== index)
        : [...previous, index]);
      suppressNextCellBoxClickRef.current = true;
      return;
    }
    const indices = selectedCellBoxIndices.includes(index) ? selectedCellBoxIndices : [index];
    startCellBoxMove(indices, point, event.altKey || event.ctrlKey || event.metaKey);
  };

  const beginCellBoxResize = (index: number, corner: 'nw' | 'ne' | 'sw' | 'se', event: React.MouseEvent<SVGCircleElement>) => {
    if (!cellBoxReviewMode || selectedCellBoxIndices.length !== 1 || selectedCellBoxIndices[0] !== index) return;
    event.preventDefault();
    event.stopPropagation();
    const box = cellBoxes[index];
    if (!box) return;
    const anchorX = corner.includes('w') ? box.left + box.width : box.left;
    const anchorY = corner.includes('n') ? box.top + box.height : box.top;
    resizingCellBoxRef.current = { index, anchorX, anchorY };
    movingCellBoxesRef.current = null;
    setIsDragging(false);
    setIsMovingCellBoxes(true);
  };

  useEffect(() => {
    if (!cellBoxReviewMode || !selectedCellBoxIndices.length) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Delete' && event.key !== 'Backspace') return;
      const target = event.target;
      if (target instanceof HTMLElement && target.closest('input, textarea, select, [contenteditable="true"]')) return;
      event.preventDefault();
      event.stopPropagation();
      deleteSelectedCellBoxes();
    };
    window.addEventListener('keydown', onKeyDown, true);
    return () => window.removeEventListener('keydown', onKeyDown, true);
  }, [cellBoxReviewMode, selectedCellBoxIndices, cellBoxes]);

  const saveCellBoxes = async () => {
    if (!cellBoxes.length || savingCellBoxes) return;
    setSavingCellBoxes(true);
    try {
      if (!result?.imageId || !result.fileName || !result.imageWidth || !result.imageHeight) throw new Error('This frame has no completed inference identity/dimensions. Re-run inspection before saving YOLO labels.');
      await onSaveCellBoxes(cellBoxes, { imageId: result.imageId, fileName: result.fileName, coordinateWidth: result.imageWidth, coordinateHeight: result.imageHeight });
      setSelectedCellBoxIndices([]);
      setCopiedCellBoxes([]);
      setCellBoxAnnotationMode(false);
      setGridError(`Saved ${cellBoxes.length} reviewed cell box(es). This complete set replaces the previous labels for this image; removed false detections will stay removed. Open Training Centre and select Train cell model when the dataset is ready.`);
    } catch (error) { setGridError(error instanceof Error ? error.message : 'Could not save cell boxes.'); }
    finally { setSavingCellBoxes(false); }
  };

  const pendingCrackCopies = workflowMode === 'crackLabel' ? Math.max(0, cellBoxes.length - baseCellCount) : 0;
  const saveCrackCopies = async () => {
    if (!pendingCrackCopies || savingCellBoxes) return;
    setSavingCellBoxes(true); setCrackCopyNote('');
    try {
      if (!result?.imageId || !result.fileName || !result.imageWidth || !result.imageHeight) throw new Error('This frame has no completed inference identity/dimensions. Re-run inspection before saving cell boxes.');
      await onSaveCellBoxes(cellBoxes, { imageId: result.imageId, fileName: result.fileName, coordinateWidth: result.imageWidth, coordinateHeight: result.imageHeight });
      setSelectedCellBoxIndices([]);
    } catch (error) { setCrackCopyNote(error instanceof Error ? error.message : 'Could not save cell boxes.'); }
    finally { setSavingCellBoxes(false); }
  };
  const discardCrackCopies = () => {
    setCellBoxes((result?.cellDetectionBoxes ?? []).map((box) => ({ ...box })));
    setSelectedCellBoxIndices([]); setCrackCopyNote('');
  };

  const displayedAnomalies: DetectedAnomaly[] = (result?.anomalies || []).filter((a) =>
    visibleCategories.has(a.category)
  );

  const baseUnit = Math.max(naturalDim.w, naturalDim.h) / (900 * camera.zoom);

  return (
    <div className="flex flex-col h-full min-h-0 bg-[#07090e] border border-[#1e2636] rounded-lg overflow-hidden select-none">
      {imageSrc && <img
        src={imageSrc}
        alt="Photovoltaic cell inspection canvas"
        referrerPolicy="no-referrer"
        onLoad={(e) => {
          const el = e.currentTarget;
          if (el.naturalWidth && el.naturalHeight && !result) {
            setNaturalDim({ w: el.naturalWidth, h: el.naturalHeight });
            setCamera({
              cx: el.naturalWidth / 2,
              cy: el.naturalHeight / 2,
              zoom: 1,
            });
          }
        }}
        onError={() => setImgError(true)}
        className="sr-only"
      />}

      {/* Canvas Overlay Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-3.5 py-2 bg-[#0f141f] border-b border-[#1e2636]">
        <div className="flex items-center gap-1 p-0.5 bg-[#07090e] border border-[#1e2636] rounded-md overflow-x-auto">
          {advancedMode && <>
          <button
            type="button"
            onClick={() => setShowGeoPoints((v) => !v)}
            className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
              showGeoPoints
                ? 'bg-[#1a2334] text-cyan-300'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Crack Origins ({displayedAnomalies.length})
          </button>
          <button
            type="button"
            onClick={() => setShowBoxes((v) => !v)}
            className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
              showBoxes
                ? 'bg-[#1a2334] text-slate-200'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Fracture Bounds
          </button>
          <button
            type="button"
            onClick={() => setShowHeatmap((v) => !v)}
            className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
              showHeatmap
                ? 'bg-[#1a2334] text-amber-300'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Crack Heatmap
          </button>
          <button
            type="button"
            onClick={() => setShowCellEdgeMap((v) => !v)}
            className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
              showCellEdgeMap
                ? 'bg-[#1a2334] text-orange-300'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Solar Cell Edge Map
          </button>
          <button
            type="button"
            onClick={() => setShowTileGrid((v) => !v)}
            className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
              showTileGrid
                ? 'bg-[#1a2334] text-cyan-300'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Inspection Tiles ({result?.pipelineSummary.vlmInspectedTiles ?? 0})
          </button>
          <button
            type="button"
            onClick={() => setShowModules((v) => !v)}
            className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
              showModules
                ? 'bg-[#1a2334] text-slate-200'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            PV-region and cell-grid diagnostics
          </button>
          </>}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {advancedMode && autoCellRegions.length > 1 && (
            <select
              aria-label="Select automatically detected PV region"
              value={autoCellRegions.find((region) => region.id === selectedModule?.id || manualCellRegions.some((item) => item.id === selectedModule?.id && item.moduleId === region.id))?.id ?? autoCellRegions[0].id}
              onChange={(event) => {
                const region = autoCellRegions.find((candidate) => candidate.id === event.target.value);
                if (region) selectDetectedRegion(region);
              }}
              className="max-w-[220px] px-2 py-1 text-xs bg-[#07090e] border border-[#1e2636] rounded-md text-slate-200"
            >
              {autoCellRegions.map((region, index) => <option key={region.id} value={region.id}>{region.label || `PV region ${index + 1}`} · {region.cellGrid?.cells.length ?? 0} cells</option>)}
            </select>
          )}
          {workflowMode === 'crackLabel' && imageSrc && (
            <span className="inline-flex flex-wrap items-center gap-2 rounded border border-amber-700/60 bg-amber-950/30 px-2 py-1 text-[10px] text-amber-200" title="Hold Alt and drag a cell onto a neighbouring position to add a missed cell. Copies snap to the cell pitch. Save them to turn them into real cells you can then label.">
              <span>Missed a cell? Hold <b>Alt</b> and drag a cell to copy it.</span>
              {pendingCrackCopies > 0 && (
                <>
                  <b>{pendingCrackCopies} new cell box(es)</b>
                  <button type="button" disabled={savingCellBoxes} onClick={() => void saveCrackCopies()} className="px-2.5 py-1 text-xs font-semibold text-slate-950 bg-amber-300 rounded disabled:opacity-40">{savingCellBoxes ? 'Saving…' : 'Save cell boxes'}</button>
                  <button type="button" disabled={savingCellBoxes} onClick={discardCrackCopies} className="px-2 py-1 text-xs text-slate-200 border border-slate-600 rounded disabled:opacity-40">Discard</button>
                </>
              )}
              {crackCopyNote && <span className="basis-full text-rose-300">{crackCopyNote}</span>}
            </span>
          )}
          {showBoxTools && imageSrc && (
            <button type="button" onClick={() => { setCellBoxAnnotationMode((value) => !value); setCellBoxReviewMode(false); setCellBoxDraft(null); setCellSelectMode(false); }} title="Draw a tight rectangle around each individual PV cell to create direct YOLO cell-detection annotations." className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md border whitespace-nowrap shrink-0 ${cellBoxAnnotationMode ? 'bg-amber-500/20 border-amber-500/60 text-amber-200' : 'bg-[#07090e] border-[#1e2636] text-slate-400 hover:text-white'}`}>
              {cellBoxAnnotationMode ? 'Drawing cell boxes…' : 'Draw cell boxes'}
            </button>
          )}
          {workflowMode === 'cells' && result && result.cellDetectionSource !== 'yolo11n' && (
            <div className="basis-full rounded-md border border-amber-900/60 bg-amber-950/20 px-3 py-2 text-[11px] text-amber-100">
              <strong>Cell-only review mode:</strong> model source is {result.cellDetectionSource ?? 'not reported'}. You can still draw individual PV-cell boxes manually, then save the complete set for this image.
            </div>
          )}
          {showBoxTools && result?.cellDetectionSource === 'yolo11n' && (
            <div className="basis-full rounded-md border border-emerald-900/60 bg-emerald-950/20 px-3 py-2 text-[11px] text-emerald-100">
              <strong>Cell detector ran:</strong> {cellBoxes.length} visible candidate box(es). Yellow dashed boxes are model detections; cyan boxes are selected. Review and correct these boxes before training again.
              <details className="mt-2 rounded border border-emerald-900/60 bg-black/10 p-2"><summary className="cursor-pointer font-semibold text-emerald-100">How to review and train this image</summary><ol className="mt-2 list-decimal pl-5 space-y-1 text-slate-300"><li>Yellow dashed rectangles are YOLO's proposed individual PV-cell boxes on the image currently shown.</li><li>Choose <strong>Edit / remove detections</strong>. Click a box to select it; drag it to move it; drag a cyan corner to resize it. Shift-click selects multiple boxes. Use Delete/Backspace or Delete selected to remove false boxes.</li><li>Use <strong>Draw cell boxes</strong> to add missing cells. Draw one tight rectangle per visible PV cell; do not draw a whole module rectangle. For YOLO training, annotate all visible cells in this image, not only cracked cells.</li><li>Click <strong>Save reviewed boxes</strong>. The exact inspected image and complete corrected box set are added/replaced in the YOLO localization dataset. Then open Training Centre and retrain; the new image is included alongside earlier reviewed images.</li><li>Use <strong>Label cracked / clean cells</strong> separately to train the crack classifier. Positive/negative crack labels do not train YOLO cell localization.</li></ol><p className="mt-2 text-amber-200">Do not save a PV image with no boxes as a background negative. Background negatives must contain no visible PV cells.</p></details>
              {result.cellDetectionDiagnostics && <div className="mt-2 grid grid-cols-2 lg:grid-cols-4 gap-1.5 text-[10px] text-slate-300"><span>Image: {result.fileName} · {result.cellDetectionDiagnostics.imageWidth}×{result.cellDetectionDiagnostics.imageHeight}</span><span>Model input: {result.cellDetectionDiagnostics.modelInputWidth}×{result.cellDetectionDiagnostics.modelInputHeight}</span><span>Inference passes: {result.cellDetectionDiagnostics.inferencePasses} ({result.cellDetectionDiagnostics.tilePasses} tiles)</span><span>Raw candidates: {result.cellDetectionDiagnostics.candidateCount} · output {result.cellDetectionDiagnostics.outputShape}</span><span>Top-K limited: {result.cellDetectionDiagnostics.candidatesLimited}</span><span>Above threshold: {result.cellDetectionDiagnostics.candidatesAboveThreshold}</span><span>Valid before merge: {result.cellDetectionDiagnostics.validBoxesBeforeMerge}</span><span>Kept boxes: {result.cellDetectionDiagnostics.boxesKept}</span><span>Peak confidence: {(result.cellDetectionDiagnostics.maximumScore * 100).toFixed(2)}% · threshold {(result.cellDetectionDiagnostics.confidenceThreshold * 100).toFixed(2)}%</span><span>Active model: {(result.cellDetectionDiagnostics.modelSizeBytes / (1024 * 1024)).toFixed(1)} MB</span><span>Model modified: {new Date(result.cellDetectionDiagnostics.modelModifiedAt).toLocaleString()}</span></div>}
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <button type="button" onClick={() => { setCellBoxReviewMode(true); setCellBoxAnnotationMode(false); setCellSelectMode(false); setSelectedModule(null); setSelectedCells([]); }} className={`rounded border px-2.5 py-1 text-xs ${cellBoxReviewMode ? 'border-cyan-500 bg-cyan-500/15 text-cyan-100' : 'border-slate-700 text-slate-300'}`}>Edit / remove cell detections</button>
                {workflowMode !== 'cells' && <button type="button" onClick={() => { setCellBoxReviewMode(false); setCellBoxAnnotationMode(false); setSelectedCellBoxIndices([]); }} className={`rounded border px-2.5 py-1 text-xs ${!cellBoxReviewMode ? 'border-cyan-500 bg-cyan-500/15 text-cyan-100' : 'border-slate-700 text-slate-300'}`}>Label cracked / clean cells</button>}
                <span className="text-[10px] text-slate-400">Click a box to select, drag to move, drag a cyan corner to resize, Shift-click to multi-select, then Delete/Backspace. Draw missing cells and save the complete set for this exact image.</span>
              </div>
            </div>
          )}
          {showBoxTools && cellBoxes.length > 0 && (
            <>
              <span className="text-[10px] text-amber-200">{cellBoxes.length} cell box(es) · {selectedCellBoxIndices.length} selected</span>
              <button type="button" onClick={() => { setCellBoxReviewMode(true); setSelectedCellBoxIndices(cellBoxes.map((_, index) => index)); }} className="px-2 py-1 text-xs text-slate-300 border border-[#293b5c] rounded">Select all</button>
              <span className="inline-flex items-center gap-1 rounded border border-amber-700/60 bg-amber-950/30 px-1.5 py-0.5" title="Repeat the selected cell(s) in that direction: to the image edge (0) or N copies. Existing boxes are not duplicated. Or hold Alt (Ctrl) and drag a selected cell.">
                <span className="text-[10px] text-amber-200">Duplicate ×</span>
                <input type="number" min={0} max={500} value={duplicateCount} onChange={(event) => setDuplicateCount(Math.max(0, Math.min(500, Math.round(Number(event.target.value) || 0))))} title="Number of copies. 0 = repeat all the way to the image edge." className="w-12 rounded border border-amber-800 bg-[#07090e] px-1 py-0.5 text-xs text-amber-100" />
                <span className="text-[10px] text-amber-300">{duplicateCount === 0 ? 'to edge' : 'copies'}</span>
                {([['left', '←'], ['up', '↑'], ['down', '↓'], ['right', '→']] as const).map(([direction, glyph]) => (
                  <button key={direction} type="button" disabled={!selectedCellBoxIndices.length} onClick={() => duplicateSelectedStep(direction)} className="px-1.5 py-0.5 text-xs text-amber-100 border border-amber-800 rounded disabled:opacity-40">{glyph}</button>
                ))}
              </span>
              <button type="button" disabled={!selectedCellBoxIndices.length} onClick={copySelectedCellBoxes} className="px-2 py-1 text-xs text-slate-300 border border-[#293b5c] rounded disabled:opacity-40">Copy selected</button>
              <button type="button" disabled={!copiedCellBoxes.length} onClick={pasteCellBoxes} className="px-2 py-1 text-xs text-slate-300 border border-[#293b5c] rounded disabled:opacity-40">Place copy {copiedCellBoxes.length ? `(${copiedCellBoxes.length})` : ''}</button>
              <button type="button" disabled={!selectedCellBoxIndices.length} onClick={deleteSelectedCellBoxes} className="px-2 py-1 text-xs text-rose-200 border border-rose-900 rounded disabled:opacity-40">Delete selected</button>
              <button type="button" onClick={() => { setCellBoxReviewMode(true); setCellBoxes((previous) => previous.slice(0, -1)); setSelectedCellBoxIndices([]); }} className="px-2 py-1 text-xs text-slate-300 border border-[#293b5c] rounded">Remove last box</button>
              <button type="button" disabled={savingCellBoxes} onClick={() => void saveCellBoxes()} className="px-2.5 py-1 text-xs font-semibold text-slate-950 bg-amber-300 rounded disabled:opacity-40">{savingCellBoxes ? 'Saving…' : 'Save reviewed boxes'}</button>
              <span className="w-full text-[10px] text-slate-500">Tip: select cells, then hold Alt (or Ctrl) and drag to copy them — the copy snaps to the next cell position — or use the Duplicate arrows (0 = fill to the image edge, or enter a number of copies). A saved set replaces the previous boxes for this image. To teach cracks: switch to Label cracked / clean cells, click the cells, choose Positive for cracked or Negative for clean, and Save.</span>
            </>
          )}
          {showBoxTools && result?.cellDetectionSource === 'yolo11n' && !cellBoxes.length && (
            <span className="basis-full text-[11px] text-amber-200">No YOLO boxes were returned for this frame. Use Draw cell boxes to add tight boxes around the visible PV cells, then Save reviewed boxes. The inference warning above includes the confidence diagnostics; classical detection is not silently substituted.</span>
          )}
          {advancedMode && result && (
            <button
              type="button"
              onClick={() => { setCellSelectMode((value) => !value); setCellCorners([]); setDraft(null); setCellBoxAnnotationMode(false); }}
              title="Optional cell-pitch calibration: click the four corners of one complete visible solar cell in clockwise order. Its drawn width and height become the exact cell size for this PV region; automatic cell detection remains primary elsewhere."
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md border whitespace-nowrap shrink-0 transition-colors ${cellSelectMode ? 'bg-cyan-500/20 border-cyan-500/60 text-cyan-200' : 'bg-[#07090e] border-[#1e2636] text-slate-400 hover:text-white'}`}
            >
              <MousePointer2 className="w-3.5 h-3.5" />
              {cellSelectMode ? `Cell corner ${cellCorners.length + 1}/4…` : 'Calibrate cell pitch (4 points)'}
            </button>
          )}
          {showCellAnnotationTools && workflowMode !== 'cells' && selectedModule && grid && (
            <>
              <button type="button" onClick={() => setActiveLabel(1)} className={`px-2.5 py-1 text-xs font-semibold rounded-md border ${activeLabel === 1 ? 'bg-emerald-500/20 border-emerald-500 text-emerald-200' : 'bg-[#07090e] border-[#1e2636] text-slate-400'}`}>+ Positive</button>
              <button type="button" onClick={() => setActiveLabel(0)} className={`px-2.5 py-1 text-xs font-semibold rounded-md border ${activeLabel === 0 ? 'bg-rose-500/20 border-rose-500 text-rose-200' : 'bg-[#07090e] border-[#1e2636] text-slate-400'}`}>− Negative</button>
              <span className="text-[10px] text-slate-400 whitespace-nowrap">Selectable cells</span>
              <span className="text-xs font-mono text-cyan-200">{selectedCells.length}</span>
              {result?.cellDetectionSource === 'yolo11n' && <span className="text-[10px] text-emerald-200">YOLO detections are editable annotation boxes · move, delete or redraw</span>}
              <span className="text-[10px] text-slate-400 whitespace-nowrap">Lattice</span>
              <span className="text-xs font-mono text-slate-200">{grid.rows} × {grid.columns}</span>
              <label className="inline-flex items-center gap-1 text-[10px] text-slate-300 whitespace-nowrap" title="Cells of a module that has at least one positive label, and are not labelled, are saved as negative (clean)."><input type="checkbox" checked={autoNegativeRest} onChange={(event) => setAutoNegativeRest(event.target.checked)} className="accent-rose-400" />Rest of module = negative</label>
              <button type="button" onClick={() => void saveCells()} disabled={!Object.keys(pendingCells).length || savingCells} className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold text-slate-950 bg-cyan-400 rounded-md hover:bg-cyan-300 disabled:opacity-40">
                <Save className="w-3.5 h-3.5" />{savingCells ? 'Saving…' : `Save (${Object.keys(pendingCells).length})`}
              </button>
            </>
          )}
          {isPlacingCellBoxes && <span className="basis-full rounded border border-cyan-700/60 bg-cyan-950/40 px-2 py-1.5 text-[11px] text-cyan-200">Placement mode: move the pointer to the destination cell. Press <kbd className="rounded border border-cyan-700 px-1.5 py-0.5 font-semibold">Space</kbd> or click to place {copiedCellBoxes.length} copied box(es). Press Esc to cancel.</span>}
          {gridError && <span className="basis-full text-[10px] text-slate-400 leading-relaxed">{gridError}</span>}
          {cursorCoords && (
            <span className="hidden xl:inline-block text-xs font-mono tabular-nums text-slate-400 mr-1 whitespace-nowrap">
              X {cursorCoords.x.toLocaleString()} · Y {cursorCoords.y.toLocaleString()} px
            </span>
          )}
          <div className="flex items-center bg-[#07090e] border border-[#1e2636] rounded-md">
            <button
              type="button"
              onClick={() =>
                setCamera((p) =>
                  clampCamera(p.cx, p.cy, p.zoom * 0.8, naturalDim.w, naturalDim.h)
                )
              }
              aria-label="Zoom out"
              className="p-1.5 text-slate-300 hover:text-white transition-colors"
            >
              <Minus className="w-3.5 h-3.5" />
            </button>
            <span className="px-2 text-xs font-mono tabular-nums text-slate-200 min-w-[52px] text-center">
              {Math.round(camera.zoom * 100)}%
            </span>
            <button
              type="button"
              onClick={() =>
                setCamera((p) =>
                  clampCamera(p.cx, p.cy, p.zoom * 1.25, naturalDim.w, naturalDim.h)
                )
              }
              aria-label="Zoom in"
              className="p-1.5 text-slate-300 hover:text-white transition-colors"
            >
              <Plus className="w-3.5 h-3.5" />
            </button>
          </div>
          <button
            type="button"
            onClick={() =>
              setCamera((p) => clampCamera(p.cx, p.cy, 4.0, naturalDim.w, naturalDim.h))
            }
            className="px-2.5 py-1 text-xs font-mono text-slate-300 bg-[#07090e] border border-[#1e2636] rounded-md hover:text-white transition-colors whitespace-nowrap shrink-0"
          >
            400% Edge
          </button>
          <button
            type="button"
            onClick={() =>
              setCamera({ cx: naturalDim.w / 2, cy: naturalDim.h / 2, zoom: 1 })
            }
            className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-slate-300 bg-[#07090e] border border-[#1e2636] rounded-md hover:text-white transition-colors whitespace-nowrap shrink-0"
          >
            <Maximize2 className="w-3.5 h-3.5" />
            Fit
          </button>
        </div>
      </div>

      {/* Native-Pixel SVG Optical Stage */}
      <div ref={stageRef} className="relative flex-1 min-h-0 overflow-hidden overscroll-contain flex items-stretch bg-[#06080d]" style={{ overscrollBehavior: 'contain' }}>
        <div className="relative min-w-0 min-h-0 flex-1 overflow-hidden overscroll-contain flex items-center justify-center">
        {imgError ? (
          <div className="w-full h-full flex flex-col items-center justify-center bg-[#0f141f] text-slate-400 p-8 text-center">
            <Crosshair className="w-8 h-8 text-cyan-400 mb-3" />
            <p className="text-sm font-medium text-slate-200">Optical Frame Unavailable</p>
            <p className="text-xs text-slate-400 mt-1">
              Upload an image, folder, or ZIP archive to run edge spider-web fracture detection.
            </p>
          </div>
        ) : (
          <svg
            ref={svgRef}
            viewBox={`${viewBoxRect.vx} ${viewBoxRect.vy} ${viewBoxRect.vw} ${viewBoxRect.vh}`}
            preserveAspectRatio="xMidYMid meet"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onClick={handleCanvasClick}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => {
              setIsDragging(false);
              movingCellBoxesRef.current = null;
              resizingCellBoxRef.current = null;
              setIsMovingCellBoxes(false);
              suppressNextCellBoxClickRef.current = false;
              setPastePreviewPoint(null);
              setDraft(null);
              setCellBoxDraft(null);
              setCursorCoords(null);
            }}
            className={`w-full h-full max-h-full overscroll-contain ${
              isPlacingCellBoxes ? 'cursor-copy' : isDragging || isMovingCellBoxes ? 'cursor-grabbing' : (cellSelectMode || cellBoxAnnotationMode) ? 'cursor-crosshair' : (selectedModule && grid) || autoCellRegions.length ? 'cursor-cell' : 'cursor-crosshair'
            }`}
          >
            <image
              href={imageSrc}
              x={0}
              y={0}
              width={naturalDim.w}
              height={naturalDim.h}
              preserveAspectRatio="none"
            />

            {showHeatmap && result?.fractureHeatmapUrl && (
              <image
                href={result.fractureHeatmapUrl}
                x={0}
                y={0}
                width={naturalDim.w}
                height={naturalDim.h}
                preserveAspectRatio="none"
                opacity={0.88}
              />
            )}

            {showCellEdgeMap && result?.cellBoundaryMapUrl && (
              <image
                href={result.cellBoundaryMapUrl}
                x={0}
                y={0}
                width={naturalDim.w}
                height={naturalDim.h}
                preserveAspectRatio="none"
                opacity={0.88}
              />
            )}

            {showModules && selectableRegions
              .filter((region) => !manualCellRegions.some((calibration) => calibration.moduleId === region.id))
              .map((region) => (
              <g key={region.id}>
                {(region.cellGrid?.cells ?? []).map((cell, index) => (
                  <rect
                    key={`${region.id}-visible-cell-${index}`}
                    x={cell.pixelBox.left}
                    y={cell.pixelBox.top}
                    width={cell.pixelBox.width}
                    height={cell.pixelBox.height}
                    fill="rgba(34, 211, 238, 0.018)"
                    stroke={cell.partial ? "rgba(250,204,21,0.75)" : "rgba(34, 211, 238, 0.58)"}
                    strokeWidth={baseUnit * (cell.partial ? 1.15 : 0.9)}
                    pointerEvents="none"
                  />
                ))}
              </g>
            ))}

            {cellCorners.length > 0 && <g pointerEvents="none">
              {cellCorners.length > 1 && <polyline points={cellCorners.map((point) => `${point.x},${point.y}`).join(' ')} fill="none" stroke="#facc15" strokeWidth={baseUnit * 2} strokeDasharray={`${baseUnit * 5} ${baseUnit * 3}`} />}
              {cellCorners.map((point, index) => <g key={`corner-${index}`}><circle cx={point.x} cy={point.y} r={baseUnit * 5} fill="#facc15" stroke="#111827" strokeWidth={baseUnit} /><text x={point.x + baseUnit * 6} y={point.y - baseUnit * 6} fill="#facc15" fontSize={baseUnit * 12}>{index + 1}</text></g>)}
            </g>}

            {selectedModule && <rect x={selectedModule.pixelBox.left} y={selectedModule.pixelBox.top} width={selectedModule.pixelBox.width} height={selectedModule.pixelBox.height} fill="none" stroke="#22d3ee" strokeWidth={baseUnit * 2.5} strokeDasharray={`${baseUnit * 9} ${baseUnit * 4}`} pointerEvents="none" />}

            {autoCellRegions.filter((region) => !manualCellRegions.some((calibration) => calibration.moduleId === region.id)).map((region) => (
              <g key={`annotation-region-${region.id}`} pointerEvents="none">
                {(region.cellGrid?.cells ?? []).map((cell) => {
                  const mark = pendingCells[`${region.id}-${cell.row}-${cell.column}`];
                  const active = selectedModule?.id === region.id;
                  if (!showModules && !mark && !active) return null;
                  const { left: x, top: y, width: w, height: h } = cell.pixelBox;
                  const fill = mark?.label === 1 ? 'rgba(34,197,94,0.38)' : mark?.label === 0 ? 'rgba(244,63,94,0.34)' : active ? 'rgba(255,255,255,0.025)' : 'transparent';
                  const stroke = mark?.label === 1 ? '#4ade80' : mark?.label === 0 ? '#fb7185' : cell.partial ? '#facc15' : active ? 'rgba(226,232,240,0.82)' : 'rgba(34,211,238,0.2)';
                  return <g key={`${region.id}-${cell.row}-${cell.column}`}>
                    <rect x={x} y={y} width={w} height={h} fill={fill} stroke={stroke} strokeWidth={baseUnit * (mark ? 2.1 : active ? 0.9 : 0.55)} />
                    {active && w > baseUnit * 18 && h > baseUnit * 18 && <text x={x + w / 2} y={y + h / 2} textAnchor="middle" dominantBaseline="middle" fill="rgba(226,232,240,0.82)" fontSize={baseUnit * 10}>{`R${cell.row}C${cell.column}`}</text>}
                  </g>;
                })}
              </g>
            ))}

            {manualCellRegions.map((region) => (
              <g key={`calibrated-cells-${region.id}`} pointerEvents="none">
                {region.cells.map((cell) => {
                  const mark = pendingCells[`${region.id}-${cell.row}-${cell.column}`];
                  const active = selectedModule?.id === region.id;
                  if (!showModules && !mark && !active) return null;
                  const { left: x, top: y, width: w, height: h } = cell.pixelBox;
                  const fill = mark?.label === 1 ? 'rgba(34,197,94,0.38)' : mark?.label === 0 ? 'rgba(244,63,94,0.34)' : active ? 'rgba(255,255,255,0.025)' : 'transparent';
                  const stroke = mark?.label === 1 ? '#4ade80' : mark?.label === 0 ? '#fb7185' : cell.partial ? '#facc15' : active ? 'rgba(226,232,240,0.9)' : 'rgba(34,211,238,0.58)';
                  return <g key={`${region.id}-${cell.row}-${cell.column}`}>
                    <rect x={x} y={y} width={w} height={h} fill={fill} stroke={stroke} strokeWidth={baseUnit * (mark ? 2.1 : active ? 1.1 : 0.8)} />
                    {(active || mark) && w > baseUnit * 18 && h > baseUnit * 18 && <text x={x + w / 2} y={y + h / 2} textAnchor="middle" dominantBaseline="middle" fill="rgba(226,232,240,0.9)" fontSize={baseUnit * 10}>{`R${cell.row}C${cell.column}`}</text>}
                  </g>;
                })}
              </g>
            ))}

            {showTileGrid &&
              result?.tiles.map((t) => {
                const isFocused = focusedTile?.id === t.id;
                return (
                  <g key={t.id}>
                    <rect
                      x={t.pixelBox.left}
                      y={t.pixelBox.top}
                      width={t.pixelBox.width}
                      height={t.pixelBox.height}
                      fill={
                        isFocused
                          ? 'rgba(6, 182, 212, 0.14)'
                          : t.inspectedByVlm
                            ? 'rgba(6, 182, 212, 0.03)'
                            : 'rgba(15, 23, 42, 0.34)'
                      }
                      stroke={
                        isFocused
                          ? '#22d3ee'
                          : t.inspectedByVlm
                            ? 'rgba(34, 211, 238, 0.62)'
                            : 'rgba(71, 85, 105, 0.3)'
                      }
                      strokeWidth={isFocused ? baseUnit * 2.5 : baseUnit * 1.1}
                    />
                    {t.inspectedByVlm && (
                      <text
                        x={t.pixelBox.left + baseUnit * 8}
                        y={t.pixelBox.top + t.pixelBox.height - baseUnit * 8}
                        fill="#67e8f9"
                        fontSize={baseUnit * 11}
                        fontFamily="JetBrains Mono, monospace"
                      >
                        {t.id} · E={t.offAxisEdgeScore}
                      </text>
                    )}
                  </g>
                );
              })}

            {/* Fracture Bounds & GeoJSON Point Pin Overlay */}
            {displayedAnomalies.map((anom) => {
              const style = anom.source === 'user' ? USER_STYLE : getCategoryStyle(anom.category);
              const { left, top, width, height } = anom.pixelBox;
              const isSelected = selectedAnomalyId === anom.id;
              const px = anom.pointPx.x;
              const py = anom.pointPx.y;
              const pad = baseUnit * 4;
              const pinR = baseUnit * (isSelected ? 10 : 7.5);
              const crossLen = baseUnit * (isSelected ? 17 : 13);

              return (
                <g
                  key={anom.id}
                  onClick={(e) => {
                    // Cell annotation takes precedence over an overlapping crack marker.
                    // Let the SVG-level hit tester receive clicks inside a selectable cell,
                    // even before React has finished activating the first detected region.
                    if (!cellSelectMode && !cellBoxAnnotationMode) {
                      const point = clientToSvgPoint(e.clientX, e.clientY);
                      if (point && findAnnotationCellAtPoint(autoCellRegions, manualCellRegions, { left: point.x, top: point.y })) return;
                    }
                    e.stopPropagation();
                    onSelectAnomaly(isSelected ? null : anom.id);
                  }}
                  className="cursor-pointer"
                >
                  {showBoxes && (
                    <rect
                      x={left - pad}
                      y={top - pad}
                      width={width + pad * 2}
                      height={height + pad * 2}
                      fill={isSelected ? 'rgba(6, 182, 212, 0.14)' : style.fill}
                      stroke={isSelected ? '#22d3ee' : style.stroke}
                      strokeWidth={baseUnit * (isSelected ? 2.8 : 1.9)}
                      strokeDasharray={
                        anom.isEdgeCrack ? undefined : `${baseUnit * 6} ${baseUnit * 3}`
                      }
                    />
                  )}

                  {showGeoPoints && (
                    <g>
                      {/* Outer halo ring around exact GeoJSON Point coordinate */}
                      <circle
                        cx={px}
                        cy={py}
                        r={pinR * 1.75}
                        fill="rgba(7, 9, 14, 0.45)"
                        stroke={isSelected ? '#22d3ee' : style.stroke}
                        strokeWidth={baseUnit * 1.8}
                      />
                      {/* Crosshair axes */}
                      <line
                        x1={px - crossLen}
                        y1={py}
                        x2={px + crossLen}
                        y2={py}
                        stroke={isSelected ? '#67e8f9' : style.stroke}
                        strokeWidth={baseUnit * 1.7}
                      />
                      <line
                        x1={px}
                        y1={py - crossLen}
                        x2={px}
                        y2={py + crossLen}
                        stroke={isSelected ? '#67e8f9' : style.stroke}
                        strokeWidth={baseUnit * 1.7}
                      />
                      {/* Exact Point Core */}
                      <circle
                        cx={px}
                        cy={py}
                        r={baseUnit * 3.5}
                        fill={isSelected ? '#67e8f9' : '#ffffff'}
                        stroke={isSelected ? '#0891b2' : style.stroke}
                        strokeWidth={baseUnit * 1.5}
                      />

                      <rect
                        x={px + baseUnit * 10}
                        y={Math.max(0, py - baseUnit * 20)}
                        width={baseUnit * (anom.isEdgeCrack ? 174 : 138)}
                        height={baseUnit * 18}
                        fill="rgba(7, 9, 14, 0.88)"
                      />
                      <text
                        x={px + baseUnit * 14}
                        y={Math.max(baseUnit * 13, py - baseUnit * 7)}
                        fill={isSelected ? '#67e8f9' : style.stroke}
                        fontSize={baseUnit * 11.5}
                        fontWeight="600"
                        fontFamily="JetBrains Mono, monospace"
                      >
                        {style.glyph} {anom.id}
                        {anom.isEdgeCrack ? ` · ${anom.edgeSide}` : ''} (
                        {Math.round(anom.confidence * 100)}%)
                      </text>
                    </g>
                  )}
                </g>
              );
            })}

            {workflowMode === 'crackLabel' && !showCellAnnotationTools && cellBoxes.slice(baseCellCount).map((box, index) => (
              <g key={`crack-copy-${index}`} pointerEvents="none">
                <rect x={box.left} y={box.top} width={box.width} height={box.height} fill="rgba(251,191,36,0.16)" stroke="#fbbf24" strokeWidth={baseUnit * 2.6} strokeDasharray={`${baseUnit * 8} ${baseUnit * 4}`} />
                <text x={box.left + baseUnit * 5} y={box.top + baseUnit * 18} fill="#fde68a" fontSize={baseUnit * 14}>new cell (unsaved)</text>
              </g>
            ))}

            {showCellAnnotationTools && cellBoxes.map((box, index) => {
              const selected = selectedCellBoxIndices.includes(index);
              return (
                <g key={`cell-annotation-${index}`} pointerEvents="auto">
                  <rect x={box.left} y={box.top} width={box.width} height={box.height} fill={selected ? 'rgba(34,211,238,0.16)' : crackedBox(box) ? 'rgba(244,63,94,0.28)' : 'rgba(16,185,129,0.08)'} stroke={selected ? '#22d3ee' : crackedBox(box) ? '#f43f5e' : '#34d399'} strokeWidth={baseUnit * (selected ? 3.2 : 2.2)} strokeDasharray={`${baseUnit * 8} ${baseUnit * 4}`} pointerEvents={cellBoxReviewMode ? 'all' : 'none'} onMouseDown={(event) => beginCellBoxEditDrag(index, event)} onClick={(event) => { if (cellBoxReviewMode) { event.preventDefault(); event.stopPropagation(); } }} />
                  <text x={box.left + baseUnit * 5} y={box.top + baseUnit * 18} fill={selected ? '#67e8f9' : crackedBox(box) ? '#fda4af' : '#6ee7b7'} fontSize={baseUnit * 14} pointerEvents="none">{result?.cellDetectionSource === 'yolo11n' ? 'YOLO cell' : 'PV cell'} {index + 1}{selected ? ' · selected' : ''}</text>
                  {selected && cellBoxReviewMode && (['nw','ne','sw','se'] as const).map((corner) => {
                    const x = corner.includes('w') ? box.left : box.left + box.width;
                    const y = corner.includes('n') ? box.top : box.top + box.height;
                    return <circle key={`${corner}-${index}`} cx={x} cy={y} r={baseUnit * 5} fill="#22d3ee" stroke="#082f49" strokeWidth={baseUnit * 1.2} pointerEvents="all" onMouseDown={(event) => beginCellBoxResize(index, corner, event)} />;
                  })}
                </g>
              );
            })}
            {showCellAnnotationTools && isPlacingCellBoxes && pastePreviewPoint && placeCellBoxesAt(copiedCellBoxes, pastePreviewPoint, { width: naturalDim.w, height: naturalDim.h }).map((box, index) => (
              <g key={`cell-copy-preview-${index}`} pointerEvents="none">
                <rect x={box.left} y={box.top} width={box.width} height={box.height} fill="rgba(34,211,238,0.18)" stroke="#22d3ee" strokeWidth={baseUnit * 3} strokeDasharray={`${baseUnit * 8} ${baseUnit * 4}`} />
                <text x={box.left + baseUnit * 5} y={box.top + baseUnit * 18} fill="#67e8f9" fontSize={baseUnit * 14}>Place copy {index + 1}</text>
              </g>
            ))}
            {cellBoxDraft && (
              <rect x={Math.min(cellBoxDraft.x0, cellBoxDraft.x1)} y={Math.min(cellBoxDraft.y0, cellBoxDraft.y1)} width={Math.abs(cellBoxDraft.x1 - cellBoxDraft.x0)} height={Math.abs(cellBoxDraft.y1 - cellBoxDraft.y0)} fill="rgba(245,158,11,0.12)" stroke="#fbbf24" strokeWidth={baseUnit * 2.5} strokeDasharray={`${baseUnit * 7} ${baseUnit * 3}`} pointerEvents="none" />
            )}
            {draft && (
              <rect
                x={Math.min(draft.x0, draft.x1)}
                y={Math.min(draft.y0, draft.y1)}
                width={Math.abs(draft.x1 - draft.x0)}
                height={Math.abs(draft.y1 - draft.y0)}
                fill="rgba(34, 197, 94, 0.15)"
                stroke="#22c55e"
                strokeWidth={baseUnit * 2.2}
                strokeDasharray={`${baseUnit * 6} ${baseUnit * 3}`}
              />
            )}
          </svg>
        )}
        </div>

        {!imgError && (
          <div className="w-4 shrink-0 border-l border-[#1e2636] bg-[#0b1019] flex items-center justify-center py-2">
            <input
              type="range"
              min={0}
              max={1000}
              step={1}
              value={imageScroll.y}
              disabled={!imageScroll.enabled}
              onChange={(e) => scrollImage('y', Number(e.target.value))}
              aria-label="Vertical image scrollbar"
              title="Scroll image vertically"
              className="h-full w-3 accent-cyan-400 cursor-ns-resize disabled:opacity-30"
              style={{ writingMode: 'vertical-lr', direction: 'rtl' }}
            />
          </div>
        )}

        {isAnalyzing && (
          <div className="absolute inset-0 bg-[#07090e]/75 flex flex-col items-center justify-center p-6 text-center">
            <div className="bg-[#0f141f] border border-[#1e2636] rounded-lg px-6 py-5 max-w-md">
              <p className="text-sm font-semibold text-slate-100">
                Scanning Every Solar Cell for Glass Fractures
              </p>
              <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
                Detecting repeated cell boundaries, scoring individual cell surfaces, checking central and perimeter cells, and filtering out ground and hardware.
              </p>
              <div className="mt-4 h-1.5 w-full bg-[#1a2233] rounded overflow-hidden">
                <div className="h-full w-2/3 bg-cyan-500 rounded animate-pulse" />
              </div>
            </div>
          </div>
        )}
      </div>
      {!imgError && (
        <div className="h-4 shrink-0 px-3 flex items-center bg-[#0b1019] border-t border-[#1e2636]">
          <input
            type="range"
            min={0}
            max={1000}
            step={1}
            value={imageScroll.x}
            disabled={!imageScroll.enabled}
            onChange={(e) => scrollImage('x', Number(e.target.value))}
            aria-label="Horizontal image scrollbar"
            title="Scroll image horizontally"
            className="w-full h-3 accent-cyan-400 cursor-ew-resize disabled:opacity-30"
          />
        </div>
      )}

      {/* Legend & Single Image GeoJSON Point Footer Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 px-3.5 py-2 bg-[#0f141f] border-t border-[#1e2636] text-xs text-slate-400">
        <div className="flex flex-wrap items-center gap-4">
          <span className="text-rose-400 font-medium whitespace-nowrap">⊕ Spider-Web Origin</span>
          <span className="text-amber-400 font-medium whitespace-nowrap">▲ Impact Pit</span>
          <span className="text-orange-400 font-medium whitespace-nowrap">◆ Linear / Cell Crack</span>
          <span className="text-sky-400 font-medium whitespace-nowrap">○ Surface Residue</span>
        </div>
        <div className="font-mono tabular-nums text-slate-400 whitespace-nowrap">
          {result?.geoReference ? (
            <span>
              EPSG:3857 [{result.geoReference.imageGeoPoint[0]}, {result.geoReference.imageGeoPoint[1]}] ·{' '}
              {result.geoReference.crsMode} · GSD {result.geoReference.gsdMmPerPx} mm/px
            </span>
          ) : (
            <span>
              {naturalDim.w.toLocaleString()} × {naturalDim.h.toLocaleString()} px ·{' '}
              {((naturalDim.w * naturalDim.h) / 1_000_000).toFixed(1)} MP
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
