import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Crosshair, Maximize2, Minus, Plus, Save, MousePointer2 } from 'lucide-react';
import type {
  DefectCategory,
  DetectedAnomaly,
  InspectionResult,
  PixelBox,
  TileDiagnostic,
  SolarCellBox,
} from '../types/inspection';

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
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  // Distinguish a cell click from a canvas pan so a drag never labels a cell.
  const panMovedRef = useRef(false);

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
  const [cellCorners, setCellCorners] = useState<Array<{ x: number; y: number }>>([]);
  const [selectedModule, setSelectedModule] = useState<{ id: string; label: string; pixelBox: PixelBox } | null>(null);
  const [grid, setGrid] = useState<{ rows: number; columns: number; confidence: number; pixelBox: PixelBox } | null>(null);
  const [selectedCells, setSelectedCells] = useState<SolarCellBox[]>([]);
  const [activeLabel, setActiveLabel] = useState<0 | 1>(1);
  const [pendingCells, setPendingCells] = useState<Record<string, CellAnnotationInput>>({});
  const [savingCells, setSavingCells] = useState(false);
  const [gridError, setGridError] = useState('');

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
    setPendingCells({});
    setCellSelectMode(false);
    setCellCorners([]);
    setGridError('');
  }, [imageSrc]);

  const chooseReferenceCell = useCallback(async (id: string, referenceCell: PixelBox) => {
    setSelectedModule(null);
    setSelectedCells([]);
    setPendingCells({});
    setGrid(null);
    setGridError('Matching the reference cell to validated solar-cell geometry…');
    const centerX = referenceCell.left + referenceCell.width / 2;
    const centerY = referenceCell.top + referenceCell.height / 2;
    const region = result?.moduleRegions.find((candidate) =>
      candidate.cellGrid?.cells.some((cell) => {
        const box = cell.pixelBox;
        return centerX >= box.left && centerX <= box.left + box.width &&
          centerY >= box.top && centerY <= box.top + box.height;
      })
    );
    if (!region?.cellGrid || region.cellGrid.cells.length === 0) {
      setGridError('That reference cell does not overlap a validated solar-cell rectangle. Re-scan the image and select the centre of a visible cell; unverified grid slots cannot be labelled.');
      return;
    }
    setSelectedModule({ id: region.id || id, label: region.label, pixelBox: region.pixelBox });
    setSelectedCells(region.cellGrid.cells);
    setGrid({
      rows: region.cellGrid.rows,
      columns: region.cellGrid.columns,
      confidence: region.cellGrid.confidence,
      pixelBox: region.cellGrid.pixelBox,
    });
    setGridError(`Using ${region.cellGrid.cells.length} individually validated cell rectangles (${region.cellGrid.rows} × ${region.cellGrid.columns} lattice, ${Math.round(region.cellGrid.confidence * 100)}% geometry confidence). Gaps and unverified positions cannot be labelled.`);
  }, [result]);

  const toggleCell = (cell: SolarCellBox) => {
    if (!selectedModule || !grid) return;
    const key = `${cell.row}-${cell.column}`;
    setPendingCells((prev) => {
      if (prev[key]?.label === activeLabel) { const next = { ...prev }; delete next[key]; return next; }
      return {
        ...prev,
        [key]: {
          box: cell.pixelBox,
          label: activeLabel,
          moduleLabel: selectedModule.label,
          cellLocation: `R${cell.row}C${cell.column}`,
          moduleBox: selectedModule.pixelBox,
        },
      };
    });
  };

  const saveCells = async () => {
    const items = Object.values(pendingCells);
    if (!items.length || savingCells) return;
    setSavingCells(true);
    try {
      await onSaveCellAnnotations(items);
      setPendingCells({});
      setGridError(`Saved ${items.length} cell label(s) to the training feedback dataset.`);
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

  const handleWheel = useCallback(
    (e: React.WheelEvent<SVGSVGElement>) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.22 : 0.82;
      const svgPt = clientToSvgPoint(e.clientX, e.clientY);

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
    },
    [clientToSvgPoint, clampCamera, naturalDim]
  );

  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    panMovedRef.current = false;
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
        if (box.width >= 8 && box.height >= 8) void chooseReferenceCell(`reference-cell-${Date.now()}`, box);
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

  // Resolve a click to one and only one cell using the click's native-image coordinate.
  // This deliberately avoids per-cell SVG bubbling/overlap triggering adjacent cells.
  const handleCanvasClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!selectedModule || !grid || cellSelectMode || draft || panMovedRef.current) {
      panMovedRef.current = false;
      return;
    }
    const pt = clientToSvgPoint(e.clientX, e.clientY);
    if (!pt) return;
    // Only validated cell boxes are selectable. A click in a gap or on ground does nothing.
    const cell = selectedCells.find(({ pixelBox }) =>
      pt.x >= pixelBox.left && pt.x < pixelBox.left + pixelBox.width &&
      pt.y >= pixelBox.top && pt.y < pixelBox.top + pixelBox.height
    );
    if (cell) toggleCell(cell);
    panMovedRef.current = false;
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    if (draft) setDraft(null);
  };

  const displayedAnomalies: DetectedAnomaly[] = (result?.anomalies || []).filter((a) =>
    visibleCategories.has(a.category)
  );

  const baseUnit = Math.max(naturalDim.w, naturalDim.h) / (900 * camera.zoom);

  return (
    <div className="flex flex-col h-full min-h-[540px] bg-[#07090e] border border-[#1e2636] rounded-lg overflow-hidden select-none">
      <img
        src={imageSrc}
        alt="Photovoltaic module surface inspection probe"
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
      />

      {/* Canvas Overlay Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-3.5 py-2 bg-[#0f141f] border-b border-[#1e2636]">
        <div className="flex items-center gap-1 p-0.5 bg-[#07090e] border border-[#1e2636] rounded-md overflow-x-auto">
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
            Auto Cell Detection
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => { setCellSelectMode((v) => !v); setCellCorners([]); setDraft(null); setSelectedModule(null); setSelectedCells([]); setGrid(null); setPendingCells({}); }}
            title="Click the four corners of one complete solar cell: top-left, top-right, bottom-right, bottom-left."
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md border whitespace-nowrap shrink-0 transition-colors ${cellSelectMode ? 'bg-cyan-500/20 border-cyan-500/60 text-cyan-200' : 'bg-[#07090e] border-[#1e2636] text-slate-300 hover:text-white'}`}
          >
            <MousePointer2 className="w-3.5 h-3.5" />
            {cellSelectMode ? `Mark cell corner ${cellCorners.length + 1}/4…` : 'Select Reference Cell (4 points)'}
          </button>
          {selectedModule && grid && (
            <>
              <button type="button" onClick={() => setActiveLabel(1)} className={`px-2.5 py-1 text-xs font-semibold rounded-md border ${activeLabel === 1 ? 'bg-emerald-500/20 border-emerald-500 text-emerald-200' : 'bg-[#07090e] border-[#1e2636] text-slate-400'}`}>+ Positive</button>
              <button type="button" onClick={() => setActiveLabel(0)} className={`px-2.5 py-1 text-xs font-semibold rounded-md border ${activeLabel === 0 ? 'bg-rose-500/20 border-rose-500 text-rose-200' : 'bg-[#07090e] border-[#1e2636] text-slate-400'}`}>− Negative</button>
              <span className="text-[10px] text-slate-400 whitespace-nowrap">Validated cells</span>
              <span className="text-xs font-mono text-cyan-200">{selectedCells.length}</span>
              <span className="text-[10px] text-slate-400 whitespace-nowrap">Lattice</span>
              <span className="text-xs font-mono text-slate-200">{grid.rows} × {grid.columns}</span>
              <button type="button" onClick={() => void saveCells()} disabled={!Object.keys(pendingCells).length || savingCells} className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold text-slate-950 bg-cyan-400 rounded-md hover:bg-cyan-300 disabled:opacity-40">
                <Save className="w-3.5 h-3.5" />{savingCells ? 'Saving…' : `Save (${Object.keys(pendingCells).length})`}
              </button>
            </>
          )}
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
      <div className="relative flex-1 min-h-0 overflow-hidden flex items-stretch bg-[#06080d]">
        <div className="relative min-w-0 min-h-0 flex-1 overflow-hidden flex items-center justify-center">
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
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onClick={handleCanvasClick}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => {
              setIsDragging(false);
              setDraft(null);
              setCursorCoords(null);
            }}
            className={`w-full h-full max-h-[70vh] ${
              isDragging ? 'cursor-grabbing' : cellSelectMode ? 'cursor-crosshair' : selectedModule && grid ? 'cursor-cell' : 'cursor-crosshair'
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

            {showModules && result?.moduleRegions.map((region) => (
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

            {selectedModule && grid && selectedCells.map((cell) => {
              const mark = pendingCells[`${cell.row}-${cell.column}`];
              const { left: x, top: y, width: w, height: h } = cell.pixelBox;
              const fill = mark?.label === 1 ? 'rgba(34,197,94,0.38)' : mark?.label === 0 ? 'rgba(244,63,94,0.34)' : 'rgba(255,255,255,0.015)';
              const stroke = mark?.label === 1 ? '#4ade80' : mark?.label === 0 ? '#fb7185' : cell.partial ? '#facc15' : 'rgba(226,232,240,0.62)';
              return <g key={`validated-cell-${cell.row}-${cell.column}`} pointerEvents="none">
                <rect x={x} y={y} width={w} height={h} fill={fill} stroke={stroke} strokeWidth={baseUnit * (mark ? 2.1 : 0.9)} />
                {(w > baseUnit * 18 && h > baseUnit * 18) && <text x={x + w / 2} y={y + h / 2} textAnchor="middle" dominantBaseline="middle" fill={mark?.label === 0 ? '#fecdd3' : mark?.label === 1 ? '#bbf7d0' : 'rgba(226,232,240,0.82)'} fontSize={baseUnit * 10}>{`R${cell.row}C${cell.column}`}</text>}
              </g>;
            })}

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
                    // While a module grid is active, the canvas click must label exactly one cell,
                    // even when an existing anomaly marker visually overlaps that cell.
                    if (selectedModule && grid && !cellSelectMode) return;
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
