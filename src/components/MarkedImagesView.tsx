import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, Download, FileArchive } from 'lucide-react';
import JSZip from 'jszip';
import { downloadBlob, markedFileName, renderCellBoxesImage, renderMarkedImage } from '../utils/markedImage';
import type { InspectionResult, PixelBox } from '../types/inspection';

/**
 * v1.4.9 Marked Images.
 *  - Two separate stages, never mixed: CELL stage (cell boxes only) and CRACK stage (crack issues only).
 *  - Cell stage: "Model detections" (YOLO output as produced) vs "Reviewed by you" (the saved, corrected set).
 *  - Crack stage: "Detected - pending review" vs "Reviewed" - only frames you confirmed after deleting false detections and
 *    adding missed cracks are the deliverable.
 *  - Two-pane layout that fills the window: the frame list and the viewer each scroll on their own, and selecting a frame
 *    (click, arrow keys, Prev/Next) keeps the list scrolled to it.
 */
export interface MarkedFrame {
  id: string;
  label: string;
  imageSrc: string;
  result: InspectionResult | null;
  crackReviewed: boolean;
  /** Cell boxes were edited after the crack run that produced `result`. */
  crackStale: boolean;
  /** Cell-stage data (frame size the boxes live in, the model's boxes and the current reviewed set). */
  cell: { imageWidth: number; imageHeight: number; detectedBoxes: PixelBox[]; boxes: PixelBox[]; reviewed: boolean } | null;
}

interface Props {
  frames: MarkedFrame[];
  /** Jump to the Inspection Stage on this frame (delete false detections / add missed cracks). */
  onOpenFrame?: (id: string) => void;
  /** Confirm / undo the crack review of these frames. */
  onSetReviewed?: (ids: string[], reviewed: boolean) => void;
}

type Stage = 'cells' | 'cracks';
type CellKind = 'detected' | 'reviewed';
type CrackKind = 'pending' | 'reviewed';

/** What one frame renders as in the current view. `key` changes whenever the picture must be re-drawn. */
interface Item { frame: MarkedFrame; /** Always 3 entries so hook dependency arrays keep a constant length. */ key: [unknown, unknown, unknown]; summary: string; tone: 'ok' | 'warn' | 'none'; build: (opts: { maxSide?: number; quality?: number }) => Promise<Blob>; fileName: string }

const pill = (active: boolean) => `px-3 py-1 text-xs font-medium rounded whitespace-nowrap ${active ? 'bg-[#1a2436] text-cyan-300' : 'text-slate-400 hover:text-slate-200'}`;

/** Thumbnail that renders only once scrolled into view (the list scroller is the observer root), so 100s of frames stay cheap. */
const Thumb: React.FC<{ item: Item; root: React.RefObject<HTMLDivElement | null> }> = ({ item, root }) => {
  const holder = useRef<HTMLDivElement | null>(null);
  const [visible, setVisible] = useState(false);
  const [url, setUrl] = useState<string | null>(null);
  const [err, setErr] = useState(false);
  useEffect(() => {
    const node = holder.current;
    if (!node || typeof IntersectionObserver === 'undefined') { setVisible(true); return; }
    const observer = new IntersectionObserver((entries) => { if (entries.some((e) => e.isIntersecting)) { setVisible(true); observer.disconnect(); } }, { root: root.current, rootMargin: '200px' });
    observer.observe(node);
    return () => observer.disconnect();
  }, [root]);
  useEffect(() => {
    if (!visible) return;
    let alive = true, made: string | null = null;
    setErr(false);
    item.build({ maxSide: 360, quality: 0.75 })
      .then((b) => { if (!alive) return; made = URL.createObjectURL(b); setUrl(made); })
      .catch(() => alive && setErr(true));
    return () => { alive = false; if (made) URL.revokeObjectURL(made); };
    // item.build is recreated every render; item.key is the real dependency list.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, ...item.key]);
  return (
    <div ref={holder} className="w-24 h-16 shrink-0 bg-[#06080d] rounded overflow-hidden flex items-center justify-center">
      {url ? <img src={url} alt="" className="w-full h-full object-cover" /> : <span className="text-[10px] text-slate-500">{err ? 'n/a' : visible ? '…' : ''}</span>}
    </div>
  );
};

export const MarkedImagesView: React.FC<Props> = ({ frames, onOpenFrame, onSetReviewed }) => {
  const [stage, setStage] = useState<Stage>('cracks');
  const [cellKind, setCellKind] = useState<CellKind>('detected');
  const [crackKind, setCrackKind] = useState<CrackKind>('pending');
  const [onlyIssues, setOnlyIssues] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [zoom, setZoom] = useState<'fit' | 1 | 2 | 3>('fit');
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [viewUrl, setViewUrl] = useState<string | null>(null);
  const [viewState, setViewState] = useState<'idle' | 'loading' | 'error'>('idle');
  const listRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<HTMLDivElement | null>(null);
  const rowRefs = useRef<Map<string, HTMLButtonElement>>(new Map());

  // ---- counts for the pills --------------------------------------------------------------------------------------
  const counts = useMemo(() => ({
    detected: frames.filter((f) => f.cell).length,
    cellReviewed: frames.filter((f) => f.cell?.reviewed).length,
    pending: frames.filter((f) => f.result && !f.crackReviewed).length,
    reviewed: frames.filter((f) => f.result && f.crackReviewed).length,
  }), [frames]);

  // ---- items for the current view --------------------------------------------------------------------------------
  const items = useMemo<Item[]>(() => {
    if (stage === 'cells') {
      const source = cellKind === 'detected' ? frames.filter((f) => f.cell) : frames.filter((f) => f.cell?.reviewed);
      return source.map((frame): Item => {
        const cell = frame.cell!;
        const boxes = cellKind === 'detected' ? cell.detectedBoxes : cell.boxes;
        const delta = cell.boxes.length - cell.detectedBoxes.length;
        return {
          frame, key: [frame.imageSrc, boxes, cellKind],
          summary: cellKind === 'detected'
            ? `${boxes.length} cell(s) detected by the model`
            : `${boxes.length} reviewed cell(s) · model found ${cell.detectedBoxes.length}${delta ? ` (${delta > 0 ? '+' : ''}${delta} after review)` : ''}`,
          tone: 'none',
          build: (o) => renderCellBoxesImage(frame.imageSrc, { width: cell.imageWidth, height: cell.imageHeight }, boxes, frame.label, cellKind, o),
          fileName: markedFileName(frame.label).replace('_marked', cellKind === 'detected' ? '_cells_detected' : '_cells_reviewed'),
        };
      });
    }
    const source = frames.filter((f) => f.result && (crackKind === 'reviewed' ? f.crackReviewed : !f.crackReviewed) && (!onlyIssues || (f.result.anomalies.length > 0)));
    return source.map((frame): Item => {
      const n = frame.result!.anomalies.length;
      return {
        frame, key: [frame.imageSrc, frame.result, crackKind],
        summary: `${n} ${crackKind === 'reviewed' ? 'confirmed' : 'detected'} issue(s)${frame.crackStale ? ' · cells changed, re-run' : ''}`,
        tone: frame.crackStale ? 'warn' : n ? 'ok' : 'none',
        // Crack images show crack issues only; cell boxes belong to the cell stage, so the cell lattice is not drawn here.
        build: (o) => renderMarkedImage(frame.imageSrc, frame.result!, { ...o, showModules: false }),
        fileName: markedFileName(frame.label).replace('_marked', crackKind === 'reviewed' ? '_cracks_reviewed' : '_cracks_pending'),
      };
    });
  }, [stage, cellKind, crackKind, onlyIssues, frames]);

  const selectedIndex = Math.max(0, items.findIndex((item) => item.frame.id === selectedId));
  const selected = items[selectedIndex] ?? null;

  // Keep a valid selection when the view changes (e.g. a frame moves from pending to reviewed).
  useEffect(() => {
    if (!items.length) { if (selectedId !== null) setSelectedId(null); return; }
    if (!items.some((item) => item.frame.id === selectedId)) setSelectedId(items[0].frame.id);
  }, [items, selectedId]);

  // ---- viewer image: rendered at a higher resolution than the thumbnail, re-drawn when the data changes ----------------
  const selectedKey = selected ? selected.key : null;
  useEffect(() => {
    if (!selected) { setViewUrl(null); setViewState('idle'); return; }
    let alive = true, made: string | null = null;
    setViewState('loading');
    selected.build({ maxSide: 3200, quality: 0.9 })
      .then((b) => { if (!alive) return; made = URL.createObjectURL(b); setViewUrl(made); setViewState('idle'); })
      .catch(() => { if (alive) { setViewUrl(null); setViewState('error'); } });
    return () => { alive = false; if (made) URL.revokeObjectURL(made); };
    // selected.build is recreated every render; the key list is the real dependency.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.frame.id, ...(selectedKey ?? [])]);

  // A new frame always opens at the top-left of the viewer's own scroll area.
  useEffect(() => { viewerRef.current?.scrollTo({ top: 0, left: 0 }); }, [selected?.frame.id, zoom]);
  // Linked scrolling: the list follows the selection, whatever moved it (click, keys, Prev/Next, a frame leaving the view).
  useEffect(() => {
    if (selected) rowRefs.current.get(selected.frame.id)?.scrollIntoView({ block: 'nearest' });
  }, [selected?.frame.id]);

  const move = useCallback((delta: number) => {
    if (!items.length) return;
    setSelectedId(items[Math.min(items.length - 1, Math.max(0, selectedIndex + delta))].frame.id);
  }, [items, selectedIndex]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (event.key === 'ArrowDown' || event.key === 'ArrowRight') { event.preventDefault(); move(1); }
      else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') { event.preventDefault(); move(-1); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [move]);

  // ---- downloads ----------------------------------------------------------------------------------------------------
  const downloadOne = async (item: Item) => {
    setBusy(item.frame.id); setError(null);
    try { downloadBlob(item.fileName, await item.build({})); }
    catch (e) { setError(e instanceof Error ? e.message : 'Download failed.'); }
    finally { setBusy(null); }
  };
  const downloadAll = async () => {
    setBusy('zip'); setError(null);
    try {
      const zip = new JSZip();
      const used = new Set<string>();
      for (const item of items) {
        let name = item.fileName;
        for (let i = 2; used.has(name); i++) name = item.fileName.replace(/\.jpg$/, `_${i}.jpg`);
        used.add(name);
        zip.file(name, await item.build({}));
      }
      const tag = stage === 'cells' ? `cells_${cellKind}` : `cracks_${crackKind}`;
      downloadBlob(`pv_marked_${tag}_${items.length}.zip`, await zip.generateAsync({ type: 'blob', compression: 'STORE' }));
    } catch (e) { setError(e instanceof Error ? e.message : 'ZIP build failed.'); }
    finally { setBusy(null); }
  };

  const explain = stage === 'cells'
    ? (cellKind === 'detected'
      ? 'Cell boxes exactly as the model produced them (yellow). Nothing you edited appears here.'
      : 'The cell set you corrected and saved in Cell Annotation (cyan). These are the cells the crack pass inspects.')
    : (crackKind === 'pending'
      ? 'Raw crack findings awaiting your review. Open a frame in the Inspection Stage to delete false detections or add missed cracks, then confirm it.'
      : 'Crack findings you confirmed. This is the deliverable: only reviewed, real issues are drawn (red boxes, Arial 10 labels; the original EXIF is kept).');

  const pendingIds = stage === 'cracks' && crackKind === 'pending' ? items.map((item) => item.frame.id) : [];

  return (
    <div className="h-full flex flex-col gap-3 px-6 py-4 w-full">
      {/* ---- header / controls ---- */}
      <div className="flex flex-wrap items-end justify-between gap-3 shrink-0">
        <div className="min-w-[260px]">
          <h1 className="text-xl font-semibold text-slate-100">Marked Images</h1>
          <p className="text-xs text-slate-400 mt-1 max-w-3xl">{explain}</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1 p-1 bg-[#07090e] border border-[#1e2636] rounded-lg">
            <button type="button" onClick={() => setStage('cells')} className={pill(stage === 'cells')}>1 · Cell stage</button>
            <button type="button" onClick={() => setStage('cracks')} className={pill(stage === 'cracks')}>2 · Crack stage</button>
          </div>
          {stage === 'cells' ? (
            <div className="flex items-center gap-1 p-1 bg-[#07090e] border border-[#1e2636] rounded-lg">
              <button type="button" onClick={() => setCellKind('detected')} className={pill(cellKind === 'detected')}>Model detections ({counts.detected})</button>
              <button type="button" onClick={() => setCellKind('reviewed')} className={pill(cellKind === 'reviewed')}>Reviewed by you ({counts.cellReviewed})</button>
            </div>
          ) : (
            <>
              <div className="flex items-center gap-1 p-1 bg-[#07090e] border border-[#1e2636] rounded-lg">
                <button type="button" onClick={() => setCrackKind('pending')} className={pill(crackKind === 'pending')}>Detected · pending review ({counts.pending})</button>
                <button type="button" onClick={() => setCrackKind('reviewed')} className={pill(crackKind === 'reviewed')}>Reviewed ({counts.reviewed})</button>
              </div>
              <label className="flex items-center gap-1.5 text-xs text-slate-300"><input type="checkbox" checked={onlyIssues} onChange={(e) => setOnlyIssues(e.target.checked)} />Only frames with issues</label>
            </>
          )}
          {pendingIds.length > 0 && onSetReviewed && (
            <button type="button" onClick={() => onSetReviewed(pendingIds, true)} className="px-3 py-2 text-xs font-semibold rounded-lg border border-emerald-700/70 bg-emerald-950/30 text-emerald-200 whitespace-nowrap">Confirm all {pendingIds.length} as reviewed</button>
          )}
          <button type="button" disabled={items.length === 0 || busy !== null} onClick={() => void downloadAll()} className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 rounded-lg hover:bg-cyan-300 disabled:opacity-40 whitespace-nowrap">
            <FileArchive className="w-3.5 h-3.5" />{busy === 'zip' ? 'Building ZIP…' : `Download all (${items.length}) as ZIP`}
          </button>
        </div>
      </div>
      {error && <p className="text-xs text-rose-300 shrink-0">{error}</p>}

      {items.length === 0 ? (
        <div className="flex-1 min-h-0 rounded-lg border border-dashed border-[#263448] flex items-center justify-center text-center p-6 text-sm text-slate-400">
          {stage === 'cells'
            ? (cellKind === 'detected' ? 'No frame has been through cell detection yet.' : 'No frame has saved, reviewed cell boxes yet. Save them in Cell Annotation.')
            : (crackKind === 'reviewed'
              ? (counts.pending > 0 ? `Nothing confirmed yet. ${counts.pending} frame(s) are waiting in “Detected · pending review”.` : 'No reviewed crack findings yet.')
              : 'No crack findings are pending review.')}
        </div>
      ) : (
        <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[360px_minmax(0,1fr)] gap-3">
          {/* ---- list: its own scroll ---- */}
          <div ref={listRef} className="min-h-0 overflow-y-auto overscroll-contain rounded-lg border border-[#1e2636] bg-[#0b0f18] divide-y divide-[#1e2636]">
            {items.map((item) => {
              const active = item.frame.id === selected?.frame.id;
              return (
                <button
                  key={item.frame.id}
                  type="button"
                  ref={(node) => { if (node) rowRefs.current.set(item.frame.id, node); else rowRefs.current.delete(item.frame.id); }}
                  onClick={() => setSelectedId(item.frame.id)}
                  className={`w-full flex items-center gap-3 p-2 text-left ${active ? 'bg-cyan-950/30' : 'hover:bg-[#131b2b]'}`}
                >
                  <Thumb item={item} root={listRef} />
                  <span className="min-w-0 flex-1">
                    <span className="block text-xs font-mono text-slate-200 truncate" title={item.frame.label}>{item.frame.label}</span>
                    <span className={`block text-[11px] ${item.tone === 'warn' ? 'text-amber-300' : item.tone === 'ok' ? 'text-rose-300' : 'text-slate-400'}`}>{item.summary}</span>
                  </span>
                </button>
              );
            })}
          </div>

          {/* ---- viewer: tall, with its own scroll ---- */}
          <div className="min-h-0 flex flex-col rounded-lg border border-[#1e2636] bg-[#0b0f18]">
            <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 border-b border-[#1e2636] shrink-0">
              <div className="min-w-0">
                <p className="text-xs font-mono text-slate-100 truncate" title={selected?.frame.label}>{selected?.frame.label}</p>
                <p className="text-[11px] text-slate-400">{selectedIndex + 1} / {items.length} · {selected?.summary}</p>
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                <button type="button" aria-label="Previous frame" disabled={selectedIndex <= 0} onClick={() => move(-1)} className="p-1.5 rounded border border-[#24324d] text-slate-200 disabled:opacity-30"><ChevronLeft className="w-4 h-4" /></button>
                <button type="button" aria-label="Next frame" disabled={selectedIndex >= items.length - 1} onClick={() => move(1)} className="p-1.5 rounded border border-[#24324d] text-slate-200 disabled:opacity-30"><ChevronRight className="w-4 h-4" /></button>
                <div className="flex items-center gap-1 p-0.5 bg-[#07090e] border border-[#1e2636] rounded-md">
                  {(['fit', 1, 2, 3] as const).map((z) => <button key={String(z)} type="button" onClick={() => setZoom(z)} className={pill(zoom === z)}>{z === 'fit' ? 'Fit' : `${z * 100}%`}</button>)}
                </div>
                {stage === 'cracks' && selected && onOpenFrame && (
                  <button type="button" onClick={() => onOpenFrame(selected.frame.id)} className="px-2.5 py-1.5 text-xs rounded-md border border-[#24324d] text-slate-200 whitespace-nowrap">Open in Inspection Stage</button>
                )}
                {stage === 'cracks' && selected && onSetReviewed && (
                  <button type="button" onClick={() => onSetReviewed([selected.frame.id], crackKind === 'pending')} className={`px-2.5 py-1.5 text-xs font-semibold rounded-md border whitespace-nowrap ${crackKind === 'pending' ? 'border-emerald-700/70 bg-emerald-950/30 text-emerald-200' : 'border-slate-600 text-slate-200'}`}>
                    {crackKind === 'pending' ? 'Confirm reviewed' : 'Move back to pending'}
                  </button>
                )}
                {selected && (
                  <button type="button" disabled={busy !== null} onClick={() => void downloadOne(selected)} className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-cyan-300 bg-[#0b1320] border border-[#1f314d] rounded-md hover:bg-[#121e33] disabled:opacity-40 whitespace-nowrap">
                    <Download className="w-3.5 h-3.5" />{busy === selected.frame.id ? '…' : 'JPG'}
                  </button>
                )}
              </div>
            </div>
            <div ref={viewerRef} className={`flex-1 min-h-0 overflow-auto overscroll-contain ${zoom === 'fit' ? 'flex items-center justify-center' : ''} bg-[#06080d]`}>
              {viewUrl ? (
                <img
                  src={viewUrl}
                  alt={`${selected?.frame.label ?? ''} marked`}
                  draggable={false}
                  className={zoom === 'fit' ? 'max-w-full max-h-full object-contain' : 'max-w-none'}
                  style={zoom === 'fit' ? undefined : { width: `${zoom * 100}%` }}
                />
              ) : <p className="p-6 text-xs text-slate-500">{viewState === 'error' ? 'Preview unavailable.' : 'Rendering…'}</p>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
