import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  Download,
  FileArchive,
  FolderOpen,
  Pause,
  Play,
  RefreshCw,
  Search,
  Trash2,
  Undo2,
  Upload,
} from 'lucide-react';
import { InspectionCanvas, type CellAnnotationInput } from './components/InspectionCanvas';
import { ForensicDrawer } from './components/ForensicDrawer';
import { MarkedImagesView } from './components/MarkedImagesView';
import { collectFilesFromDataTransfer, normalizeSelectedFileList } from './utils/geojson';
import { downloadBlob, markedFileName, renderMarkedImage } from './utils/markedImage';
import {
  deleteFeedback,
  getFeedbackSummary,
  resetModel,
  sendFeedback,
  sendFeedbackBatch,
  trainModel,
  withAnomalies,
  type FeedbackSummary,
} from './utils/feedback';
import type {
  DefectCategory,
  DetectedAnomaly,
  InspectionConfig,
  InspectionResult,
  TileDiagnostic,
} from './types/inspection';

type ActiveTab = 'workspace' | 'batch' | 'tiles' | 'marked';
type FilterCategory = 'all' | 'spider_web_crack' | 'edge_spider_web' | 'surface_residue';

interface FrameEntry {
  id: string;
  label: string;
  sourceType: 'sample' | 'upload';
  sampleKey?:
    | 'pcs011-right-edge'
    | 'edge-spiderweb-jfif'
    | 'dji-rtk-0074'
    | 'edge-closeup-png';
  fileObj?: File;
  imageSrc: string;
  result: InspectionResult | null;
  status: 'queued' | 'analyzing' | 'completed' | 'error';
  errorMessage?: string;
  /** Detections the user deleted as false; kept so they can be restored (feedbackId = stored training negative) */
  rejected?: { anomaly: DetectedAnomaly; feedbackId: string }[];
  /** anomaly id -> stored training positive, for issues the user added */
  feedbackIds?: Record<string, string>;
}

function getAnomalyBadgeText(
  category: DefectCategory,
  isEdgeCrack: boolean
): {
  glyph: string;
  label: string;
  colorClass: string;
} {
  switch (category) {
    case 'spider_web_crack':
      return {
        glyph: '⊕',
        label: isEdgeCrack ? 'Edge Spider-Web' : 'Spider-Web Crack',
        colorClass: 'text-rose-400',
      };
    case 'impact_point':
      return {
        glyph: '▲',
        label: 'Impact Pit',
        colorClass: 'text-amber-400',
      };
    case 'linear_fracture':
      return {
        glyph: '◆',
        label: 'Linear / Cell Crack',
        colorClass: 'text-orange-400',
      };
    case 'surface_residue':
      return {
        glyph: '○',
        label: 'Surface Residue',
        colorClass: 'text-sky-400',
      };
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('workspace');
  const [config, setConfig] = useState<InspectionConfig>({
    tileSizePx: 1360,
    overlapRatio: 0.18,
    sensitivity: 'high',
    maxTilesToInspect: 12,
    includeSurfaceResidue: true,
    contrastEnhancement: true,
    edgeFocusMode: true,
    onlyImagesWithIssuesInGeoJson: true,
    rejectGround: true,
    useSiteReferenceForNonGps: false,
    siteReferenceLat: 40.45084487,
    siteReferenceLon: -104.84752746,
  });

  const [frames, setFrames] = useState<FrameEntry[]>([
    {
      id: 'sample-pcs011-right-edge',
      label: 'PCS011-R2-T11_RightEdge.png',
      sourceType: 'sample',
      sampleKey: 'pcs011-right-edge',
      imageSrc: '/assets/aistudio/image_ecd4bcb0.png',
      result: null,
      status: 'queued',
    },
    {
      id: 'sample-edge-jfif',
      label: 'f81d2698_0074_EdgeSpiderWeb.jfif',
      sourceType: 'sample',
      sampleKey: 'edge-spiderweb-jfif',
      imageSrc: '/assets/aistudio/f81d2698-4dfd-4e1a-b917-de5433d27e2a_848c4df5.jfif',
      result: null,
      status: 'queued',
    },
    {
      id: 'sample-dji-rtk',
      label: 'DJI_20261011114408_0074_Z.JPG (RTK GPS)',
      sourceType: 'sample',
      sampleKey: 'dji-rtk-0074',
      imageSrc: '/assets/aistudio/DJI_20261011114408_0074_Z_9902db27.JPG',
      result: null,
      status: 'queued',
    },
    {
      id: 'sample-edge-closeup',
      label: 'Edge_SpiderWeb_CloseUp.png',
      sourceType: 'sample',
      sampleKey: 'edge-closeup-png',
      imageSrc: '/assets/aistudio/image_2e5da07a.png',
      result: null,
      status: 'queued',
    },
  ]);

  const [activeFrameId, setActiveFrameId] = useState<string>('sample-pcs011-right-edge');
  const [selectedAnomalyId, setSelectedAnomalyId] = useState<string | null>(null);
  const [focusedTile, setFocusedTile] = useState<TileDiagnostic | null>(null);

  const [categoryFilter, setCategoryFilter] = useState<FilterCategory>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [tileFilterMode, setTileFilterMode] = useState<'all' | 'inspected' | 'anomalies'>('all');

  const [batchRunning, setBatchRunning] = useState<boolean>(false);
  const [batchConcurrency, setBatchConcurrency] = useState<number>(2);
  const [isExtractingArchive, setIsExtractingArchive] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const zipInputRef = useRef<HTMLInputElement | null>(null);
  const metaInputRef = useRef<HTMLInputElement | null>(null);
  const [metaStatus, setMetaStatus] = useState<string>('');

  // Uploads raw-image metadata JSON (camera GPS per filename) so images without embedded GPS get an image-centre Point
  const handleMetadataFile = useCallback(async (file: File | undefined) => {
    if (!file) return;
    try {
      const res = await fetch('/api/metadata', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: await file.text(),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Metadata upload failed.');
      setMetaStatus(`${data.records} records loaded - re-scan frames to apply`);
    } catch (err: unknown) {
      setMetaStatus(err instanceof Error ? err.message : 'Metadata upload failed.');
    }
  }, []);

  const initialRunTriggered = useRef<boolean>(false);
  const framesRef = useRef<FrameEntry[]>(frames);
  framesRef.current = frames;
  const configRef = useRef<InspectionConfig>(config);
  configRef.current = config;
  const batchPausedRef = useRef<boolean>(false);

  const activeFrame = useMemo(
    () => frames.find((f) => f.id === activeFrameId) || frames[0],
    [frames, activeFrameId]
  );

  const analyzeSingleFrameEntry = useCallback(
    async (entry: FrameEntry, forceRefresh = false): Promise<InspectionResult | null> => {
      const cfg = configRef.current;
      setFrames((prev) =>
        prev.map((f) =>
          f.id === entry.id ? { ...f, status: 'analyzing', errorMessage: undefined } : f
        )
      );

      try {
        let result: InspectionResult;
        if (entry.sourceType === 'sample' && entry.sampleKey) {
          const res = await fetch('/api/inspect/sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              sampleId: entry.sampleKey,
              config: cfg,
              forceRefresh,
            }),
          });
          const data = await res.json();
          if (!res.ok) throw new Error(data.error || 'Sample inspection failed.');
          result = data as InspectionResult;
        } else {
          const arrayBuf = entry.fileObj
            ? await entry.fileObj.arrayBuffer()
            : await (await fetch(entry.imageSrc)).arrayBuffer();

          const res = await fetch('/api/inspect/upload', {
            method: 'POST',
            headers: {
              'Content-Type': entry.fileObj?.type || 'application/octet-stream',
              'x-file-name': encodeURIComponent(entry.label),
              'x-inspection-config': JSON.stringify(cfg),
              'x-include-preview': 'false',
            },
            body: arrayBuf,
          });
          const data = await res.json();
          if (!res.ok) throw new Error(data.error || `Failed to inspect ${entry.label}`);
          result = data as InspectionResult;
        }

        setFrames((prev) =>
          prev.map((f) =>
            f.id === entry.id ? { ...f, result, status: 'completed' } : f
          )
        );
        return result;
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Frame inspection failed.';
        setFrames((prev) =>
          prev.map((f) =>
            f.id === entry.id ? { ...f, status: 'error', errorMessage: msg } : f
          )
        );
        return null;
      }
    },
    []
  );

  // Auto-inspect primary edge spider-web frame on initial load
  useEffect(() => {
    if (!initialRunTriggered.current) {
      initialRunTriggered.current = true;
      const first = framesRef.current[0];
      if (first) {
        analyzeSingleFrameEntry(first, false).then((res) => {
          if (res && res.anomalies.length > 0) {
            const firstSpider =
              res.anomalies.find((a) => a.category === 'spider_web_crack') ||
              res.anomalies[0];
            setSelectedAnomalyId(firstSpider.id);
          }
        });
      }
    }
  }, [analyzeSingleFrameEntry]);

  // Auto-trigger analysis if user selects an un-analyzed sample frame in Workspace
  const handleSelectActiveFrame = useCallback(
    (frameId: string) => {
      setActiveFrameId(frameId);
      setSelectedAnomalyId(null);
      setFocusedTile(null);
      const target = framesRef.current.find((f) => f.id === frameId);
      if (target && target.status === 'queued' && !batchRunning) {
        analyzeSingleFrameEntry(target, false).then((res) => {
          if (res && res.anomalies.length > 0) {
            const firstSpider =
              res.anomalies.find((a) => a.category === 'spider_web_crack') ||
              res.anomalies[0];
            setSelectedAnomalyId(firstSpider.id);
          }
        });
      } else if (target?.result?.anomalies.length) {
        const firstSpider =
          target.result.anomalies.find((a) => a.category === 'spider_web_crack') ||
          target.result.anomalies[0];
        setSelectedAnomalyId(firstSpider.id);
      }
    },
    [analyzeSingleFrameEntry, batchRunning]
  );

  // Concurrent Batch Processing Runner for Folder / ZIP / Multi-Image Queues
  const startBatchProcessing = useCallback(
    async (explicitTargets?: FrameEntry[]) => {
      batchPausedRef.current = false;
      setBatchRunning(true);

      const getNextQueued = (): FrameEntry | undefined => {
        if (batchPausedRef.current) return undefined;
        if (explicitTargets) {
          return explicitTargets.find((t) => {
            const cur = framesRef.current.find((f) => f.id === t.id);
            return cur?.status === 'queued';
          });
        }
        return framesRef.current.find((f) => f.status === 'queued');
      };

      const workerCount = Math.max(1, Math.min(3, batchConcurrency));
      const workers = Array.from({ length: workerCount }, async () => {
        while (!batchPausedRef.current) {
          const next = getNextQueued();
          if (!next) break;
          // Optimistically mark analyzing so another worker doesn't grab the same frame
          framesRef.current = framesRef.current.map((f) =>
            f.id === next.id ? { ...f, status: 'analyzing' } : f
          );
          await analyzeSingleFrameEntry(next, false);
        }
      });

      await Promise.all(workers);
      setBatchRunning(false);
    },
    [analyzeSingleFrameEntry, batchConcurrency]
  );

  const enqueueFilesAndProcess = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;

      const created: FrameEntry[] = files.map((file, idx) => ({
        id: `batch-${Date.now()}-${idx}-${Math.random().toString(36).slice(2, 6)}`,
        label: file.name,
        sourceType: 'upload',
        fileObj: file,
        imageSrc: URL.createObjectURL(file),
        result: null,
        status: 'queued',
      }));

      setFrames((prev) => {
        const updated = [...prev, ...created];
        framesRef.current = updated;
        return updated;
      });

      if (created.length === 1) {
        setActiveFrameId(created[0].id);
        setSelectedAnomalyId(null);
        setActiveTab('workspace');
        const res = await analyzeSingleFrameEntry(created[0], false);
        if (res && res.anomalies.length > 0) {
          setSelectedAnomalyId(res.anomalies[0].id);
        }
      } else {
        setActiveTab('batch');
        startBatchProcessing();
      }
    },
    [analyzeSingleFrameEntry, startBatchProcessing]
  );

  const handleInputSelection = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    setIsExtractingArchive(true);
    try {
      const normalized = await normalizeSelectedFileList(fileList);
      await enqueueFilesAndProcess(normalized);
    } finally {
      setIsExtractingArchive(false);
    }
  };

  const handleDropTransfer = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsExtractingArchive(true);
    try {
      const collected = await collectFilesFromDataTransfer(e.dataTransfer);
      await enqueueFilesAndProcess(collected);
    } finally {
      setIsExtractingArchive(false);
    }
  };

  const handlePauseBatch = () => {
    batchPausedRef.current = true;
    setBatchRunning(false);
  };

  const handleRetryErrors = () => {
    setFrames((prev) => {
      const updated = prev.map((f) =>
        f.status === 'error' ? { ...f, status: 'queued' as const, errorMessage: undefined } : f
      );
      framesRef.current = updated;
      return updated;
    });
    setTimeout(() => startBatchProcessing(), 50);
  };

  const visibleCategorySet = useMemo<Set<DefectCategory>>(() => {
    if (categoryFilter === 'spider_web_crack' || categoryFilter === 'edge_spider_web') {
      return new Set<DefectCategory>(['spider_web_crack']);
    }
    if (categoryFilter === 'surface_residue') {
      return new Set<DefectCategory>(['surface_residue']);
    }
    return new Set<DefectCategory>([
      'spider_web_crack',
      'impact_point',
      'linear_fracture',
      'surface_residue',
    ]);
  }, [categoryFilter]);

  const filteredAnomalies = useMemo<DetectedAnomaly[]>(() => {
    const list = activeFrame?.result?.anomalies || [];
    const q = searchQuery.trim().toLowerCase();
    return list.filter((a) => {
      if (categoryFilter === 'edge_spider_web' && (!a.isEdgeCrack || a.category !== 'spider_web_crack')) {
        return false;
      }
      if (!visibleCategorySet.has(a.category)) return false;
      if (!q) return true;
      return (
        a.id.toLowerCase().includes(q) ||
        a.title.toLowerCase().includes(q) ||
        a.edgeSide.toLowerCase().includes(q) ||
        a.moduleIndex.toLowerCase().includes(q) ||
        a.cellLocation.toLowerCase().includes(q)
      );
    });
  }, [activeFrame?.result?.anomalies, categoryFilter, visibleCategorySet, searchQuery]);

  const selectedAnomaly = useMemo<DetectedAnomaly | null>(() => {
    if (!selectedAnomalyId || !activeFrame?.result) return null;
    return activeFrame.result.anomalies.find((a) => a.id === selectedAnomalyId) || null;
  }, [selectedAnomalyId, activeFrame?.result]);

  const completedResults = useMemo<InspectionResult[]>(
    () => frames.map((f) => f.result).filter((r): r is InspectionResult => r !== null),
    [frames]
  );

  const framesWithIssues = useMemo(
    () => completedResults.filter((r) => r.anomalies.length > 0).length,
    [completedResults]
  );

  // ---------- user feedback: add missed issues / delete false detections ----------
  const [feedback, setFeedback] = useState<FeedbackSummary | null>(null);
  const [notice, setNotice] = useState<string>('');
  const [training, setTraining] = useState<boolean>(false);
  const [markedBusy, setMarkedBusy] = useState<boolean>(false);
  const noticeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const flash = useCallback((msg: string, ms = 7000) => {
    setNotice(msg);
    if (noticeTimer.current) clearTimeout(noticeTimer.current);
    noticeTimer.current = setTimeout(() => setNotice(''), ms);
  }, []);
  const errMsg = (e: unknown, fallback: string) => (e instanceof Error ? e.message : fallback);

  useEffect(() => {
    getFeedbackSummary().then(setFeedback).catch(() => undefined);
  }, []);

  const patchFrame = useCallback((id: string, fn: (f: FrameEntry) => FrameEntry) => {
    setFrames((prev) => {
      const next = prev.map((f) => (f.id === id ? fn(f) : f));
      framesRef.current = next;
      return next;
    });
  }, []);

  const frameBytes = (f: FrameEntry): Promise<ArrayBuffer> =>
    f.fileObj ? f.fileObj.arrayBuffer() : fetch(f.imageSrc).then((r) => r.arrayBuffer());

  const handleSaveCellAnnotations = async (items: CellAnnotationInput[]) => {
    const frame = activeFrame;
    const res = frame?.result;
    if (!frame || !res || !items.length) throw new Error('Select and inspect an image before saving cell labels.');
    const image = await frameBytes(frame);
    const saved = await sendFeedbackBatch(image, frame.label, items.map((item) => ({
      box: item.box,
      label: item.label,
      category: 'spider_web_crack',
      moduleBox: item.moduleBox,
    })));
    const positives = items.filter((item) => item.label === 1);
    if (positives.length) {
      const newAnomalies: DetectedAnomaly[] = positives.map((item, index) => {
        const box = item.box;
        const cx = box.left + box.width / 2;
        const cy = box.top + box.height / 2;
        const seq = `${Date.now()}-${index + 1}`;
        return {
          id: `USERCELL-${seq}`,
          category: 'spider_web_crack',
          isEdgeCrack: false,
          edgeSide: 'interior',
          severity: 'moderate',
          confidence: 1,
          title: 'Manually labelled positive cell',
          description: 'Positive training label assigned to this reference-sized solar cell.',
          moduleIndex: item.moduleLabel,
          cellLocation: item.cellLocation,
          pointPx: { x: Math.round(cx), y: Math.round(cy) },
          box: {
            xmin: box.left / res.imageWidth,
            ymin: box.top / res.imageHeight,
            xmax: (box.left + box.width) / res.imageWidth,
            ymax: (box.top + box.height) / res.imageHeight,
          },
          pixelBox: box,
          tileId: 'USER-CELL',
          offAxisEdgeScore: 0,
          source: 'user',
        };
      });
      patchFrame(frame.id, (f) => ({
        ...f,
        result: f.result ? withAnomalies(f.result, [...f.result.anomalies, ...newAnomalies]) : f.result,
        feedbackIds: {
          ...f.feedbackIds,
          ...Object.fromEntries(newAnomalies.map((anomaly, index) => {
            const originalIndex = items.findIndex((item) => item === positives[index]);
            const match = saved.items.find((entry) => entry.index === originalIndex);
            return [anomaly.id, match?.id ?? ''];
          }).filter((entry) => Boolean(entry[1]))),
        },
      }));
    }
    setFeedback(await getFeedbackSummary());
    const negatives = items.length - positives.length;
    flash(`Saved ${items.length} cell label(s) for training: ${positives.length} positive, ${negatives} negative. Select Retrain model when your dataset is ready.`);
  };

  const handleDeleteAnomaly = async (anom: DetectedAnomaly) => {
    const frame = activeFrame;
    if (!frame?.result) return;
    const without = frame.result.anomalies.filter((a) => a.id !== anom.id);
    if (selectedAnomalyId === anom.id) setSelectedAnomalyId(null);

    if (anom.source === 'user') {
      patchFrame(frame.id, (f) => (f.result ? { ...f, result: withAnomalies(f.result, without) } : f));
      const fb = frame.feedbackIds?.[anom.id];
      if (fb) await deleteFeedback(fb).catch(() => undefined);
      getFeedbackSummary().then(setFeedback).catch(() => undefined);
      flash(`${anom.id} removed.`);
      return;
    }
    patchFrame(frame.id, (f) =>
      f.result
        ? { ...f, result: withAnomalies(f.result, without), rejected: [...(f.rejected ?? []), { anomaly: anom, feedbackId: '' }] }
        : f
    );
    try {
      const id = await sendFeedback(await frameBytes(frame), frame.label, 0, anom.pixelBox, anom.category);
      patchFrame(frame.id, (f) => ({
        ...f,
        rejected: f.rejected?.map((r) => (r.anomaly.id === anom.id ? { ...r, feedbackId: id } : r)),
      }));
      getFeedbackSummary().then(setFeedback).catch(() => undefined);
      flash(`${anom.id} deleted as a false detection and saved as a training negative.`);
    } catch (e) {
      // could not store the negative: put the detection back so the UI never disagrees with what will be trained on
      patchFrame(frame.id, (f) =>
        f.result
          ? { ...f, result: withAnomalies(f.result, [...f.result.anomalies, anom]), rejected: f.rejected?.filter((r) => r.anomaly.id !== anom.id) }
          : f
      );
      flash(`Could not delete ${anom.id}: ${errMsg(e, 'server error')}`);
    }
  };

  const handleRestoreAnomaly = async (anomId: string) => {
    const frame = activeFrame;
    const entry = frame?.rejected?.find((r) => r.anomaly.id === anomId);
    if (!frame || !entry) return;
    patchFrame(frame.id, (f) =>
      f.result
        ? { ...f, result: withAnomalies(f.result, [...f.result.anomalies, entry.anomaly]), rejected: f.rejected?.filter((r) => r.anomaly.id !== anomId) }
        : f
    );
    if (entry.feedbackId) await deleteFeedback(entry.feedbackId).catch(() => undefined);
    getFeedbackSummary().then(setFeedback).catch(() => undefined);
  };

  const handleTrain = async () => {
    setTraining(true);
    try {
      const r = await trainModel();
      setFeedback(r);
      const pct = (v: number) => `${Math.round(v * 100)}%`;
      flash(
        r.accepted
          ? `Model updated using ${r.samples} training windows and ${r.validationImages} held-out images. Held-out crack recall: ${pct(r.addedRecallBefore)} → ${pct(r.addedRecallAfter)}; false-positive rejection: ${pct(r.rejectedClearedBefore)} → ${pct(r.rejectedClearedAfter)}. Re-scan frames to apply.`
          : `Training candidate rejected. No model was changed. Held-out recall: ${pct(r.addedRecallBefore)} → ${pct(r.addedRecallAfter)}; false-positive rejection: ${pct(r.rejectedClearedBefore)} → ${pct(r.rejectedClearedAfter)}. ${r.reason ?? ''}`,
        16000
      );
    } catch (e) {
      flash(errMsg(e, 'Training failed.'));
    } finally {
      setTraining(false);
    }
  };

  const handleResetModel = async () => {
    try {
      setFeedback(await resetModel());
      flash('Reset to the original base model. Saved positive/negative feedback was preserved. Re-scan frames to apply.');
    } catch (e) {
      flash(errMsg(e, 'Reset failed.'));
    }
  };

  const handleDownloadMarked = async () => {
    if (!activeFrame.result) return;
    setMarkedBusy(true);
    try {
      downloadBlob(markedFileName(activeFrame.label), await renderMarkedImage(activeFrame.imageSrc, activeFrame.result));
    } catch (e) {
      flash(errMsg(e, 'Could not build the marked image.'));
    } finally {
      setMarkedBusy(false);
    }
  };

  const summary = activeFrame?.result?.pipelineSummary;
  const queuedCount = frames.filter((f) => f.status === 'queued').length;
  const analyzingCount = frames.filter((f) => f.status === 'analyzing').length;
  const errorCount = frames.filter((f) => f.status === 'error').length;

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0e14] text-slate-100">
      {/* Hidden File, Folder (webkitdirectory), and ZIP Archive Inputs */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".jpg,.jpeg,.jfif,.png,.webp,.tif,.tiff,.zip"
        multiple
        onChange={(e) => {
          handleInputSelection(e.target.files);
          e.target.value = '';
        }}
        className="hidden"
      />
      <input
        ref={folderInputRef}
        type="file"
        multiple
        {...({ webkitdirectory: '', directory: '' } as Record<string, string>)}
        onChange={(e) => {
          handleInputSelection(e.target.files);
          e.target.value = '';
        }}
        className="hidden"
      />
      <input
        ref={metaInputRef}
        type="file"
        accept=".json,.geojson,application/json"
        onChange={(e) => {
          void handleMetadataFile(e.target.files?.[0]);
          e.target.value = '';
        }}
        className="hidden"
      />
      <input
        ref={zipInputRef}
        type="file"
        accept=".zip,application/zip"
        multiple
        onChange={(e) => {
          handleInputSelection(e.target.files);
          e.target.value = '';
        }}
        className="hidden"
      />

      {/* Strict 3-Zone Top Bar Contract */}
      <header className="flex items-center justify-between gap-8 px-6 py-3.5 bg-[#0f141f] border-b border-[#1e2636]">
        {/* Zone 1: Single-Line Wordmark */}
        <a
          href="#workspace"
          onClick={(e) => {
            e.preventDefault();
            setActiveTab('workspace');
          }}
          className="text-base font-bold tracking-tight text-slate-100 whitespace-nowrap shrink-0"
        >
          PV CrackVision
        </a>

        {/* Zone 2: 4 Clean Single-Line Text Navigation Links */}
        <nav className="flex items-center gap-6 text-sm font-medium text-slate-400 overflow-x-auto">
          <button
            type="button"
            onClick={() => setActiveTab('workspace')}
            className={`py-0.5 transition-colors whitespace-nowrap shrink-0 border-b-2 ${
              activeTab === 'workspace'
                ? 'text-slate-100 border-cyan-400'
                : 'border-transparent hover:text-slate-200 hover:border-slate-600'
            }`}
          >
            Inspection Stage
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('batch')}
            className={`py-0.5 transition-colors whitespace-nowrap shrink-0 border-b-2 ${
              activeTab === 'batch'
                ? 'text-slate-100 border-cyan-400'
                : 'border-transparent hover:text-slate-200 hover:border-slate-600'
            }`}
          >
            Batch · Folder & ZIP ({completedResults.length}/{frames.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('tiles')}
            className={`py-0.5 transition-colors whitespace-nowrap shrink-0 border-b-2 ${
              activeTab === 'tiles'
                ? 'text-slate-100 border-cyan-400'
                : 'border-transparent hover:text-slate-200 hover:border-slate-600'
            }`}
          >
            Inspection Tiles
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('marked')}
            className={`py-0.5 transition-colors whitespace-nowrap shrink-0 border-b-2 ${
              activeTab === 'marked'
                ? 'text-slate-100 border-cyan-400'
                : 'border-transparent hover:text-slate-200 hover:border-slate-600'
            }`}
          >
            Marked Images ({framesWithIssues}/{completedResults.length})
          </button>
        </nav>

        {/* Zone 3: 1 Primary Action Button */}
        <div className="flex items-center shrink-0">
          <button
            type="button"
            onClick={() => setActiveTab('batch')}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 rounded-lg hover:bg-cyan-300 transition-colors whitespace-nowrap shrink-0"
          >
            <Upload className="w-3.5 h-3.5" />
            Batch Upload
          </button>
        </div>
      </header>

      {/* Main Asymmetric Split Console Workspace */}
      {activeTab === 'workspace' && (
        <main className="flex-1 grid grid-cols-1 lg:grid-cols-[315px_minmax(0,1fr)_365px] divide-y lg:divide-y-0 lg:divide-x divide-[#1e2636]">
          {/* Left Parameter & Upload Control Column */}
          <aside className="bg-[#0e131d] p-5 flex flex-col justify-between gap-6">
            <div className="space-y-5">
              {/* Active Frame Selector */}
              <div className="space-y-2">
                <label
                  htmlFor="active-frame-select"
                  className="block text-xs font-medium text-slate-400"
                >
                  Active Inspection Frame ({frames.length} in Queue)
                </label>
                <select
                  id="active-frame-select"
                  value={activeFrame.id}
                  onChange={(e) => handleSelectActiveFrame(e.target.value)}
                  className="w-full px-3 py-2 text-xs font-mono bg-[#07090e] border border-[#1e2636] rounded-md text-slate-200 focus:outline-none focus:border-cyan-400"
                >
                  {frames.map((f) => (
                    <option key={f.id} value={f.id}>
                      {f.status === 'completed' ? '✓ ' : f.status === 'analyzing' ? '… ' : '○ '}
                      {f.label}
                    </option>
                  ))}
                </select>

                {/* Direct Ingestion Triggers: Images, Folder, ZIP */}
                <div className="grid grid-cols-3 gap-1.5 pt-1">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="inline-flex items-center justify-center gap-1 py-1.5 px-2 text-xs font-medium text-slate-200 bg-[#141d2e] border border-[#24324d] rounded hover:bg-[#1a263c] transition-colors whitespace-nowrap"
                  >
                    <Upload className="w-3 h-3 text-cyan-400" />
                    Files
                  </button>
                  <button
                    type="button"
                    onClick={() => folderInputRef.current?.click()}
                    className="inline-flex items-center justify-center gap-1 py-1.5 px-2 text-xs font-medium text-slate-200 bg-[#141d2e] border border-[#24324d] rounded hover:bg-[#1a263c] transition-colors whitespace-nowrap"
                  >
                    <FolderOpen className="w-3 h-3 text-cyan-400" />
                    Folder
                  </button>
                  <button
                    type="button"
                    onClick={() => zipInputRef.current?.click()}
                    className="inline-flex items-center justify-center gap-1 py-1.5 px-2 text-xs font-medium text-slate-200 bg-[#141d2e] border border-[#24324d] rounded hover:bg-[#1a263c] transition-colors whitespace-nowrap"
                  >
                    <FileArchive className="w-3 h-3 text-cyan-400" />
                    .ZIP
                  </button>
                </div>
                <button
                  type="button"
                  onClick={() => metaInputRef.current?.click()}
                  className="mt-2 w-full inline-flex items-center justify-center gap-1 py-1.5 px-2 text-xs font-medium text-slate-200 bg-[#141d2e] border border-[#24324d] rounded hover:bg-[#1a263c] transition-colors whitespace-nowrap"
                >
                  <Upload className="w-3 h-3 text-cyan-400" />
                  Load metadata JSON (camera GPS)
                </button>
                {metaStatus && <p className="mt-1 text-[10px] text-slate-400">{metaStatus}</p>}
              </div>

              {/* Edge Spider-Web CV Calibration Controls */}
              <div className="pt-4 border-t border-[#1e2636] space-y-4">
                <h2 className="text-xs font-semibold text-slate-200">
                  Edge Spider-Web Crack Calibration
                </h2>

                <div className="space-y-1.5">
                  <span className="block text-xs text-slate-400">Detection Sensitivity</span>
                  <div className="grid grid-cols-3 gap-1 p-1 bg-[#07090e] border border-[#1e2636] rounded-md">
                    {(['high', 'balanced', 'strict'] as const).map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        onClick={() => setConfig((c) => ({ ...c, sensitivity: mode }))}
                        className={`py-1.5 text-xs font-medium rounded transition-colors capitalize whitespace-nowrap ${
                          config.sensitivity === mode
                            ? 'bg-[#1a2436] text-cyan-300'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {mode}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <label htmlFor="max-tiles-slider" className="text-slate-400">
                      Max Scanned Perimeter Windows
                    </label>
                    <span className="font-mono tabular-nums text-slate-200">
                      {config.maxTilesToInspect} tiles
                    </span>
                  </div>
                  <input
                    id="max-tiles-slider"
                    type="range"
                    min={6}
                    max={16}
                    step={1}
                    value={config.maxTilesToInspect}
                    onChange={(e) =>
                      setConfig((c) => ({
                        ...c,
                        maxTilesToInspect: Number(e.target.value),
                      }))
                    }
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                </div>

                <div className="space-y-2.5 pt-1">
                  <label className="flex items-center justify-between gap-2 text-xs text-slate-300 cursor-pointer">
                    <span>Quadrant Perimeter Window Slicing</span>
                    <input
                      type="checkbox"
                      checked={config.edgeFocusMode}
                      onChange={(e) =>
                        setConfig((c) => ({ ...c, edgeFocusMode: e.target.checked }))
                      }
                      className="accent-cyan-400 rounded"
                    />
                  </label>
                  <label className="flex items-center justify-between gap-2 text-xs text-slate-300 cursor-pointer">
                    <span>Adaptive Silicon Off-Axis Ridge Boost</span>
                    <input
                      type="checkbox"
                      checked={config.contrastEnhancement}
                      onChange={(e) =>
                        setConfig((c) => ({ ...c, contrastEnhancement: e.target.checked }))
                      }
                      className="accent-cyan-400 rounded"
                    />
                  </label>
                  <label className="flex items-center justify-between gap-2 text-xs text-slate-300 cursor-pointer">
                    <span>Reject Ground / Off-Array Detections</span>
                    <input
                      type="checkbox"
                      checked={config.rejectGround}
                      onChange={(e) => setConfig((c) => ({ ...c, rejectGround: e.target.checked }))}
                      className="accent-cyan-400 rounded"
                    />
                  </label>
                  <label className="flex items-center justify-between gap-2 text-xs text-slate-300 cursor-pointer">
                    <span>EPSG:3857 Site Fallback (Non-EXIF)</span>
                    <input
                      type="checkbox"
                      checked={config.useSiteReferenceForNonGps}
                      onChange={(e) =>
                        setConfig((c) => ({
                          ...c,
                          useSiteReferenceForNonGps: e.target.checked,
                        }))
                      }
                      className="accent-cyan-400 rounded"
                    />
                  </label>
                </div>

                <button
                  type="button"
                  disabled={activeFrame.status === 'analyzing'}
                  onClick={() => analyzeSingleFrameEntry(activeFrame, true)}
                  className="w-full inline-flex items-center justify-center gap-2 py-2.5 px-4 text-xs font-semibold text-slate-100 bg-[#172236] border border-[#293b5c] rounded-lg hover:bg-[#1e2c46] disabled:opacity-50 transition-colors whitespace-nowrap"
                >
                  {activeFrame.status === 'analyzing' ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      Scanning Solar Cells...
                    </>
                  ) : (
                    <>
                      <Play className="w-3.5 h-3.5 text-cyan-400" />
                      Re-Scan Active Frame
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Marked-image output + model training from user feedback */}
            <div className="pt-4 border-t border-[#1e2636] space-y-3">
              <button
                type="button"
                disabled={!activeFrame.result || markedBusy}
                onClick={() => void handleDownloadMarked()}
                className="w-full inline-flex items-center justify-center gap-2 py-2 px-3 text-xs font-medium text-cyan-300 bg-[#0b1320] border border-[#1f314d] rounded-lg hover:bg-[#121e33] disabled:opacity-40 transition-colors whitespace-nowrap"
              >
                <Download className="w-3.5 h-3.5" />
                {markedBusy ? 'Rendering…' : 'Download marked image (.jpg)'}
              </button>

              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-200">Model training data</span>
                  <span className="font-mono tabular-nums text-slate-400">
                    {feedback ? `+${feedback.positives} / −${feedback.negatives}` : '—'}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 leading-relaxed">
                  Label individual cells as Positive (cracked) or Negative (clean). Training uses hard false-positive mining for negative labels and validates on images held out from training. {feedback?.userTuned ? ' Using your tuned model.' : ' Using the base model.'}
                </p>
                {feedback && (
                  <div className={`rounded-md border px-2.5 py-2 text-[10px] leading-relaxed ${feedback.trainingReady ? 'border-emerald-800/60 bg-emerald-950/20 text-emerald-200' : 'border-amber-800/60 bg-amber-950/20 text-amber-200'}`}>
                    <div className="font-semibold mb-1">{feedback.trainingReady ? 'Training requirements met' : 'More reviewed labels required'}</div>
                    <div>Positive {feedback.positives}/{feedback.trainingRequirements.positives} · Negative {feedback.negatives}/{feedback.trainingRequirements.negatives}</div>
                    <div>Images {feedback.images}/{feedback.trainingRequirements.images} · Positive-label images {feedback.positiveImages}/{feedback.trainingRequirements.positiveImages} · Negative-label images {feedback.negativeImages}/{feedback.trainingRequirements.negativeImages}</div>
                    <div>Annotated PV regions {feedback.annotatedRegions}/{feedback.trainingRequirements.annotatedRegions}</div>
                    {!feedback.trainingReady && <div className="mt-1 text-amber-300/90">{feedback.trainingMissing.join(' · ')}</div>}
                  </div>
                )}
                <div className="grid grid-cols-2 gap-1.5">
                  <button
                    type="button"
                    disabled={training || !feedback?.trainingReady}
                    onClick={() => void handleTrain()}
                    className="inline-flex items-center justify-center gap-1.5 py-1.5 px-2 text-xs font-medium text-emerald-300 bg-[#0b1a14] border border-emerald-800/60 rounded hover:bg-[#10261c] disabled:opacity-40 transition-colors whitespace-nowrap"
                  >
                    {training && <RefreshCw className="w-3 h-3 animate-spin" />}
                    {training ? 'Training…' : 'Retrain model'}
                  </button>
                  <button
                    type="button"
                    disabled={training || !feedback?.userTuned}
                    onClick={() => void handleResetModel()}
                    className="inline-flex items-center justify-center py-1.5 px-2 text-xs font-medium text-slate-300 bg-[#141d2e] border border-[#24324d] rounded hover:bg-[#1a263c] disabled:opacity-40 transition-colors whitespace-nowrap"
                  >
                    Reset to base
                  </button>
                </div>
              </div>
            </div>
          </aside>

          {/* Center Optical Stage */}
          <section className="p-5 flex flex-col gap-4 bg-[#0b0e14] min-w-0">
            {activeFrame.result?.warnings?.map((w) => (
              <div
                key={w}
                className="flex items-center gap-2.5 px-4 py-2.5 bg-amber-950/40 border border-amber-800/60 rounded-lg text-xs text-amber-200"
              >
                <AlertCircle className="w-4 h-4 text-amber-400 shrink-0" />
                <span>{w}</span>
              </div>
            ))}
            {activeFrame.status === 'error' && (
              <div className="flex items-center justify-between gap-4 px-4 py-3 bg-rose-950/40 border border-rose-800/60 rounded-lg text-xs text-rose-200">
                <div className="flex items-center gap-2.5">
                  <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                  <span>{activeFrame.errorMessage}</span>
                </div>
                <button
                  type="button"
                  onClick={() => analyzeSingleFrameEntry(activeFrame, true)}
                  className="px-3 py-1 bg-rose-900/60 hover:bg-rose-800/70 rounded text-rose-100 font-medium whitespace-nowrap shrink-0"
                >
                  Retry Scan
                </button>
              </div>
            )}

            {/* Quantitative Telemetry Strip */}
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-4 pb-3.5 border-b border-[#1e2636]">
              <div>
                <span className="text-xs text-slate-400 block">Edge Spider-Web</span>
                <div className="mt-0.5 flex items-baseline gap-1.5">
                  <span
                    className={`text-2xl font-mono font-semibold tabular-nums ${
                      (summary?.edgeSpiderWebCracksCount ?? 0) > 0
                        ? 'text-rose-400'
                        : 'text-emerald-400'
                    }`}
                  >
                    {summary ? summary.edgeSpiderWebCracksCount : '—'}
                  </span>
                  <span className="text-xs font-mono text-slate-500">perimeter</span>
                </div>
              </div>

              <div>
                <span className="text-xs text-slate-400 block">Total Spider-Web</span>
                <div className="mt-0.5 flex items-baseline gap-1.5">
                  <span
                    className={`text-2xl font-mono font-semibold tabular-nums ${
                      (summary?.spiderWebCracksCount ?? 0) > 0
                        ? 'text-rose-400'
                        : 'text-emerald-400'
                    }`}
                  >
                    {summary ? summary.spiderWebCracksCount : '—'}
                  </span>
                  <span className="text-xs font-mono text-slate-500">points</span>
                </div>
              </div>

              <div>
                <span className="text-xs text-slate-400 block">Impact / Linear</span>
                <div className="mt-0.5 flex items-baseline gap-1.5">
                  <span className="text-2xl font-mono font-semibold tabular-nums text-amber-400">
                    {summary
                      ? summary.impactPointsCount + summary.linearFracturesCount
                      : '—'}
                  </span>
                  <span className="text-xs font-mono text-slate-500">cells</span>
                </div>
              </div>

              <div>
                <span className="text-xs text-slate-400 block">Surface Residue</span>
                <div className="mt-0.5 flex items-baseline gap-1.5">
                  <span className="text-2xl font-mono font-semibold tabular-nums text-sky-400">
                    {summary ? summary.surfaceResidueCount : '—'}
                  </span>
                  <span className="text-xs font-mono text-slate-500">isolated</span>
                </div>
              </div>

              <div>
                <span className="text-xs text-slate-400 block">GeoJSON CRS</span>
                <div className="mt-0.5 flex items-baseline gap-1.5">
                  <span
                    title={activeFrame.result?.geoReference.crsName}
                    className="text-sm font-mono font-semibold tabular-nums text-cyan-300 truncate"
                  >
                    {activeFrame.result ? activeFrame.result.geoReference.epsgCode : 'EPSG:3857'}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex-1 min-h-0">
              <InspectionCanvas
                imageSrc={activeFrame.imageSrc}
                result={activeFrame.result}
                isAnalyzing={activeFrame.status === 'analyzing'}
                selectedAnomalyId={selectedAnomalyId}
                onSelectAnomaly={setSelectedAnomalyId}
                focusedTile={focusedTile}
                visibleCategories={visibleCategorySet}
                onSaveCellAnnotations={handleSaveCellAnnotations}
              />
            </div>
          </section>

          {/* Right Findings & GeoJSON Point Forensics Column */}
          <aside className="bg-[#0e131d] flex flex-col max-h-[calc(100vh-57px)]">
            <div className="p-4 border-b border-[#1e2636] space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-slate-100">
                  Image Crack Attributes
                </h2>
                <span className="text-xs font-mono tabular-nums text-slate-400">
                  {filteredAnomalies.length} found
                </span>
              </div>

              {/* Category Filter Buttons */}
              <div className="grid grid-cols-4 gap-1 p-1 bg-[#07090e] border border-[#1e2636] rounded-md">
                {(
                  [
                    { id: 'all', label: 'All' },
                    { id: 'edge_spider_web', label: 'Edge Cracks' },
                    { id: 'spider_web_crack', label: 'Spider-Web' },
                    { id: 'surface_residue', label: 'Residue' },
                  ] as const
                ).map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setCategoryFilter(tab.id)}
                    className={`py-1 px-1.5 text-[11px] font-medium rounded transition-colors whitespace-nowrap truncate ${
                      categoryFilter === tab.id
                        ? 'bg-[#1a2436] text-cyan-300'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              <div className="relative">
                <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Filter by edge side, array, cell..."
                  className="w-full pl-8 pr-3 py-1.5 text-xs bg-[#07090e] border border-[#1e2636] rounded-md text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-400"
                />
              </div>
            </div>

            {notice && (
              <div className="px-4 py-2 text-[11px] leading-relaxed text-emerald-200 bg-emerald-950/40 border-b border-emerald-900/60">
                {notice}
              </div>
            )}

            <div className="flex-1 overflow-y-auto divide-y divide-[#1e2636]">
              {activeFrame.status === 'analyzing' ? (
                <div className="p-4 space-y-3">
                  {[1, 2, 3, 4].map((skeleton) => (
                    <div key={skeleton} className="space-y-2 py-2">
                      <div className="h-3 w-2/5 bg-[#182032] rounded animate-pulse" />
                      <div className="h-3.5 w-4/5 bg-[#182032] rounded animate-pulse" />
                      <div className="h-2.5 w-3/5 bg-[#182032] rounded animate-pulse" />
                    </div>
                  ))}
                </div>
              ) : filteredAnomalies.length === 0 ? (
                <div className="p-6 text-center space-y-2">
                  <p className="text-sm font-medium text-slate-200">
                    No Matching Points in View
                  </p>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {activeFrame.status === 'queued'
                      ? 'Click Re-Scan Active Frame to run perimeter spider-web crack detection on this image.'
                      : 'Switch category filter to "All" or scan another frame from the queue.'}
                  </p>
                </div>
              ) : (
                filteredAnomalies.map((anom) => {
                  const badge = getAnomalyBadgeText(anom.category, anom.isEdgeCrack);
                  const isSelected = selectedAnomalyId === anom.id;
                  return (
                    <div key={anom.id} className="relative">
                    <button
                      type="button"
                      onClick={() => setSelectedAnomalyId(anom.id)}
                      className={`w-full text-left px-4 py-3 pr-11 transition-colors ${
                        isSelected ? 'bg-[#151e30]' : 'hover:bg-[#111827]'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2 text-xs font-mono tabular-nums">
                        <div className="flex items-center gap-1.5 min-w-0 truncate">
                          <span className={`${badge.colorClass} font-medium`}>
                            {badge.glyph} {badge.label}
                          </span>
                          <span aria-hidden="true" className="text-slate-600">·</span>
                          <span className="text-slate-400">{anom.edgeSide}</span>
                        </div>
                        <span className="text-slate-300 shrink-0">
                          {Math.round(anom.confidence * 100)}%
                        </span>
                      </div>

                      <p className="text-xs font-semibold text-slate-100 mt-1 truncate">
                        {anom.title}
                      </p>

                      <div className="flex items-center gap-1.5 text-[11px] text-slate-400 font-mono tabular-nums mt-1 truncate">
                        <span>{anom.moduleIndex}</span>
                        <span aria-hidden="true">·</span>
                        <span>{anom.cellLocation}</span>
                        <span aria-hidden="true">·</span>
                        <span>
                          px({anom.pointPx.x}, {anom.pointPx.y})
                        </span>
                      </div>
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleDeleteAnomaly(anom)}
                      title={anom.source === 'user' ? 'Remove this issue you added' : 'Delete: not a real issue (saved as a training negative)'}
                      aria-label={`Delete ${anom.id}`}
                      className="absolute right-2.5 bottom-2.5 p-1.5 text-slate-500 hover:text-rose-300 hover:bg-rose-950/40 rounded transition-colors"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                    </div>
                  );
                })
              )}
            </div>

            {(activeFrame.rejected?.length ?? 0) > 0 && (
              <div className="border-t border-[#1e2636] px-4 py-2.5 space-y-1.5 max-h-36 overflow-y-auto">
                <span className="text-[11px] font-medium text-slate-400">
                  Deleted as false ({activeFrame.rejected!.length})
                </span>
                {activeFrame.rejected!.map((r) => (
                  <div key={r.anomaly.id} className="flex items-center justify-between gap-2 text-[11px] font-mono text-slate-500">
                    <span className="truncate line-through">{r.anomaly.id} · {r.anomaly.title}</span>
                    <button
                      type="button"
                      onClick={() => void handleRestoreAnomaly(r.anomaly.id)}
                      className="inline-flex items-center gap-1 text-cyan-400 hover:text-cyan-300 shrink-0"
                    >
                      <Undo2 className="w-3 h-3" /> Restore
                    </button>
                  </div>
                ))}
              </div>
            )}

            <ForensicDrawer
              anomaly={selectedAnomaly}
              imageGeoFeature={activeFrame.result?.geoJson.features[0] || null}
              onFocusOnCanvas={(id) => {
                setSelectedAnomalyId(null);
                requestAnimationFrame(() => setSelectedAnomalyId(id));
              }}
            />
          </aside>
        </main>
      )}

      {/* Batch Processing, Folder & ZIP Upload Console */}
      {activeTab === 'batch' && (
        <main className="max-w-6xl w-full mx-auto py-8 px-6 space-y-6">
          <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 pb-5 border-b border-[#1e2636]">
            <div>
              <h1 className="text-2xl font-semibold text-slate-100">
                Batch Processing · Folder & ZIP Ingestion
              </h1>
              <p className="text-sm text-slate-400 mt-1">
                Upload drone flight folders, <code className="text-cyan-300">.zip</code> archives, or multi-image batches to detect edge spider-web cracks and get back <strong className="text-slate-200">marked images</strong> with every issue drawn on the frame.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2.5 shrink-0">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-slate-200 bg-[#131b2b] border border-[#24324d] rounded-lg hover:bg-[#19243a] transition-colors whitespace-nowrap"
              >
                <Upload className="w-3.5 h-3.5 text-cyan-400" />
                Select Images
              </button>
              <button
                type="button"
                onClick={() => folderInputRef.current?.click()}
                className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-slate-200 bg-[#131b2b] border border-[#24324d] rounded-lg hover:bg-[#19243a] transition-colors whitespace-nowrap"
              >
                <FolderOpen className="w-3.5 h-3.5 text-cyan-400" />
                Upload Folder
              </button>
              <button
                type="button"
                onClick={() => zipInputRef.current?.click()}
                className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-slate-200 bg-[#131b2b] border border-[#24324d] rounded-lg hover:bg-[#19243a] transition-colors whitespace-nowrap"
              >
                <FileArchive className="w-3.5 h-3.5 text-cyan-400" />
                Upload .ZIP Archive
              </button>
              <button
                type="button"
                disabled={completedResults.length === 0}
                onClick={() => setActiveTab('marked')}
                className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 rounded-lg hover:bg-cyan-300 disabled:opacity-40 transition-colors whitespace-nowrap"
              >
                <Download className="w-3.5 h-3.5" />
                View / Download Marked Images ({completedResults.length})
              </button>
            </div>
          </div>

          {/* Drag-and-Drop Folder / ZIP / Multi-Image Zone */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDropTransfer}
            onClick={() => folderInputRef.current?.click()}
            className="border border-dashed border-[#2a3752] bg-[#0e131d] hover:border-cyan-400/60 rounded-lg p-7 text-center cursor-pointer transition-colors"
          >
            <FolderOpen className="w-6 h-6 text-cyan-400 mx-auto mb-2" />
            <p className="text-sm font-medium text-slate-200">
              {isExtractingArchive
                ? 'Unpacking folder / ZIP archive into batch queue...'
                : 'Drag and drop a flight folder, .zip archive, or multiple DJI / PV surface images here'}
            </p>
            <p className="text-xs text-slate-400 mt-1">
              Recursively scans nested subfolders and .zip packages (.jpg, .jpeg, .jfif, .png, .webp, .tif) · Outputs a marked image per frame
            </p>
          </div>

          {/* Batch Execution Control & Progress Bar */}
          <div className="flex flex-wrap items-center justify-between gap-4 px-4 py-3 bg-[#0e131d] border border-[#1e2636] rounded-lg">
            <div className="flex flex-wrap items-center gap-4 text-xs font-mono tabular-nums">
              <span className="text-slate-200">
                Completed: {completedResults.length}/{frames.length}
              </span>
              <span aria-hidden="true" className="text-slate-600">·</span>
              <span className="text-slate-400">Queued: {queuedCount}</span>
              <span aria-hidden="true" className="text-slate-600">·</span>
              <span className="text-cyan-400">
                Marked Images: {completedResults.length}
              </span>
              <span aria-hidden="true" className="text-slate-600">·</span>
              <span className="text-rose-400 font-semibold">
                Cracked Images: {framesWithIssues}
              </span>
            </div>

            <div className="flex items-center gap-2.5">
              <div className="flex items-center gap-1.5 text-xs text-slate-400">
                <span>Workers:</span>
                <select
                  value={batchConcurrency}
                  onChange={(e) => setBatchConcurrency(Number(e.target.value))}
                  className="px-2 py-1 text-xs font-mono bg-[#07090e] border border-[#1e2636] rounded text-slate-200"
                >
                  <option value={1}>1×</option>
                  <option value={2}>2×</option>
                  <option value={3}>3×</option>
                </select>
              </div>

              {errorCount > 0 && (
                <button
                  type="button"
                  onClick={handleRetryErrors}
                  className="px-3 py-1.5 text-xs font-medium text-rose-200 bg-rose-950/60 border border-rose-800/60 rounded-md hover:bg-rose-900/60 transition-colors whitespace-nowrap"
                >
                  Retry Failed ({errorCount})
                </button>
              )}

              {batchRunning ? (
                <button
                  type="button"
                  onClick={handlePauseBatch}
                  className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-medium text-amber-300 bg-[#1a1810] border border-amber-700/50 rounded-md hover:bg-[#262112] transition-colors whitespace-nowrap"
                >
                  <Pause className="w-3.5 h-3.5" />
                  Pause Batch
                </button>
              ) : (
                <button
                  type="button"
                  disabled={queuedCount === 0}
                  onClick={() => startBatchProcessing()}
                  className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold text-slate-950 bg-cyan-400 rounded-md hover:bg-cyan-300 disabled:opacity-40 transition-colors whitespace-nowrap"
                >
                  <Play className="w-3.5 h-3.5" />
                  Process Queued ({queuedCount})
                </button>
              )}
            </div>
          </div>

          {/* Batch Queue Table */}
          <div className="border border-[#1e2636] rounded-lg overflow-hidden bg-[#0e131d]">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[#1e2636] bg-[#111724] text-xs font-medium text-slate-400">
                  <th className="py-3 px-4">Image / Archive Path</th>
                  <th className="py-3 px-4">Georeference CRS</th>
                  <th className="py-3 px-4 text-right">Edge Spider-Web</th>
                  <th className="py-3 px-4 text-right">Total Spider-Web</th>
                  <th className="py-3 px-4 text-right">Other Points</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1e2636] text-xs">
                {frames.map((frame) => {
                  const s = frame.result?.pipelineSummary;
                  const g = frame.result?.geoReference;
                  return (
                    <tr key={frame.id} className="hover:bg-[#131b2b] transition-colors">
                      <td className="py-3 px-4 font-mono text-slate-200 max-w-[280px] truncate">
                        {frame.label}
                      </td>
                      <td className="py-3 px-4 font-mono tabular-nums text-slate-400">
                        {g ? `${g.epsgCode} (${g.crsMode})` : '—'}
                      </td>
                      <td className="py-3 px-4 text-right font-mono tabular-nums">
                        {s ? (
                          <span
                            className={
                              s.edgeSpiderWebCracksCount > 0
                                ? 'text-rose-400 font-semibold'
                                : 'text-slate-400'
                            }
                          >
                            {s.edgeSpiderWebCracksCount}
                          </span>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="py-3 px-4 text-right font-mono tabular-nums">
                        {s ? (
                          <span
                            className={
                              s.spiderWebCracksCount > 0
                                ? 'text-rose-400 font-semibold'
                                : 'text-emerald-400'
                            }
                          >
                            {s.spiderWebCracksCount}
                          </span>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="py-3 px-4 text-right font-mono tabular-nums text-amber-400">
                        {s
                          ? s.impactPointsCount +
                            s.linearFracturesCount +
                            s.surfaceResidueCount
                          : '—'}
                      </td>
                      <td className="py-3 px-4 font-mono">
                        {frame.status === 'completed' && (
                          <span className="text-emerald-400">● Completed</span>
                        )}
                        {frame.status === 'analyzing' && (
                          <span className="text-cyan-400">▲ Scanning</span>
                        )}
                        {frame.status === 'error' && (
                          <span className="text-rose-400">✖ Error</span>
                        )}
                        {frame.status === 'queued' && (
                          <span className="text-slate-400">○ Queued</span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-right">
                        <button
                          type="button"
                          onClick={() => {
                            handleSelectActiveFrame(frame.id);
                            setActiveTab('workspace');
                          }}
                          className="px-3 py-1 text-xs font-medium text-cyan-300 bg-[#162033] border border-[#253655] rounded hover:bg-[#1d2b44] transition-colors whitespace-nowrap"
                        >
                          Inspect Stage
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </main>
      )}

      {/* Solar-Cell Inspection Tile Telemetry Tab */}
      {activeTab === 'tiles' && (
        <main className="max-w-6xl w-full mx-auto py-8 px-6 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 pb-5 border-b border-[#1e2636]">
            <div>
              <h1 className="text-2xl font-semibold text-slate-100">
                Solar-Cell Inspection Tile Telemetry
              </h1>
              <p className="text-sm text-slate-400 mt-1">
                High-magnification array sub-regions and dark-silicon hairline scores for {activeFrame.label}.
              </p>
            </div>

            <div className="flex items-center gap-1 p-1 bg-[#07090e] border border-[#1e2636] rounded-lg">
              {(
                [
                  { id: 'all', label: 'All Windows' },
                  { id: 'inspected', label: 'VLM Scanned' },
                  { id: 'anomalies', label: 'With Crack Points' },
                ] as const
              ).map((f) => (
                <button
                  key={f.id}
                  type="button"
                  onClick={() => setTileFilterMode(f.id)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                    tileFilterMode === f.id
                      ? 'bg-[#1a2436] text-cyan-300'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {f.label}
                </button>
              ))}
            </div>
          </div>

          {!activeFrame.result ? (
            <div className="p-10 text-center border border-[#1e2636] rounded-lg bg-[#0e131d] text-sm text-slate-400">
              Scan the active frame to inspect solar-cell coverage and tile diagnostics.
            </div>
          ) : (
            <div className="border border-[#1e2636] rounded-lg overflow-hidden bg-[#0e131d]">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[#1e2636] bg-[#111724] text-xs font-medium text-slate-400">
                    <th className="py-3 px-4">Window ID</th>
                    <th className="py-3 px-4">Native Pixel Bounds</th>
                    <th className="py-3 px-4">Zone</th>
                    <th className="py-3 px-4 text-right">PV Coverage</th>
                    <th className="py-3 px-4 text-right">Hairline Ridge Score</th>
                    <th className="py-3 px-4 text-right">Points Found</th>
                    <th className="py-3 px-4 text-right">Focus</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1e2636] text-xs font-mono tabular-nums">
                  {activeFrame.result.tiles
                    .filter((t) => {
                      if (tileFilterMode === 'inspected') return t.inspectedByVlm;
                      if (tileFilterMode === 'anomalies') return t.detectionsCount > 0;
                      return true;
                    })
                    .map((tile) => (
                      <tr key={tile.id} className="hover:bg-[#131b2b] transition-colors">
                        <td className="py-2.5 px-4 text-slate-200 font-semibold">{tile.id}</td>
                        <td className="py-2.5 px-4 text-slate-400">
                          {tile.pixelBox.left},{tile.pixelBox.top} ({tile.pixelBox.width}×
                          {tile.pixelBox.height}px)
                        </td>
                        <td className="py-2.5 px-4 font-sans text-slate-300">
                          {tile.isPerimeterTile ? 'Array Perimeter' : 'Interior Cells'}
                        </td>
                        <td className="py-2.5 px-4 text-right text-slate-300">
                          {(tile.pvCoverageRatio * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-4 text-right text-cyan-300">
                          {tile.offAxisEdgeScore.toFixed(2)}
                        </td>
                        <td className="py-2.5 px-4 text-right">
                          <span
                            className={
                              tile.detectionsCount > 0
                                ? 'text-rose-400 font-semibold'
                                : 'text-slate-500'
                            }
                          >
                            {tile.detectionsCount}
                          </span>
                        </td>
                        <td className="py-2.5 px-4 text-right font-sans">
                          <button
                            type="button"
                            onClick={() => {
                              setFocusedTile(tile);
                              setActiveTab('workspace');
                            }}
                            className="px-2.5 py-1 text-xs font-medium text-slate-200 bg-[#162033] border border-[#253655] rounded hover:bg-[#1d2b44] transition-colors whitespace-nowrap"
                          >
                            Focus Window
                          </button>
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
        </main>
      )}

      {/* Marked-image output tab */}
      {activeTab === 'marked' && (
        <main className="flex-1">
          <MarkedImagesView frames={frames} />
        </main>
      )}
    </div>
  );
}
