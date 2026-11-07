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
import { CellTrainingMonitor } from './components/CellTrainingMonitor';
import { CrackTrainingMonitor } from './components/CrackTrainingMonitor';
import { ForensicDrawer } from './components/ForensicDrawer';
import { MarkedImagesView } from './components/MarkedImagesView';
import { SettingsView, type AppInfo } from './components/SettingsView';
import { collectFilesFromDataTransfer, normalizeSelectedFileList } from './utils/geojson';
import { downloadBlob, markedFileName, renderMarkedImage } from './utils/markedImage';
import {
  deleteFeedback,
  getFeedbackSummary,
  resetModel,
  sendFeedbackBatch,
  getExistingImageFeedback,
  trainModel,
  getCrackTrainingStatus,
  withAnomalies,
  type FeedbackSummary,
  type CrackTrainingJobStatus,
} from './utils/feedback';
import type { TrainingSettings, CellDetectorTrainingSettings, CrackClassifierTrainingSettings, CellInferenceSettings, CellTypeFilterSettings, CellTypeProfile, CrackDetectionSettings } from './types/trainingSettings';
import type {
  DefectCategory,
  DetectedAnomaly,
  InspectionConfig,
  InspectionResult,
  PixelBox,
  TileDiagnostic,
} from './types/inspection';

type ActiveTab = 'workspace' | 'detections' | 'batch' | 'tiles' | 'crackAnnotation' | 'marked' | 'training' | 'trainingSettings' | 'dataModels';
type DetectionMode = 'cells' | 'cracks';
interface YoloInputPreview { kind: 'full' | 'tile'; window: PixelBox; modelWidth: number; modelHeight: number; scale: number; padX: number; padY: number; dataUrl: string }
interface CellDetectionState {
  status: 'queued' | 'running' | 'completed' | 'error';
  imageId: string;
  imageWidth?: number; imageHeight?: number; originalWidth?: number; originalHeight?: number;
  boxes: PixelBox[]; source?: 'yolo11n' | 'classical'; warning?: string; error?: string;
  diagnostics?: InspectionResult['cellDetectionDiagnostics']; reviewed?: boolean;
  /** Exact letterboxed tensors the YOLO model received (hidden viewer, Ctrl+Shift+Y). */
  inputPreviews?: YoloInputPreview[];
}
type FilterCategory = 'all' | 'spider_web_crack' | 'edge_spider_web' | 'surface_residue';

interface FrameEntry {
  id: string;
  label: string;
  sourceType: 'upload';
  fileObj?: File;
  imageSrc: string;
  result: InspectionResult | null;
  status: 'queued' | 'analyzing' | 'completed' | 'error';
  errorMessage?: string;
  /** Detections the user deleted as false; kept so they can be restored (feedbackId = stored training negative) */
  rejected?: { anomaly: DetectedAnomaly; feedbackId: string }[];
  cellDetection?: CellDetectionState;
}

const EMPTY_FRAME: FrameEntry = {
  id: 'empty-workspace', label: 'No image uploaded', sourceType: 'upload', imageSrc: '',
  result: null, status: 'queued',
};

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
  const [advancedMode, setAdvancedMode] = useState(false);
  const [detectionMode, setDetectionMode] = useState<DetectionMode>('cells');
  const [cellBatchRunning, setCellBatchRunning] = useState(false);
  const [cellSingleRunning, setCellSingleRunning] = useState(false);
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

  const [frames, setFrames] = useState<FrameEntry[]>([]);
  const [activeFrameId, setActiveFrameId] = useState<string>('');
  const [selectedAnomalyId, setSelectedAnomalyId] = useState<string | null>(null);
  const [focusedTile, setFocusedTile] = useState<TileDiagnostic | null>(null);

  const [categoryFilter, setCategoryFilter] = useState<FilterCategory>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [tileFilterMode, setTileFilterMode] = useState<'all' | 'inspected' | 'anomalies'>('all');

  const [batchRunning, setBatchRunning] = useState<boolean>(false);
  const [notice, setNotice] = useState<string>('');
  // v1.4.8.6: three-way choice when an image is already in the crack training data.
  const [saveChoice, setSaveChoice] = useState<{ label: string; positives: number; negatives: number; count: number; resolve: (choice: 'replace' | 'add' | 'cancel') => void } | null>(null);
  const askSaveChoice = (label: string, existing: { positives: number; negatives: number }, count: number) =>
    new Promise<'replace' | 'add' | 'cancel'>((resolve) => setSaveChoice({ label, positives: existing.positives, negatives: existing.negatives, count, resolve }));
  const answerSaveChoice = (choice: 'replace' | 'add' | 'cancel') => { saveChoice?.resolve(choice); setSaveChoice(null); };
  const noticeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const flash = useCallback((msg: string, ms = 7000) => {
    setNotice(msg);
    if (noticeTimer.current) clearTimeout(noticeTimer.current);
    noticeTimer.current = setTimeout(() => setNotice(''), ms);
  }, []);
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

  const framesRef = useRef<FrameEntry[]>(frames);
  framesRef.current = frames;
  const configRef = useRef<InspectionConfig>(config);
  configRef.current = config;
  const batchPausedRef = useRef<boolean>(false);

  const activeFrame = useMemo(
    () => frames.find((f) => f.id === activeFrameId) || frames[0] || EMPTY_FRAME,
    [frames, activeFrameId]
  );

  const analyzeSingleFrameEntry = useCallback(
    async (entry: FrameEntry, forceRefresh = false): Promise<InspectionResult | null> => {
      const currentFrame = framesRef.current.find((frame) => frame.id === entry.id);
      if (!currentFrame?.cellDetection || currentFrame.cellDetection.status !== 'completed') {
        setDetectionMode('cells');
        setActiveTab('detections');
        flash('Complete Cell detections for every uploaded image before starting Crack detection.', 10000);
        return null;
      }
      const cfg = configRef.current;
      setFrames((prev) =>
        prev.map((f) =>
          f.id === entry.id ? { ...f, status: 'analyzing', errorMessage: undefined } : f
        )
      );

      try {
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
            'x-detection-stage': 'cracks',
          },
          body: arrayBuf,
        });
        const data = await res.json().catch(() => ({} as any)); // non-JSON 5xx body must yield a readable error
        if (!res.ok) throw new Error(data.error || `Failed to inspect ${entry.label}`);
        const result = data as InspectionResult;

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

  // Inspect a user-uploaded frame when it is selected for the first time.
  const handleSelectActiveFrame = useCallback(
    (frameId: string) => {
      setActiveFrameId(frameId);
      setSelectedAnomalyId(null);
      setFocusedTile(null);
      // Selecting an image never launches a detector implicitly. Cell and crack
      // passes are explicit, separate operator actions in the Detections workspace.
      const target = framesRef.current.find((f) => f.id === frameId);
      if (target?.result?.anomalies.length) {
        const firstSpider = target.result.anomalies.find((a) => a.category === 'spider_web_crack') || target.result.anomalies[0];
        setSelectedAnomalyId(firstSpider.id);
      }
    },
    []
  );

  // Concurrent Batch Processing Runner for Folder / ZIP / Multi-Image Queues
  const startBatchProcessing = useCallback(
    async (explicitTargets?: FrameEntry[]) => {
      const allFrames = framesRef.current.filter((frame) => frame.imageSrc);
      if (allFrames.length && allFrames.some((frame) => frame.cellDetection?.status !== 'completed')) {
        setDetectionMode('cells');
        setActiveTab('detections');
        flash('Run Cell detections for every uploaded image before starting the Crack detection stage.', 10000);
        return;
      }
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
    [analyzeSingleFrameEntry, batchConcurrency, flash]
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

      setActiveFrameId(created[0].id);
      setSelectedAnomalyId(null);
      setDetectionMode('cells');
      setActiveTab('detections');
    },
    []
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

  const toggleAdvancedMode = () => {
    if (advancedMode) {
      setAdvancedMode(false);
      if (activeTab === 'tiles') setActiveTab('workspace');
    } else {
      setAdvancedMode(true);
    }
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
  const [cellTrainingStatus, setCellTrainingStatus] = useState<any>(null);
  const [cellDetectorStatus, setCellDetectorStatus] = useState<any>(null);
  const [cellSetupBusy, setCellSetupBusy] = useState(false);
  const [trainingSettings, setTrainingSettings] = useState<TrainingSettings | null>(null);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsSync, setSettingsSync] = useState<'idle' | 'dirty' | 'saving' | 'saved' | 'error'>('idle');
  const lastSavedSettings = useRef<string>('');
  const latestSettings = useRef<TrainingSettings | null>(null);
  const [appInfo, setAppInfo] = useState<AppInfo | null>(null);
  const [folderBannerDismissed, setFolderBannerDismissed] = useState(() => { try { return localStorage.getItem('pvcv.folderBannerDismissed') === '1'; } catch { return false; } });
  const [negativeUploadBusy, setNegativeUploadBusy] = useState(false);
  const [training, setTraining] = useState<boolean>(false);
  const [crackTrainingJob, setCrackTrainingJob] = useState<CrackTrainingJobStatus | null>(null);
  const crackTrainingNoticeJobRef = useRef<string>('');
  const [markedBusy, setMarkedBusy] = useState<boolean>(false);
  const errMsg = (e: unknown, fallback: string) => (e instanceof Error ? e.message : fallback);

  useEffect(() => {
    getFeedbackSummary().then(setFeedback).catch(() => undefined);
    fetch('/api/cell-detector').then((response) => response.json()).then(setCellDetectorStatus).catch(() => undefined);
    fetch('/api/cell-training/status').then((response) => response.json()).then(setCellTrainingStatus).catch((error) => setCellTrainingStatus({ status: 'not_ready', statusPollError: error instanceof Error ? error.message : 'Could not read the training status from the local server.' }));
    fetch('/api/training-settings', { cache: 'no-store' }).then((response) => response.json()).then((loaded) => { lastSavedSettings.current = JSON.stringify(loaded); setTrainingSettings(loaded); setSettingsSync('saved'); }).catch(() => undefined);
    fetch('/api/app/info', { cache: 'no-store' }).then((response) => response.json()).then(setAppInfo).catch(() => undefined);
    getCrackTrainingStatus().then((job) => { setCrackTrainingJob(job); setTraining(['preparing', 'training', 'validating'].includes(job.status)); }).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!cellTrainingStatus || !['setting_up', 'training'].includes(cellTrainingStatus.status)) return;
    const timer = window.setInterval(() => {
      fetch('/api/cell-training/status').then((response) => response.json()).then((data) => {
        setCellTrainingStatus(data);
        if (['completed', 'failed'].includes(data.status)) fetch('/api/cell-detector').then((response) => response.json()).then(setCellDetectorStatus).catch(() => undefined);
      }).catch((error) => setCellTrainingStatus((current: any) => ({ ...current, statusPollError: error instanceof Error ? error.message : 'Training status update failed; retrying automatically.' })));
    }, 2000);
    return () => window.clearInterval(timer);
  }, [cellTrainingStatus?.status]);

  useEffect(() => {
    if (!crackTrainingJob || !['preparing', 'training', 'validating'].includes(crackTrainingJob.status)) return;
    const timer = window.setInterval(() => {
      getCrackTrainingStatus().then((job) => {
        setCrackTrainingJob(job);
        const active = ['preparing', 'training', 'validating'].includes(job.status);
        setTraining(active);
        if (!active && job.jobId && crackTrainingNoticeJobRef.current === job.jobId) {
          crackTrainingNoticeJobRef.current = '';
          getFeedbackSummary().then(setFeedback).catch(() => undefined);
          if (job.status === 'completed' && job.report) {
            const report = job.report;
            const pct = (value: number) => `${Math.round(value * 100)}%`;
            flash(report.accepted
              ? `${report.algorithm ?? 'Crack'} classifier trained and activated. Held-out precision ${pct(report.validationAfter.precision)}, recall ${pct(report.validationAfter.recall)}, specificity ${pct(report.validationAfter.specificity)}, balanced accuracy ${pct(report.validationAfter.balancedAccuracy)}.`
              : `Training completed, but the candidate did not pass the validation gate. Previous active model retained. ${report.reason ?? ''}`, 14000);
          } else if (job.status === 'failed') flash(job.error || 'Crack-classifier training failed. Open the training log for details.', 16000);
        }
      }).catch(() => undefined);
    }, 1200);
    return () => window.clearInterval(timer);
  }, [crackTrainingJob?.status, crackTrainingJob?.jobId, flash]);

  const changeCellDetectorMode = async (mode: string) => {
    try {
      const response = await fetch('/api/cell-detector', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not change cell detector mode.');
      setCellDetectorStatus(data);
      flash(`Cell detector mode set to ${mode}. Re-scan an image to apply it.`);
    } catch (error) { flash(errMsg(error, 'Could not change cell detector mode.'), 10000); }
  };

  const setupCellTraining = async () => {
    setCellSetupBusy(true);
    try {
      const response = await fetch('/api/cell-training/setup', { method: 'POST' });
      const data = await response.json(); setCellTrainingStatus(data);
      if (!response.ok) throw new Error(data.error || 'Could not set up the YOLO engine.');
      flash('YOLO11 training runtime verified. Existing packages were reused when healthy; missing or broken packages were repaired.');
    } catch (error) { flash(errMsg(error, 'YOLO setup failed.'), 14000); }
    finally { setCellSetupBusy(false); }
  };

  const startCellTraining = async () => {
    try {
      if (!trainingSettings) throw new Error('Load Training Settings before starting a run.');
      const response = await fetch('/api/cell-training/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(trainingSettings.cellDetector) });
      const data = await response.json(); setCellTrainingStatus(data);
      if (!response.ok) throw new Error(data.error || 'Could not start cell training.');
      flash(`YOLO11 cell training started for ${trainingSettings.cellDetector.epochs} epochs. ${trainingSettings.cellDetector.patience === 0 ? "Early stopping is disabled; use Cancel to stop manually." : `Early stopping is enabled with patience ${trainingSettings.cellDetector.patience}.`}`, 14000);
    } catch (error) { flash(errMsg(error, 'Cell detector training failed to start.'), 14000); }
  };

  const cancelCellTraining = async () => {
    try {
      const response = await fetch('/api/cell-training/cancel', { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not cancel the cell training job.');
      setCellTrainingStatus(data);
    } catch (error) { flash(errMsg(error, 'Could not cancel the cell training job.'), 12000); }
  };

  // v1.4.8.0: settings persist automatically (debounced) instead of only on the Save button.
  useEffect(() => {
    latestSettings.current = trainingSettings;
    if (!trainingSettings) return;
    const body = JSON.stringify(trainingSettings);
    if (body === lastSavedSettings.current) return;
    setSettingsSync('dirty');
    const timer = window.setTimeout(async () => {
      setSettingsSync('saving');
      try {
        const response = await fetch('/api/training-settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Could not save training settings.');
        lastSavedSettings.current = JSON.stringify(data);
        if (latestSettings.current === trainingSettings && JSON.stringify(data) !== body) setTrainingSettings(data); // show server-clamped values
        setSettingsSync('saved');
      } catch { setSettingsSync('error'); }
    }, 900);
    return () => window.clearTimeout(timer);
  }, [trainingSettings]);
  const settingsBadge = (
    <span className={`text-[11px] ${settingsSync === 'error' ? 'text-rose-300' : settingsSync === 'saved' ? 'text-emerald-300' : 'text-slate-400'}`}>
      {settingsSync === 'saving' ? 'Saving…' : settingsSync === 'dirty' ? 'Unsaved changes…' : settingsSync === 'error' ? 'Auto-save failed — use Save' : settingsSync === 'saved' ? '✓ All changes saved automatically' : ''}
    </span>
  );

  const saveAllTrainingSettings = async () => {
    if (!trainingSettings) return;
    setSettingsSaving(true);
    try {
      const response = await fetch('/api/training-settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(trainingSettings) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not save training settings.');
      setTrainingSettings(data);
      lastSavedSettings.current = JSON.stringify(data); setSettingsSync('saved'); flash('Training settings saved to your data folder and shared by every version.');
    } catch (error) { flash(errMsg(error, 'Could not save training settings.'), 12000); }
    finally { setSettingsSaving(false); }
  };

  const uploadNegativeCellSamples = async (files: FileList | null) => {
    if (!files?.length) return;
    setNegativeUploadBusy(true);
    let saved = 0;
    const failures: string[] = [];
    try {
      for (const file of Array.from(files)) {
        try {
          const header = new TextEncoder().encode(JSON.stringify({ fileName: file.name }) + '\n');
          const image = new Uint8Array(await file.arrayBuffer());
          const payload = new Uint8Array(header.length + image.length);
          payload.set(header, 0); payload.set(image, header.length);
          const response = await fetch('/api/cell-annotations/negative', { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: payload });
          const data = await response.json();
          if (!response.ok) throw new Error(data.error || 'Upload rejected.');
          saved++;
        } catch (error) { failures.push(`${file.name}: ${errMsg(error, 'upload failed')}`); }
      }
      const latest = await fetch('/api/cell-training/status').then((response) => response.json()).catch(() => null); // save already succeeded; don't report failure on status refresh
      if (latest) setCellTrainingStatus(latest);
      flash(`${saved} background-only negative sample(s) saved.${failures.length ? `\n${failures.join('\n')}` : ''}`, failures.length ? 14000 : 8000);
    } finally { setNegativeUploadBusy(false); }
  };


  const patchFrame = useCallback((id: string, fn: (f: FrameEntry) => FrameEntry) => {
    setFrames((prev) => {
      const next = prev.map((f) => (f.id === id ? fn(f) : f));
      framesRef.current = next;
      return next;
    });
  }, []);

  const frameBytes = (f: FrameEntry): Promise<ArrayBuffer> =>
    f.fileObj ? f.fileObj.arrayBuffer() : fetch(f.imageSrc).then((r) => r.arrayBuffer());

  const detectCellsForFrame = async (entry: FrameEntry): Promise<boolean> => {
    patchFrame(entry.id, (frame) => ({ ...frame, cellDetection: { status: 'running', imageId: frame.id, boxes: [] } }));
    try {
      const bytes = await frameBytes(entry);
      const response = await fetch('/api/cell-detect/upload', {
        method: 'POST',
        headers: { 'Content-Type': entry.fileObj?.type || 'application/octet-stream', 'x-file-name': encodeURIComponent(entry.label), 'x-frame-id': entry.id, 'x-debug-inputs': '1' },
        body: bytes,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Cell detection failed for ${entry.label}.`);
      patchFrame(entry.id, (frame) => ({ ...frame, cellDetection: {
        status: 'completed', imageId: entry.id, imageWidth: data.imageWidth, imageHeight: data.imageHeight,
        originalWidth: data.originalWidth, originalHeight: data.originalHeight,
        boxes: Array.isArray(data.boxes) ? data.boxes : [], source: data.source, warning: data.warning, diagnostics: data.diagnostics,
        inputPreviews: Array.isArray(data.inputPreviews) ? data.inputPreviews : undefined,
      } }));
      return true;
    } catch (error) {
      patchFrame(entry.id, (frame) => ({ ...frame, cellDetection: { status: 'error', imageId: frame.id, boxes: [], error: errMsg(error, 'Cell detection failed.') } }));
      return false;
    }
  };

  const runCellDetectionsForAll = async () => {
    const targets = framesRef.current.filter((frame) => frame.imageSrc);
    if (!targets.length || cellBatchRunning || cellSingleRunning) return;
    setCellBatchRunning(true);
    let cursor = 0;
    let succeeded = 0;
    let failed = 0;
    const worker = async () => {
      while (cursor < targets.length) {
        const index = cursor++;
        const ok = await detectCellsForFrame(targets[index]);
        if (ok) succeeded++; else failed++;
      }
    };
    try {
      await Promise.all(Array.from({ length: Math.min(2, targets.length) }, () => worker()));
      flash(`Cell detection pass complete: ${succeeded} image(s) completed, ${failed} failed. Review and save corrected boxes for each image before training.`, failed ? 14000 : 10000);
    } finally { setCellBatchRunning(false); }
  };

  const runCellDetectionsForSelected = async () => {
    const target = framesRef.current.find((frame) => frame.id === activeFrameId && frame.imageSrc)
      ?? framesRef.current.find((frame) => frame.imageSrc);
    if (!target || cellBatchRunning || cellSingleRunning) return;
    setActiveFrameId(target.id);
    setCellSingleRunning(true);
    try {
      const ok = await detectCellsForFrame(target);
      const latest = framesRef.current.find((frame) => frame.id === target.id)?.cellDetection;
      if (!ok) {
        flash(`Cell detection failed for ${target.label}: ${latest?.error || 'Unknown detector error.'}`, 14000);
      } else {
        flash(`Cell detection completed for ${target.label}. Review and save the corrected boxes for this image.`, 10000);
      }
    } finally {
      setCellSingleRunning(false);
    }
  };

  const runCrackDetectionsForAll = async () => {
    const targets = framesRef.current.filter((frame) => frame.imageSrc);
    if (!targets.length || batchRunning) return;
    const reset = targets.map((frame) => ({ ...frame, status: 'queued' as const, result: null, errorMessage: undefined }));
    const resetById = new Map(reset.map((frame) => [frame.id, frame]));
    const nextFrames = framesRef.current.map((frame) => resetById.get(frame.id) ?? frame);
    framesRef.current = nextFrames;
    setFrames(nextFrames);
    setDetectionMode('cracks');
    setTimeout(() => { void startBatchProcessing(reset); }, 0);
  };

  const handleSaveCellBoxes = async (boxes: PixelBox[], source: { imageId: string; fileName: string; coordinateWidth: number; coordinateHeight: number }) => {
    // Resolve by the imageId that produced the detections, never by whichever frame
    // happens to be active when an asynchronous save is clicked.
    const frame = framesRef.current.find((item) => item.id === source.imageId || item.result?.imageId === source.imageId || item.cellDetection?.imageId === source.imageId);
    if (!frame || !frame.imageSrc || !frame.cellDetection?.imageWidth || !frame.cellDetection?.imageHeight) throw new Error('The source image could not be matched to a completed cell-detection pass. Run Cell detections for this image first.');
    const image = await frameBytes(frame);
    const metadata = new TextEncoder().encode(JSON.stringify({ fileName: source.fileName || frame.label, imageId: source.imageId, coordinateWidth: source.coordinateWidth, coordinateHeight: source.coordinateHeight, boxes }));
    const body = new Blob([metadata, new Uint8Array([10]), new Uint8Array(image)]);
    const response = await fetch('/api/cell-annotations', { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || 'Could not save cell box annotations.');
    setFrames((previous) => { const next = previous.map((item) => {
      if (item.id !== frame.id || !item.cellDetection) return item;
      // Keep the corrected set as the frame's cell base (crack labelling + Inspection read it).
      // Convert from the editor's coordinate space to the detection space if they differ.
      const det = item.cellDetection;
      const kx = (det.imageWidth ?? source.coordinateWidth) / Math.max(1, source.coordinateWidth);
      const ky = (det.imageHeight ?? source.coordinateHeight) / Math.max(1, source.coordinateHeight);
      const corrected = boxes.map((b) => ({ left: b.left * kx, top: b.top * ky, width: b.width * kx, height: b.height * ky }));
      return { ...item, cellDetection: { ...det, boxes: corrected, reviewed: true } };
    }); framesRef.current = next; return next; });
    const latest = await fetch('/api/cell-training/status').then((r) => r.json()).catch(() => ({} as any)); // save already succeeded; tolerate status refresh failure
    if (latest.status) setCellTrainingStatus(latest);
    flash(`Saved ${data.boxes} reviewed PV-cell boxes from ${source.fileName}. The complete corrected set for this exact image replaces its prior YOLO labels (${data.positiveImages ?? latest.positiveImages ?? latest.annotatedImages} positive images total).`);
  };

  const handleSaveCellAnnotations = async (items: CellAnnotationInput[]) => {
    const frame = activeFrame;
    // v1.4.8.3: the Crack Annotation tab only has a cell-detection pass (no crack `result`), so a
    // missing `result` must not block saving. Boxes live in whichever frame the canvas displayed.
    const coordinateSpace = frame?.result
      ? { width: frame.result.imageWidth, height: frame.result.imageHeight }
      : frame?.cellDetection?.status === 'completed' && frame.cellDetection.imageWidth && frame.cellDetection.imageHeight
        ? { width: frame.cellDetection.imageWidth, height: frame.cellDetection.imageHeight }
        : null;
    if (!frame || !frame.imageSrc || !coordinateSpace || !items.length) throw new Error('Select an image and run Cell detections (or inspect it) before saving cell labels.');
    const image = await frameBytes(frame);
    // v1.4.8.5: the same image already in the crack training set is REPLACED, never merged, after a warning.
    const existing = await getExistingImageFeedback(image);
    let replace = true;
    if (existing && existing.positives + existing.negatives > 0) {
      const choice = await askSaveChoice(frame.label, existing, items.length);
      if (choice === 'cancel') throw new Error('Save cancelled. The existing training labels for this image were kept.');
      replace = choice === 'replace';
    }
    const saved = await sendFeedbackBatch(image, frame.label, items.map((item) => ({
      box: item.box,
      label: item.label,
      category: 'spider_web_crack',
      moduleBox: item.moduleBox,
    })), coordinateSpace, replace);
    const replacedCount = (saved.replaced?.positives ?? 0) + (saved.replaced?.negatives ?? 0);
    // Saved annotations stay in the training dataset; they are not injected into the
    // detection list. A positive label is ground truth, not an inference result.
    setFeedback(await getFeedbackSummary());
    const positives = items.filter((item) => item.label === 1);
    const negatives = items.length - positives.length;
    flash(`${replace ? '' : 'Added to the existing labels. '}Added to the crack training annotations: ${items.length} cell label(s) from ${frame.label} (${positives.length} cracked, ${negatives} clean).${replacedCount ? ` Replaced ${replacedCount} earlier label(s) for this image.` : ''} Dataset now: ${saved.positives} cracked / ${saved.negatives} clean.`, 9000);
  };

  const handleDeleteAnomaly = async (anom: DetectedAnomaly) => {
    const frame = activeFrame;
    if (!frame?.result) return;
    const without = frame.result.anomalies.filter((a) => a.id !== anom.id);
    if (selectedAnomalyId === anom.id) setSelectedAnomalyId(null);

    if (anom.source === 'user') {
      patchFrame(frame.id, (f) => (f.result ? { ...f, result: withAnomalies(f.result, without) } : f));
      getFeedbackSummary().then(setFeedback).catch(() => undefined);
      flash(`${anom.id} removed.`);
      return;
    }

    // A model-training negative must be one exact, validated PV-cell crop. A detection
    // rectangle (especially a VLM multi-cell box) is never stored as a cell label.
    const cx = anom.pointPx.x;
    const cy = anom.pointPx.y;
    const matchedRegion = anom.category === 'spider_web_crack'
      ? frame.result.moduleRegions.find((region) => region.cellGrid?.cells.some((cell) => {
          const b = cell.pixelBox;
          return cx >= b.left && cx < b.left + b.width && cy >= b.top && cy < b.top + b.height;
        }))
      : undefined;
    const matchedCell = matchedRegion?.cellGrid?.cells.find((cell) => {
      const b = cell.pixelBox;
      return cx >= b.left && cx < b.left + b.width && cy >= b.top && cy < b.top + b.height;
    });
    patchFrame(frame.id, (f) =>
      f.result
        ? { ...f, result: withAnomalies(f.result, without), rejected: [...(f.rejected ?? []), { anomaly: anom, feedbackId: '' }] }
        : f
    );
    if (anom.category !== 'spider_web_crack' || !matchedRegion || !matchedCell) {
      flash(`${anom.id} removed but not used for training. Only spider-web crack detections mapped to one validated cell can be saved this way; label the exact cell as Negative to add a training example.`);
      return;
    }
    try {
      const saved = await sendFeedbackBatch(await frameBytes(frame), frame.label, [{ box: matchedCell.pixelBox, label: 0, category: 'spider_web_crack', moduleBox: matchedRegion.pixelBox }]);
      const id = saved.items[0]?.id;
      if (!id) throw new Error('The cell-negative annotation was not returned by the server.');
      patchFrame(frame.id, (f) => ({
        ...f,
        rejected: f.rejected?.map((r) => (r.anomaly.id === anom.id ? { ...r, feedbackId: id } : r)),
      }));
      getFeedbackSummary().then(setFeedback).catch(() => undefined);
      flash(`${anom.id} deleted; its exact validated cell was saved as a training negative.`);
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
      const job = await trainModel(trainingSettings?.crackClassifier);
      setCrackTrainingJob(job);
      crackTrainingNoticeJobRef.current = job.jobId ?? '';
      flash(`Crack-classifier training started (${job.algorithm ?? trainingSettings?.crackClassifier.algorithm ?? 'configured model'}). Live progress, epoch analytics and logs are shown below.`, 9000);
    } catch (e) {
      setTraining(false);
      flash(errMsg(e, 'Could not start crack-classifier training.'));
    }
  };

  const handleResetModel = async () => {
    const confirmed = window.confirm('Permanently clear all saved positive/negative cell labels, feedback source images, and the trained cell model? This cannot be undone.');
    if (!confirmed) return;
    try {
      setFeedback(await resetModel());
      setFrames((previous) => {
        const next = previous.map((frame) => ({ ...frame, result: null, status: 'queued' as const, rejected: [] }));
        framesRef.current = next;
        return next;
      });
      setSelectedAnomalyId(null);
      setFocusedTile(null);
      flash('Training data and trained model cleared. The next training run will start from zero weights.');
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
  const cellReviewResult = useMemo<InspectionResult | null>(() => {
    const detection = activeFrame?.cellDetection;
    if (!detection || detection.status !== 'completed' || !detection.imageWidth || !detection.imageHeight) return null;
    return {
      imageId: detection.imageId, fileName: activeFrame.label, imageWidth: detection.imageWidth, imageHeight: detection.imageHeight,
      megapixels: Number(((detection.imageWidth * detection.imageHeight) / 1_000_000).toFixed(1)), analyzedAt: '', processingTimeMs: 0,
      modelUsed: detection.source === 'yolo11n' ? 'YOLO11n cell-only pass' : 'Classical cell geometry',
      geoReference: null as any,
      pipelineSummary: { totalGridTiles: 0, pvActiveTiles: 0, vlmInspectedTiles: 0, spiderWebCracksCount: 0, edgeSpiderWebCracksCount: 0, impactPointsCount: 0, linearFracturesCount: 0, surfaceResidueCount: 0 },
      moduleRegions: [], cellDetectionBoxes: detection.boxes, cellDetectionSource: detection.source,
      cellDetectionDiagnostics: detection.diagnostics, tiles: [], anomalies: [], geoJson: { type: 'FeatureCollection', features: [] },
      warnings: detection.warning ? [detection.warning] : undefined,
    } as unknown as InspectionResult;
  }, [activeFrame?.id, activeFrame?.label, activeFrame?.cellDetection]);
  // Crack annotation is built on the (reviewed) cell boxes: they become the selectable cells.
  const crackLabelResult = useMemo<InspectionResult | null>(
    () => (cellReviewResult && cellReviewResult.cellDetectionBoxes?.length ? { ...cellReviewResult, cellDetectionSource: 'yolo11n' } as InspectionResult : null),
    [cellReviewResult],
  );
  // Inspection Stage shows BOTH layers: crack findings (from the crack pass) and the
  // cell boxes (from the cell pass, which the crack pass deliberately does not repeat).
  const inspectionResult = useMemo<InspectionResult | null>(() => {
    const result = activeFrame?.result ?? null;
    const detection = activeFrame?.cellDetection;
    if (!result || !detection || detection.status !== 'completed' || !detection.boxes.length || !detection.imageWidth || !detection.imageHeight) return result;
    if (result.cellDetectionSource === 'yolo11n' && result.cellDetectionBoxes?.length) return result;
    const kx = result.imageWidth / detection.imageWidth, ky = result.imageHeight / detection.imageHeight;
    return { ...result, cellDetectionSource: 'yolo11n', cellDetectionBoxes: detection.boxes.map((b) => ({ left: b.left * kx, top: b.top * ky, width: b.width * kx, height: b.height * ky })) };
  }, [activeFrame?.result, activeFrame?.cellDetection]);
  const [yoloViewerOpen, setYoloViewerOpen] = useState(false);
  // Hidden developer viewer: Ctrl+Shift+Y toggles it; Esc closes it.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === 'y') { event.preventDefault(); setYoloViewerOpen((open) => !open); }
      else if (event.key === 'Escape') setYoloViewerOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  const cellDoneCount = frames.filter((frame) => frame.cellDetection?.status === 'completed').length;
  const cellErrorCount = frames.filter((frame) => frame.cellDetection?.status === 'error').length;
  const cellQueuedCount = frames.filter((frame) => !frame.cellDetection || frame.cellDetection.status === 'queued').length;
  const queuedCount = frames.filter((f) => f.status === 'queued').length;
  const analyzingCount = frames.filter((f) => f.status === 'analyzing').length;
  const errorCount = frames.filter((f) => f.status === 'error').length;

  const updateCellTrainingSetting = (key: keyof CellDetectorTrainingSettings, value: number | boolean | string) => {
    setTrainingSettings((current) => current ? { ...current, cellDetector: { ...current.cellDetector, [key]: value } as CellDetectorTrainingSettings } : current);
  };
  const updateCrackTrainingSetting = (key: keyof CrackClassifierTrainingSettings, value: number | boolean | string) => {
    setTrainingSettings((current) => current ? { ...current, crackClassifier: { ...current.crackClassifier, [key]: value } as CrackClassifierTrainingSettings } : current);
  };
  // v1.4.7.0: inference-time cell parameters, cell-type filter and crack ground gate (all persisted with Save all training settings).
  const updateCellInferenceSetting = (key: keyof CellInferenceSettings, value: number | boolean) => {
    setTrainingSettings((current) => current ? { ...current, cellInference: { ...current.cellInference, [key]: value } as CellInferenceSettings } : current);
  };
  const updateCrackDetectionSetting = (key: keyof CrackDetectionSettings, value: number) => {
    setTrainingSettings((current) => current ? { ...current, crackDetection: { ...current.crackDetection, [key]: value } as CrackDetectionSettings } : current);
  };
  const updateCellTypeFilter = (patch: Partial<CellTypeFilterSettings>) => {
    setTrainingSettings((current) => current ? { ...current, cellTypeFilter: { ...current.cellTypeFilter, ...patch } } : current);
  };
  const updateCellTypeProfile = (id: string, patch: Partial<CellTypeProfile>) => {
    setTrainingSettings((current) => current ? { ...current, cellTypeFilter: { ...current.cellTypeFilter, profiles: current.cellTypeFilter.profiles.map((profile) => profile.id === id ? { ...profile, ...patch } : profile) } } : current);
  };
  const addCellTypeProfile = (mode: 'accept' | 'reject') => {
    setTrainingSettings((current) => current ? { ...current, cellTypeFilter: { ...current.cellTypeFilter, profiles: [...current.cellTypeFilter.profiles, { id: `type-${Date.now().toString(36)}`, name: mode === 'accept' ? 'New cell type' : 'Negative type', mode, enabled: true, aspect: 1, aspectTolerance: 0.3, minLongPx: 0, maxLongPx: 0 }] } } : current);
  };
  const removeCellTypeProfile = (id: string) => {
    setTrainingSettings((current) => current ? { ...current, cellTypeFilter: { ...current.cellTypeFilter, profiles: current.cellTypeFilter.profiles.filter((profile) => profile.id !== id) } } : current);
  };
  const learnCellTypeFromAnnotations = async () => {
    try {
      const response = await fetch('/api/cell-type-filter/learn', { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not learn the cell type.');
      setTrainingSettings(data.settings);
      flash(`Learned cell type from ${data.learned.samples} saved boxes: aspect ${data.learned.aspect} ± ${(data.learned.aspectTolerance * 100).toFixed(0)}%. Now the default accept type.`);
    } catch (error) { flash(errMsg(error, 'Could not learn the cell type.'), 10000); }
  };
  const numberField = (label: string, value: number, onChange: (next: number) => void, min: number, max: number, step = 1, help?: string) => (
    <label className="block space-y-1 text-xs text-slate-300"><span>{label}</span><input type="number" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} className="w-full rounded-md border border-[#263448] bg-[#07090e] px-2.5 py-2 text-xs text-slate-100" />{help && <span className="block text-[10px] leading-4 text-slate-500">{help}</span>}</label>
  );
  const selectField = (label: string, value: string | number, options: Array<{ value: string | number; label: string }>, onChange: (next: string) => void, help?: string) => (
    <label className="block space-y-1 text-xs text-slate-300"><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)} className="w-full rounded-md border border-[#263448] bg-[#07090e] px-2.5 py-2 text-xs text-slate-100">{options.map((option) => <option key={String(option.value)} value={option.value}>{option.label}</option>)}</select>{help && <span className="block text-[10px] leading-4 text-slate-500">{help}</span>}</label>
  );
  const toggleField = (label: string, checked: boolean, onChange: (next: boolean) => void, help?: string) => (
    <label className="flex min-h-10 items-start gap-2 rounded-md border border-[#263448] bg-[#07090e] px-3 py-2 text-xs text-slate-200"><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} className="mt-0.5 accent-cyan-400" /><span>{label}{help && <small className="mt-1 block text-[10px] leading-4 text-slate-500">{help}</small>}</span></label>
  );

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0e14] text-slate-100">
      {saveChoice && (
        <div role="dialog" aria-modal="true" className="fixed inset-0 z-[110] flex items-center justify-center bg-black/60 p-4">
          <div className="w-full max-w-md rounded-xl border border-amber-700/70 bg-[#0e131d] p-5 shadow-2xl space-y-3">
            <h2 className="text-sm font-semibold text-amber-200">Image already in the crack training data</h2>
            <p className="text-xs leading-relaxed text-slate-300"><b className="text-slate-100">{saveChoice.label}</b> already has {saveChoice.positives} cracked + {saveChoice.negatives} clean cell labels stored. You are saving {saveChoice.count} label(s) now.</p>
            <ul className="text-[11px] leading-relaxed text-slate-400 list-disc pl-4 space-y-1">
              <li><b className="text-slate-200">Replace</b> deletes the stored labels for this image and keeps only the new ones.</li>
              <li><b className="text-slate-200">Add to existing</b> keeps the stored labels; a cell at the same position is updated to the new label.</li>
            </ul>
            <div className="flex flex-wrap justify-end gap-2 pt-1">
              <button type="button" onClick={() => answerSaveChoice('cancel')} className="px-3 py-1.5 text-xs rounded-md border border-slate-600 text-slate-200">Cancel</button>
              <button type="button" onClick={() => answerSaveChoice('add')} className="px-3 py-1.5 text-xs font-semibold rounded-md bg-cyan-400 text-slate-950">Add to existing</button>
              <button type="button" onClick={() => answerSaveChoice('replace')} className="px-3 py-1.5 text-xs font-semibold rounded-md bg-amber-300 text-slate-950">Replace</button>
            </div>
          </div>
        </div>
      )}
      {notice && (
        <div role="status" className="fixed top-20 right-4 z-[100] max-w-md rounded-lg border border-emerald-700/70 bg-emerald-950/95 px-4 py-3 text-xs leading-relaxed text-emerald-100 shadow-2xl flex items-start gap-3">
          <span className="flex-1">{notice}</span>
          <button type="button" aria-label="Dismiss" onClick={() => setNotice('')} className="text-emerald-300 hover:text-white">✕</button>
        </div>
      )}
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
          <button type="button" onClick={() => { setDetectionMode('cells'); setActiveTab('detections'); }} className={`py-0.5 transition-colors whitespace-nowrap shrink-0 border-b-2 ${activeTab === 'detections' && detectionMode === 'cells' ? 'text-slate-100 border-cyan-400' : 'border-transparent hover:text-slate-200 hover:border-slate-600'}`}>1 · Cell Annotation</button>
          <button type="button" onClick={() => { setActiveTab('crackAnnotation'); }} className={`py-0.5 transition-colors whitespace-nowrap shrink-0 border-b-2 ${activeTab === 'crackAnnotation' ? 'text-slate-100 border-cyan-400' : 'border-transparent hover:text-slate-200 hover:border-slate-600'}`}>2 · Crack Annotation</button>
          <button type="button" onClick={() => { setDetectionMode('cracks'); setActiveTab('detections'); }} className={`py-0.5 transition-colors whitespace-nowrap shrink-0 border-b-2 ${activeTab === 'detections' && detectionMode === 'cracks' ? 'text-slate-100 border-cyan-400' : 'border-transparent hover:text-slate-200 hover:border-slate-600'}`}>Crack Detection</button>
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
          {advancedMode && <button
            type="button"
            onClick={() => setActiveTab('tiles')}
            className={`py-0.5 transition-colors whitespace-nowrap shrink-0 border-b-2 ${
              activeTab === 'tiles'
                ? 'text-slate-100 border-cyan-400'
                : 'border-transparent hover:text-slate-200 hover:border-slate-600'
            }`}
          >
            Detection Diagnostics
          </button>}
          <button type="button" onClick={() => setActiveTab('training')} className={`py-0.5 transition-colors whitespace-nowrap shrink-0 border-b-2 ${activeTab === 'training' ? 'text-slate-100 border-cyan-400' : 'border-transparent hover:text-slate-200 hover:border-slate-600'}`}>Training</button>
          <button type="button" onClick={() => setActiveTab('trainingSettings')} className={`py-0.5 transition-colors whitespace-nowrap shrink-0 border-b-2 ${activeTab === 'trainingSettings' ? 'text-slate-100 border-cyan-400' : 'border-transparent hover:text-slate-200 hover:border-slate-600'}`}>Training Settings</button>
          <button type="button" onClick={() => setActiveTab('dataModels')} className={`py-0.5 transition-colors whitespace-nowrap shrink-0 border-b-2 inline-flex items-center gap-1.5 ${activeTab === 'dataModels' ? 'text-slate-100 border-cyan-400' : 'border-transparent hover:text-slate-200 hover:border-slate-600'}`}>Data &amp; Models{appInfo?.dataFolder.isDefault && !appInfo.dataFolder.envOverride && <span className="h-1.5 w-1.5 rounded-full bg-amber-400" title="Choose a permanent data folder" />}{appInfo?.dataFolder.restartRequired && <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" title="Restart required" />}</button>
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

        <button type="button" onClick={toggleAdvancedMode} className={`shrink-0 px-3 py-2 text-[11px] rounded-lg border ${advancedMode ? 'border-amber-500/60 bg-amber-500/10 text-amber-200' : 'border-[#263448] text-slate-400 hover:text-slate-200'}`}>{advancedMode ? 'Advanced mode: ON' : 'Advanced mode'}</button>
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

      {appInfo?.dataFolder.isDefault && !appInfo.dataFolder.envOverride && !folderBannerDismissed && activeTab !== 'dataModels' && (
        <div className="flex flex-wrap items-center gap-3 border-b border-amber-600/30 bg-amber-500/10 px-4 py-2 text-xs text-amber-100">
          <span className="flex-1 min-w-[240px]">Your annotations, models and settings are stored next to this release. Choose a permanent data folder once so every future version finds them automatically.</span>
          <button type="button" onClick={() => setActiveTab('dataModels')} className="rounded-md border border-amber-500/60 bg-amber-500/20 px-3 py-1.5 font-semibold">Choose folder</button>
          <button type="button" onClick={() => { setFolderBannerDismissed(true); try { localStorage.setItem('pvcv.folderBannerDismissed', '1'); } catch { /* private mode */ } }} className="rounded-md px-2 py-1.5 text-amber-200/80 hover:text-amber-100">Later</button>
        </div>
      )}

      {activeTab === 'dataModels' && (
        <SettingsView
          flash={flash}
          onInfo={setAppInfo}
          onModelChanged={() => { fetch('/api/cell-detector').then((r) => r.json()).then(setCellDetectorStatus).catch(() => undefined); fetch('/api/cell-training/status').then((r) => r.json()).then(setCellTrainingStatus).catch(() => undefined); }}
          onSettingsReplaced={(next) => { lastSavedSettings.current = JSON.stringify(next); setTrainingSettings(next); setSettingsSync('saved'); }}
        />
      )}

      {/* Two-stage detection operations: cell localization first, crack analysis second. */}
      {activeTab === 'detections' && (
        <main className="flex-1 min-h-0 overflow-auto bg-[#090d14] p-4 md:p-6">
          <div className="max-w-[1600px] mx-auto space-y-4">
            <header className="flex flex-wrap items-start justify-between gap-4">
              <div><p className="text-[11px] uppercase tracking-[0.2em] text-cyan-400">Two-stage inspection workflow</p><h1 className="text-2xl font-semibold text-slate-100 mt-1">Detections</h1><p className="text-sm text-slate-400 mt-2">Run cell localization across the complete image set first. Review, correct and save those boxes. Then run crack detection across the same images as a separate pass.</p></div>
              <div className="flex gap-2"><button type="button" onClick={() => setActiveTab('batch')} className="px-3 py-2 rounded-lg border border-slate-700 bg-slate-800 text-xs text-slate-200">Upload / manage images</button><span className="px-3 py-2 rounded-lg border border-[#263448] bg-[#0e131d] text-xs text-slate-300">{frames.length} image(s) in queue</span></div>
            </header>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <button type="button" onClick={() => setDetectionMode('cells')} className={`text-left rounded-xl border p-4 transition-colors ${detectionMode === 'cells' ? 'border-cyan-500/70 bg-cyan-950/25' : 'border-[#263448] bg-[#0e131d] hover:border-slate-500'}`}><span className="text-[10px] font-semibold uppercase tracking-widest text-cyan-300">Stage 1</span><span className="mt-1 block text-lg font-semibold text-slate-100">Cell detections</span><span className="mt-1 block text-xs text-slate-400">Find individual PV-cell boxes on every image. Remove false boxes, move or resize detections, draw missed cells and save the reviewed boxes to the YOLO dataset.</span></button>
              <button type="button" onClick={() => setDetectionMode('cracks')} className={`text-left rounded-xl border p-4 transition-colors ${detectionMode === 'cracks' ? 'border-rose-500/70 bg-rose-950/20' : 'border-[#263448] bg-[#0e131d] hover:border-slate-500'}`}><span className="text-[10px] font-semibold uppercase tracking-widest text-rose-300">Stage 2</span><span className="mt-1 block text-lg font-semibold text-slate-100">Crack detection</span><span className="mt-1 block text-xs text-slate-400">Run the separate crack-analysis pipeline across the same image set after the cell-detection pass. Review crack findings per image.</span></button>
            </div>
            {detectionMode === 'cells' ? (
              <section className="space-y-3">
                <div className="rounded-xl border border-[#263448] bg-[#0e131d] p-4 flex flex-wrap items-center justify-between gap-3">
                  <div className="min-w-[240px]"><h2 className="text-sm font-semibold text-slate-100">Cell detection batch</h2><p className="mt-1 text-xs text-slate-400">Completed {cellDoneCount}/{frames.length} · Queued {cellQueuedCount} · Failed {cellErrorCount}. Cell detection does not run crack analysis.</p><p className="mt-1 text-[10px] text-cyan-200">Selected image: <span className="font-semibold">{activeFrame.imageSrc ? activeFrame.label : 'None selected'}</span></p></div>
                  <div className="flex flex-wrap gap-2">
                    <button type="button" disabled={!activeFrame.imageSrc || cellBatchRunning || cellSingleRunning} onClick={() => void runCellDetectionsForSelected()} className="px-4 py-2 rounded-lg border border-cyan-600/70 bg-cyan-950/40 text-cyan-100 text-xs font-semibold disabled:opacity-40">{cellSingleRunning ? 'Detecting selected image…' : 'Run selected image only'}</button>
                    <button type="button" disabled={!frames.length || cellBatchRunning || cellSingleRunning} onClick={() => void runCellDetectionsForAll()} className="px-4 py-2 rounded-lg bg-cyan-400 text-slate-950 text-xs font-semibold disabled:opacity-40">{cellBatchRunning ? 'Detecting cells across images…' : 'Run cell detections for all images'}</button>
                  </div>
                </div>
                <CellTrainingMonitor status={cellTrainingStatus} onOpenTraining={() => setActiveTab('training')} />
                <div className="grid grid-cols-1 xl:grid-cols-[340px_minmax(0,1fr)] gap-3 min-h-[560px]">
                  <section className="rounded-xl border border-[#263448] bg-[#0e131d] overflow-hidden">
                    <div className="px-3 py-2 border-b border-[#263448] text-xs font-semibold text-slate-200">Image queue · cell status</div>
                    <div className="max-h-[650px] overflow-auto divide-y divide-[#1e2636]">{frames.map((frame) => <button key={frame.id} type="button" onClick={() => setActiveFrameId(frame.id)} className={`w-full text-left p-3 ${activeFrame.id === frame.id ? 'bg-cyan-950/30' : 'hover:bg-[#131b2b]'}`}><span className="block truncate text-xs font-medium text-slate-100">{frame.label}</span><span className="mt-1 flex items-center justify-between gap-2 text-[10px] text-slate-400"><span>{frame.cellDetection?.status === 'completed' ? `${frame.cellDetection.boxes.length} box(es) · ${frame.cellDetection.source ?? 'unknown'}` : frame.cellDetection?.status === 'running' ? 'Detecting…' : frame.cellDetection?.status === 'error' ? `Failed: ${frame.cellDetection.error}` : 'Not run'}</span><span>{frame.cellDetection?.reviewed ? '✓ Reviewed' : frame.cellDetection?.status === 'completed' ? 'Review needed' : ''}</span></span></button>)}{!frames.length && <p className="p-5 text-xs text-slate-400">Upload one or more PV images to begin.</p>}</div>
                  </section>
                  <section className="min-h-[560px] rounded-xl border border-[#263448] bg-[#0e131d] p-2 flex flex-col gap-2">{activeFrame.imageSrc && cellReviewResult ? <InspectionCanvas key={`cells-${activeFrame.id}`} imageSrc={activeFrame.imageSrc} result={cellReviewResult} isAnalyzing={activeFrame.cellDetection?.status === 'running'} selectedAnomalyId={null} onSelectAnomaly={() => undefined} focusedTile={null} visibleCategories={visibleCategorySet} onSaveCellAnnotations={handleSaveCellAnnotations} onSaveCellBoxes={handleSaveCellBoxes} advancedMode={false} workflowMode="cells" /> : activeFrame.imageSrc ? <div className="flex-1 min-h-[420px] flex flex-col items-center justify-center gap-3 text-center p-6 text-xs text-slate-400"><p>This image has not been through cell localization yet.</p><button type="button" disabled={cellBatchRunning || cellSingleRunning} onClick={() => void runCellDetectionsForSelected()} className="rounded-lg bg-cyan-400 px-4 py-2 font-semibold text-slate-950 disabled:opacity-40">{cellSingleRunning ? 'Detecting this image…' : 'Run cell detection for this image'}</button><p>No crack analysis runs in Stage 1.</p></div> : <div className="flex-1 flex items-center justify-center text-xs text-slate-400">Upload images to begin.</div>}{activeFrame.cellDetection?.warning && <p className="rounded-lg border border-amber-900/60 bg-amber-950/20 p-3 text-xs text-amber-200">{activeFrame.cellDetection.warning}</p>}{activeFrame.cellDetection?.error && <p className="rounded-lg border border-rose-900/60 bg-rose-950/20 p-3 text-xs text-rose-200">{activeFrame.cellDetection.error}</p>}</section>
                </div>
                <div className="rounded-lg border border-[#263448] bg-[#0e131d] p-3 text-xs text-slate-400"><strong className="text-slate-200">Cell review guide:</strong> Yellow boxes are model proposals. Click to select, drag to move, drag a cyan corner to resize, Shift-click to multi-select, Delete to remove false boxes. Use Draw cell boxes to add missed cells. Save the complete reviewed set for each image. These labels train cell localization only; they do not label cracks.</div>
              </section>
            ) : (
              <section className="space-y-3">
                <div className="rounded-xl border border-[#263448] bg-[#0e131d] p-4 flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-sm font-semibold text-slate-100">Crack detection batch</h2><p className="mt-1 text-xs text-slate-400">Completed {completedResults.length}/{frames.length} · Queued {queuedCount} · Failed {errorCount}. This pass uses crack-analysis processing and does not repeat the direct YOLO cell-detection stage.</p></div><button type="button" disabled={!frames.length || batchRunning || cellDoneCount !== frames.length} onClick={() => void runCrackDetectionsForAll()} className="px-4 py-2 rounded-lg bg-rose-400 text-slate-950 text-xs font-semibold disabled:opacity-40">{batchRunning ? 'Detecting cracks…' : 'Run crack detection for all images'}</button></div>
                {cellDoneCount !== frames.length && frames.length > 0 && <p className="text-xs text-amber-200 rounded-lg border border-amber-900/60 bg-amber-950/20 p-3">Complete Cell detections for all {frames.length} images first. Crack detection unlocks after the first stage has finished for every image.</p>}
                <div className="rounded-xl border border-[#263448] bg-[#0e131d] overflow-hidden"><table className="w-full text-left text-xs"><thead className="bg-[#111724] text-slate-400"><tr><th className="p-3">Image</th><th className="p-3">Crack pass</th><th className="p-3 text-right">Findings</th><th className="p-3 text-right">Action</th></tr></thead><tbody className="divide-y divide-[#1e2636]">{frames.map((frame) => <tr key={frame.id}><td className="p-3 text-slate-200 max-w-[420px] truncate">{frame.label}</td><td className="p-3 text-slate-400">{frame.status === 'completed' ? 'Completed' : frame.status === 'analyzing' ? 'Running…' : frame.status === 'error' ? `Failed · ${frame.errorMessage ?? ''}` : 'Not run'}</td><td className="p-3 text-right text-rose-200">{frame.result?.anomalies.length ?? '—'}</td><td className="p-3 text-right"><button type="button" disabled={!frame.result} onClick={() => { setActiveFrameId(frame.id); setActiveTab('workspace'); }} className="px-2.5 py-1.5 rounded border border-slate-700 text-slate-200 disabled:opacity-40">Review cracks</button></td></tr>)}</tbody></table>{!frames.length && <p className="p-6 text-center text-xs text-slate-400">Upload image(s) and complete Cell detections first.</p>}</div>
              </section>
            )}
          </div>
        </main>
      )}

      {/* Main Asymmetric Split Console Workspace */}
      {activeTab === 'workspace' && (
        <main className="flex-1 min-h-0 overflow-hidden grid grid-cols-1 lg:grid-cols-[315px_minmax(0,1fr)_365px] divide-y lg:divide-y-0 lg:divide-x divide-[#1e2636]">
          {/* v1.4.8.7 — left column reordered into three numbered steps: 1 Images → 2 Scan & export → 3 Detection settings */}
          <aside className="bg-[#0e131d] p-4 flex flex-col gap-5 min-h-0 overflow-y-auto">
            {/* 1 · Images */}
            <section className="space-y-2">
              <div className="flex items-center justify-between">
                <h2 className="text-[11px] font-semibold uppercase tracking-wider text-slate-300">1 · Images</h2>
                <span className="text-[10px] font-mono text-slate-500">{frames.length} in queue</span>
              </div>
              <label htmlFor="active-frame-select" className="sr-only">Active inspection frame</label>
              <select
                id="active-frame-select"
                value={activeFrame.id}
                disabled={frames.length === 0}
                onChange={(e) => handleSelectActiveFrame(e.target.value)}
                className="w-full px-3 py-2 text-xs font-mono bg-[#07090e] border border-[#1e2636] rounded-md text-slate-200 focus:outline-none focus:border-cyan-400 disabled:opacity-60"
              >
                {frames.length === 0 && <option value="empty-workspace">No images uploaded yet</option>}
                {frames.map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.status === 'completed' ? '✓ ' : f.status === 'analyzing' ? '… ' : '○ '}
                    {f.label}
                  </option>
                ))}
              </select>
              <div className="grid grid-cols-3 gap-1.5">
                <button type="button" onClick={() => fileInputRef.current?.click()} className="inline-flex items-center justify-center gap-1 py-1.5 px-2 text-xs font-medium text-slate-200 bg-[#141d2e] border border-[#24324d] rounded hover:bg-[#1a263c] transition-colors">
                  <Upload className="w-3 h-3 text-cyan-400" />Files
                </button>
                <button type="button" onClick={() => folderInputRef.current?.click()} className="inline-flex items-center justify-center gap-1 py-1.5 px-2 text-xs font-medium text-slate-200 bg-[#141d2e] border border-[#24324d] rounded hover:bg-[#1a263c] transition-colors">
                  <FolderOpen className="w-3 h-3 text-cyan-400" />Folder
                </button>
                <button type="button" onClick={() => zipInputRef.current?.click()} className="inline-flex items-center justify-center gap-1 py-1.5 px-2 text-xs font-medium text-slate-200 bg-[#141d2e] border border-[#24324d] rounded hover:bg-[#1a263c] transition-colors">
                  <FileArchive className="w-3 h-3 text-cyan-400" />.ZIP
                </button>
              </div>
              <details className="group rounded-md border border-[#1e2636] bg-[#0b0f18]">
                <summary className="cursor-pointer select-none px-2.5 py-1.5 text-[11px] text-slate-400 hover:text-slate-200">Camera GPS metadata (optional)</summary>
                <div className="px-2.5 pb-2.5 space-y-1">
                  <button type="button" onClick={() => metaInputRef.current?.click()} className="w-full inline-flex items-center justify-center gap-1 py-1.5 px-2 text-xs font-medium text-slate-200 bg-[#141d2e] border border-[#24324d] rounded hover:bg-[#1a263c] transition-colors">
                    <Upload className="w-3 h-3 text-cyan-400" />Load metadata JSON
                  </button>
                  <p className="text-[10px] text-slate-500">Used for non-EXIF images when the georeferencing fallback below is off.</p>
                </div>
              </details>
              {metaStatus && <p className="text-[10px] text-slate-400">{metaStatus}</p>}
            </section>

            {/* 2 · Scan & export */}
            <section className="space-y-2 pt-4 border-t border-[#1e2636]">
              <h2 className="text-[11px] font-semibold uppercase tracking-wider text-slate-300">2 · Scan &amp; export</h2>
              <button
                type="button"
                disabled={frames.length === 0 || activeFrame.status === 'analyzing'}
                onClick={() => analyzeSingleFrameEntry(activeFrame, true)}
                className="w-full inline-flex items-center justify-center gap-2 py-2.5 px-4 text-xs font-semibold text-slate-100 bg-[#172236] border border-[#293b5c] rounded-lg hover:bg-[#1e2c46] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {activeFrame.status === 'analyzing' ? (
                  <><RefreshCw className="w-3.5 h-3.5 animate-spin" />Scanning Solar Cells...</>
                ) : (
                  <><Play className="w-3.5 h-3.5 text-cyan-400" />Re-Scan Active Frame</>
                )}
              </button>
              <button
                type="button"
                disabled={!activeFrame.result || markedBusy}
                onClick={() => void handleDownloadMarked()}
                className="w-full inline-flex items-center justify-center gap-2 py-2 px-3 text-xs font-medium text-cyan-300 bg-[#0b1320] border border-[#1f314d] rounded-lg hover:bg-[#121e33] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                {markedBusy ? 'Rendering…' : 'Download marked image (.jpg)'}
              </button>
              {/* Training controls are isolated in the dedicated Training Centre tab. */}
            </section>

            {/* 3 · Detection settings */}
            <section className="space-y-3 pt-4 border-t border-[#1e2636]">
              <h2 className="text-[11px] font-semibold uppercase tracking-wider text-slate-300">3 · Detection settings</h2>

              <div className="space-y-1.5">
                <span className="block text-xs text-slate-400">Detection Sensitivity</span>
                <div className="grid grid-cols-3 gap-1 p-1 bg-[#07090e] border border-[#1e2636] rounded-md">
                  {(['high', 'balanced', 'strict'] as const).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => setConfig((c) => ({ ...c, sensitivity: mode }))}
                      className={`py-1.5 text-xs font-medium rounded transition-colors capitalize whitespace-nowrap ${
                        config.sensitivity === mode ? 'bg-[#1a2436] text-cyan-300' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="max-tiles-slider" className="text-slate-400">Maximum Inspection Tiles</label>
                  <span className="font-mono tabular-nums text-slate-200">{config.maxTilesToInspect} tiles</span>
                </div>
                <input
                  id="max-tiles-slider"
                  type="range"
                  min={6}
                  max={16}
                  step={1}
                  value={config.maxTilesToInspect}
                  onChange={(e) => setConfig((c) => ({ ...c, maxTilesToInspect: Number(e.target.value) }))}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
              </div>

              <div className="space-y-2 rounded-md border border-[#1e2636] bg-[#0b0f18] p-2.5">
                <span className="block text-[10px] uppercase tracking-wider text-slate-500">Crack search</span>
                <label className="flex items-center justify-between gap-2 text-xs text-slate-300 cursor-pointer">
                  <span>Quadrant Perimeter Window Slicing</span>
                  <input type="checkbox" checked={config.edgeFocusMode} onChange={(e) => setConfig((c) => ({ ...c, edgeFocusMode: e.target.checked }))} className="accent-cyan-400 rounded" />
                </label>
                <label className="flex items-center justify-between gap-2 text-xs text-slate-300 cursor-pointer">
                  <span>Adaptive Silicon Off-Axis Ridge Boost</span>
                  <input type="checkbox" checked={config.contrastEnhancement} onChange={(e) => setConfig((c) => ({ ...c, contrastEnhancement: e.target.checked }))} className="accent-cyan-400 rounded" />
                </label>
                <label className="flex items-center justify-between gap-2 text-xs text-slate-300 cursor-pointer">
                  <span>Reject Ground / Off-Array Detections</span>
                  <input type="checkbox" checked={config.rejectGround} onChange={(e) => setConfig((c) => ({ ...c, rejectGround: e.target.checked }))} className="accent-cyan-400 rounded" />
                </label>
              </div>

              <div className="space-y-2 rounded-md border border-[#1e2636] bg-[#0b0f18] p-2.5">
                <span className="block text-[10px] uppercase tracking-wider text-slate-500">Georeferencing</span>
                <label className="flex items-center justify-between gap-2 text-xs text-slate-300 cursor-pointer">
                  <span>EPSG:3857 Site Fallback (Non-EXIF)</span>
                  <input type="checkbox" checked={config.useSiteReferenceForNonGps} onChange={(e) => setConfig((c) => ({ ...c, useSiteReferenceForNonGps: e.target.checked }))} className="accent-cyan-400 rounded" />
                </label>
              </div>
            </section>
          </aside>

          {/* Center Optical Stage */}
          <section className="p-5 flex flex-col gap-4 bg-[#0b0e14] min-w-0 min-h-0 overflow-hidden">
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

            {!activeFrame.imageSrc && (
              <div className="flex flex-col items-center justify-center gap-3 min-h-[260px] border border-dashed border-[#2a3752] rounded-lg bg-[#0e131d] text-center px-6 py-10">
                <Upload className="w-8 h-8 text-cyan-400" />
                <div>
                  <h2 className="text-sm font-semibold text-slate-100">Clean inspection workspace</h2>
                  <p className="text-xs text-slate-400 mt-1">No sample images or previous detections are preloaded. Upload your own PV images to begin cell detection and annotation.</p>
                </div>
                <button type="button" onClick={() => fileInputRef.current?.click()} className="px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 rounded-lg hover:bg-cyan-300">Upload inspection images</button>
              </div>
            )}
            {/* Quantitative Telemetry Strip */}
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pb-3 border-b border-[#1e2636]">
              <div>
                <span className="text-xs text-slate-400 block">Edge Spider-Web</span>
                <div className="mt-0.5 flex items-baseline gap-1.5">
                  <span
                    className={`text-xl font-mono font-semibold tabular-nums ${
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
                    className={`text-xl font-mono font-semibold tabular-nums ${
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
                  <span className="text-xl font-mono font-semibold tabular-nums text-amber-400">
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
                  <span className="text-xl font-mono font-semibold tabular-nums text-sky-400">
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
              <div className="flex flex-wrap items-center gap-3 px-3 py-1.5 text-[11px] text-slate-300 bg-[#0b0f18] border-b border-[#1e2636]">
                <span className="font-semibold text-slate-200">Legend</span>
                <span className="inline-flex items-center gap-1"><i className="inline-block w-3 h-3 border-2 border-dashed border-emerald-400" />Clean cell</span>
                <span className="inline-flex items-center gap-1"><i className="inline-block w-3 h-3 border-2 border-rose-500 bg-rose-500/30" />Cracked cell</span>
                <span className="inline-flex items-center gap-1"><i className="inline-block w-3 h-3 border-2 border-dashed border-cyan-400" />Selected</span>
                <span className="text-slate-500">Missed something? Draw cell boxes or mark cracked/clean cells, then Save. Both feed the next training run.</span>
              </div>
              <InspectionCanvas
                imageSrc={activeFrame.imageSrc}
                result={inspectionResult}
                isAnalyzing={activeFrame.status === 'analyzing'}
                selectedAnomalyId={selectedAnomalyId}
                onSelectAnomaly={setSelectedAnomalyId}
                focusedTile={focusedTile}
                visibleCategories={visibleCategorySet}
                onSaveCellAnnotations={handleSaveCellAnnotations}
                onSaveCellBoxes={handleSaveCellBoxes}
                advancedMode={false}
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
                    {advancedMode && (
                      <button
                        type="button"
                        onClick={() => void handleDeleteAnomaly(anom)}
                        title={anom.source === 'user' ? 'Remove this issue you added' : 'Delete: not a real issue (saved as a training negative)'}
                        aria-label={`Delete ${anom.id}`}
                        className="absolute right-2.5 bottom-2.5 p-1.5 text-slate-500 hover:text-rose-300 hover:bg-rose-950/40 rounded transition-colors"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                    </div>
                  );
                })
              )}
            </div>

            {advancedMode && (activeFrame.rejected?.length ?? 0) > 0 && (
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
      {/* Crack annotation: separate from cell annotation. Cells (reviewed or model-proposed) are the base;
          the operator marks cracked (Positive) / clean (Negative) cells for the crack classifier. */}
      {activeTab === 'crackAnnotation' && (
        <main className="flex-1 min-h-0 overflow-hidden bg-[#090d14] p-4 md:p-6 flex flex-col gap-3">
          <header className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-[11px] uppercase tracking-[0.2em] text-rose-300">Stage 2 · Crack annotation</p>
              <h1 className="text-xl font-semibold text-slate-100 mt-1">Crack annotation</h1>
              <p className="text-xs text-slate-400 mt-1">Cell boxes from the cell model (corrected in Cell Annotation) are the base. Click a cell, choose Positive (cracked) or Negative (clean), then Save. These labels train only the crack classifier, never the YOLO cell model.</p>
            </div>
            <div className="flex items-center gap-2">
              <label htmlFor="crack-annotation-frame-select" className="text-xs text-slate-400">Source image</label>
              <select id="crack-annotation-frame-select" value={activeFrame.id} onChange={(event) => handleSelectActiveFrame(event.target.value)} className="max-w-[320px] px-3 py-2 rounded-lg bg-[#07090e] border border-[#263448] text-xs text-slate-200">
                {frames.length === 0 && <option value={EMPTY_FRAME.id}>No image loaded</option>}
                {frames.map((frame) => <option key={frame.id} value={frame.id}>{frame.label} · {frame.cellDetection?.status === 'completed' ? `${frame.cellDetection.boxes.length} cells${frame.cellDetection.reviewed ? ' ✓' : ''}` : 'no cell pass'}</option>)}
              </select>
            </div>
          </header>
          <div className="flex-1 h-[calc(100vh-210px)] min-h-[680px]">
            {activeFrame.imageSrc && crackLabelResult ? (
              <InspectionCanvas
                key={`crack-label-${activeFrame.id}-${crackLabelResult.cellDetectionBoxes?.length ?? 0}`}
                imageSrc={activeFrame.imageSrc}
                result={crackLabelResult}
                isAnalyzing={false}
                selectedAnomalyId={null}
                onSelectAnomaly={() => undefined}
                focusedTile={null}
                visibleCategories={visibleCategorySet}
                onSaveCellAnnotations={handleSaveCellAnnotations}
                onSaveCellBoxes={handleSaveCellBoxes}
                advancedMode={false}
                workflowMode="crackLabel"
              />
            ) : (
              <div className="h-full rounded-lg border border-dashed border-[#263448] flex items-center justify-center text-center p-6 text-sm text-slate-400">
                {activeFrame.imageSrc ? 'This image has no cell boxes yet. Run Cell detections (or draw/save cell boxes) in the Cell Annotation tab first.' : 'Upload images in Batch, then run the cell stage.'}
              </div>
            )}
          </div>
        </main>
      )}

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
              <p className="text-[11px] uppercase tracking-[0.2em] text-amber-300">Advanced detection controls</p>
              <h1 className="text-2xl font-semibold text-slate-100">Detection Diagnostics</h1>
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

          <section className="rounded-xl border border-[#263448] bg-[#0e131d] p-4 flex flex-wrap items-center justify-between gap-4">
              <div className="min-w-0">
                <h2 className="text-sm font-semibold text-slate-100">Direct cell detector</h2>
                <p className="text-xs text-slate-400 mt-1">{cellDetectorStatus?.message || 'Checking local cell detector status…'}</p>
                <p className="text-[11px] text-slate-500 mt-1">Active path: {cellDetectorStatus?.active || '—'} · Model: {cellDetectorStatus?.modelExists ? 'trained cell.onnx found' : 'no trained ONNX model; classical geometry fallback'}</p>
              </div>
              <label className="flex items-center gap-2 text-xs text-slate-300">Detector mode
                <select value={cellDetectorStatus?.selected || 'auto'} onChange={(event) => void changeCellDetectorMode(event.target.value)} className="px-3 py-2 rounded-lg bg-[#07090e] border border-[#263448] text-xs text-slate-200">
                  <option value="auto">Auto (recommended)</option><option value="yolo11n">YOLO11n direct cells</option><option value="classical">Classical cell geometry</option>
                </select>
              </label>
            </section>

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

      {activeTab === 'training' && (
        <main className="flex-1 min-h-0 overflow-auto bg-[#090d14] p-5 md:p-8">
          <div className="max-w-6xl mx-auto space-y-6">
            <header><p className="text-[11px] uppercase tracking-[0.2em] text-cyan-400">Model operations</p><h1 className="text-2xl font-semibold text-slate-100 mt-1">Training Centre</h1><p className="text-sm text-slate-400 mt-2">Direct cell localization and crack classification are separate model families with separate datasets. Saving a reviewed YOLO box set adds that exact inspected field image to the localization dataset; the next training run rebuilds from all saved reviewed images and continues from the latest app-trained checkpoint. Crack/clean labels train only the separate classifier.</p></header>
            {cellTrainingStatus?.statusPollError && <div className="rounded-lg border border-amber-800/60 bg-amber-950/20 px-3 py-2 text-xs text-amber-200">Training status could not be refreshed: {cellTrainingStatus.statusPollError}</div>}
            <CrackTrainingMonitor job={crackTrainingJob} feedback={feedback} training={training} />
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
              <section className="rounded-xl border border-cyan-900/60 bg-[#0b1420] p-5 space-y-4">
                <div className="flex items-center justify-between"><h2 className="text-base font-semibold text-cyan-100">01 · Direct cell detection (YOLO11n)</h2><span className="text-xs text-slate-400">{cellTrainingStatus?.status || 'checking'}</span></div>
                <p className="text-sm text-slate-400">Train direct cell localization from complete individual-cell box sets saved from the images actually inspected, including new field images that were not in the previous training run. No module detector is used. At least 2 distinct positive source images are required so validation is held out by image; annotate every visible cell in each PV image. Add separate background-only negatives in Training Settings. Candidate activation requires held-out mAP50 ≥ 0.15. Subsequent runs continue from the latest app-trained best.pt checkpoint when one is available.</p>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2"><div className="rounded-lg bg-[#080d15] p-3"><div className="text-xl font-semibold text-slate-100">{cellTrainingStatus?.annotatedImages ?? 0}</div><div className="text-[11px] text-slate-500">Positive images</div></div><div className="rounded-lg bg-[#080d15] p-3"><div className="text-xl font-semibold text-slate-100">{cellTrainingStatus?.annotatedCells ?? 0}</div><div className="text-[11px] text-slate-500">Cell boxes</div></div><div className="rounded-lg bg-[#080d15] p-3"><div className="text-xl font-semibold text-slate-100">{cellTrainingStatus?.negativeSamples ?? 0}</div><div className="text-[11px] text-slate-500">Negative images</div></div><div className="rounded-lg bg-[#080d15] p-3"><div className="text-xl font-semibold text-slate-100">{cellTrainingStatus?.activeModelExists ? 'Ready' : '—'}</div><div className="text-[11px] text-slate-500">Active model</div></div></div>
                <details className="rounded-lg border border-[#25364d] bg-[#080d15] p-3"><summary className="cursor-pointer text-xs font-semibold text-slate-200">Review YOLO training source images ({cellTrainingStatus?.trainingSources?.length ?? 0})</summary><p className="mt-2 text-[10px] text-slate-500">This list is the actual full-image localization dataset. After saving corrected boxes from a different field image, verify that image's filename appears here before starting the next training run.</p><div className="mt-2 max-h-48 overflow-auto space-y-1">{(cellTrainingStatus?.trainingSources ?? []).slice(-30).map((source: any) => <div key={source.imageFile} className="flex items-center justify-between gap-3 border-t border-[#1d2b3d] py-1 text-[10px]"><span className="min-w-0 truncate text-slate-300" title={source.imageFile}>{source.fileName}</span><span className="shrink-0 font-mono text-cyan-200">{source.boxes} boxes · {source.sampleType}</span></div>)}{!(cellTrainingStatus?.trainingSources?.length) && <p className="text-[10px] text-slate-500">No reviewed images have been saved yet.</p>}</div></details>
                <p className="text-[10px] text-slate-500">{cellTrainingStatus?.checkpointReusable ? `Next training run will continue from ${cellTrainingStatus.reusableCheckpoint || 'the previous best checkpoint'}.` : 'No previous PV cell checkpoint found; the first run starts from the official YOLO11n base weights.'}</p>
                <p className="text-[11px] text-slate-500">{Math.max(0, 2 - (cellTrainingStatus?.annotatedImages ?? 0))} more distinct annotated image(s) required before training can start. Only complete, full-image YOLO cell-box annotations count here; cracked/clean feedback is excluded from YOLO training and trains the separate crack classifier. First-time setup downloads the training runtime and YOLO packages; training currently runs on CPU. Early-stopping patience is {trainingSettings?.cellDetector.patience ?? 0}; zero disables early stopping, while a positive value opts in to plateau-based stopping.</p>
                {!advancedMode && <div className="rounded-lg border border-amber-800/50 bg-amber-950/20 p-3 text-sm text-amber-100 flex flex-wrap items-center justify-between gap-3"><span>Create training labels in Cell Annotation (cell boxes) and Crack Annotation (cracked / clean cells).</span><button type="button" onClick={() => { setDetectionMode('cells'); setActiveTab('detections'); }} className="px-3 py-2 rounded-lg border border-amber-700/70 bg-amber-500/10 text-amber-100 text-xs">Open Cell Annotation</button></div>}
                <div className="flex flex-wrap items-center gap-2"><button type="button" disabled={cellSetupBusy || ['setting_up','training'].includes(cellTrainingStatus?.status)} onClick={() => void setupCellTraining()} className="px-3 py-2 rounded-lg bg-cyan-500/15 border border-cyan-700/60 text-cyan-100 text-xs disabled:opacity-40">{cellSetupBusy || cellTrainingStatus?.status === 'setting_up' ? 'Checking training runtime…' : cellTrainingStatus?.runtimeReady ? 'Verify / reuse training runtime' : 'Set up in-app training runtime'}</button><button type="button" onClick={() => setActiveTab('trainingSettings')} className="px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-200 text-xs">Edit all detector parameters</button><span className="text-[11px] text-slate-400">{trainingSettings ? `${trainingSettings.cellDetector.epochs} epochs · ${trainingSettings.cellDetector.imageSize}px · batch ${trainingSettings.cellDetector.batch}` : 'Loading training settings…'}</span></div>
                {['training','setting_up'].includes(cellTrainingStatus?.status) && <div className="space-y-2"><div className="h-2 rounded-full bg-slate-800 overflow-hidden"><div className="h-full bg-cyan-400 transition-[width] duration-500" style={{width:`${cellTrainingStatus.progress || 0}%`}}/></div><p className="text-xs text-slate-300">{cellTrainingStatus.message} · {cellTrainingStatus.progress || 0}%</p><p className="text-[11px] text-emerald-300">Job state is saved on this computer. Refreshing the browser does not cancel a running training worker. Keep the PV CrackVision server running.</p></div>}
                <div className="flex gap-2"><button type="button" disabled={!trainingSettings || !cellTrainingStatus?.runtimeReady || (cellTrainingStatus?.positiveImages || cellTrainingStatus?.annotatedImages || 0) < 2 || ['training','setting_up'].includes(cellTrainingStatus?.status)} onClick={() => void startCellTraining()} className="px-4 py-2 rounded-lg bg-emerald-500/15 border border-emerald-700/60 text-emerald-100 text-xs disabled:opacity-40">Start cell detector training</button><button type="button" disabled={cellTrainingStatus?.status !== 'training'} onClick={() => void cancelCellTraining()} className="px-4 py-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-200 text-xs disabled:opacity-40">Cancel</button></div>
                {cellTrainingStatus?.error && <div className="rounded-lg border border-rose-900/60 bg-rose-950/20 p-3"><p className="text-xs text-rose-200 whitespace-pre-wrap">{cellTrainingStatus.error}</p>{cellTrainingStatus?.errorTraceback && <details className="mt-2 text-[11px] text-rose-300"><summary>Full error traceback</summary><pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap">{cellTrainingStatus.errorTraceback}</pre></details>}</div>}
                <section className="rounded-lg border border-[#25364d] bg-[#080d15] p-3 space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2"><div><h3 className="text-sm font-semibold text-slate-100">Epoch analytics</h3><p className="text-[10px] text-slate-500">Metrics are recorded after each epoch's validation pass.</p></div><div className="flex gap-2">{cellTrainingStatus?.metricsDownloadUrl && <a href={cellTrainingStatus.metricsDownloadUrl} className="px-2.5 py-1.5 rounded border border-cyan-800/70 text-[10px] text-cyan-200 hover:bg-cyan-950/40">Download CSV</a>}{cellTrainingStatus?.logDownloadUrl && <a href={cellTrainingStatus.logDownloadUrl} className="px-2.5 py-1.5 rounded border border-slate-700 text-[10px] text-slate-200 hover:bg-slate-800">Full worker log</a>}</div></div>
                  <p className="text-[10px] text-slate-500">Precision, recall and mAP measure validation quality. Confidence is an inference threshold/score, not a per-epoch training metric.</p>
                  <div className="overflow-auto max-h-[360px] rounded border border-[#1d2b3d]"><table className="min-w-[1180px] w-full text-[11px] text-right tabular-nums"><thead className="sticky top-0 bg-[#111b2a] text-slate-400"><tr><th className="p-2 text-left">Epoch</th><th className="p-2">Precision</th><th className="p-2">Recall</th><th className="p-2">mAP50</th><th className="p-2">mAP50–95</th><th className="p-2">Train box</th><th className="p-2">Train cls</th><th className="p-2">Train DFL</th><th className="p-2">Val box</th><th className="p-2">Val cls</th><th className="p-2">Val DFL</th><th className="p-2">Timestamp</th></tr></thead><tbody>{(cellTrainingStatus?.epochMetrics || []).map((row: any) => <tr key={row.epoch} className="border-t border-[#1d2b3d] hover:bg-[#111b2a]"><td className="p-2 text-left text-slate-200 font-semibold">{row.epoch}</td><td className="p-2 text-cyan-200">{typeof row.precision === 'number' ? row.precision.toFixed(4) : '—'}</td><td className="p-2 text-cyan-200">{typeof row.recall === 'number' ? row.recall.toFixed(4) : '—'}</td><td className="p-2 text-emerald-200">{typeof row.map50 === 'number' ? row.map50.toFixed(4) : '—'}</td><td className="p-2 text-emerald-200">{typeof row.map50_95 === 'number' ? row.map50_95.toFixed(4) : '—'}</td><td className="p-2 text-slate-300">{typeof row.box_loss === 'number' ? row.box_loss.toFixed(4) : '—'}</td><td className="p-2 text-slate-300">{typeof row.cls_loss === 'number' ? row.cls_loss.toFixed(4) : '—'}</td><td className="p-2 text-slate-300">{typeof row.dfl_loss === 'number' ? row.dfl_loss.toFixed(4) : '—'}</td><td className="p-2 text-violet-200">{typeof row.val_box_loss === 'number' ? row.val_box_loss.toFixed(4) : '—'}</td><td className="p-2 text-violet-200">{typeof row.val_cls_loss === 'number' ? row.val_cls_loss.toFixed(4) : '—'}</td><td className="p-2 text-violet-200">{typeof row.val_dfl_loss === 'number' ? row.val_dfl_loss.toFixed(4) : '—'}</td><td className="p-2 text-slate-500">{row.timestamp ? new Date(row.timestamp).toLocaleString() : '—'}</td></tr>)}{!(cellTrainingStatus?.epochMetrics?.length) && <tr><td colSpan={12} className="p-6 text-center text-slate-500">No epoch metrics yet. Start training to populate this table after the first validation epoch.</td></tr>}</tbody></table></div>
                </section>
                <details open className="rounded-lg border border-[#25364d] bg-[#080d15] p-3 text-xs text-slate-400"><summary className="cursor-pointer font-semibold text-slate-200">Live training log ({cellTrainingStatus?.log?.length || 0} saved entries)</summary><pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-words font-mono text-[10px] leading-5">{(cellTrainingStatus?.log || []).slice(-100).join('\n') || 'Waiting for worker output…'}</pre></details>
                {cellTrainingStatus?.metrics && <div className="rounded-lg bg-[#080d15] p-3"><div className="text-xs font-semibold text-slate-200 mb-2">Final held-out validation summary</div><div className="grid grid-cols-2 gap-2">{Object.entries(cellTrainingStatus.metrics).map(([key, value]) => <div key={key} className="text-[11px] text-slate-400">{key.replaceAll('_',' ')} <span className="text-slate-100 font-mono">{Number(value).toFixed(4)}</span></div>)}</div></div>}
                {cellTrainingStatus?.status === 'completed' && <p className="text-xs text-emerald-300">Cell detector exported to ONNX and activated. {cellTrainingStatus?.onnxRuntimeReady ? 'The ONNX inference runtime is available; re-scan an image to use the trained detector.' : 'Close and relaunch PV CrackVision with start.bat (Windows) or start.sh (macOS/Linux) once; the launcher will install the ONNX inference runtime before opening the app.'}</p>}
                {cellTrainingStatus?.status === 'cancelled' && <p className="text-xs text-amber-300">Training was stopped manually. Completed-epoch checkpoints and the previously active model are retained.</p>}
                {cellTrainingStatus?.previousModelExists && <button type="button" onClick={async () => { try { const response = await fetch('/api/cell-training/rollback', { method: 'POST' }); const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Rollback failed.'); setCellTrainingStatus(data); flash('Previous cell detector restored.'); } catch (error) { flash(errMsg(error, 'Rollback failed.'), 12000); } }} className="px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-200 text-xs">Roll back previous cell model</button>}
              </section>
              <section className="rounded-xl border border-rose-900/50 bg-[#120e14] p-5 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2"><h2 className="text-base font-semibold text-rose-100">02 · Crack classification</h2><div className="flex items-center gap-2"><span className="text-xs text-slate-400">{feedback?.userTuned ? 'Trained' : 'Not trained'}</span><button type="button" onClick={() => setActiveTab('trainingSettings')} className="rounded-md border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-[10px] text-slate-200">Edit classifier parameters</button></div></div>
                <p className="text-sm text-slate-400">This is a separate cell-level classifier trained from reviewed cracked and clean cell crops. Its labels are not used by the YOLO cell detector.</p>
                {feedback && <div className="grid grid-cols-3 gap-2"><div className="rounded-lg bg-[#0b0b11] p-3"><div className="text-xl text-slate-100">{feedback.positives}</div><div className="text-[11px] text-slate-500">Cracked</div></div><div className="rounded-lg bg-[#0b0b11] p-3"><div className="text-xl text-slate-100">{feedback.negatives}</div><div className="text-[11px] text-slate-500">Clean</div></div><div className="rounded-lg bg-[#0b0b11] p-3"><div className="text-xl text-slate-100">{feedback.images}</div><div className="text-[11px] text-slate-500">Images</div></div></div>}
                {feedback && <div className="text-xs text-slate-400">{feedback.trainingReady ? 'Training requirements met.' : `More reviewed labels required: ${feedback.trainingMissing.join(' · ')}`}</div>}
                <div className="flex flex-wrap items-center gap-2"><button type="button" disabled={training || !feedback?.trainingReady} onClick={() => void handleTrain()} className="px-4 py-2 rounded-lg bg-emerald-500/15 border border-emerald-700/60 text-emerald-100 text-xs disabled:opacity-40">{training ? 'Training…' : 'Train crack classifier'}</button><button type="button" disabled={training || !feedback || feedback.total === 0} onClick={() => void handleResetModel()} className="px-4 py-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-200 text-xs disabled:opacity-40">Clear crack dataset + model</button><a href="#crack-classifier-monitor" className="px-3 py-2 rounded-lg border border-rose-800/70 bg-rose-500/10 text-rose-100 text-xs">View live analytics ↑</a></div>

              </section>
            </div>
            <p className="text-[11px] text-slate-500">Validation is performed on held-out source images. A model should only be activated after its validation has completed successfully. The existing crack classifier and YOLO cell detector are separate model families.</p>
          </div>
        </main>
      )}

      {activeTab === 'trainingSettings' && (
        <main className="flex-1 min-h-0 overflow-auto bg-[#090d14] p-5 md:p-8">
          <div className="max-w-6xl mx-auto space-y-6">
            <header className="space-y-2">
              <p className="text-[11px] uppercase tracking-[0.2em] text-cyan-400">Persistent model configuration</p>
              <h1 className="text-2xl font-semibold text-slate-100">Training Settings</h1>
              <p className="text-sm text-slate-400">Configure both model families independently. Changes are saved automatically to your data folder (change it under Data &amp; Models) and reused by every version.</p>
              <div className="flex flex-wrap gap-2"><button type="button" onClick={() => void saveAllTrainingSettings()} disabled={!trainingSettings || settingsSaving} className="rounded-md border border-emerald-700/60 bg-emerald-500/15 px-4 py-2 text-xs font-semibold text-emerald-100 disabled:opacity-40">{settingsSaving ? 'Saving settings…' : 'Save all training settings'}</button><button type="button" onClick={() => setActiveTab('training')} className="rounded-md border border-slate-700 bg-slate-800 px-4 py-2 text-xs text-slate-200">Back to Training Centre</button>{settingsBadge}</div>
            </header>
            {!trainingSettings && <div className="rounded-lg border border-amber-800/60 bg-amber-950/20 p-4 text-sm text-amber-100">Could not load training settings from the local server. Refresh the page after checking that PV CrackVision is running.</div>}
            {trainingSettings && <>
              <section className="rounded-xl border border-cyan-900/60 bg-[#0b1420] p-5 space-y-4">
                <div><h2 className="text-lg font-semibold text-cyan-100">01 · Cell detector — YOLO11n</h2><p className="mt-1 text-xs text-slate-400">Detection training, transfer learning, optimizer, loss weights and augmentation parameters. The runtime is CPU-only in this release.</p></div>
                <div className="rounded-lg border border-emerald-900/60 bg-emerald-950/20 p-3 text-xs text-emerald-100"><strong>Early stopping:</strong> patience 0 (the default) disables automatic early stopping, so training continues to the configured epoch count unless you press Cancel or a real worker/system failure occurs. Set patience above 0 only when you explicitly want plateau-based early stopping. The validation activation threshold never stops the training loop.</div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {numberField('Epochs', trainingSettings.cellDetector.epochs, (v) => updateCellTrainingSetting('epochs', v), 1, 500, 1, 'Requested training epoch limit.') }
                  {numberField('Early-stopping patience', trainingSettings.cellDetector.patience, (v) => updateCellTrainingSetting('patience', v), 0, 500, 1, '0 = disabled; training reaches the requested epoch count. A positive value enables early stopping after that many epochs without validation-fitness improvement.')}
                  {selectField('Image size', trainingSettings.cellDetector.imageSize, [416,512,640,768,960].map((v) => ({value:v,label:`${v} × ${v}`})), (v) => updateCellTrainingSetting('imageSize', Number(v)), 'Larger sizes preserve smaller PV-cell details but need more RAM and time.')}
                  {numberField('Batch size', trainingSettings.cellDetector.batch, (v) => updateCellTrainingSetting('batch', v), 1, 16, 1, 'CPU memory dependent.')}
                  {numberField('Freeze first N layers', trainingSettings.cellDetector.freeze, (v) => updateCellTrainingSetting('freeze', v), 0, 30, 1, '0 = no frozen layers. Useful for transfer learning from a stable checkpoint.')}
                  {selectField('Optimizer', trainingSettings.cellDetector.optimizer, ['auto','SGD','Adam','AdamW','RMSProp'].map((v) => ({value:v,label:v})), (v) => updateCellTrainingSetting('optimizer', v), 'Use AdamW/Adam to make the explicit learning rate take effect; auto chooses optimizer settings itself.')}
                  {numberField('Initial learning rate (lr0)', trainingSettings.cellDetector.lr0, (v) => updateCellTrainingSetting('lr0', v), 0.000001, 1, 0.0001)}
                  {numberField('Final LR factor (lrf)', trainingSettings.cellDetector.lrf, (v) => updateCellTrainingSetting('lrf', v), 0.000001, 1, 0.001)}
                  {numberField('Momentum / beta1', trainingSettings.cellDetector.momentum, (v) => updateCellTrainingSetting('momentum', v), 0, 0.999, 0.001)}
                  {numberField('Weight decay', trainingSettings.cellDetector.weightDecay, (v) => updateCellTrainingSetting('weightDecay', v), 0, 1, 0.0001)}
                  {numberField('Warmup epochs', trainingSettings.cellDetector.warmupEpochs, (v) => updateCellTrainingSetting('warmupEpochs', v), 0, 50, 0.5)}
                  {numberField('Warmup momentum', trainingSettings.cellDetector.warmupMomentum, (v) => updateCellTrainingSetting('warmupMomentum', v), 0, 0.999, 0.001)}
                  {numberField('Warmup bias LR', trainingSettings.cellDetector.warmupBiasLr, (v) => updateCellTrainingSetting('warmupBiasLr', v), 0, 1, 0.01)}
                  {numberField('Close mosaic (last N epochs)', trainingSettings.cellDetector.closeMosaic, (v) => updateCellTrainingSetting('closeMosaic', v), 0, 500, 1)}
                  {numberField('Mosaic probability', trainingSettings.cellDetector.mosaic, (v) => updateCellTrainingSetting('mosaic', v), 0, 1, 0.05)}
                  {numberField('MixUp probability', trainingSettings.cellDetector.mixup, (v) => updateCellTrainingSetting('mixup', v), 0, 1, 0.05)}
                  {numberField('Rotation degrees', trainingSettings.cellDetector.degrees, (v) => updateCellTrainingSetting('degrees', v), 0, 180, 1)}
                  {numberField('Translation fraction', trainingSettings.cellDetector.translate, (v) => updateCellTrainingSetting('translate', v), 0, 1, 0.01)}
                  {numberField('Scale gain', trainingSettings.cellDetector.scale, (v) => updateCellTrainingSetting('scale', v), 0, 1, 0.05)}
                  {numberField('Horizontal flip', trainingSettings.cellDetector.fliplr, (v) => updateCellTrainingSetting('fliplr', v), 0, 1, 0.05)}
                  {numberField('Vertical flip', trainingSettings.cellDetector.flipud, (v) => updateCellTrainingSetting('flipud', v), 0, 1, 0.05)}
                  {numberField('HSV hue', trainingSettings.cellDetector.hsvH, (v) => updateCellTrainingSetting('hsvH', v), 0, 0.5, 0.001)}
                  {numberField('HSV saturation', trainingSettings.cellDetector.hsvS, (v) => updateCellTrainingSetting('hsvS', v), 0, 1, 0.05)}
                  {numberField('HSV value', trainingSettings.cellDetector.hsvV, (v) => updateCellTrainingSetting('hsvV', v), 0, 1, 0.05)}
                  {numberField('Box loss gain', trainingSettings.cellDetector.boxLoss, (v) => updateCellTrainingSetting('boxLoss', v), 0.1, 50, 0.1)}
                  {numberField('Class loss gain', trainingSettings.cellDetector.clsLoss, (v) => updateCellTrainingSetting('clsLoss', v), 0.1, 50, 0.1)}
                  {numberField('DFL loss gain', trainingSettings.cellDetector.dflLoss, (v) => updateCellTrainingSetting('dflLoss', v), 0.1, 50, 0.1)}
                  {numberField('Multi-scale fraction', trainingSettings.cellDetector.multiScale, (v) => updateCellTrainingSetting('multiScale', v), 0, 1, 0.05)}
                  {numberField('Random seed', trainingSettings.cellDetector.seed, (v) => updateCellTrainingSetting('seed', v), 0, 2147483647, 1)}
                  {numberField('Checkpoint save period', trainingSettings.cellDetector.savePeriod, (v) => updateCellTrainingSetting('savePeriod', v), -1, 500, 1, '-1 disables periodic copies; best/last checkpoints are still saved.')}
                  {numberField('Minimum mAP50 to activate model', trainingSettings.cellDetector.minimumMap50, (v) => updateCellTrainingSetting('minimumMap50', v), 0, 0.95, 0.01, 'The candidate must meet this held-out threshold after all requested epochs; it never stops training early.')}
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
                  {toggleField('Cosine LR schedule', trainingSettings.cellDetector.cosLr, (v) => updateCellTrainingSetting('cosLr', v))}
                  {toggleField('Deterministic training', trainingSettings.cellDetector.deterministic, (v) => updateCellTrainingSetting('deterministic', v))}
                  {toggleField('Cache dataset in memory', trainingSettings.cellDetector.cache, (v) => updateCellTrainingSetting('cache', v), 'Can increase memory pressure on large images.')}
                  {toggleField('Use pretrained weights', trainingSettings.cellDetector.pretrained, (v) => updateCellTrainingSetting('pretrained', v))}
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-slate-400"><div className="rounded-md bg-[#080d15] p-3">Device: <strong className="text-slate-200">CPU (fixed)</strong></div><div className="rounded-md bg-[#080d15] p-3">Workers: <strong className="text-slate-200">0 (Windows-safe)</strong></div><div className="rounded-md bg-[#080d15] p-3">AMP: <strong className="text-slate-200">Off for CPU training</strong></div><div className="rounded-md bg-[#080d15] p-3">Validation: <strong className="text-slate-200">Every epoch</strong></div><div className="rounded-md bg-[#080d15] p-3">Early stopping: <strong className={trainingSettings.cellDetector.patience === 0 ? "text-emerald-300" : "text-amber-300"}>{trainingSettings.cellDetector.patience === 0 ? "Off (patience 0)" : `On (patience ${trainingSettings.cellDetector.patience})`}</strong></div><div className="rounded-md bg-[#080d15] p-3">Time limit: <strong className="text-slate-200">None</strong></div><div className="rounded-md bg-[#080d15] p-3">Checkpoints: <strong className="text-slate-200">best.pt / last.pt</strong></div><div className="rounded-md bg-[#080d15] p-3">Activation gate: <strong className="text-slate-200">mAP50 ≥ {trainingSettings.cellDetector.minimumMap50.toFixed(2)}</strong></div></div>
              </section>

              <section className="rounded-xl border border-rose-900/50 bg-[#120e14] p-5 space-y-4">
                <div><h2 className="text-lg font-semibold text-rose-100">02 · Crack classifier</h2><p className="mt-1 text-xs text-slate-400">v1.4.8.7: every sample is one cell crop. <strong>Gradient boosting</strong> runs on richer features (edge orientation + crack continuity) and a <strong>small CNN</strong> runs on the cell crop itself; the ensemble averages both. Boosting trains in seconds; the CNN adds a few minutes. The legacy logistic model remains selectable.</p></div>
                <div className="rounded-lg border border-amber-900/60 bg-amber-950/20 p-3 text-xs text-amber-100"><strong>Freeze layers do not apply here.</strong> There are no convolutional/backbone layers in this classifier. The controls below are its actual optimization and validation parameters.</div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {selectField('Classifier', trainingSettings.crackClassifier.algorithm, [
                    { value: 'ensemble', label: 'Ensemble — gradient boosting + small CNN (recommended)' },
                    { value: 'gbm', label: 'Gradient boosting only (fastest)' },
                    { value: 'cnn', label: 'Small CNN only' },
                    { value: 'logistic', label: 'Legacy logistic regression (v1.4.8.6)' },
                  ], (v) => updateCrackTrainingSetting('algorithm', v))}
                  {numberField('Boosting trees', trainingSettings.crackClassifier.gbmTrees, (v) => updateCrackTrainingSetting('gbmTrees', v), 10, 1000, 10)}
                  {numberField('Boosting tree depth', trainingSettings.crackClassifier.gbmDepth, (v) => updateCrackTrainingSetting('gbmDepth', v), 1, 8, 1)}
                  {numberField('Boosting learning rate', trainingSettings.crackClassifier.gbmLearningRate, (v) => updateCrackTrainingSetting('gbmLearningRate', v), 0.005, 0.5, 0.005)}
                  {numberField('Boosting min samples per leaf', trainingSettings.crackClassifier.gbmMinLeaf, (v) => updateCrackTrainingSetting('gbmMinLeaf', v), 1, 50, 1)}
                  {numberField('Boosting row subsample', trainingSettings.crackClassifier.gbmSubsample, (v) => updateCrackTrainingSetting('gbmSubsample', v), 0.3, 1, 0.05)}
                  {numberField('CNN epochs', trainingSettings.crackClassifier.cnnEpochs, (v) => updateCrackTrainingSetting('cnnEpochs', v), 1, 200, 1)}
                  {numberField('CNN learning rate', trainingSettings.crackClassifier.cnnLearningRate, (v) => updateCrackTrainingSetting('cnnLearningRate', v), 0.0001, 0.05, 0.001)}
                  {numberField('CNN batch size', trainingSettings.crackClassifier.cnnBatchSize, (v) => updateCrackTrainingSetting('cnnBatchSize', v), 4, 128, 4)}
                  {numberField('Gradient iterations (logistic only)', trainingSettings.crackClassifier.iterations, (v) => updateCrackTrainingSetting('iterations', v), 100, 50000, 100)}
                  {numberField('Learning rate', trainingSettings.crackClassifier.learningRate, (v) => updateCrackTrainingSetting('learningRate', v), 0.000001, 1, 0.001)}
                  {numberField('L2 regularization', trainingSettings.crackClassifier.l2, (v) => updateCrackTrainingSetting('l2', v), 0, 10, 0.001)}
                  {numberField('Validation fraction', trainingSettings.crackClassifier.validationFraction, (v) => updateCrackTrainingSetting('validationFraction', v), 0.1, 0.4, 0.05)}
                  {numberField('Threshold minimum', trainingSettings.crackClassifier.thresholdMin, (v) => updateCrackTrainingSetting('thresholdMin', v), 0.05, 0.8, 0.01)}
                  {numberField('Threshold maximum', trainingSettings.crackClassifier.thresholdMax, (v) => updateCrackTrainingSetting('thresholdMax', v), 0.2, 0.95, 0.01)}
                  {numberField('Threshold search step', trainingSettings.crackClassifier.thresholdStep, (v) => updateCrackTrainingSetting('thresholdStep', v), 0.005, 0.2, 0.005)}
                  {numberField('Minimum held-out balanced accuracy', trainingSettings.crackClassifier.minimumBalancedAccuracy, (v) => updateCrackTrainingSetting('minimumBalancedAccuracy', v), 0.5, 0.95, 0.01)}
                </div>
                {toggleField('Balance positive and negative classes', trainingSettings.crackClassifier.classBalance, (v) => updateCrackTrainingSetting('classBalance', v), 'Applies inverse-frequency sample weights to every classifier (boosting, CNN, logistic).')}
              </section>

              <section className="rounded-xl border border-amber-900/50 bg-[#15120b] p-5 space-y-3">
                <div><h2 className="text-lg font-semibold text-amber-100">03 · Negative cell-detector samples</h2><p className="mt-1 text-xs text-slate-400">Add whole images that contain no visible PV cells to detect. YOLO receives an empty label file for each image and learns that background should not produce cell boxes.</p></div>
                <div className="rounded-lg border border-rose-900/60 bg-rose-950/20 p-3 text-xs text-rose-100"><strong>Important:</strong> Do not upload a PV-module image with unboxed cells as a negative sample. That would teach the detector to treat real PV cells as background. Use images such as ground, roof, vegetation, sky, roads or other scenes with no target cells.</div>
                <div className="flex flex-wrap items-center gap-3"><label className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-amber-700/70 bg-amber-500/10 px-3 py-2 text-xs font-semibold text-amber-100">{negativeUploadBusy ? 'Adding negative samples…' : 'Add background-only images'}<input type="file" accept="image/*" multiple disabled={negativeUploadBusy} onChange={(event) => { void uploadNegativeCellSamples(event.target.files); event.currentTarget.value = ''; }} className="hidden" /></label><span className="text-xs text-slate-400">Saved negatives: <strong className="text-slate-100">{cellTrainingStatus?.negativeSamples ?? 0}</strong></span><span className="text-xs text-slate-400">Positive cell images: <strong className="text-slate-100">{cellTrainingStatus?.positiveImages ?? cellTrainingStatus?.annotatedImages ?? 0}</strong></span></div>
                <p className="text-[10px] text-slate-500">Images are stored in the shared Working Folder/data/training/cell_annotations folder and persist across versions. Negative images are split across training and validation when two or more are present.</p>
              </section>
              <section className="rounded-xl border border-sky-900/50 bg-[#0b1320] p-5 space-y-4">
                <div><h2 className="text-lg font-semibold text-sky-100">04 · Cell detection — inference parameters (NMS &amp; limits)</h2><p className="mt-1 text-xs text-slate-400">Every value here used to be fixed in code. Defaults reproduce the previous behaviour. Applied immediately on the next Cell detections run (after Save).</p></div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {numberField('Confidence threshold', trainingSettings.cellInference.confidenceThreshold, (v) => updateCellInferenceSetting('confidenceThreshold', v), 0.0001, 0.95, 0.005, 'Low = high recall (more false boxes to delete). Raise to cut noise.')}
                  {numberField('NMS IoU — per pass', trainingSettings.cellInference.decoderNmsIou, (v) => updateCellInferenceSetting('decoderNmsIou', v), 0.05, 0.95, 0.05, 'Overlap above which two boxes in one pass count as duplicates.')}
                  {numberField('NMS IoU — merge passes', trainingSettings.cellInference.mergeNmsIou, (v) => updateCellInferenceSetting('mergeNmsIou', v), 0.05, 0.95, 0.05, 'Same, when merging the full-image pass with the detail tiles.')}
                  {numberField('Max candidates before NMS', trainingSettings.cellInference.maxCandidatesBeforeNms, (v) => updateCellInferenceSetting('maxCandidatesBeforeNms', v), 10, 5000, 10)}
                  {numberField('Max boxes per pass', trainingSettings.cellInference.maxDecoderBoxes, (v) => updateCellInferenceSetting('maxDecoderBoxes', v), 10, 5000, 10)}
                  {numberField('Max boxes per image', trainingSettings.cellInference.maxFinalBoxes, (v) => updateCellInferenceSetting('maxFinalBoxes', v), 10, 10000, 10)}
                  {numberField('Tile size (× model input)', trainingSettings.cellInference.tileScale, (v) => updateCellInferenceSetting('tileScale', v), 1, 6, 0.1)}
                  {numberField('Tile overlap', trainingSettings.cellInference.tileOverlap, (v) => updateCellInferenceSetting('tileOverlap', v), 0.1, 0.45, 0.01)}
                  {numberField('Max tiles per image', trainingSettings.cellInference.maxTiles, (v) => updateCellInferenceSetting('maxTiles', v), 1, 64, 1)}
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {toggleField('Detail tiles in addition to the whole-image pass', trainingSettings.cellInference.useTiles, (v) => updateCellInferenceSetting('useTiles', v))}
                  {toggleField('Reject large boxes that enclose repeating small cells', trainingSettings.cellInference.enclosingSuppression, (v) => updateCellInferenceSetting('enclosingSuppression', v))}
                </div>
              </section>

              <section className="rounded-xl border border-emerald-900/50 bg-[#0b1511] p-5 space-y-4">
                <div><h2 className="text-lg font-semibold text-emerald-100">05 · Cell types — only these YOLO detections pass</h2><p className="mt-1 text-xs text-slate-400">Aspect = long side ÷ short side of the box (1.0 = square). A detection passes only if it matches an enabled <strong>accept</strong> type and no enabled <strong>negative</strong> type, and its long side is within the relative window of the image's median cell. Everything else is rejected before you see it.</p></div>
                {toggleField('Enable cell-type filter', trainingSettings.cellTypeFilter.enabled, (v) => updateCellTypeFilter({ enabled: v }))}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {numberField('Relative size min (× median)', trainingSettings.cellTypeFilter.relativeMin, (v) => updateCellTypeFilter({ relativeMin: v }), 0, 1, 0.05, '0 = off')}
                  {numberField('Relative size max (× median)', trainingSettings.cellTypeFilter.relativeMax, (v) => updateCellTypeFilter({ relativeMax: v }), 0, 20, 0.05, '0 = off')}
                  {numberField('Reject smaller than: min short side (px)', trainingSettings.cellTypeFilter.minShortPx, (v) => updateCellTypeFilter({ minShortPx: v }), 0, 100000, 1, '0 = off')}
                  {numberField('Reject smaller than: min area (× typical cell)', trainingSettings.cellTypeFilter.minAreaRelative, (v) => updateCellTypeFilter({ minAreaRelative: v }), 0, 1, 0.05, '0 = off; typical cell = confident half of detections')}
                  {numberField('Reject smaller than: min short side (× typical cell)', trainingSettings.cellTypeFilter.minShortRelative, (v) => updateCellTypeFilter({ minShortRelative: v }), 0, 1, 0.05, '0 = off')}
                  {numberField('Reject below confidence', trainingSettings.cellTypeFilter.minConfidence, (v) => updateCellTypeFilter({ minConfidence: v }), 0, 0.95, 0.01, '0 = off')}
                </div>
                <div className="space-y-3">
                  {trainingSettings.cellTypeFilter.profiles.map((profile) => (
                    <div key={profile.id} className={`rounded-lg border p-3 space-y-2 ${profile.mode === 'accept' ? 'border-emerald-900/60 bg-emerald-950/10' : 'border-rose-900/60 bg-rose-950/10'}`}>
                      <div className="flex flex-wrap items-center gap-2">
                        <input value={profile.name} onChange={(event) => updateCellTypeProfile(profile.id, { name: event.target.value })} className="min-w-48 flex-1 rounded-md border border-[#263448] bg-[#07090e] px-2.5 py-1.5 text-xs text-slate-100" />
                        <select value={profile.mode} onChange={(event) => updateCellTypeProfile(profile.id, { mode: event.target.value as 'accept' | 'reject' })} className="rounded-md border border-[#263448] bg-[#07090e] px-2 py-1.5 text-xs text-slate-100"><option value="accept">Accept (positive)</option><option value="reject">Reject (negative)</option></select>
                        <label className="flex items-center gap-1 text-xs text-slate-300"><input type="checkbox" checked={profile.enabled} onChange={(event) => updateCellTypeProfile(profile.id, { enabled: event.target.checked })} className="accent-emerald-400" />Enabled</label>
                        <button type="button" onClick={() => removeCellTypeProfile(profile.id)} className="rounded-md border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-xs text-slate-200">Remove</button>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {numberField('Aspect (long ÷ short)', profile.aspect, (v) => updateCellTypeProfile(profile.id, { aspect: v }), 1, 20, 0.05)}
                        {numberField('Aspect tolerance (±fraction)', profile.aspectTolerance, (v) => updateCellTypeProfile(profile.id, { aspectTolerance: v }), 0.02, 3, 0.01)}
                        {numberField('Min long side (px, 0 = off)', profile.minLongPx, (v) => updateCellTypeProfile(profile.id, { minLongPx: v }), 0, 100000, 1)}
                        {numberField('Max long side (px, 0 = off)', profile.maxLongPx, (v) => updateCellTypeProfile(profile.id, { maxLongPx: v }), 0, 100000, 1)}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="flex flex-wrap gap-2">
                  <button type="button" onClick={() => addCellTypeProfile('accept')} className="rounded-md border border-emerald-700/60 bg-emerald-500/15 px-3 py-2 text-xs text-emerald-100">+ Add cell type (accept)</button>
                  <button type="button" onClick={() => addCellTypeProfile('reject')} className="rounded-md border border-rose-700/60 bg-rose-500/15 px-3 py-2 text-xs text-rose-100">+ Add negative type (reject)</button>
                  <button type="button" onClick={() => void learnCellTypeFromAnnotations()} className="rounded-md border border-cyan-700/60 bg-cyan-500/15 px-3 py-2 text-xs text-cyan-100">Learn default type from my saved cell boxes</button>
                </div>
              </section>

              <section className="rounded-xl border border-orange-900/50 bg-[#16100b] p-5 space-y-4">
                <div><h2 className="text-lg font-semibold text-orange-100">06 · Crack detection — ground gate</h2><p className="mt-1 text-xs text-slate-400">Fractions of a crack box that decide whether it sits on ground instead of a PV module. Defaults reproduce the previous fixed values.</p></div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {numberField('Max soil fraction', trainingSettings.crackDetection.groundMaxSoilFraction, (v) => updateCrackDetectionSetting('groundMaxSoilFraction', v), 0, 1, 0.05, 'Reject if more soil than this.')}
                  {numberField('Min PV fraction', trainingSettings.crackDetection.groundMinPvFraction, (v) => updateCrackDetectionSetting('groundMinPvFraction', v), 0, 1, 0.05, 'Reject if less silicon than this (module located).')}
                  {numberField('Min PV fraction — fallback', trainingSettings.crackDetection.groundMinPvFractionFallback, (v) => updateCrackDetectionSetting('groundMinPvFractionFallback', v), 0, 1, 0.05, 'Same when no module could be localised (whole-image fallback).')}
                </div>
                <p className="text-[11px] text-slate-500">Crack-classifier training parameters (iterations, learning rate, L2, threshold search, class balance) are in section 02 above.</p>
              </section>
              <div className="flex flex-wrap items-center gap-2"><button type="button" onClick={() => void saveAllTrainingSettings()} disabled={settingsSaving} className="rounded-md border border-emerald-700/60 bg-emerald-500/15 px-4 py-2 text-xs font-semibold text-emerald-100 disabled:opacity-40">{settingsSaving ? 'Saving settings…' : 'Save all training settings'}</button><button type="button" onClick={() => setActiveTab('training')} className="rounded-md border border-slate-700 bg-slate-800 px-4 py-2 text-xs text-slate-200">Back to Training Centre</button>{settingsBadge}</div>
            </>}
          </div>
        </main>
      )}

      {/* Marked-image output tab */}
      {activeTab === 'marked' && (
        <main className="flex-1">
          <MarkedImagesView frames={frames} />
        </main>
      )}
      {yoloViewerOpen && (
        <div className="fixed inset-0 z-[100] bg-black/85 overflow-auto p-4" onClick={() => setYoloViewerOpen(false)}>
          <div className="max-w-[1500px] mx-auto rounded-xl border border-cyan-700/50 bg-[#0b0f18] p-4" onClick={(event) => event.stopPropagation()}>
            <div className="flex items-center justify-between gap-3">
              <div><h2 className="text-sm font-semibold text-cyan-200">YOLO input viewer (hidden · Ctrl+Shift+Y)</h2><p className="text-[11px] text-slate-400">Exactly what the cell model receives for “{activeFrame.label}”: letterboxed to the model size, grey (114) padding. Boxes are decoded from these tensors back into image pixels.</p></div>
              <button type="button" onClick={() => setYoloViewerOpen(false)} className="px-3 py-1.5 text-xs rounded border border-slate-600 text-slate-200">Close (Esc)</button>
            </div>
            {activeFrame.cellDetection?.inputPreviews?.length ? (
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                {activeFrame.cellDetection.inputPreviews.map((preview, index) => (
                  <figure key={index} className="rounded-lg border border-[#263448] bg-[#07090e] p-2">
                    <img src={preview.dataUrl} alt={`YOLO input ${index + 1}`} className="w-full h-auto rounded" />
                    <figcaption className="mt-1 text-[10px] text-slate-400">{preview.kind === 'full' ? 'Full frame' : `Tile ${index}`} · crop {Math.round(preview.window.left)},{Math.round(preview.window.top)} {Math.round(preview.window.width)}×{Math.round(preview.window.height)} → {preview.modelWidth}×{preview.modelHeight} · scale {preview.scale.toFixed(3)} · pad {preview.padX},{preview.padY}</figcaption>
                  </figure>
                ))}
              </div>
            ) : <p className="mt-4 text-xs text-slate-400">No model input captured for this image yet. Run Cell detections (needs a trained cell.onnx; the classical fallback feeds nothing to YOLO).</p>}
            <p className="mt-3 text-[10px] text-slate-500">The crack stage is a handcrafted-feature classifier on cell crops, not a YOLO network, so it has no tensor to preview; its inputs are the cell boxes you see in Crack Annotation.</p>
          </div>
        </div>
      )}
    </div>
  );
}
