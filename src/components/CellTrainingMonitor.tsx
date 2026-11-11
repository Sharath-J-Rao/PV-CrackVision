type EpochMetric = {
  epoch: number;
  timestamp?: string;
  precision?: number | null;
  recall?: number | null;
  map50?: number | null;
  map50_95?: number | null;
  box_loss?: number | null;
  cls_loss?: number | null;
  dfl_loss?: number | null;
  val_box_loss?: number | null;
  val_cls_loss?: number | null;
  val_dfl_loss?: number | null;
};

type CellTrainingStatus = {
  status?: string;
  message?: string;
  progress?: number;
  updatedAt?: string;
  startedAt?: string;
  finishedAt?: string;
  annotatedImages?: number;
  positiveImages?: number;
  annotatedCells?: number;
  negativeSamples?: number;
  activeModelExists?: boolean;
  epochMetrics?: EpochMetric[];
  log?: string[];
  metrics?: Record<string, number>;
  metricsDownloadUrl?: string;
  logDownloadUrl?: string;
  error?: string;
  errorTraceback?: string;
  statusPollError?: string;
  checkpointReusable?: boolean;
  reusableCheckpoint?: string;
};

interface Props {
  status: CellTrainingStatus | null;
  onOpenTraining?: () => void;
}

const metric = (value: number | null | undefined) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '—';

export function CellTrainingMonitor({ status, onOpenTraining }: Props) {
  const rows = status?.epochMetrics ?? [];
  const progress = Math.max(0, Math.min(100, Number(status?.progress) || 0));
  const isActive = status?.status === 'training' || status?.status === 'setting_up' || status?.status === 'paused';

  return (
    <details open className="rounded-xl border border-cyan-900/60 bg-[#0b1420]">
      <summary className="flex cursor-pointer flex-wrap items-center justify-between gap-2 p-4">
        <div>
          <p className="text-[10px] uppercase tracking-[0.18em] text-cyan-300">Cell detector · YOLO11n</p>
          <h2 className="mt-1 text-sm font-semibold text-slate-100">Training progress, epoch analytics &amp; worker log</h2>
          <p className="mt-1 text-[10px] text-slate-500">Live view of the same persisted training job shown in Training Centre. Detection runs and model training are separate operations.</p>
        </div>
        <span className="rounded-md border border-slate-700 bg-[#080d15] px-2.5 py-1.5 text-[10px] uppercase tracking-wide text-slate-200">{status?.status ?? 'checking'} · {rows.length} epoch(s)</span>
      </summary>
      <div className="space-y-3 border-t border-cyan-900/50 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-[10px] text-slate-500">Training dataset, metrics and diagnostics</div>
        <div className="flex flex-wrap items-center gap-2">
          {status?.metricsDownloadUrl && <a href={status.metricsDownloadUrl} className="rounded border border-cyan-800/70 px-2.5 py-1.5 text-[10px] text-cyan-200 hover:bg-cyan-950/40">Download epoch CSV</a>}
          {status?.logDownloadUrl && <a href={status.logDownloadUrl} className="rounded border border-slate-700 px-2.5 py-1.5 text-[10px] text-slate-200 hover:bg-slate-800">Download full worker log</a>}
          {onOpenTraining && <button type="button" onClick={onOpenTraining} className="rounded border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-[10px] text-slate-200">Training Centre</button>}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <div className="rounded-lg bg-[#080d15] p-3"><div className="text-xl font-semibold text-slate-100">{status?.annotatedImages ?? status?.positiveImages ?? 0}</div><div className="text-[10px] text-slate-500">Positive source images</div></div>
        <div className="rounded-lg bg-[#080d15] p-3"><div className="text-xl font-semibold text-slate-100">{status?.annotatedCells ?? 0}</div><div className="text-[10px] text-slate-500">Reviewed cell boxes</div></div>
        <div className="rounded-lg bg-[#080d15] p-3"><div className="text-xl font-semibold text-slate-100">{status?.negativeSamples ?? 0}</div><div className="text-[10px] text-slate-500">Background-only negatives</div></div>
        <div className="rounded-lg bg-[#080d15] p-3"><div className="text-xl font-semibold text-slate-100">{status?.activeModelExists ? 'Ready' : '—'}</div><div className="text-[10px] text-slate-500">Active ONNX model</div></div>
      </div>

      {isActive ? (
        <div className="space-y-1.5 rounded-lg border border-cyan-900/50 bg-[#080d15] p-3">
          <div className="flex items-center justify-between gap-3 text-xs"><span className="text-slate-200">{status?.message || 'Training worker is running…'}</span><strong className="text-cyan-200 tabular-nums">{progress.toFixed(0)}%</strong></div>
          <div className="h-2 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full bg-cyan-400 transition-[width] duration-500" style={{ width: `${progress}%` }} /></div>
          <p className="text-[10px] text-slate-500">The worker writes progress and metrics to disk. This panel refreshes while setup or training is active; a browser refresh does not cancel the worker.</p>
        </div>
      ) : (
        <p className="rounded-lg bg-[#080d15] px-3 py-2 text-[11px] text-slate-400">{status?.message || (status ? `Last job status: ${status.status}.` : 'Waiting for the local training status endpoint…')} {status?.updatedAt ? `Updated ${new Date(status.updatedAt).toLocaleString()}.` : ''}</p>
      )}

      {status?.statusPollError && <p className="rounded-lg border border-amber-800/60 bg-amber-950/20 p-2 text-xs text-amber-200">Training status refresh issue: {status.statusPollError}</p>}
      {status?.error && <div className="rounded-lg border border-rose-900/60 bg-rose-950/20 p-3"><p className="whitespace-pre-wrap text-xs text-rose-200">{status.error}</p>{status.errorTraceback && <details className="mt-2 text-[10px] text-rose-300"><summary className="cursor-pointer">Full worker traceback</summary><pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words">{status.errorTraceback}</pre></details>}</div>}

      <div className="space-y-2">
        <div className="flex flex-wrap items-end justify-between gap-2"><div><h3 className="text-xs font-semibold text-slate-100">Epoch analytics</h3><p className="text-[10px] text-slate-500">Precision, recall and mAP are held-out validation metrics; loss columns are training and validation losses.</p></div><span className="text-[10px] text-slate-500">{rows.length} epoch row(s) saved</span></div>
        <div className="max-h-[340px] overflow-auto rounded border border-[#1d2b3d]">
          <table className="min-w-[1180px] w-full text-right text-[10px] tabular-nums">
            <thead className="sticky top-0 bg-[#111b2a] text-slate-400"><tr><th className="p-2 text-left">Epoch</th><th className="p-2">Precision</th><th className="p-2">Recall</th><th className="p-2">mAP50</th><th className="p-2">mAP50–95</th><th className="p-2">Train box</th><th className="p-2">Train cls</th><th className="p-2">Train DFL</th><th className="p-2">Val box</th><th className="p-2">Val cls</th><th className="p-2">Val DFL</th><th className="p-2">Timestamp</th></tr></thead>
            <tbody>{rows.map((row) => <tr key={row.epoch} className="border-t border-[#1d2b3d] hover:bg-[#111b2a]"><td className="p-2 text-left font-semibold text-slate-200">{row.epoch}</td><td className="p-2 text-cyan-200">{metric(row.precision)}</td><td className="p-2 text-cyan-200">{metric(row.recall)}</td><td className="p-2 text-emerald-200">{metric(row.map50)}</td><td className="p-2 text-emerald-200">{metric(row.map50_95)}</td><td className="p-2 text-slate-300">{metric(row.box_loss)}</td><td className="p-2 text-slate-300">{metric(row.cls_loss)}</td><td className="p-2 text-slate-300">{metric(row.dfl_loss)}</td><td className="p-2 text-violet-200">{metric(row.val_box_loss)}</td><td className="p-2 text-violet-200">{metric(row.val_cls_loss)}</td><td className="p-2 text-violet-200">{metric(row.val_dfl_loss)}</td><td className="p-2 text-slate-500">{row.timestamp ? new Date(row.timestamp).toLocaleString() : '—'}</td></tr>)}{rows.length === 0 && <tr><td colSpan={12} className="p-5 text-center text-slate-500">No epoch metrics have been recorded for the current/most recent training job yet. Metrics appear after each epoch's validation pass.</td></tr>}</tbody>
          </table>
        </div>
      </div>

      <details open className="rounded-lg border border-[#25364d] bg-[#080d15] p-3 text-xs text-slate-400">
        <summary className="cursor-pointer font-semibold text-slate-200">Live training log ({status?.log?.length ?? 0} saved entries)</summary>
        <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-words font-mono text-[10px] leading-5">{status?.log?.slice(-120).join('\n') || 'No worker log entries have been captured yet. Start or set up the cell detector from Training Centre to see each worker step here.'}</pre>
      </details>

      {status?.metrics && <div className="rounded-lg bg-[#080d15] p-3"><div className="mb-2 text-xs font-semibold text-slate-200">Final held-out validation summary</div><div className="grid grid-cols-2 md:grid-cols-3 gap-2">{Object.entries(status.metrics).map(([key, value]) => <div key={key} className="text-[10px] text-slate-400">{key.replaceAll('_', ' ')} <span className="font-mono text-slate-100">{Number(value).toFixed(4)}</span></div>)}</div></div>}
      {status?.checkpointReusable && <p className="text-[10px] text-slate-500">Reusable fine-tuning checkpoint: <span className="font-mono text-cyan-200">{status.reusableCheckpoint || 'cell-best.pt'}</span></p>}
      </div>
    </details>
  );
}
