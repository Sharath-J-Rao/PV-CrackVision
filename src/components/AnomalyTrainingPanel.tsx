import React, { useCallback, useEffect, useState } from 'react';

/** Train the crack anomaly detector (default crack engine since v1.4.17). It learns from reviewed CLEAN cells only; cracked cells just validate the threshold. */
interface Job { status: string; progress?: number; message?: string; error?: string; errorTraceback?: string; log?: string[]; startedAt?: string; finishedAt?: string; report?: Record<string, any> }
interface Snapshot {
  anom: Job;
  installed: { anom: Record<string, any> | null; anomBundled?: boolean; anomWarning?: string };
  readiness: { anom: { ready: boolean; missing: string[]; warnings?: string[]; counts?: { positives: number; negatives: number; images: number; positiveImages: number } } };
}
const ACTIVE = ['preparing', 'training', 'validating'];
const pct = (v: unknown) => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '—');
const num = (v: unknown, d = 3) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—');

export const AnomalyTrainingPanel: React.FC<{ onOpenSettings?: () => void }> = ({ onOpenSettings }) => {
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [error, setError] = useState('');
  const refresh = useCallback(async () => {
    try { const r = await fetch('/api/classifier/status', { cache: 'no-store' }); if (!r.ok) throw new Error(`HTTP ${r.status}`); setSnap(await r.json()); setError(''); }
    catch (e) { setError(e instanceof Error ? e.message : 'Status unavailable.'); }
  }, []);
  const job = snap?.anom; const active = Boolean(job && ACTIVE.includes(job.status));
  useEffect(() => { void refresh(); const t = setInterval(() => void refresh(), active ? 2500 : 10000); return () => clearInterval(t); }, [refresh, active]);

  const call = async (url: string) => {
    try {
      const r = await fetch(url, { method: 'POST' }); const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
      setError('');
    } catch (e) { setError(e instanceof Error ? e.message : 'Request failed.'); }
    void refresh();
  };
  const clear = () => {
    if (window.confirm(`Remove your trained anomaly detector and its generated training data (job history, logs, work files)?\n\nYour reviewed cracked/clean labels and source images are kept, so you can retrain at any time. ${hasBundled ? 'Crack detection goes back to the model bundled with the app.' : 'Until it is trained again, crack detection reports that no model is available.'}`)) void call('/api/classifier/clear?scope=anomaly');
  };

  // v1.4.17.6: a bundled model ships with the app, so training is optional
  const ready = snap?.readiness.anom; const installed = snap?.installed.anom; const usingBundled = installed?.source === 'bundled'; const hasBundled = Boolean(snap?.installed.anomBundled); const report = job?.report; const v = report?.validation as Record<string, number> | undefined;
  return (
    <div className="space-y-3">
      {error && <p className="rounded-lg border border-amber-800/60 bg-amber-950/20 px-3 py-2 text-xs text-amber-200">{error}</p>}
      {snap?.installed.anomWarning && <p className="rounded-lg border border-amber-800/60 bg-amber-950/20 px-3 py-2 text-xs text-amber-200">{snap.installed.anomWarning}</p>}
      <div className="rounded-lg border border-[#2a1d33] bg-[#0d0811] p-4 space-y-3">
        <p className="text-[11px] text-slate-400">Learns what a <strong className="text-slate-200">clean</strong> cell looks like from your reviewed clean cells (patch memory bank on a frozen pretrained backbone). A cell with patches unlike anything clean is flagged as a crack, so new crack types are caught without cracked examples. Cracked labels are never fitted; if you have enough, they validate the threshold and report recall.</p>
        <div className="text-[11px] text-slate-300 space-y-0.5">
          <div>Installed: <span className="font-mono text-fuchsia-200">{installed ? installed.version : 'none'}</span>{installed && <span className="ml-2 rounded border border-slate-700 px-1.5 py-0.5 text-[10px] text-slate-300">{usingBundled ? 'bundled with the app' : 'trained by you'}</span>}</div>
          {usingBundled && <div className="text-[10px] text-slate-400">Crack detection works out of the box with the bundled model. Training is optional: a model you train overrides it until you revert.</div>}
          {installed && <div>{installed.memoryBank} memory-bank patches · threshold mode {installed.thresholdMode} · {String(installed.msPerCell)} ms/cell (CPU){installed.cracksVerified ? '' : ' · crack recall not verified'}</div>}
        </div>
        {ready?.counts && <div className="text-[10px] text-slate-400">Reviewed labels: <span className="text-emerald-200">{ready.counts.negatives} clean (used for fitting)</span> · <span className="text-rose-200">{ready.counts.positives} cracked (validation only)</span> · {ready.counts.images} source images</div>}
        {ready && !ready.ready && <div className="text-[11px] text-amber-300">{installed ? 'To retrain, needs' : 'Needs'}: {ready.missing.join(', ')}</div>}
        {ready?.warnings?.length ? <div className="rounded border border-amber-900/60 bg-amber-950/20 p-2 space-y-1 text-[10px] text-amber-200">{ready.warnings.map((w) => <p key={w}>{w}</p>)}</div> : null}
        {job && job.status !== 'idle' && (
          <div className="space-y-1">
            <div className="h-1.5 rounded bg-[#241831] overflow-hidden"><div className="h-full bg-fuchsia-400 transition-all" style={{ width: `${Math.max(0, Math.min(100, job.progress ?? 0))}%` }} /></div>
            <div className="text-[11px] text-slate-400">{job.status} · {job.message}</div>
            {job.error && job.status === 'failed' && <div className="rounded border border-rose-800 bg-rose-950/20 p-2 text-[11px] text-rose-100"><strong>{job.error}</strong>{job.errorTraceback && <details className="mt-1"><summary className="cursor-pointer">Full error traceback</summary><pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap text-[10px]">{job.errorTraceback}</pre></details>}</div>}
          </div>
        )}
        {report && job?.status === 'completed' && v && (
          <div className="rounded border border-[#29232d] p-3 space-y-1">
            <div className="flex justify-between text-xs font-semibold text-slate-100"><span>Held-out validation</span><span className={report.accepted ? 'text-emerald-300' : 'text-amber-200'}>{report.accepted ? 'Accepted and activated' : 'Rejected; previous model kept'}</span></div>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-[10px] text-slate-500">
              {[['Clean cells kept clean', v.specificity], ...(report.cracksVerified ? [['Crack recall', v.recall], ['Precision (field ratio)', v.precisionAtRealisticRatio], ['Balanced acc.', v.balancedAccuracy], ['AUROC', report.auroc]] : [])].map(([k, x]) => <div key={String(k)}>{k}<div className="text-sm text-slate-100">{pct(x)}</div></div>)}
            </div>
            <p className="text-[10px] text-slate-400">Held-out: {report.valCells?.clean} clean / {report.valCells?.cracked} cracked cells · {num(v.falsePositivesPer1000, 2)} false alarms / 1,000 cells at the field ratio · raw threshold {num(report.rawThreshold, 3)}</p>
            <p className="text-[10px] text-slate-300">{report.evaluationNote}</p>
          </div>
        )}
        <div className="flex flex-wrap gap-2">
          <button type="button" disabled={active || !ready?.ready} onClick={() => void call('/api/classifier/anom/start')} className="rounded-md border border-cyan-700/60 bg-cyan-500/15 px-3 py-1.5 text-xs font-semibold text-cyan-100 disabled:opacity-40">{usingBundled ? 'Retrain on my data (overrides bundled)' : 'Train anomaly detector'}</button>
          <button type="button" disabled={!active} onClick={() => void call('/api/classifier/anom/cancel')} className="rounded-md border border-rose-800/60 px-3 py-1.5 text-xs text-rose-200 disabled:opacity-40">Cancel</button>
          <button type="button" disabled={active || (!installed?.source || usingBundled) && job?.status === 'idle'} onClick={clear} className="rounded-md border border-rose-800/60 bg-rose-950/30 px-3 py-1.5 text-xs text-rose-200 disabled:opacity-40">{hasBundled ? 'Revert to bundled model' : 'Clear model & training data'}</button>
          <a href="/api/classifier/anom/log" className="rounded-md border border-[#2a1d33] px-3 py-1.5 text-xs text-slate-300">Log</a>
          {onOpenSettings && <button type="button" onClick={onOpenSettings} className="rounded-md border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs text-slate-200">Anomaly parameters</button>}
        </div>
        <details className="rounded border border-[#29232d] p-3"><summary className="cursor-pointer text-xs font-semibold text-slate-200">Training log ({job?.log?.length ?? 0} entries)</summary><pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-words font-mono text-[10px] leading-5 text-slate-400">{(job?.log ?? []).slice(-150).join('\n') || 'No worker log entries yet.'}</pre></details>
      </div>
    </div>
  );
};
