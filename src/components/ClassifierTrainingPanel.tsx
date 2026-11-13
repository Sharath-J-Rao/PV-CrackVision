import React, { useCallback, useEffect, useState } from 'react';

/** v1.4.14: train the pretrained crack models. A (crop classifier) decides; B (tiled detector) is a shadow comparison. They can train at the same time. */
type Mode = 'cls' | 'det';
interface Job { status: string; progress?: number; message?: string; epoch?: number; epochs?: number; error?: string; errorTraceback?: string; report?: Record<string, any>; metrics?: Array<Record<string, number | null>>; log?: string[]; startedAt?: string; finishedAt?: string }
interface Snapshot {
  cls: Job; det: Job;
  installed: { cls: Record<string, any> | null; det: Record<string, any> | null; warning?: string };
  readiness: Record<Mode, { ready: boolean; missing: string[] }>;
}
const ACTIVE = ['preparing', 'training', 'validating'];
/** A finished/failed/cancelled run leaves job history behind even when no model was installed; that is also 'data to clear'. */
const hasGeneratedData = (snap: Snapshot) => (['cls', 'det'] as Mode[]).some((m) => snap[m].status !== 'idle');
const pct = (v: unknown) => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '—');

const CARDS: Record<Mode, { title: string; blurb: string }> = {
  cls: { title: 'A · Cell crop classifier (decides)', blurb: 'Pretrained yolo11n-cls on one crop per cell. Uses your cracked/clean cell labels, the hard negatives you confirmed, outer-ring and missed-crack weights. Replaces the old boosted-tree/CNN model once trained.' },
  det: { title: 'B · Tiled crack detector (shadow)', blurb: 'Pretrained YOLO detector on native-resolution tiles. Background tiles come only from cells you labelled clean, so unlabelled cells are never taught as clean. Compared with A, never shown as a result.' },
};


const num = (v: unknown, d = 4) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—');

/** v1.4.14.4: epoch analytics, held-out validation and training log for the pretrained models. */
const CrackNetMonitor: React.FC<{ snap: Snapshot }> = ({ snap }) => {
  const [mode, setMode] = useState<Mode>('cls');
  const job = snap[mode]; const rows = job.metrics ?? []; const v = job.report?.validation as Record<string, number> | undefined;
  return (
    <div id="crack-net-monitor" className="scroll-mt-4 rounded-lg border border-[#2a1d33] bg-[#0d0811] p-4 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div><p className="text-[10px] uppercase tracking-[0.18em] text-fuchsia-300">Live training telemetry</p><h3 className="text-sm font-semibold text-slate-100">Epoch analytics &amp; training log</h3></div>
        <div className="flex gap-2">{(['cls', 'det'] as Mode[]).map((m) => <button key={m} type="button" onClick={() => setMode(m)} className={`rounded-md border px-3 py-1 text-xs ${m === mode ? 'border-fuchsia-600 bg-fuchsia-500/20 text-fuchsia-100' : 'border-[#2a1d33] text-slate-400'}`}>{m === 'cls' ? 'A · classifier' : 'B · detector'}</button>)}
          <a href={`/api/classifier/${mode}/log`} className="rounded-md border border-slate-700 px-3 py-1 text-xs text-slate-200">Download log</a></div>
      </div>
      <div className="text-[11px] text-slate-400">{job.status.toUpperCase()}{job.epochs ? ` · epoch ${job.epoch ?? 0}/${job.epochs}` : ''} · {job.message}{job.startedAt ? ` · started ${new Date(job.startedAt).toLocaleString()}` : ''}{job.finishedAt ? ` · finished ${new Date(job.finishedAt).toLocaleString()}` : ''}</div>
      <div className="max-h-64 overflow-auto rounded border border-[#29232d]"><table className="w-full min-w-[480px] text-[10px] text-right tabular-nums"><thead className="sticky top-0 bg-[#1b151e] text-slate-400"><tr><th className="p-2 text-left">Epoch</th><th className="p-2">Train loss</th>{mode === 'cls' ? <><th className="p-2">Val loss</th><th className="p-2">Top-1 acc</th></> : <><th className="p-2">Precision</th><th className="p-2">Recall</th><th className="p-2">mAP50</th></>}</tr></thead>
        <tbody>{rows.map((r) => <tr key={String(r.epoch)} className="border-t border-[#29232d] text-slate-300"><td className="p-2 text-left text-slate-200">{r.epoch}</td><td className="p-2">{num(r.trainLoss as number)}</td>{mode === 'cls' ? <><td className="p-2">{num(r.validationLoss as number)}</td><td className="p-2 text-cyan-200">{pct(r.top1)}</td></> : <><td className="p-2">{pct(r.precision)}</td><td className="p-2">{pct(r.recall)}</td><td className="p-2 text-cyan-200">{pct(r.map50)}</td></>}</tr>)}
          {!rows.length && <tr><td colSpan={5} className="p-4 text-center text-slate-500">No epochs recorded yet. Rows appear after each epoch's validation pass.</td></tr>}</tbody></table></div>
      {v && <div className="rounded border border-[#29232d] p-3 space-y-1"><div className="flex justify-between text-xs font-semibold text-slate-100"><span>Final held-out validation</span><span className={job.report?.accepted ? 'text-emerald-300' : 'text-amber-200'}>{job.report?.accepted ? 'Accepted and activated' : 'Rejected; previous model kept'}</span></div>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-[10px] text-slate-500">{(mode === 'cls' ? [['Precision', v.precision], ['Recall', v.recall], ['Specificity', v.specificity], ['F1', v.f1], ['Balanced acc.', v.balancedAccuracy]] : [['Precision', v.precision], ['Recall', v.recall], ['mAP50', v.mAP50]]).map(([k, x]) => <div key={String(k)}>{k}<div className="text-sm text-slate-100">{pct(x)}</div></div>)}</div>
        {mode === 'cls' && <p className="text-[10px] text-slate-400">Confusion (held-out cells): TP {v.tp} · FP {v.fp} · TN {v.tn} · FN {v.fn} · threshold {num(v.threshold, 3)} · cracked {v.positives} / clean {v.negatives}</p>}</div>}
      {job.error && job.status === 'failed' && <div className="rounded border border-rose-800 bg-rose-950/20 p-3 text-[11px] text-rose-100"><strong>{job.error}</strong>{job.errorTraceback && <details className="mt-2"><summary className="cursor-pointer">Full error traceback</summary><pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap text-[10px]">{job.errorTraceback}</pre></details>}</div>}
      <details open className="rounded border border-[#29232d] p-3"><summary className="cursor-pointer text-xs font-semibold text-slate-200">Training log ({job.log?.length ?? 0} entries)</summary><pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-words font-mono text-[10px] leading-5 text-slate-400">{(job.log ?? []).slice(-150).join('\n') || 'No worker log entries yet.'}</pre></details>
    </div>
  );
};

/** `embedded` drops the outer card chrome so the panel can live inside the Training Centre's '02 · Crack classification' card. */
export const ClassifierTrainingPanel: React.FC<{ embedded?: boolean }> = ({ embedded = false }) => {
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [error, setError] = useState('');
  const [clearing, setClearing] = useState(false);
  const refresh = useCallback(async () => {
    try { const r = await fetch('/api/classifier/status', { cache: 'no-store' }); if (!r.ok) throw new Error(`HTTP ${r.status}`); setSnap(await r.json()); setError(''); }
    catch (e) { setError(e instanceof Error ? e.message : 'Status unavailable.'); }
  }, []);
  const busy = Boolean(snap && (ACTIVE.includes(snap.cls.status) || ACTIVE.includes(snap.det.status)));
  useEffect(() => { void refresh(); const t = setInterval(() => void refresh(), busy ? 2500 : 10000); return () => clearInterval(t); }, [refresh, busy]);

  const act = async (mode: Mode, action: 'start' | 'cancel') => {
    try {
      const r = await fetch(`/api/classifier/${mode}/${action}`, { method: 'POST' });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
      setError('');
    } catch (e) { setError(e instanceof Error ? e.message : 'Request failed.'); }
    void refresh();
  };
  /** v1.4.14.4: remove the installed A/B models and everything their trainers generated. Reviewed labels are never touched. */
  const clearAll = async () => {
    if (!window.confirm('Clear the pretrained crack models A and B and their generated training data (cropped/tiled datasets, runs, logs, job history)?\n\nYour reviewed cracked/clean labels and source images are kept, so you can retrain at any time. Until A is trained again, crack detection uses the bundled classifier.')) return;
    setClearing(true);
    try {
      const r = await fetch('/api/classifier/clear', { method: 'POST' });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
      setSnap(body as Snapshot); setError('');
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not clear the pretrained crack models.'); }
    finally { setClearing(false); }
  };
  // Both jobs are independent processes, so "train both" is two starts; one failing (e.g. not ready) must not block the other.
  const startBoth = async () => { for (const m of ['cls', 'det'] as Mode[]) if (snap?.readiness[m].ready && !ACTIVE.includes(snap[m].status)) await act(m, 'start'); };

  return (
    <section className={embedded ? 'space-y-4' : 'rounded-xl border border-fuchsia-900/60 bg-[#150d18] p-5 space-y-4'}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-fuchsia-100">Pretrained crack models · A and B</h2>
        <div className="flex flex-wrap gap-2">
          <button type="button" disabled={busy || clearing || !snap || (!snap.installed.cls && !snap.installed.det && !hasGeneratedData(snap))} onClick={() => void clearAll()} title="Removes the installed A/B models and their generated training data. Reviewed labels are kept." className="rounded-md border border-rose-800/60 bg-rose-950/30 px-3 py-1.5 text-xs text-rose-200 disabled:opacity-40">{clearing ? 'Clearing…' : 'Clear A + B model & training data'}</button>
          <button type="button" disabled={!snap || (!snap.readiness.cls.ready && !snap.readiness.det.ready)} onClick={() => void startBoth()} className="rounded-md border border-fuchsia-700/60 bg-fuchsia-500/15 px-3 py-1.5 text-xs font-semibold text-fuchsia-100 disabled:opacity-40">Train A and B together</button>
        </div>
      </div>
      {error && <p className="rounded-lg border border-amber-800/60 bg-amber-950/20 px-3 py-2 text-xs text-amber-200">{error}</p>}
      {snap?.installed.warning && <p className="rounded-lg border border-amber-800/60 bg-amber-950/20 px-3 py-2 text-xs text-amber-200">{snap.installed.warning}</p>}
      {snap && !snap.installed.cls && <p className="text-xs text-slate-400">No classifier of your own is trained yet: crack detection uses the classifier bundled with the app. Train A to replace it.</p>}
      <div className="grid grid-cols-1 gap-4">
        {(['cls', 'det'] as Mode[]).map((mode) => {
          const job = snap?.[mode]; const installed = snap?.installed[mode]; const ready = snap?.readiness[mode]; const active = Boolean(job && ACTIVE.includes(job.status));
          const v = installed?.validation as Record<string, number> | undefined; const report = job?.report;
          return (
            <div key={mode} className="rounded-lg border border-[#2a1d33] bg-[#0d0811] p-4 space-y-3">
              <div><h3 className="text-sm font-semibold text-slate-100">{CARDS[mode].title}</h3><p className="mt-1 text-[11px] text-slate-400">{CARDS[mode].blurb}</p></div>
              <div className="text-[11px] text-slate-300 space-y-0.5">
                <div>Installed: <span className="font-mono text-fuchsia-200">{installed ? installed.version : 'none'}</span></div>
                {installed && mode === 'cls' && v && <div>Held-out: precision {pct(v.precision)} · recall {pct(v.recall)} · threshold {String(installed.threshold)} · {String(installed.msPerCell)} ms/cell (CPU)</div>}
                {installed && mode === 'det' && v && <div>Held-out tiles: precision {pct(v.precision)} · recall {pct(v.recall)} · mAP50 {pct(v.mAP50)}</div>}
              </div>
              {job && job.status !== 'idle' && (
                <div className="space-y-1">
                  <div className="h-1.5 rounded bg-[#241831] overflow-hidden"><div className="h-full bg-fuchsia-400 transition-all" style={{ width: `${Math.max(0, Math.min(100, job.progress ?? 0))}%` }} /></div>
                  <div className="text-[11px] text-slate-400">{job.status}{job.epochs ? ` · epoch ${job.epoch ?? 0}/${job.epochs}` : ''} · {job.message}</div>
                  {job.error && job.status === 'failed' && <div className="text-[11px] text-rose-300">{job.error}</div>}
                  {report && job.status === 'completed' && <div className="text-[11px] text-slate-300">{report.accepted ? 'Activated.' : 'Not activated (gate not met).'} {report.evaluationNote}</div>}
                </div>
              )}
              {ready && !ready.ready && <div className="text-[11px] text-amber-300">Needs: {ready.missing.join(', ')}</div>}
              <div className="flex flex-wrap gap-2">
                <button type="button" disabled={active || !ready?.ready} onClick={() => void act(mode, 'start')} className="rounded-md border border-cyan-700/60 bg-cyan-500/15 px-3 py-1.5 text-xs font-semibold text-cyan-100 disabled:opacity-40">{`Train ${mode === 'cls' ? 'A' : 'B'}`}</button>
                <button type="button" disabled={!active} onClick={() => void act(mode, 'cancel')} className="rounded-md border border-rose-800/60 px-3 py-1.5 text-xs text-rose-200 disabled:opacity-40">Cancel</button>
                <a href={`/api/classifier/${mode}/log`} className="rounded-md border border-[#2a1d33] px-3 py-1.5 text-xs text-slate-300">Log</a>
              </div>
            </div>
          );
        })}
      </div>
      {snap && <CrackNetMonitor snap={snap} />}
    </section>
  );
};
