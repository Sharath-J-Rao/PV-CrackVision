import React, { useCallback, useEffect, useMemo, useState } from 'react';

/** v1.4.17.7: inspect the reviewed cracked/clean cells the crack models train on, see which model + device is active, and recover models from earlier releases. */
interface Item { id: string; fileName: string; imageFile: string; label: 0 | 1; source: string; modelScore?: number; createdAt: string }
interface ImageRow { imageFile: string; fileName: string; cracked: number; clean: number; lastAt: string }
interface Summary {
  counts: { total: number; positives: number; negatives: number; images: number; positiveImages: number; negativeImages: number; trainingReady: boolean; trainingMissing: string[] };
  bySource: Record<string, { cracked: number; clean: number }>; images: ImageRow[]; items: Item[];
}
interface Recovery { ranAt: string; searched: string[]; installed: Array<{ model: string; from: string; version?: string }>; kept: Array<{ model: string; reason: string }>; warnings: string[]; found: Array<{ model: string; dir: string; version?: string }> }
interface Status { installed: { anom: Record<string, any> | null; cls: Record<string, any> | null; anomWarning?: string; warning?: string }; devices?: Record<'anomaly' | 'classifier', { provider: string | null; gpu: boolean }> }

const SOURCE_LABEL: Record<string, string> = { reviewed: 'Reviewed by you', 'rejected-detection': 'False detection removed (hard negative)', 'missed-crack': 'Missed crack you added', 'bulk-negative': 'Bulk clean', 'auto-negative': 'Auto clean (module has a crack elsewhere)' };
const PAGE = 48;
const deviceText = (d?: { provider: string | null; gpu: boolean }) => !d || !d.provider ? 'not loaded yet' : d.gpu ? `GPU (${d.provider})` : 'CPU';
const when = (v?: string) => { const t = Date.parse(v ?? ''); return Number.isFinite(t) ? new Date(t).toLocaleString() : '—'; };

export const CrackTrainingDataPanel: React.FC = () => {
  const [data, setData] = useState<Summary | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [recovery, setRecovery] = useState<Recovery | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [label, setLabel] = useState<'all' | '1' | '0'>('all');
  const [source, setSource] = useState('all');
  const [image, setImage] = useState('all');
  const [page, setPage] = useState(0);

  const load = useCallback(async () => {
    try {
      const [s, c, r] = await Promise.all([fetch('/api/crack-training/summary', { cache: 'no-store' }), fetch('/api/classifier/status', { cache: 'no-store' }), fetch('/api/models/recovery', { cache: 'no-store' })]);
      if (!s.ok) throw new Error((await s.json().catch(() => ({}))).error || `HTTP ${s.status}`);
      setData(await s.json()); if (c.ok) setStatus(await c.json()); if (r.ok) setRecovery(await r.json()); setError('');
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not load the crack training data.'); }
  }, []);
  useEffect(() => { void load(); const t = setInterval(() => void load(), 15000); return () => clearInterval(t); }, [load]);

  const recover = async () => {
    setBusy(true);
    try {
      const r = await fetch('/api/models/recover', { method: 'POST', headers: { 'x-pv-action': '1' } }); const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
      setRecovery(body.report); setStatus(body.status); setError('');
    } catch (e) { setError(e instanceof Error ? e.message : 'Recovery failed.'); }
    setBusy(false);
  };
  const remove = async (item: Item) => {
    if (!window.confirm(`Remove this ${item.label ? 'cracked' : 'clean'} training cell from ${item.fileName}? The source image and other cells are kept. Retrain to apply.`)) return;
    try { const r = await fetch(`/api/feedback/${item.id}`, { method: 'DELETE', headers: { 'x-pv-action': '1' } }); if (!r.ok) throw new Error(`HTTP ${r.status}`); await load(); }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not remove the cell.'); }
  };

  const filtered = useMemo(() => (data?.items ?? []).filter((i) => (label === 'all' || String(i.label) === label) && (source === 'all' || i.source === source) && (image === 'all' || i.imageFile === image)), [data, label, source, image]);
  useEffect(() => setPage(0), [label, source, image]);
  const pages = Math.max(1, Math.ceil(filtered.length / PAGE)), shown = filtered.slice(page * PAGE, page * PAGE + PAGE);
  const c = data?.counts; const anom = status?.installed.anom, cls = status?.installed.cls;
  const card = (title: string, value: React.ReactNode, hint?: string) => <div className="rounded-lg border border-[#263448] bg-[#0e131d] p-3"><p className="text-[10px] uppercase tracking-wider text-slate-500">{title}</p><p className="text-xl font-semibold text-slate-100 mt-1">{value}</p>{hint && <p className="text-[10px] text-slate-500 mt-1">{hint}</p>}</div>;
  const model = (name: string, m: Record<string, any> | null | undefined, device?: { provider: string | null; gpu: boolean }, warn?: string) => (
    <div className="rounded-lg border border-[#263448] bg-[#0e131d] p-3 text-xs space-y-1">
      <p className="text-slate-200 font-semibold">{name}</p>
      {m ? <>
        <p className="text-slate-400">Version <span className="text-slate-200">{m.version ?? '—'}</span> · {m.source === 'bundled' ? 'bundled with the app' : 'trained by you / imported'}</p>
        <p className="text-slate-400">Trained {when(m.trainedAt)} · threshold {typeof m.threshold === 'number' ? m.threshold : '—'} · {m.msPerCell ? `${m.msPerCell} ms/cell` : ''}</p>
        <p className="text-slate-400">Held-out balanced accuracy {typeof m.validation?.balancedAccuracy === 'number' ? `${(m.validation.balancedAccuracy * 100).toFixed(1)}%` : '—'} · Inference device: <span className={device?.gpu ? 'text-emerald-300' : 'text-slate-200'}>{deviceText(device)}</span></p>
      </> : <p className="text-amber-300">No model installed.</p>}
      {warn && <p className="text-amber-300">{warn}</p>}
    </div>);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <header><p className="text-[11px] uppercase tracking-[0.2em] text-cyan-400">Crack models</p><h1 className="text-2xl font-semibold text-slate-100 mt-1">Crack Training Data</h1>
        <p className="text-sm text-slate-400 mt-2">Every reviewed cell the crack models learn from. The anomaly detector fits on <b>clean</b> cells only; cracked cells only validate its threshold. The archived classifier learns from both.</p></header>
      {error && <p className="rounded-lg border border-amber-800/60 bg-amber-950/20 px-3 py-2 text-xs text-amber-200">{error}</p>}

      <section className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2"><h2 className="text-base font-semibold text-slate-100">Active crack models</h2>
          <button type="button" disabled={busy} onClick={() => void recover()} className="rounded-md border border-cyan-700/60 bg-cyan-500/10 px-3 py-1.5 text-xs text-cyan-100 disabled:opacity-40">{busy ? 'Searching…' : 'Find models from previous versions'}</button></div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">{model('Anomaly detector (default engine)', anom, status?.devices?.anomaly, status?.installed.anomWarning)}{model('Archived classifier', cls, status?.devices?.classifier, status?.installed.warning)}</div>
        {recovery && <div className="rounded-lg border border-[#263448] bg-[#0b1420] p-3 text-[11px] text-slate-400 space-y-1">
          <p>Last search {when(recovery.ranAt)} · {recovery.searched.length} folder(s) checked · {recovery.found.length} earlier model file set(s) found.</p>
          {recovery.installed.map((i, n) => <p key={n} className="text-emerald-300">Installed {i.model}{i.version ? ` (${i.version})` : ''} from {i.from}</p>)}
          {!recovery.installed.length && recovery.kept.map((k, n) => <p key={n}>{k.model}: {k.reason}.</p>)}
          {recovery.warnings.map((w, n) => <p key={n} className="text-amber-300">{w}</p>)}
          <details><summary className="cursor-pointer text-slate-500">Folders searched</summary>{recovery.searched.map((d) => <p key={d} className="font-mono break-all">{d}</p>)}</details></div>}
      </section>

      <section className="space-y-3">
        <h2 className="text-base font-semibold text-slate-100">Dataset</h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {card('Cracked cells', c?.positives ?? '—', c ? `${c.positiveImages} image(s)` : undefined)}{card('Clean cells', c?.negatives ?? '—', c ? `${c.negativeImages} image(s)` : undefined)}
          {card('Images', c?.images ?? '—')}{card('Total cells', c?.total ?? '—')}{card('Training ready', c ? (c.trainingReady ? 'Yes' : 'Not yet') : '—', c && !c.trainingReady ? c.trainingMissing.join(' ') : undefined)}</div>
        {data && <div className="flex flex-wrap gap-2 text-[11px]">{Object.entries(data.bySource).map(([k, v]) => <span key={k} className="rounded-full border border-[#263448] bg-[#0e131d] px-2.5 py-1 text-slate-300">{SOURCE_LABEL[k] ?? k}: {v.cracked} cracked · {v.clean} clean</span>)}</div>}
      </section>

      <section className="space-y-3">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <select value={label} onChange={(e) => setLabel(e.target.value as any)} className="rounded border border-slate-700 bg-slate-900 px-2 py-1.5 text-slate-200"><option value="all">All labels</option><option value="1">Cracked</option><option value="0">Clean</option></select>
          <select value={source} onChange={(e) => setSource(e.target.value)} className="rounded border border-slate-700 bg-slate-900 px-2 py-1.5 text-slate-200"><option value="all">All sources</option>{Object.keys(data?.bySource ?? {}).map((k) => <option key={k} value={k}>{SOURCE_LABEL[k] ?? k}</option>)}</select>
          <select value={image} onChange={(e) => setImage(e.target.value)} className="max-w-[260px] rounded border border-slate-700 bg-slate-900 px-2 py-1.5 text-slate-200"><option value="all">All images</option>{(data?.images ?? []).map((i) => <option key={i.imageFile} value={i.imageFile}>{i.fileName} ({i.cracked}/{i.clean})</option>)}</select>
          <span className="text-slate-400">{filtered.length} cell(s)</span><button type="button" onClick={() => void load()} className="ml-auto rounded border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-slate-200">Refresh</button>
        </div>
        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
          {shown.map((i) => <figure key={i.id} className={`relative rounded-lg border bg-[#0e131d] p-1 ${i.label ? 'border-rose-700/70' : 'border-emerald-800/60'}`}>
            <img src={`/api/crack-training/cell/${i.id}`} loading="lazy" alt={`${i.label ? 'Cracked' : 'Clean'} cell from ${i.fileName}`} className="w-full aspect-square object-contain bg-black rounded" />
            <figcaption className="mt-1 text-[10px] text-slate-400 truncate" title={`${i.fileName} · ${SOURCE_LABEL[i.source] ?? i.source}`}><span className={i.label ? 'text-rose-300' : 'text-emerald-300'}>{i.label ? 'Cracked' : 'Clean'}</span> · {i.fileName}</figcaption>
            <button type="button" onClick={() => void remove(i)} title="Remove from training data" className="absolute top-1.5 right-1.5 rounded bg-black/70 px-1.5 text-[11px] text-slate-200 hover:text-rose-300">✕</button></figure>)}
        </div>
        {!filtered.length && <p className="p-6 text-center text-xs text-slate-400">{data ? 'No reviewed cells match. Label cells in Crack Annotation to build the training set.' : 'Loading…'}</p>}
        {pages > 1 && <div className="flex items-center justify-center gap-3 text-xs text-slate-300"><button type="button" disabled={page === 0} onClick={() => setPage(page - 1)} className="rounded border border-slate-700 px-3 py-1 disabled:opacity-40">Previous</button><span>Page {page + 1} / {pages}</span><button type="button" disabled={page >= pages - 1} onClick={() => setPage(page + 1)} className="rounded border border-slate-700 px-3 py-1 disabled:opacity-40">Next</button></div>}
      </section>
    </div>
  );
};
