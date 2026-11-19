import { useCallback, useEffect, useRef, useState } from 'react';
import { CheckCircle2, ChevronUp, Download, FolderOpen, FolderSearch, HardDrive, Loader2, RefreshCw, RotateCcw, Upload, AlertTriangle } from 'lucide-react';
import type { TrainingSettings } from '../types/trainingSettings';

export type DataFolderInfo = {
  current: string; source: 'env' | 'config' | 'default'; isDefault: boolean; defaultRoot: string; configFile: string;
  envOverride: boolean; pendingRoot?: string; restartRequired: boolean; writable: boolean; exists: boolean;
};
export type AppInfo = { version: string; packaged: boolean; dataFolder: DataFolderInfo };
type BrowseResult = { path: string; parent: string | null; dirs: Array<{ name: string; path: string }>; roots: Array<{ name: string; path: string }>; resolvedDataRoot: string };
type Usage = { training: number; models: number; runtime: number; truncated: boolean };

/** Future desktop (.exe) shells can expose a native folder dialog here; the in-app picker is used otherwise. */
declare global { interface Window { pvCrackVisionDesktop?: { selectFolder?: () => Promise<string | null> } } }

async function api<T = any>(url: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(url, { ...init, headers: { 'x-pv-action': '1', ...(init.headers ?? {}) } });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data?.error || `Request failed (${response.status}).`);
  return data as T;
}
const mb = (bytes: number) => bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : `${Math.max(0.1, bytes / 1024 ** 2).toFixed(bytes < 1024 ** 2 ? 1 : 0)} MB`;
const card = 'rounded-xl border border-[#263448] bg-[#0e131d] p-5 space-y-4';
const btn = 'rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-slate-200 hover:bg-slate-700 disabled:opacity-40 inline-flex items-center gap-2';
const btnPrimary = 'rounded-md border border-emerald-700/60 bg-emerald-500/15 px-3 py-2 text-xs font-semibold text-emerald-100 hover:bg-emerald-500/25 disabled:opacity-40 inline-flex items-center gap-2';

function saveBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob); const a = document.createElement('a');
  a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function SettingsView({ flash, onModelChanged, onSettingsReplaced, onInfo }: {
  flash: (message: string, ms?: number) => void; onModelChanged: () => void; onSettingsReplaced: (settings: TrainingSettings) => void; onInfo?: (info: AppInfo) => void;
}) {
  const [info, setInfo] = useState<AppInfo | null>(null);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [browse, setBrowse] = useState<BrowseResult | null>(null);
  const [typedPath, setTypedPath] = useState('');
  const [copyExisting, setCopyExisting] = useState(true);
  const [includeRuntime, setIncludeRuntime] = useState(false);
  const [busy, setBusy] = useState('');
  const [restarting, setRestarting] = useState(false);
  const [modelResult, setModelResult] = useState<{ ok: boolean; text: string } | null>(null);
  const modelInput = useRef<HTMLInputElement>(null);
  const settingsInput = useRef<HTMLInputElement>(null);
  const fail = (error: unknown, fallback: string) => flash(error instanceof Error ? error.message : fallback, 12000);

  const refresh = useCallback(async () => {
    try { const next = await api<AppInfo>('/api/app/info'); setInfo(next); onInfo?.(next); } catch { /* server restarting */ }
    api<Usage>('/api/data-folder/usage').then(setUsage).catch(() => undefined);
  }, [onInfo]);
  useEffect(() => { void refresh(); }, [refresh]);

  const load = async (target?: string) => {
    try { const result = await api<BrowseResult>(`/api/data-folder/browse${target ? `?path=${encodeURIComponent(target)}` : ''}`); setBrowse(result); setTypedPath(result.path); }
    catch (error) { fail(error, 'Could not list folders.'); }
  };
  const openPicker = async () => {
    const native = window.pvCrackVisionDesktop?.selectFolder;
    if (native) { const picked = await native(); if (picked) { setPickerOpen(true); await load(picked); } return; }
    setPickerOpen(true); await load(info?.dataFolder.current);
  };
  const applyFolder = async () => {
    if (!browse) return;
    setBusy('folder');
    try {
      const result = await api('/api/data-folder', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: browse.path, copyExisting, includeRuntime }) });
      setPickerOpen(false); setInfo(result.info); onInfo?.(result.info);
      flash(result.restartRequired ? 'Data folder saved. Restart to start using it.' : 'Data folder confirmed.', 9000);
    } catch (error) { fail(error, 'Could not set the data folder.'); } finally { setBusy(''); }
  };
  const resetFolder = async () => {
    setBusy('folder');
    try { const result = await api('/api/data-folder/reset', { method: 'POST' }); setInfo(result.info); onInfo?.(result.info); flash('Using the default data folder after restart.'); }
    catch (error) { fail(error, 'Could not reset the data folder.'); } finally { setBusy(''); }
  };
  const restart = async () => {
    setRestarting(true);
    try { await api('/api/app/restart', { method: 'POST' }); } catch (error) { setRestarting(false); fail(error, 'Could not restart.'); return; }
    const started = Date.now();
    const poll = async () => {
      try { const r = await fetch('/api/health', { cache: 'no-store' }); if (r.ok && Date.now() - started > 1500) { window.location.reload(); return; } } catch { /* still down */ }
      if (Date.now() - started > 45000) { setRestarting(false); flash('PV CrackVision did not come back by itself. Close it and start it again; your new data folder is already saved.', 15000); return; }
      setTimeout(poll, 1000);
    };
    setTimeout(poll, 1500);
  };

  const importModel = async (file: File | undefined) => {
    if (!file) return;
    setBusy('model'); setModelResult(null);
    try {
      const result = await api('/api/cell-model/import', { method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'x-file-name': encodeURIComponent(file.name) }, body: file });
      setModelResult({ ok: true, text: `${result.message}${result.warning ? ` ${result.warning}` : ''}` }); onModelChanged(); void refresh();
    } catch (error) { setModelResult({ ok: false, text: error instanceof Error ? error.message : 'Could not load the model.' }); }
    finally { setBusy(''); if (modelInput.current) modelInput.current.value = ''; }
  };
  const exportModel = async () => {
    try { const response = await fetch('/api/cell-model/export', { headers: { 'x-pv-action': '1' } }); if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || 'No model to export.'); saveBlob(await response.blob(), 'cell.onnx'); }
    catch (error) { fail(error, 'Could not export the model.'); }
  };

  const exportSettings = async () => {
    try { const data = await (await fetch('/api/training-settings', { cache: 'no-store' })).json(); saveBlob(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }), 'pv-crackvision-training-settings.json'); }
    catch (error) { fail(error, 'Could not export settings.'); }
  };
  const importSettings = async (file: File | undefined) => {
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      const saved = await (await fetch('/api/training-settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(parsed) })).json();
      onSettingsReplaced(saved); flash('Training settings imported and saved. Invalid values were clamped to safe limits.', 9000);
    } catch (error) { fail(error, 'That file is not a valid settings export.'); }
    finally { if (settingsInput.current) settingsInput.current.value = ''; }
  };
  const resetSettings = async () => {
    if (!window.confirm('Reset all training and inference settings to their defaults?')) return;
    try { onSettingsReplaced(await api<TrainingSettings>('/api/training-settings/reset', { method: 'POST' })); flash('Training settings reset to defaults.'); }
    catch (error) { fail(error, 'Could not reset settings.'); }
  };

  const folder = info?.dataFolder;
  return (
    <main className="flex-1 min-h-0 overflow-auto bg-[#090d14] p-4 md:p-6">
      <div className="max-w-4xl mx-auto space-y-5">
        <header>
          <p className="text-[11px] uppercase tracking-[0.2em] text-cyan-400">Setup</p>
          <h1 className="text-2xl font-semibold text-slate-100 mt-1">Data &amp; Models</h1>
          <p className="text-sm text-slate-400 mt-1">Everything you create — annotations, trained models, settings — lives in one folder you choose. It stays the same across versions, upgrades and the future desktop app.{info ? ` · v${info.version}` : ''}</p>
        </header>

        {/* ---- Data folder ---- */}
        <section className={card}>
          <div className="flex items-start gap-3"><HardDrive className="w-5 h-5 text-cyan-300 mt-0.5 shrink-0" />
            <div className="min-w-0 flex-1">
              <h2 className="text-sm font-semibold text-slate-100">Data folder</h2>
              <p className="mt-1 break-all rounded-md bg-[#080d15] px-3 py-2 font-mono text-xs text-slate-200">{folder?.current ?? 'Loading…'}</p>
              {folder && (
                <div className="mt-2 flex flex-wrap gap-2 text-[11px]">
                  {folder.source === 'config' && <span className="rounded-full border border-emerald-700/50 bg-emerald-500/10 px-2 py-0.5 text-emerald-200">Chosen by you · permanent across versions</span>}
                  {folder.source === 'default' && <span className="rounded-full border border-amber-600/50 bg-amber-500/10 px-2 py-0.5 text-amber-200">Default location · tied to where this release was extracted</span>}
                  {folder.source === 'env' && <span className="rounded-full border border-slate-600 bg-slate-800 px-2 py-0.5 text-slate-300">Set by PV_CRACKVISION_DATA_DIR</span>}
                  <span className={`rounded-full border px-2 py-0.5 ${folder.writable ? 'border-slate-700 text-slate-300' : 'border-rose-600/50 bg-rose-500/10 text-rose-200'}`}>{folder.writable ? 'Writable' : 'Not writable'}</span>
                  {usage && <span className="rounded-full border border-slate-700 px-2 py-0.5 text-slate-400">Training {mb(usage.training)} · Models {mb(usage.models)} · Python runtime {mb(usage.runtime)}{usage.truncated ? '+' : ''}</span>}
                </div>
              )}
            </div>
          </div>
          {folder?.isDefault && !folder.envOverride && <p className="rounded-lg border border-amber-600/40 bg-amber-500/10 p-3 text-xs text-amber-100">Pick a permanent folder once and every future version will find your annotations, models and settings there — no more copying between releases.</p>}
          {folder?.restartRequired && (
            <div className="rounded-lg border border-cyan-600/40 bg-cyan-500/10 p-3 text-xs text-cyan-100 space-y-2">
              <p>New data folder saved: <span className="font-mono break-all">{folder.pendingRoot}</span>. Restart to start using it.</p>
              <button type="button" onClick={() => void restart()} disabled={restarting} className={btnPrimary}>{restarting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}{restarting ? 'Restarting…' : 'Restart now'}</button>
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => void openPicker()} disabled={folder?.envOverride || busy === 'folder'} className={btnPrimary}><FolderOpen className="w-3.5 h-3.5" />Change data folder…</button>
            {folder?.source === 'config' && <button type="button" onClick={() => void resetFolder()} disabled={busy === 'folder'} className={btn}><RotateCcw className="w-3.5 h-3.5" />Use default location</button>}
          </div>
          <p className="text-[11px] text-slate-500">Your choice is stored per user in <span className="font-mono">{folder?.configFile ?? '…'}</span>, outside any program folder.</p>
        </section>

        {/* ---- Models ---- */}
        <section className={card}>
          <div className="flex items-start gap-3"><FolderSearch className="w-5 h-5 text-cyan-300 mt-0.5 shrink-0" />
            <div className="flex-1">
              <h2 className="text-sm font-semibold text-slate-100">Load or share a model</h2>
              <p className="mt-1 text-xs text-slate-400">Models are plain files in the data folder, so they load the same way in the desktop app — nothing is bundled inside the program. Load one that was trained elsewhere, or export yours to use on another computer.</p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <input ref={modelInput} type="file" accept=".onnx,.json,.pt" className="hidden" onChange={(event) => void importModel(event.target.files?.[0])} />
            <button type="button" onClick={() => modelInput.current?.click()} disabled={busy === 'model'} className={btnPrimary}>{busy === 'model' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Upload className="w-3.5 h-3.5" />}Load model file…</button>
            <button type="button" onClick={() => void exportModel()} className={btn}><Download className="w-3.5 h-3.5" />Export active cell model</button>
          </div>
          <ul className="text-[11px] text-slate-500 space-y-0.5"><li><span className="text-slate-300">.onnx</span> — cell detector (verified by loading it before it replaces the active one; the previous model is kept for roll-back)</li><li><span className="text-slate-300">.json</span> — crack classifier</li><li><span className="text-slate-300">.pt</span> — training checkpoint; the next training run continues from it</li></ul>
          {modelResult && <p className={`rounded-lg border p-3 text-xs flex gap-2 ${modelResult.ok ? 'border-emerald-700/50 bg-emerald-500/10 text-emerald-100' : 'border-rose-600/50 bg-rose-500/10 text-rose-100'}`}>{modelResult.ok ? <CheckCircle2 className="w-4 h-4 shrink-0" /> : <AlertTriangle className="w-4 h-4 shrink-0" />}<span>{modelResult.text}</span></p>}
        </section>

        {/* ---- Settings persistence ---- */}
        <section className={card}>
          <div><h2 className="text-sm font-semibold text-slate-100">Training &amp; inference settings</h2>
            <p className="mt-1 text-xs text-slate-400">Saved automatically to the data folder every time you change them, and reused by every version. Export them to back up or copy to another computer.</p></div>
          <div className="flex flex-wrap gap-2">
            <input ref={settingsInput} type="file" accept=".json,application/json" className="hidden" onChange={(event) => void importSettings(event.target.files?.[0])} />
            <button type="button" onClick={() => void exportSettings()} className={btn}><Download className="w-3.5 h-3.5" />Export settings</button>
            <button type="button" onClick={() => settingsInput.current?.click()} className={btn}><Upload className="w-3.5 h-3.5" />Import settings…</button>
            <button type="button" onClick={() => void resetSettings()} className={btn}><RotateCcw className="w-3.5 h-3.5" />Reset to defaults</button>
          </div>
        </section>
      </div>

      {/* ---- Folder picker ---- */}
      {pickerOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onClick={() => setPickerOpen(false)}>
          <div className="w-full max-w-2xl rounded-xl border border-[#263448] bg-[#0e131d] p-5 space-y-3" onClick={(event) => event.stopPropagation()}>
            <h3 className="text-sm font-semibold text-slate-100">Choose where PV CrackVision keeps its data</h3>
            <div className="flex gap-2">
              <button type="button" disabled={!browse?.parent} onClick={() => browse?.parent && void load(browse.parent)} className={btn} title="Up one level"><ChevronUp className="w-3.5 h-3.5" /></button>
              <input value={typedPath} onChange={(event) => setTypedPath(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') void load(typedPath); }} className="flex-1 rounded-md border border-slate-700 bg-[#080d15] px-3 py-2 font-mono text-xs text-slate-100 outline-none focus:border-cyan-500" placeholder="Paste a folder path and press Enter" />
            </div>
            <div className="flex flex-wrap gap-1.5">{browse?.roots.map((root) => <button key={root.path} type="button" onClick={() => void load(root.path)} className="rounded-full border border-slate-700 px-2.5 py-1 text-[11px] text-slate-300 hover:bg-slate-800">{root.name}</button>)}</div>
            <div className="h-56 overflow-auto rounded-lg border border-[#1e2636] bg-[#080d15] divide-y divide-[#1e2636]">
              {browse?.dirs.length ? browse.dirs.map((dir) => <button key={dir.path} type="button" onClick={() => void load(dir.path)} className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-slate-200 hover:bg-slate-800/70"><FolderOpen className="w-3.5 h-3.5 text-slate-500" />{dir.name}</button>) : <p className="p-4 text-xs text-slate-500">No sub-folders here. You can still use this folder.</p>}
            </div>
            <p className="rounded-md bg-[#080d15] px-3 py-2 text-[11px] text-slate-400">Data will be stored in <span className="font-mono text-slate-200 break-all">{browse?.resolvedDataRoot}</span></p>
            <label className="flex items-start gap-2 text-xs text-slate-300"><input type="checkbox" checked={copyExisting} onChange={(event) => setCopyExisting(event.target.checked)} className="mt-0.5" /><span>Copy my existing annotations, models and settings into the new folder <span className="text-slate-500">(originals are kept)</span></span></label>
            <label className="flex items-start gap-2 text-xs text-slate-300"><input type="checkbox" checked={includeRuntime} disabled={!copyExisting} onChange={(event) => setIncludeRuntime(event.target.checked)} className="mt-0.5" /><span>Also copy the Python training environment <span className="text-slate-500">(multi-GB and not always relocatable; leave off and rebuild it with one click in Training)</span></span></label>
            <div className="flex justify-end gap-2 pt-1"><button type="button" onClick={() => setPickerOpen(false)} className={btn}>Cancel</button><button type="button" onClick={() => void applyFolder()} disabled={!browse || busy === 'folder'} className={btnPrimary}>{busy === 'folder' && <Loader2 className="w-3.5 h-3.5 animate-spin" />}{busy === 'folder' ? 'Copying…' : 'Use this folder'}</button></div>
          </div>
        </div>
      )}
    </main>
  );
}
