import React, { useEffect, useMemo, useState } from 'react';
import { Download, FileArchive, X } from 'lucide-react';
import JSZip from 'jszip';
import { downloadBlob, markedFileName, renderMarkedImage } from '../utils/markedImage';
import type { InspectionResult } from '../types/inspection';

export interface MarkedFrame {
  id: string;
  label: string;
  imageSrc: string;
  result: InspectionResult | null;
}

interface Props {
  frames: MarkedFrame[];
}

/** Thumbnail rendered lazily; re-rendered whenever the anomaly list changes (adds / deletes). */
const Thumb: React.FC<{ frame: MarkedFrame; onOpen: () => void }> = ({ frame, onOpen }) => {
  const [url, setUrl] = useState<string | null>(null);
  const [err, setErr] = useState(false);
  const r = frame.result;
  useEffect(() => {
    if (!r) return;
    let alive = true, made: string | null = null;
    setErr(false);
    renderMarkedImage(frame.imageSrc, r, { maxSide: 720, quality: 0.8 })
      .then((b) => {
        if (!alive) return;
        made = URL.createObjectURL(b);
        setUrl(made);
      })
      .catch(() => alive && setErr(true));
    return () => {
      alive = false;
      if (made) URL.revokeObjectURL(made);
    };
  }, [frame.imageSrc, r]);
  return (
    <button type="button" onClick={onOpen} className="block w-full aspect-[4/3] bg-[#06080d] overflow-hidden">
      {url ? (
        <img src={url} alt={`${frame.label} marked`} className="w-full h-full object-contain" />
      ) : (
        <span className="text-xs text-slate-500">{err ? 'Preview unavailable' : 'Rendering…'}</span>
      )}
    </button>
  );
};

export const MarkedImagesView: React.FC<Props> = ({ frames }) => {
  const [onlyIssues, setOnlyIssues] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<{ name: string; url: string } | null>(null);

  const done = useMemo(() => frames.filter((f) => f.result), [frames]);
  const shown = useMemo(() => done.filter((f) => !onlyIssues || (f.result?.anomalies.length ?? 0) > 0), [done, onlyIssues]);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview.url); }, [preview]);

  const downloadOne = async (f: MarkedFrame) => {
    if (!f.result) return;
    setBusy(f.id);
    setError(null);
    try {
      downloadBlob(markedFileName(f.label), await renderMarkedImage(f.imageSrc, f.result));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Download failed.');
    } finally {
      setBusy(null);
    }
  };

  const openPreview = async (f: MarkedFrame) => {
    if (!f.result) return;
    try {
      const blob = await renderMarkedImage(f.imageSrc, f.result, { maxSide: 2400, quality: 0.9 });
      setPreview({ name: f.label, url: URL.createObjectURL(blob) });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Preview failed.');
    }
  };

  const downloadAll = async () => {
    setBusy('zip');
    setError(null);
    try {
      const zip = new JSZip();
      const used = new Set<string>();
      for (const f of shown) {
        let name = markedFileName(f.label);
        for (let i = 2; used.has(name); i++) name = markedFileName(f.label).replace('_marked', `_marked_${i}`);
        used.add(name);
        zip.file(name, await renderMarkedImage(f.imageSrc, f.result!));
      }
      downloadBlob(`pv_marked_images_${shown.length}.zip`, await zip.generateAsync({ type: 'blob', compression: 'STORE' }));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'ZIP build failed.');
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="max-w-6xl mx-auto py-8 px-6 space-y-6 w-full">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-5 border-b border-[#1e2636]">
        <div>
          <h1 className="text-2xl font-semibold text-slate-100">Marked Images</h1>
          <p className="text-sm text-slate-400 mt-1.5 max-w-2xl">
            Each inspected frame with its detected issues drawn on it (red = detected, green = added by you). Edits you make in the
            Inspection Stage — added issues and deleted false detections — are reflected here immediately.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <div className="flex items-center gap-1 p-1 bg-[#07090e] border border-[#1e2636] rounded-lg">
            {([false, true] as const).map((v) => (
              <button
                key={String(v)}
                type="button"
                onClick={() => setOnlyIssues(v)}
                className={`px-3 py-1 text-xs font-medium rounded whitespace-nowrap ${onlyIssues === v ? 'bg-[#1a2436] text-cyan-300' : 'text-slate-400 hover:text-slate-200'}`}
              >
                {v ? 'Images With Issues' : 'All Inspected'}
              </button>
            ))}
          </div>
          <button
            type="button"
            disabled={shown.length === 0 || busy !== null}
            onClick={downloadAll}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 rounded-lg hover:bg-cyan-300 disabled:opacity-40 whitespace-nowrap"
          >
            <FileArchive className="w-3.5 h-3.5" />
            {busy === 'zip' ? 'Building ZIP…' : `Download all (${shown.length}) as ZIP`}
          </button>
        </div>
      </div>

      {error && <p className="text-xs text-rose-300">{error}</p>}

      {shown.length === 0 ? (
        <p className="text-sm text-slate-400 py-10 text-center">
          {done.length === 0 ? 'No frames inspected yet.' : 'No inspected frame has issues. Switch to “All Inspected” to see every frame.'}
        </p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {shown.map((f) => (
            <div key={f.id} className="bg-[#0e131d] border border-[#1e2636] rounded-lg overflow-hidden">
              <Thumb frame={f} onOpen={() => void openPreview(f)} />
              <div className="flex items-center justify-between gap-2 px-3 py-2.5">
                <div className="min-w-0">
                  <p className="text-xs font-mono text-slate-200 truncate" title={f.label}>{f.label}</p>
                  <p className={`text-[11px] ${(f.result?.anomalies.length ?? 0) ? 'text-rose-400' : 'text-emerald-400'}`}>
                    {f.result?.anomalies.length ?? 0} issue(s) marked
                  </p>
                </div>
                <button
                  type="button"
                  disabled={busy !== null}
                  onClick={() => void downloadOne(f)}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-cyan-300 bg-[#0b1320] border border-[#1f314d] rounded-md hover:bg-[#121e33] disabled:opacity-40 whitespace-nowrap shrink-0"
                >
                  <Download className="w-3.5 h-3.5" />
                  {busy === f.id ? '…' : 'JPG'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {preview && (
        <div className="fixed inset-0 z-50 bg-black/85 flex flex-col items-center justify-center p-6" onClick={() => setPreview(null)}>
          <button type="button" aria-label="Close preview" className="absolute top-4 right-4 p-2 text-slate-200 hover:text-white">
            <X className="w-5 h-5" />
          </button>
          <img src={preview.url} alt={`${preview.name} marked`} className="max-w-full max-h-[90vh] object-contain" />
        </div>
      )}
    </div>
  );
};
