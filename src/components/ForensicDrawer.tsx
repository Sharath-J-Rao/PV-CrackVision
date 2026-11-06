import React, { useState } from 'react';
import { Check, Copy, Focus } from 'lucide-react';
import type { DetectedAnomaly, GeoJsonPointFeature } from '../types/inspection';

interface ForensicDrawerProps {
  anomaly: DetectedAnomaly | null;
  imageGeoFeature: GeoJsonPointFeature | null;
  onFocusOnCanvas: (id: string) => void;
}

function formatSeverityLabel(severity: DetectedAnomaly['severity']): {
  text: string;
  tone: string;
} {
  switch (severity) {
    case 'critical':
      return { text: '✖ Critical Fracture', tone: 'text-rose-400' };
    case 'moderate':
      return { text: '▲ Moderate Defect', tone: 'text-amber-400' };
    case 'minor':
      return { text: '● Minor Anomaly', tone: 'text-emerald-400' };
    case 'informational':
      return { text: '○ Surface Deposit', tone: 'text-sky-400' };
  }
}

export const ForensicDrawer: React.FC<ForensicDrawerProps> = ({
  anomaly,
  imageGeoFeature,
  onFocusOnCanvas,
}) => {
  const [copiedGeo, setCopiedGeo] = useState(false);

  if (!anomaly) {
    return (
      <div className="p-4 border-t border-[#1e2636] bg-[#0b0e14] text-xs text-slate-400">
        Select any detected crack attribute to inspect its contrast-enhanced 1:1 crop, off-axis hairline map, and parent Image GeoJSON Point attributes.
      </div>
    );
  }

  const sev = formatSeverityLabel(anomaly.severity);

  const handleCopyImageGeoJson = async () => {
    if (!imageGeoFeature) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(imageGeoFeature, null, 2));
      setCopiedGeo(true);
      setTimeout(() => setCopiedGeo(false), 1800);
    } catch {
      setCopiedGeo(false);
    }
  };

  return (
    <div className="p-4 border-t border-[#1e2636] bg-[#0b0e14] space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-xs font-mono tabular-nums">
            <span className="text-slate-200 font-semibold">{anomaly.id}</span>
            <span aria-hidden="true" className="text-slate-600">·</span>
            <span className={sev.tone}>{sev.text}</span>
          </div>
          <h4 className="text-sm font-semibold text-slate-100 mt-0.5 truncate">
            {anomaly.title}
          </h4>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {imageGeoFeature && (
            <button
              type="button"
              onClick={handleCopyImageGeoJson}
              className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-slate-200 bg-[#131b2b] border border-[#24314a] rounded hover:bg-[#19243a] transition-colors whitespace-nowrap"
            >
              {copiedGeo ? (
                <Check className="w-3 h-3 text-emerald-400" />
              ) : (
                <Copy className="w-3 h-3" />
              )}
              {copiedGeo ? 'Copied' : 'Image GeoJSON'}
            </button>
          )}
          <button
            type="button"
            onClick={() => onFocusOnCanvas(anomaly.id)}
            className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-cyan-300 bg-[#131b2b] border border-[#24314a] rounded hover:bg-[#19243a] transition-colors whitespace-nowrap"
          >
            <Focus className="w-3 h-3" />
            Zoom
          </button>
        </div>
      </div>

      {/* Dual 1:1 Dark-Silicon Enhanced Crop vs Hairline Ridge Map */}
      <div className="grid grid-cols-2 gap-2.5">
        <div className="space-y-1">
          <div className="aspect-square w-full bg-[#07090e] border border-[#1e2636] rounded overflow-hidden">
            {anomaly.cropDataUrl ? (
              <img
                src={anomaly.cropDataUrl}
                alt={`Enhanced 1:1 crop for ${anomaly.id}`}
                referrerPolicy="no-referrer"
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-xs text-slate-500">
                1:1 Patch
              </div>
            )}
          </div>
          <p className="text-[11px] font-mono text-slate-400">
            Contrast-Enhanced 1:1 Crop
          </p>
        </div>

        <div className="space-y-1">
          <div className="aspect-square w-full bg-[#07090e] border border-[#1e2636] rounded overflow-hidden">
            {anomaly.edgeMapDataUrl ? (
              <img
                src={anomaly.edgeMapDataUrl}
                alt={`Hairline fracture ridge map for ${anomaly.id}`}
                referrerPolicy="no-referrer"
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-xs text-slate-500">
                Ridge Map
              </div>
            )}
          </div>
          <p className="text-[11px] font-mono text-slate-400">
            Off-Axis Hairline Filter
          </p>
        </div>
      </div>

      <p className="text-xs text-slate-300 leading-relaxed">
        {anomaly.description}
      </p>

      {/* Attribute Details inside Parent Image GeoJSON Point */}
      <div className="grid grid-cols-2 gap-x-3 gap-y-2 pt-2 border-t border-[#1e2636] text-xs">
        {imageGeoFeature && (
          <div className="col-span-2">
            <span className="text-slate-500 block">
              Image GeoJSON Point ({imageGeoFeature.properties.crs})
            </span>
            <span className="font-mono tabular-nums text-cyan-300">
              [{imageGeoFeature.geometry.coordinates[0]}, {imageGeoFeature.geometry.coordinates[1]}] m
            </span>
          </div>
        )}
        <div>
          <span className="text-slate-500 block">Perimeter / Cell</span>
          <span className="font-mono tabular-nums text-slate-200">
            {anomaly.edgeSide} · {anomaly.cellLocation}
          </span>
        </div>
        <div>
          <span className="text-slate-500 block">Origin Pixel & Span</span>
          <span className="font-mono tabular-nums text-slate-200">
            ({anomaly.pointPx.x}, {anomaly.pointPx.y}) · ~{anomaly.estimatedDiameterMm ?? 0} mm
          </span>
        </div>
      </div>
    </div>
  );
};
