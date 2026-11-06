import fs from 'node:fs';

/** Camera position record from a raw-image metadata GeoJSON (e.g. exiftool export of DJI frames). */
export interface MetaRecord {
  lat: number;
  lon: number;
  altM: number | null;
  model: string | null;
}

const store = new Map<string, MetaRecord>();

/** dji_20261011114408_0074_z_9902db27.JPG -> dji_20261011114408_0074_z (folder, extension, _hash stripped) */
export function normalizeImageKey(name: string): string {
  const base = name.split(/[\\/]/).pop() ?? name;
  return base
    .toLowerCase()
    .replace(/\.[a-z0-9]+$/, '')
    .replace(/\s*\(\d+\)$/, '') // browser/OS duplicate suffix: "DJI_..._Z (1).JPG" must still match its metadata record
    .replace(/_[0-9a-f]{8}$/, '');
}

const num = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;

/** Accepts a GeoJSON FeatureCollection (properties.filename + GPSLatitude/GPSLongitude, or Point geometry). Returns records loaded. */
export function loadMetadata(json: unknown): number {
  const feats = (json as { features?: unknown[] } | null)?.features;
  if (!Array.isArray(feats)) throw new Error('Metadata JSON must be a GeoJSON FeatureCollection with a "features" array.');
  store.clear();
  for (const f of feats) {
    const ft = f as {
      geometry?: { coordinates?: unknown[] } | null;
      properties?: Record<string, unknown>;
    };
    const p = ft.properties ?? {};
    const name = typeof p.filename === 'string' ? p.filename : typeof p.key === 'string' ? p.key : null;
    const lat = num(p.GPSLatitude) ?? num(ft.geometry?.coordinates?.[1]);
    const lon = num(p.GPSLongitude) ?? num(ft.geometry?.coordinates?.[0]);
    if (!name || lat === null || lon === null || Math.abs(lat) > 90 || Math.abs(lon) > 180) continue;
    store.set(normalizeImageKey(name), {
      lat,
      lon,
      altM: num(p.GPSAltitude),
      model: typeof p.Model === 'string' ? p.Model : null,
    });
  }
  return store.size;
}

export function loadMetadataFile(path: string): number {
  return loadMetadata(JSON.parse(fs.readFileSync(path, 'utf8')));
}

export const lookupMetadata = (fileName: string): MetaRecord | null =>
  store.get(normalizeImageKey(fileName)) ?? null;

export const metadataCount = (): number => store.size;
