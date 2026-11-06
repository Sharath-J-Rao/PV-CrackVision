import sharp from 'sharp';
import { buildValidatedCellGeometry } from './cellGeometry.ts';
import ExifReader from 'exifreader';
import { lookupMetadata } from './metadataStore.ts';
import { loadCrackModel, scanForEdgeWebs, nmsWindows } from './crackModel.ts';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import type {
  BoundingBox,
  CentroidMethod,
  CrsMode,
  DefectCategory,
  DetectedAnomaly,
  EdgeSide,
  GeoJsonCrackAttributeItem,
  GeoJsonFeatureCollection,
  GeoJsonPointFeature,
  GeoReferenceInfo,
  InspectionConfig,
  InspectionResult,
  ModuleRegion,
  PixelBox,
  SolarCellGrid,
  SeverityLevel,
  TileDiagnostic,
} from '../src/types/inspection.ts';

const EPSG3857_LABEL = 'EPSG:3857 - WGS 84 / Pseudo-Mercator';
const EPSG3857_URN = 'urn:ogc:def:crs:EPSG::3857';
const EARTH_RADIUS_M = 6378137.0;

/**
 * Projects WGS84 [longitude, latitude] degrees into EPSG:3857 (WGS 84 / Pseudo-Mercator) meters [X, Y].
 */
export function lonLatToEpsg3857(lon: number, lat: number): [number, number] {
  const clampedLat = Math.max(-85.05112878, Math.min(85.05112878, lat));
  const x = EARTH_RADIUS_M * lon * (Math.PI / 180);
  const y =
    EARTH_RADIUS_M * Math.log(Math.tan(Math.PI / 4 + (clampedLat * Math.PI) / 360));
  return [Number(x.toFixed(3)), Number(y.toFixed(3))];
}

/**
 * Inverse projection from EPSG:3857 meters [X, Y] to WGS84 [longitude, latitude] degrees.
 */
export function epsg3857ToLonLat(x: number, y: number): [number, number] {
  const lon = (x / EARTH_RADIUS_M) * (180 / Math.PI);
  const lat = (2 * Math.atan(Math.exp(y / EARTH_RADIUS_M)) - Math.PI / 2) * (180 / Math.PI);
  return [Number(lon.toFixed(8)), Number(lat.toFixed(8))];
}

// Returns null when no usable key -> pipeline falls back to classical-only (offline) detection
function getGenAIClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  const placeholder = !apiKey || /^MY_GEMINI_API_KEY$/i.test(apiKey);
  if (placeholder || process.env.CV_ENGINE === 'classical') return null;
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// Model names are env-overridable instead of hard-coded
const PRIMARY_MODEL = process.env.GEMINI_PRIMARY_MODEL || 'gemini-3.8-flash';
const FALLBACK_MODEL = process.env.GEMINI_FALLBACK_MODEL || 'gemini-3.1-flash-lite';

// Quota-exhausted models are skipped until the next UTC day (was: forever)
const dailyExhaustedModels = new Set<string>();
let exhaustedDay = new Date().toISOString().slice(0, 10);
function resetExhaustedIfNewDay(): void {
  const today = new Date().toISOString().slice(0, 10);
  if (today !== exhaustedDay) {
    dailyExhaustedModels.clear();
    exhaustedDay = today;
  }
}

const detectionItemSchema = {
  type: Type.OBJECT,
  properties: {
    category: {
      type: Type.STRING,
      description:
        'Must be one of: spider_web_crack, impact_point, linear_fracture, surface_residue',
    },
    isEdgeCrack: {
      type: Type.BOOLEAN,
      description:
        'True only if the fracture touches the outermost solar-cell row/column or array frame; interior-cell fractures must be false.',
    },
    edgeSide: {
      type: Type.STRING,
      description:
        'One of: bottom-left, top-left, bottom-right, top-right, left-edge, right-edge, top-edge, bottom-edge, interior',
    },
    severity: {
      type: Type.STRING,
      description: 'Must be one of: critical, moderate, minor, informational',
    },
    confidence: {
      type: Type.NUMBER,
      description: 'Confidence score between 0.50 and 0.99',
    },
    title: {
      type: Type.STRING,
      description:
        'Concise technical defect label identifying the affected solar cell(s), e.g. Interior Solar-Cell Hairline Fracture or Edge-Originated Spider-Web Glass Crack.',
    },
    description: {
      type: Type.STRING,
      description:
        '1-2 sentence empirical description of the spider-web hairline fracture arcs, edge origin, affected cell columns/rows, or surface anomaly.',
    },
    radialBranches: {
      type: Type.INTEGER,
      description:
        'Count of visible sweeping fracture arcs or branching filaments (e.g. 2..14 for spider_web_crack).',
    },
    pointY: {
      type: Type.INTEGER,
      description:
        'Normalized 0..1000 Y coordinate of the primary crack nexus or edge initiation point within that tile.',
    },
    pointX: {
      type: Type.INTEGER,
      description:
        'Normalized 0..1000 X coordinate of the primary crack nexus or edge initiation point within that tile.',
    },
    ymin: {
      type: Type.INTEGER,
      description: 'Top bounding box edge normalized 0..1000 in that tile.',
    },
    xmin: {
      type: Type.INTEGER,
      description: 'Left bounding box edge normalized 0..1000 in that tile.',
    },
    ymax: {
      type: Type.INTEGER,
      description: 'Bottom bounding box edge normalized 0..1000 in that tile.',
    },
    xmax: {
      type: Type.INTEGER,
      description: 'Right bounding box edge normalized 0..1000 in that tile.',
    },
  },
  required: [
    'category',
    'isEdgeCrack',
    'edgeSide',
    'severity',
    'confidence',
    'title',
    'description',
    'radialBranches',
    'pointY',
    'pointX',
    'ymin',
    'xmin',
    'ymax',
    'xmax',
  ],
};

const tileDetectionSchema = {
  type: Type.OBJECT,
  properties: {
    tileResults: {
      type: Type.ARRAY,
      description:
        'One inspection result object per provided TILE_INDEX (0 through N-1), in order.',
      items: {
        type: Type.OBJECT,
        properties: {
          tileIndex: {
            type: Type.INTEGER,
            description: '0-based TILE_INDEX of the inspected tile.',
          },
          detections: {
            type: Type.ARRAY,
            description:
              'All spider-web cracks (both wide sweeping arcs AND single-column/multi-column border-cell spider-web networks near left/right/top/bottom edges), impact pits, linear fractures, or surface residue visible in this tile.',
            items: detectionItemSchema,
          },
        },
        required: ['tileIndex', 'detections'],
      },
    },
  },
  required: ['tileResults'],
};

function parseXmpFloat(val: unknown): number | null {
  if (val === null || val === undefined) return null;
  if (typeof val === 'number' && Number.isFinite(val)) return val;
  if (typeof val === 'object') {
    const rec = val as Record<string, unknown>;
    const raw = rec.description ?? rec.value;
    if (typeof raw === 'number' && Number.isFinite(raw)) return raw;
    if (typeof raw === 'string') {
      const n = Number.parseFloat(raw);
      return Number.isFinite(n) ? n : null;
    }
  }
  if (typeof val === 'string') {
    const n = Number.parseFloat(val);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function parseExifString(val: unknown): string | null {
  if (!val) return null;
  if (typeof val === 'string') return val;
  if (typeof val === 'object') {
    const rec = val as Record<string, unknown>;
    if (typeof rec.description === 'string') return rec.description;
    if (typeof rec.value === 'string') return rec.value;
  }
  return null;
}

function parseNumberArrayTag(val: unknown): number[] | null {
  if (!val || typeof val !== 'object') return null;
  const rec = val as Record<string, unknown>;
  const raw = rec.value ?? rec.description;
  if (Array.isArray(raw)) {
    const nums = raw.map((v) => Number(v)).filter((n) => Number.isFinite(n));
    return nums.length > 0 ? nums : null;
  }
  return null;
}

/**
 * Extracts GeoTIFF tiepoints, DJI RTK XMP, or EXIF GPS metadata and projects image coordinates
 * into EPSG:3857 - WGS 84 / Pseudo-Mercator meters [X_easting, Y_northing].
 */
function extractGeoReference(
  imageBuffer: Buffer,
  imgWidth: number,
  imgHeight: number,
  refModWidthPx: number,
  config: InspectionConfig,
  fileName: string
): GeoReferenceInfo {
  let hasEmbeddedGps = false;
  let crsMode: CrsMode = 'PIXEL_LOCAL';
  let centroidMethod: CentroidMethod = 'none';
  // Camera (drone) position vs. image centroid are tracked separately: the GeoJSON Point is the centroid
  let camLat: number | null = null;
  let camLon: number | null = null;
  let lrfLat: number | null = null;
  let lrfLon: number | null = null;
  let lrfOk = false;
  let gimbalPitchDeg: number | null = null;
  let metaUsed = false;
  let centerLat: number | null = null;
  let centerLon: number | null = null;
  let epsg3857X: number | null = null;
  let epsg3857Y: number | null = null;
  let altitudeM: number | null = null;
  let relativeAltitudeM: number | null = null;
  let gimbalYawDeg = 0;
  let cameraModel: string | null = null;
  let capturedAt: string | null = null;

  // Estimate GSD from known standard utility-scale 144-cell PV module landscape width (~2.278 m)
  const moduleBasedGsdMm = refModWidthPx > 250 ? 2278 / refModWidthPx : 2278 / (imgWidth * 0.44);
  let gsdMmPerPx = Number(moduleBasedGsdMm.toFixed(3));

  try {
    const tags = ExifReader.load(imageBuffer, { expanded: true });
    const xmp = (tags.xmp || {}) as Record<string, unknown>;
    const exif = (tags.exif || {}) as Record<string, unknown>;
    const gps = (tags.gps || {}) as Record<string, unknown>;

    // 1. Check GeoTIFF ModelTiepointTag / ModelPixelScaleTag (e.g., EPSG:3857 orthomosaic tiles)
    const tiepoint = parseNumberArrayTag(exif.ModelTiepointTag);
    const pixelScale = parseNumberArrayTag(exif.ModelPixelScaleTag);
    if (tiepoint && tiepoint.length >= 6) {
      const tieI = tiepoint[0];
      const tieJ = tiepoint[1];
      const tieX = tiepoint[3];
      const tieY = tiepoint[4];
      const scaleX = pixelScale && pixelScale.length >= 2 ? pixelScale[0] : 0;
      const scaleY = pixelScale && pixelScale.length >= 2 ? pixelScale[1] : 0;
      const centerX = tieX + (imgWidth / 2 - tieI) * scaleX;
      const centerY = tieY - (imgHeight / 2 - tieJ) * scaleY;

      if (Math.abs(centerX) > 180 || Math.abs(centerY) > 90) {
        epsg3857X = Number(centerX.toFixed(3));
        epsg3857Y = Number(centerY.toFixed(3));
        const [lon, lat] = epsg3857ToLonLat(epsg3857X, epsg3857Y);
        centerLon = lon;
        centerLat = lat;
        hasEmbeddedGps = true;
        centroidMethod = 'geotiff_center';
        crsMode = 'EPSG3857_GEOTIFF';
        if (scaleX > 0.00005 && scaleX < 0.05) {
          gsdMmPerPx = Number((scaleX * 1000).toFixed(3));
        }
      } else if (Math.abs(centerX) <= 180 && Math.abs(centerY) <= 90) {
        centerLon = centerX;
        centerLat = centerY;
        const [mx, my] = lonLatToEpsg3857(centerLon, centerLat);
        epsg3857X = mx;
        epsg3857Y = my;
        hasEmbeddedGps = true;
        centroidMethod = 'geotiff_center';
        crsMode = 'EPSG3857_GEOTIFF';
      }
    }

    // 2. Camera position: DJI RTK XMP > EXIF GPS (centroid is derived after this try block)
    if (!hasEmbeddedGps) {
      const xmpLat = parseXmpFloat(xmp.GpsLatitude);
      const xmpLon = parseXmpFloat(xmp.GpsLongitude);
      const exifLat = typeof gps.Latitude === 'number' ? gps.Latitude : null;
      const exifLon = typeof gps.Longitude === 'number' ? gps.Longitude : null;
      if (xmpLat !== null && xmpLon !== null) {
        camLat = xmpLat;
        camLon = xmpLon;
        crsMode = 'EPSG3857_RTK';
      } else if (exifLat !== null && exifLon !== null) {
        camLat = exifLat;
        camLon = exifLon;
        crsMode = 'EPSG3857_EXIF';
      }
      // Laser-rangefinder target = ground point under the image centre (true centroid when valid)
      lrfLat = parseXmpFloat(xmp.LRFTargetLat);
      lrfLon = parseXmpFloat(xmp.LRFTargetLon);
      lrfOk = (parseExifString(xmp.LRFStatus) ?? '').toLowerCase() === 'normal';
      gimbalPitchDeg = parseXmpFloat(xmp.GimbalPitchDegree);
    }

    altitudeM =
      parseXmpFloat(xmp.AbsoluteAltitude) ??
      (typeof gps.Altitude === 'number' ? gps.Altitude : null);
    relativeAltitudeM =
      parseXmpFloat(xmp.RelativeAltitude) ?? parseXmpFloat(xmp.LRFTargetAlt); // RelativeAltitude = true AGL height

    const yaw =
      parseXmpFloat(xmp.GimbalYawDegree) ?? parseXmpFloat(xmp.FlightYawDegree) ?? 0;
    gimbalYawDeg = yaw;

    const make = parseExifString(exif.Make);
    const model = parseExifString(exif.Model);
    cameraModel = [make, model].filter(Boolean).join(' ') || null;
    capturedAt = parseExifString(exif.DateTimeOriginal) ?? parseExifString(exif.DateTime);

    const focal35 = parseXmpFloat(exif.FocalLengthIn35mmFilm);
    if (focal35 && relativeAltitudeM && relativeAltitudeM > 1) {
      const opticalGsdMm = ((relativeAltitudeM * 36.0) / (focal35 * imgWidth)) * 1000;
      if (opticalGsdMm > 0.15 && opticalGsdMm < 25) {
        gsdMmPerPx = Number((opticalGsdMm * 0.5 + moduleBasedGsdMm * 0.5).toFixed(3));
      }
    }
  } catch {
    // Non-EXIF image format (e.g. JFIF / PNG)
  }

  // 3. No embedded GPS: look the camera position up in the loaded raw-image metadata JSON by filename
  if (centroidMethod === 'none' && camLat === null && lrfLat === null) {
    const meta = lookupMetadata(fileName);
    if (meta) {
      camLat = meta.lat;
      camLon = meta.lon;
      altitudeM = altitudeM ?? meta.altM;
      cameraModel = cameraModel ?? meta.model;
      crsMode = 'EPSG3857_METADATA';
      metaUsed = true;
    }
  }

  // 4. Derive the image centroid: LRF target > camera shifted along gimbal > camera (nadir assumed)
  if (centroidMethod === 'none' && (camLat !== null || lrfLat !== null)) {
    const MPD = 111320; // metres per degree latitude
    const cosLat = Math.cos(((camLat ?? lrfLat ?? 0) * Math.PI) / 180);
    let cLat = camLat;
    let cLon = camLon;
    let method: CentroidMethod = metaUsed ? 'metadata_json_nadir' : 'camera_nadir';
    const lrfValid =
      lrfOk &&
      lrfLat !== null &&
      lrfLon !== null &&
      lrfLat !== 0 &&
      lrfLon !== 0 &&
      Math.abs(lrfLat) <= 90 &&
      Math.abs(lrfLon) <= 180 &&
      (camLat === null ||
        camLon === null ||
        Math.hypot((lrfLat - camLat) * MPD, (lrfLon - camLon) * MPD * cosLat) <= 300); // sanity: target near camera
    if (lrfValid) {
      cLat = lrfLat;
      cLon = lrfLon;
      method = 'lrf_target';
    } else if (
      camLat !== null &&
      camLon !== null &&
      !metaUsed &&
      gimbalPitchDeg !== null &&
      gimbalPitchDeg > -87 && // within 3 deg of straight down counts as nadir
      gimbalPitchDeg <= -20 && // too close to horizon: offset unreliable
      relativeAltitudeM !== null &&
      relativeAltitudeM > 1
    ) {
      // Oblique shot: image centre lies relAlt * tan(90deg + pitch) ahead of the camera along gimbal yaw
      const d = relativeAltitudeM * Math.tan(((90 + gimbalPitchDeg) * Math.PI) / 180);
      const yawRad = (gimbalYawDeg * Math.PI) / 180;
      cLat = camLat + (d * Math.cos(yawRad)) / MPD;
      cLon = camLon + (d * Math.sin(yawRad)) / (MPD * cosLat);
      method = 'gimbal_offset';
    }
    if (cLat !== null && cLon !== null) {
      centerLat = cLat;
      centerLon = cLon;
      const [mx, my] = lonLatToEpsg3857(cLon, cLat);
      epsg3857X = mx;
      epsg3857Y = my;
      hasEmbeddedGps = !metaUsed;
      centroidMethod = method;
    }
  }

  if (centroidMethod === 'none' && config.useSiteReferenceForNonGps) {
    // Support either WGS84 (Lon/Lat) or direct EPSG:3857 meters in siteReferenceLon/Lat
    if (Math.abs(config.siteReferenceLon) > 180 || Math.abs(config.siteReferenceLat) > 90) {
      epsg3857X = Number(config.siteReferenceLon.toFixed(3));
      epsg3857Y = Number(config.siteReferenceLat.toFixed(3));
      const [lon, lat] = epsg3857ToLonLat(epsg3857X, epsg3857Y);
      centerLon = lon;
      centerLat = lat;
    } else {
      centerLat = config.siteReferenceLat;
      centerLon = config.siteReferenceLon;
      const [mx, my] = lonLatToEpsg3857(centerLon, centerLat);
      epsg3857X = mx;
      epsg3857Y = my;
    }
    crsMode = 'EPSG3857_SITE_REF';
    centroidMethod = 'site_reference';
  }

  const isEpsg3857 = epsg3857X !== null && epsg3857Y !== null && crsMode !== 'PIXEL_LOCAL';
  const imageGeoPoint: [number, number] = isEpsg3857
    ? [epsg3857X!, epsg3857Y!]
    : [Math.round(imgWidth / 2), -Math.round(imgHeight / 2)];

  return {
    centroidMethod,
    cameraLat: camLat,
    cameraLon: camLon,
    hasEmbeddedGps,
    crsMode,
    epsgCode: isEpsg3857 ? 'EPSG:3857' : 'PIXEL_LOCAL',
    crsName: isEpsg3857 ? EPSG3857_LABEL : 'Local Image Pixel Coordinates',
    imageGeoPoint,
    epsg3857X,
    epsg3857Y,
    centerLat,
    centerLon,
    altitudeM,
    relativeAltitudeM,
    gimbalYawDeg,
    cameraModel,
    capturedAt,
    gsdMmPerPx,
  };
}

interface DownsampledAnalysis {
  scaleWidth: number;
  scaleHeight: number;
  pvMask: Uint8Array;
  offAxisEnergy: Float32Array;
  impactEnergy: Float32Array;
  moduleRegions: ModuleRegion[];
  fractureHeatmapUrl: string;
  cellBoundaryMapUrl: string;
  activeBounds: PixelBox;
  /** Per-pixel (downsampled) masks used to veto detections that sit on ground */
  siliconMask: Uint8Array;
  soilMask: Uint8Array;
}

/**
 * Neutralizes saturated user annotation lines (blue/red/green boxes) so neither classical CV
 * nor VLM is blinded or distracted by drawn markup.
 */
function sanitizeUserMarkupRgb(
  r: number,
  g: number,
  b: number
): { r: number; g: number; b: number; wasMarkup: boolean } {
  const isBlueBox = b - r > 46 && b > 120;
  const isRedBox = r - g > 75 && r - b > 75 && r > 170;
  const isGreenBox = g - r > 60 && g - b > 50 && g > 145;
  const isOrangeBox = r > 165 && g > 75 && g < 205 && b < 115 && r - g > 35 && g - b > 20;
  const isYellowBox = r > 185 && g > 155 && b < 125 && r - b > 65 && g - b > 45;
  if (isBlueBox || isRedBox || isGreenBox || isOrangeBox || isYellowBox) {
    const neutral = Math.min(80, Math.round(Math.min(r, g, b) * 0.85));
    return { r: neutral, g: neutral, b: neutral, wasMarkup: true };
  }
  return { r, g, b, wasMarkup: false };
}

/**
 * Build the directional edge maps used for cell-boundary geometry.
 * `vertical` responds to vertical cell seams (X derivative); `horizontal` responds
 * to horizontal seams (Y derivative). These are intentionally separate from the
 * off-axis fracture heatmap, which suppresses regular grid structure for crack scoring.
 */
function buildSolarCellDirectionalEdgeMaps(
  gray: Float32Array,
  sw: number,
  sh: number
): { vertical: Float32Array; horizontal: Float32Array } {
  const vertical = new Float32Array(sw * sh);
  const horizontal = new Float32Array(sw * sh);
  for (let y = 1; y < sh - 1; y++) {
    for (let x = 1; x < sw - 1; x++) {
      const i = y * sw + x;
      const tl = gray[i - sw - 1];
      const tr = gray[i - sw + 1];
      const ml = gray[i - 1];
      const mr = gray[i + 1];
      const bl = gray[i + sw - 1];
      const br = gray[i + sw + 1];
      const tc = gray[i - sw];
      const bc = gray[i + sw];
      const gx = Math.abs(-tl + tr - 2 * ml + 2 * mr - bl + br);
      const gy = Math.abs(-tl - 2 * tc - tr + bl + 2 * bc + br);
      // Suppress weak cell-finger texture while preserving the high-contrast cell seams.
      vertical[i] = Math.max(0, gx - 14);
      horizontal[i] = Math.max(0, gy - 14);
    }
  }
  return { vertical, horizontal };
}

interface CellAxisCandidate {
  pitch: number;
  confidence: number;
}

/** Return several real periodicity peaks so the two axes can be resolved together. */
function estimateCellAxisPitchCandidates(profile: Float32Array): CellAxisCandidate[] {
  const n = profile.length;
  if (n < 36) return [];

  const smooth = new Float32Array(n);
  let mean = 0;
  let maxValue = 0;
  for (let i = 0; i < n; i++) {
    const value = (
      profile[Math.max(0, i - 1)] +
      profile[i] +
      profile[Math.min(n - 1, i + 1)]
    ) / 3;
    smooth[i] = value;
    mean += value;
    if (value > maxValue) maxValue = value;
  }
  mean /= n;
  if (maxValue - mean < 0.025) return [];

  const centered = new Float32Array(n);
  for (let i = 0; i < n; i++) centered[i] = smooth[i] - mean;

  const minLag = Math.max(6, Math.round(n * 0.015));
  const maxLag = Math.min(Math.floor(n / 3), Math.round(n * 0.36));
  const correlations: number[] = new Array(maxLag + 1).fill(-1);
  let best = -1;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let dot = 0;
    let aa = 0;
    let bb = 0;
    for (let i = 0; i < n - lag; i++) {
      const a = centered[i];
      const b = centered[i + lag];
      dot += a * b;
      aa += a * a;
      bb += b * b;
    }
    const correlation = dot / (Math.sqrt(aa * bb) + 1e-8);
    correlations[lag] = correlation;
    if (correlation > best) best = correlation;
  }
  if (best < 0.12) return [];

  const peaks: CellAxisCandidate[] = [];
  for (let lag = minLag + 1; lag < maxLag; lag++) {
    const score = correlations[lag];
    if (
      score >= best * 0.52 &&
      score >= correlations[lag - 1] &&
      score >= correlations[lag + 1]
    ) {
      peaks.push({ pitch: lag, confidence: Math.max(0, Math.min(1, score)) });
    }
  }

  // Keep distinct local maxima; a true cell pitch and its harmonic (2x, 3x) remain
  // available for joint selection, while adjacent one-pixel maxima are collapsed.
  peaks.sort((a, b) => b.confidence - a.confidence);
  const selected: CellAxisCandidate[] = [];
  for (const candidate of peaks) {
    if (selected.every((existing) => Math.abs(existing.pitch - candidate.pitch) >= 3)) {
      selected.push(candidate);
    }
    if (selected.length >= 12) break;
  }
  return selected;
}

/**
 * Use the same directional edge map that is exposed in the UI to measure cell pitch.
 * The X-derivative map gives vertical boundaries; the Y-derivative map gives horizontal
 * boundaries. We jointly choose the pair so a strong half-pitch busbar/finger repeat on
 * one axis is not accidentally combined with a whole-cell repeat on the other.
 */
function estimateSolarCellGrid(
  verticalEdgeMap: Float32Array,
  horizontalEdgeMap: Float32Array,
  whiteGridPixel: Uint8Array,
  sw: number,
  sh: number,
  imageWidth: number,
  imageHeight: number,
  moduleBox: PixelBox,
  pvSiliconMask: Uint8Array,
  sharedPitch?: { pitchX: number; pitchY: number; confidence: number }
): SolarCellGrid | undefined {
  const x0 = Math.max(0, Math.min(sw - 1, Math.floor((moduleBox.left / imageWidth) * sw)));
  const y0 = Math.max(0, Math.min(sh - 1, Math.floor((moduleBox.top / imageHeight) * sh)));
  const x1 = Math.max(x0 + 1, Math.min(sw, Math.ceil(((moduleBox.left + moduleBox.width) / imageWidth) * sw)));
  const y1 = Math.max(y0 + 1, Math.min(sh, Math.ceil(((moduleBox.top + moduleBox.height) / imageHeight) * sh)));
  const width = x1 - x0;
  const height = y1 - y0;
  if (width < 36 || height < 36) return undefined;

  const edgeColumnProfile = new Float32Array(width);
  const edgeRowProfile = new Float32Array(height);
  const xBandStart = x0 + Math.max(2, Math.round(width * 0.06));
  const xBandEnd = x1 - Math.max(2, Math.round(width * 0.06));
  const yBandStart = y0 + Math.max(2, Math.round(height * 0.06));
  const yBandEnd = y1 - Math.max(2, Math.round(height * 0.06));

  for (let y = yBandStart; y < yBandEnd; y++) {
    const rowOffset = y * sw;
    for (let x = x0; x < x1; x++) {
      edgeColumnProfile[x - x0] += verticalEdgeMap[rowOffset + x];
    }
  }
  for (let y = y0; y < y1; y++) {
    const rowOffset = y * sw;
    for (let x = xBandStart; x < xBandEnd; x++) {
      edgeRowProfile[y - y0] += horizontalEdgeMap[rowOffset + x];
    }
  }
  for (let x = 0; x < width; x++) edgeColumnProfile[x] /= Math.max(1, yBandEnd - yBandStart);
  for (let y = 0; y < height; y++) edgeRowProfile[y] /= Math.max(1, xBandEnd - xBandStart);

  const xCandidates = estimateCellAxisPitchCandidates(edgeColumnProfile);
  const yCandidates = estimateCellAxisPitchCandidates(edgeRowProfile);
  let selectedPair: { x: CellAxisCandidate; y: CellAxisCandidate; score: number } | undefined;

  // PV cells are commonly square or half-cut rectangles. Permit both layouts without
  // hard-coding a row/column count, but reject implausible edge harmonics.
  const commonCellAspectRatios = [0.5, 1, 2];
  for (const xCandidate of xCandidates) {
    for (const yCandidate of yCandidates) {
      const columns = Math.round(width / xCandidate.pitch);
      const rows = Math.round(height / yCandidate.pitch);
      if (columns < 2 || columns > 96 || rows < 2 || rows > 48) continue;
      const cellAspect = (width / columns) / (height / rows);
      if (cellAspect < 0.38 || cellAspect > 2.65) continue;
      const aspectPenalty = Math.min(...commonCellAspectRatios.map((ratio) =>
        Math.abs(Math.log(cellAspect / ratio))
      ));
      const score = xCandidate.confidence + yCandidate.confidence - aspectPenalty * 0.12;
      if (!selectedPair || score > selectedPair.score) {
        selectedPair = { x: xCandidate, y: yCandidate, score };
      }
    }
  }

  let method: SolarCellGrid['method'] = 'edge-map-periodicity';
  let pitchX = sharedPitch ? sharedPitch.pitchX * sw / imageWidth : selectedPair?.x.pitch;
  let pitchY = sharedPitch ? sharedPitch.pitchY * sh / imageHeight : selectedPair?.y.pitch;
  let confidence = sharedPitch
    ? sharedPitch.confidence
    : selectedPair ? (selectedPair.x.confidence + selectedPair.y.confidence) / 2 : 0;

  // When a frame-level pitch has been calibrated, retain it across every visible cell
  // field. This prevents a dim/partial row band from choosing a 2x harmonic and turning
  // a standard cell lattice into a false 8x14 (or similar) grid.
  if (!sharedPitch && (!pitchX || !pitchY)) {
    const whiteColumnProfile = new Float32Array(width);
    const whiteRowProfile = new Float32Array(height);
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        if (!whiteGridPixel[y * sw + x]) continue;
        whiteColumnProfile[x - x0]++;
        whiteRowProfile[y - y0]++;
      }
    }
    for (let x = 0; x < width; x++) whiteColumnProfile[x] /= height;
    for (let y = 0; y < height; y++) whiteRowProfile[y] /= width;
    const whiteX = estimateCellAxisPitchCandidates(whiteColumnProfile)[0];
    const whiteY = estimateCellAxisPitchCandidates(whiteRowProfile)[0];
    if (!whiteX || !whiteY) return undefined;
    pitchX = whiteX.pitch;
    pitchY = whiteY.pitch;
    confidence = Math.min(whiteX.confidence, whiteY.confidence);
    method = 'gridline-periodicity';
  }

  // Re-check that the shared pitch is actually present in this local band. A band
  // with weak local edge evidence is omitted rather than forcing the frame pitch onto it.
  if (sharedPitch) {
    const correlationAt = (profile: Float32Array, pitch: number): number => {
      const lag = Math.round(pitch);
      if (lag < 2 || lag >= profile.length - 2) return 0;
      let mean = 0;
      for (let i = 0; i < profile.length; i++) mean += profile[i];
      mean /= profile.length;
      let dot = 0, aa = 0, bb = 0;
      for (let i = 0; i < profile.length - lag; i++) {
        const a = profile[i] - mean;
        const b = profile[i + lag] - mean;
        dot += a * b; aa += a * a; bb += b * b;
      }
      return Math.max(0, dot / (Math.sqrt(aa * bb) + 1e-8));
    };
    const localX = correlationAt(edgeColumnProfile, pitchX);
    const localY = correlationAt(edgeRowProfile, pitchY);
    confidence = Math.min(sharedPitch.confidence, (localX + localY) / 2);
    if (localX < 0.12 || localY < 0.12) return undefined;
  }

  // Recover lattice phase as well as pitch. This matters for cropped/half panels: the
  // visible image edge can cut through a cell, so the grid must not restart at the ROI edge.
  const bestPhase = (profile: Float32Array, pitch: number): number => {
    let bestOffset = 0;
    let bestScore = -Infinity;
    for (let offset = 0; offset < Math.max(1, Math.round(pitch)); offset++) {
      let score = 0;
      for (let pos = offset; pos < profile.length; pos += pitch) {
        score += Math.max(
          profile[Math.max(0, Math.floor(pos) - 1)] || 0,
          profile[Math.min(profile.length - 1, Math.round(pos))] || 0,
          profile[Math.min(profile.length - 1, Math.ceil(pos) + 1)] || 0
        );
      }
      if (score > bestScore) { bestScore = score; bestOffset = offset; }
    }
    return bestOffset;
  };
  const phaseX = bestPhase(edgeColumnProfile, pitchX);
  const phaseY = bestPhase(edgeRowProfile, pitchY);
  // Treat a near-zero phase as a boundary already at the ROI start. Otherwise use the
  // preceding lattice boundary so an image crop through a cell keeps that partial cell.
  const gridX0 = x0 + (phaseX <= pitchX * 0.18 ? phaseX : phaseX - pitchX);
  const gridY0 = y0 + (phaseY <= pitchY * 0.18 ? phaseY : phaseY - pitchY);
  const inferredColumns = Math.max(1, Math.ceil((x1 - gridX0) / pitchX));
  const inferredRows = Math.max(1, Math.ceil((y1 - gridY0) / pitchY));
  if (inferredColumns > 120 || inferredRows > 60) return undefined;
  const gridX1 = gridX0 + inferredColumns * pitchX;
  const gridY1 = gridY0 + inferredRows * pitchY;
  const alignedPixelBox: PixelBox = {
    left: Math.round((gridX0 / sw) * imageWidth),
    top: Math.round((gridY0 / sh) * imageHeight),
    width: Math.max(1, Math.round(((gridX1 - gridX0) / sw) * imageWidth)),
    height: Math.max(1, Math.round(((gridY1 - gridY0) / sh) * imageHeight)),
  };

  // A lattice position is only a proposal. Validate every cell independently using
  // the PV-silicon mask before returning coordinates to any overlay or detector.
  const geometry = buildValidatedCellGeometry({
    gridX0,
    gridY0,
    pitchX,
    pitchY,
    rows: inferredRows,
    columns: inferredColumns,
    width: sw,
    height: sh,
    imageWidth,
    imageHeight,
    pvSiliconMask,
    verticalEdgeMap,
    horizontalEdgeMap,
    periodicityConfidence: confidence,
    clipBounds: { left: x0, top: y0, right: x1, bottom: y1 },
  });
  if (geometry.cells.length < 4) return undefined;
  const detectedRows = Math.max(...geometry.cells.map((cell) => cell.row));
  const detectedColumns = Math.max(...geometry.cells.map((cell) => cell.column));
  return {
    rows: detectedRows,
    columns: detectedColumns,
    pixelBox: alignedPixelBox,
    cells: geometry.cells,
    pitchX: pitchX * imageWidth / sw,
    pitchY: pitchY * imageHeight / sh,
    confidence: geometry.confidence,
    method,
  };
}

/** Render the exact directional map used for grid estimation, for on-canvas inspection. */
async function renderCellBoundaryMap(
  verticalEdgeMap: Float32Array,
  horizontalEdgeMap: Float32Array,
  sw: number,
  sh: number,
  imageWidth: number,
  imageHeight: number,
  moduleRegions: ModuleRegion[]
): Promise<string> {
  const rgba = Buffer.alloc(sw * sh * 4);
  const regions = moduleRegions.filter((region) => !region.isFallback);
  const samples: number[] = [];
  for (const region of regions) {
    const box = region.pixelBox;
    const x0 = Math.max(0, Math.floor((box.left / imageWidth) * sw));
    const y0 = Math.max(0, Math.floor((box.top / imageHeight) * sh));
    const x1 = Math.min(sw, Math.ceil(((box.left + box.width) / imageWidth) * sw));
    const y1 = Math.min(sh, Math.ceil(((box.top + box.height) / imageHeight) * sh));
    for (let y = y0; y < y1; y += 2) {
      for (let x = x0; x < x1; x += 2) {
        const i = y * sw + x;
        samples.push(Math.max(verticalEdgeMap[i], horizontalEdgeMap[i]));
      }
    }
  }
  if (samples.length === 0) return '';
  samples.sort((a, b) => a - b);
  const p95 = samples[Math.floor(samples.length * 0.95)] || 1;
  const scale = Math.max(24, p95 * 1.15);

  for (const region of regions) {
    const box = region.pixelBox;
    const x0 = Math.max(0, Math.floor((box.left / imageWidth) * sw));
    const y0 = Math.max(0, Math.floor((box.top / imageHeight) * sh));
    const x1 = Math.min(sw, Math.ceil(((box.left + box.width) / imageWidth) * sw));
    const y1 = Math.min(sh, Math.ceil(((box.top + box.height) / imageHeight) * sh));
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const i = y * sw + x;
        const vx = Math.min(1, verticalEdgeMap[i] / scale);
        const hy = Math.min(1, horizontalEdgeMap[i] / scale);
        const strength = Math.max(vx, hy);
        if (strength < 0.12) continue;
        const p = i * 4;
        // Vertical seams are orange; horizontal seams are cyan.
        rgba[p] = Math.round(255 * vx + 15 * hy);
        rgba[p + 1] = Math.round(105 * vx + 220 * hy);
        rgba[p + 2] = Math.round(18 * vx + 255 * hy);
        rgba[p + 3] = Math.round(35 + strength * 210);
      }
    }
  }
  const png = await sharp(rgba, { raw: { width: sw, height: sh, channels: 4 } }).png().toBuffer();
  return `data:image/png;base64,${png.toString('base64')}`;
}


/**
 * Separable box blur of a float plane (prefix sums, clamped edges).
 * Used to turn sparse gridline pixels into a smooth "structure density" map.
 */
function boxBlurPlane(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  const acc = new Float64Array(Math.max(w, h) + 1);
  for (let y = 0; y < h; y++) {
    acc[0] = 0;
    for (let x = 0; x < w; x++) acc[x + 1] = acc[x] + src[y * w + x];
    for (let x = 0; x < w; x++) {
      const a = Math.max(0, x - r);
      const b = Math.min(w - 1, x + r);
      tmp[y * w + x] = (acc[b + 1] - acc[a]) / (b - a + 1);
    }
  }
  for (let x = 0; x < w; x++) {
    acc[0] = 0;
    for (let y = 0; y < h; y++) acc[y + 1] = acc[y] + tmp[y * w + x];
    for (let y = 0; y < h; y++) {
      const a = Math.max(0, y - r);
      const b = Math.min(h - 1, y + r);
      out[y * w + x] = (acc[b + 1] - acc[a]) / (b - a + 1);
    }
  }
  return out;
}

/** 9x9 square erosion (isMin) or dilation (!isMin), separable; edges clamped. */
function rankFilter9(src: Float32Array, w: number, h: number, isMin: boolean): Float32Array {
  const R = 4;
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = src[y * w + x];
      for (let k = Math.max(0, x - R); k <= Math.min(w - 1, x + R); k++) {
        const s = src[y * w + k];
        if (isMin ? s < v : s > v) v = s;
      }
      tmp[y * w + x] = v;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = tmp[y * w + x];
      for (let k = Math.max(0, y - R); k <= Math.min(h - 1, y + R); k++) {
        const s = tmp[k * w + x];
        if (isMin ? s < v : s > v) v = s;
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

/**
 * Per-pixel "is there PV gridline structure around here?" mask (1 = yes), from long straight bright lines.
 *
 * Why: colour alone cannot tell a dark tracker shadow on soil/grass from dark silicon (both are dark and
 * near-neutral), so Stage 1 used to count shadow as silicon and stretch the module box across it, which then
 * let "cracks" be reported on the ground. Real modules are covered by long straight bright gridlines (cell
 * gaps, busbars, frame); a shadow has essentially none (measured density ~0 vs >= 0.03 on modules), because the
 * top-hat of a dark region is flat. Sunlit straw does have streaks, but it is already rejected by colour, so
 * this mask is used only as a VETO on silicon-coloured pixels.
 *
 * Returns null when almost nothing is structured (blurry / tiny close-up crops) so the caller keeps the
 * previous colour-only behaviour instead of rejecting every pixel.
 */
function computeGridStructureMask(gray: Float32Array, sw: number, sh: number): Uint8Array | null {
  // White top-hat isolates thin bright features (1-3 px lines) irrespective of exposure
  const opened = rankFilter9(rankFilter9(gray, sw, sh, true), sw, sh, false);
  const minRun = Math.max(14, Math.round(sw * 0.013));
  const ver = new Float32Array(sw * sh);
  const hor = new Float32Array(sw * sh);
  const isBright = (i: number): boolean => gray[i] - opened[i] > 38;
  // Keep only LONG straight runs; grass blades / sticks are short and randomly oriented
  for (let x = 0; x < sw; x++) {
    let runStart = -1;
    for (let y = 0; y <= sh; y++) {
      const on = y < sh && isBright(y * sw + x);
      if (on && runStart === -1) runStart = y;
      if (!on && runStart !== -1) {
        if (y - runStart >= minRun) for (let k = runStart; k < y; k++) ver[k * sw + x] = 1;
        runStart = -1;
      }
    }
  }
  for (let y = 0; y < sh; y++) {
    let runStart = -1;
    for (let x = 0; x <= sw; x++) {
      const on = x < sw && isBright(y * sw + x);
      if (on && runStart === -1) runStart = x;
      if (!on && runStart !== -1) {
        if (x - runStart >= minRun) for (let k = runStart; k < x; k++) hor[y * sw + k] = 1;
        runStart = -1;
      }
    }
  }
  const r = Math.max(8, Math.round(sw * 0.02));
  const dv = boxBlurPlane(ver, sw, sh, r);
  const dh = boxBlurPlane(hor, sw, sh, r);
  const mask = new Uint8Array(sw * sh);
  let structured = 0;
  for (let i = 0; i < sw * sh; i++) {
    // Either direction suffices (a module weakly lit in one direction still shows the other); shadow has neither
    if (dv[i] > 0.02 || dh[i] > 0.02) {
      mask[i] = 1;
      structured++;
    }
  }
  if (process.env.CV_DEBUG) {
    console.log(`[CV_DEBUG] gridline structure covers ${((structured / (sw * sh)) * 100).toFixed(1)}% of frame`);
  }
  return structured < sw * sh * 0.01 ? null : mask;
}

/**
 * Stage 1: Exposure-Invariant PV-surface localization + tracker-shadow veto +
 * Orthogonal Row+Column Median Grid-Removed Off-Axis Hairline Energy Map.
 */
export async function runClassicalStage1(
  rawRgb: Buffer,
  imgWidth: number,
  imgHeight: number
): Promise<DownsampledAnalysis> {
  const maxAnalysisDim = 1280;
  const scale = Math.min(1, maxAnalysisDim / Math.max(imgWidth, imgHeight));
  const sw = Math.max(64, Math.round(imgWidth * scale));
  const sh = Math.max(64, Math.round(imgHeight * scale));

  const smallRgb = await sharp(rawRgb, {
    raw: { width: imgWidth, height: imgHeight, channels: 3 },
  })
    .resize(sw, sh, { fit: 'fill', kernel: 'lanczos3' })
    .raw()
    .toBuffer();

  const gray = new Float32Array(sw * sh);
  const rawPvPixel = new Uint8Array(sw * sh);
  // Stricter cell-interior mask: shadows on soil can be neutral gray and pass the
  // broad localization mask, but real PV silicon in this dataset is blue-enriched.
  // Keep this separate so localization remains tolerant while cell validation is strict.
  const rawCellPixel = new Uint8Array(sw * sh);
  const whiteGridPixel = new Uint8Array(sw * sh);
  const soilMask = new Uint8Array(sw * sh);
  let pvPixelCount = 0;

  for (let i = 0; i < sw * sh; i++) {
    const clean = sanitizeUserMarkupRgb(
      smallRgb[i * 3],
      smallRgb[i * 3 + 1],
      smallRgb[i * 3 + 2]
    );
    const { r, g, b, wasMarkup } = clean;
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    gray[i] = lum;

    // Captures BOTH dark monocrystalline silicon (lum 15..50) AND sunlit/sky-reflected PV glass (lum 50..162)
    // while rejecting warm dry grass/soil (R > B + 5) and bright frames/toolbars.
    const isPvSilicon =
      !wasMarkup &&
      b >= r - 4 &&
      b - r < 46 &&
      lum < 162 &&
      r < 152 &&
      !(r > g + 8 && lum > 68);
    if (!wasMarkup && r > b + 8 && lum > 68) soilMask[i] = 1; // warm dry soil / grass
    if (isPvSilicon) {
      rawPvPixel[i] = 1;
      pvPixelCount++;
      if (b >= r + 3) rawCellPixel[i] = 1;
    }

    // Crisp neutral-white PV cell inter-gaps & aluminum frames
    if (!wasMarkup && lum > 190 && Math.abs(r - b) < 30 && Math.abs(r - g) < 24) {
      whiteGridPixel[i] = 1;
    }
  }

  // Build the directional cell-boundary edge map once; grid estimation and the UI use this same map.
  const cellDirectionalEdges = buildSolarCellDirectionalEdgeMaps(gray, sw, sh);

  // Veto silicon-coloured pixels that have no gridline structure around them (dark tracker shadow on soil/grass).
  // Done before pvRatio / column / row analysis so every later step (module boxes, heatmap, ground gate) sees
  // shadow as non-module. null => no usable structure in this frame, keep the colour-only result.
  const gridStructure = computeGridStructureMask(gray, sw, sh);
  if (gridStructure) {
    for (let i = 0; i < sw * sh; i++) {
      if (rawPvPixel[i] && !gridStructure[i]) {
        rawPvPixel[i] = 0;
        pvPixelCount--;
      }
      if (!gridStructure[i]) rawCellPixel[i] = 0;
    }
  }

  const pvRatio = pvPixelCount / (sw * sh);
  const pvMask = new Uint8Array(sw * sh);
  const moduleRegions: ModuleRegion[] = [];
  let activeBounds: PixelBox = {
    left: 0,
    top: 0,
    width: imgWidth,
    height: imgHeight,
  };

  // 0.01 (was 0.05): a frame with ONE small module on bare soil (e.g. 5% of the image) used to skip module search
  // entirely and fall back to "whole image is module", so soil got scanned. The gridline-structure veto above now
  // keeps shadow/soil out of rawPvPixel, so a low ratio is safe to search.
  if (pvRatio >= 0.01) {
    const colSilicon = new Float32Array(sw);
    const colWhiteGrid = new Float32Array(sw);
    for (let y = 0; y < sh; y++) {
      for (let x = 0; x < sw; x++) {
        const idx = y * sw + x;
        colSilicon[x] += rawPvPixel[idx];
        colWhiteGrid[x] += whiteGridPixel[idx];
      }
    }

    // Smooth column silicon + verify presence of periodic white PV cell gridlines in window
    // Rejects dark tracker shadows cast on soil/grass!
    const winHalf = Math.max(14, Math.round(sw * 0.02));
    const bridgeCols = Math.max(10, Math.round(sw * 0.014));

    // Longest run of "module" columns. hNorm is the image height the column silicon/grid counts are normalised by:
    // sh normally; a smaller value for frames where the module only covers a thin band of rows (see retry below).
    const scanColumns = (hNorm: number): { start: number; end: number } => {
      let bStart = -1;
      let bEnd = -1;
      let cStart = -1;
      for (let x = 0; x < sw; x++) {
        let winGrid = 0;
        const w0 = Math.max(0, x - winHalf);
        const w1 = Math.min(sw - 1, x + winHalf);
        for (let wx = w0; wx <= w1; wx++) winGrid += colWhiteGrid[wx];
        const gridDensity = winGrid / ((w1 - w0 + 1) * hNorm);
        const silDensity = colSilicon[x] / hNorm;

        if (silDensity > 0.16 && gridDensity > 0.01) {
          if (cStart === -1) cStart = x;
          if (bStart === -1 || x - cStart > bEnd - bStart) {
            bStart = cStart;
            bEnd = x;
          }
        } else if (cStart !== -1) {
          let bridge = false;
          for (let k = 1; k <= bridgeCols && x + k < sw; k++) {
            if (colSilicon[x + k] / hNorm > 0.2) {
              bridge = true;
              break;
            }
          }
          if (!bridge) cStart = -1;
        }
      }
      return { start: bStart, end: bEnd };
    };

    let { start: bestStart, end: bestEnd } = scanColumns(sh);
    if (bestStart === -1) {
      // Retry only when the normal scan found nothing (so frames that already worked are untouched): a small module
      // on bare soil covers <16% of the rows, so no column can reach the full-height density. Normalise by the
      // number of rows that actually hold silicon instead.
      let silRows = 0;
      for (let y = 0; y < sh; y++) {
        let cnt = 0;
        for (let x = 0; x < sw; x++) cnt += rawPvPixel[y * sw + x];
        if (cnt / sw > 0.12) silRows++;
      }
      if (silRows >= sh * 0.03) ({ start: bestStart, end: bestEnd } = scanColumns(silRows));
    }

    if (process.env.CV_DEBUG) {
      console.log(`[CV_DEBUG] stage1 pvRatio=${pvRatio.toFixed(3)} cols=${bestStart}..${bestEnd} of ${sw}`);
    }
    if (bestStart !== -1 && bestEnd - bestStart > sw * 0.12) {
      const padX = Math.round(sw * 0.012);
      const minCol = Math.max(0, bestStart - padX);
      const maxCol = Math.min(sw - 1, bestEnd + padX);
      const colSpan = Math.max(1, maxCol - minCol + 1);

      const rowSiliconRatio = new Float32Array(sh);
      for (let y = 0; y < sh; y++) {
        let cnt = 0;
        for (let x = minCol; x <= maxCol; x++) {
          cnt += rawPvPixel[y * sw + x];
        }
        rowSiliconRatio[y] = cnt / colSpan;
      }

      let inModule = false;
      let modStart = 0;
      // Bridge a single weak row (often a thin inter-cell busbar), but do not bridge
      // multi-row frame/air gaps between adjacent physical modules. The previous 0.6%
      // image-height look-ahead merged vertically stacked modules into one oversized ROI.
      const bridgeRows = 1;
      const rawBands: Array<{ y0: number; y1: number }> = [];
      for (let y = 0; y < sh; y++) {
        const active = rowSiliconRatio[y] > 0.28;
        if (active && !inModule) {
          inModule = true;
          modStart = y;
        } else if (!active && inModule) {
          let lookaheadActive = false;
          for (let k = 1; k <= bridgeRows && y + k < sh; k++) {
            if (rowSiliconRatio[y + k] > 0.42) {
              lookaheadActive = true;
              break;
            }
          }
          if (!lookaheadActive) {
            inModule = false;
            if (y - modStart >= Math.max(10, Math.round(sh * 0.008))) {
              rawBands.push({ y0: modStart, y1: y - 1 });
            }
          }
        }
      }
      if (inModule && sh - modStart >= Math.max(10, Math.round(sh * 0.008))) {
        rawBands.push({ y0: modStart, y1: sh - 1 });
      }

      if (rawBands.length > 0) {
        rawBands.forEach((band, idx) => {
          // No padding: bounds are trimmed to where dark silicon really starts/ends so soil margins stay outside
          const by0 = band.y0;
          const by1 = band.y1;
          const bandH = by1 - by0 + 1;
          const colHasSilicon = (x: number): boolean => {
            let cnt = 0;
            for (let y = by0; y <= by1; y++) cnt += rawPvPixel[y * sw + x];
            return cnt / bandH >= 0.45;
          };
          let bx0 = minCol;
          let bx1 = maxCol;
          while (bx0 < bx1 && !colHasSilicon(bx0)) bx0++;
          while (bx1 > bx0 && !colHasSilicon(bx1)) bx1--;
          const left = Math.round((bx0 / sw) * imgWidth);
          const right = Math.round(((bx1 + 1) / sw) * imgWidth);
          const top = Math.round((by0 / sh) * imgHeight);
          const bottom = Math.round(((by1 + 1) / sh) * imgHeight);
          moduleRegions.push({
            id: `mod-${idx + 1}`,
            label: `Visible cell field ${idx + 1}`,
            pixelBox: {
              left,
              top,
              width: Math.max(1, right - left),
              height: Math.max(1, bottom - top),
            },
            box: {
              xmin: left / imgWidth,
              ymin: top / imgHeight,
              xmax: right / imgWidth,
              ymax: bottom / imgHeight,
            },
          });

          for (let y = by0; y <= by1; y++) {
            for (let x = bx0; x <= bx1; x++) {
              pvMask[y * sw + x] = 1;
            }
          }
        });

        const minTop = moduleRegions[0].pixelBox.top;
        const maxBottom =
          moduleRegions[moduleRegions.length - 1].pixelBox.top +
          moduleRegions[moduleRegions.length - 1].pixelBox.height;
        const aLeft = Math.min(...moduleRegions.map((m) => m.pixelBox.left));
        const aRight = Math.max(...moduleRegions.map((m) => m.pixelBox.left + m.pixelBox.width));
        activeBounds = {
          left: aLeft,
          top: minTop,
          width: Math.max(1, aRight - aLeft),
          height: Math.max(1, maxBottom - minTop),
        };
      }
    }
  }

  if (moduleRegions.length === 0) {
    pvMask.fill(1);
    moduleRegions.push({
      id: 'mod-1',
      isFallback: true, // placeholder, not a detected module
      label: 'Primary Surface ROI',
      pixelBox: { left: 0, top: 0, width: imgWidth, height: imgHeight },
      box: { xmin: 0, ymin: 0, xmax: 1, ymax: 1 },
    });
  }

  // Calibrate cell pitch from the strongest region once per frame, then reuse that
  // exact cell size for every visible PV band. Local phase and per-cell occupancy are
  // still re-measured in each band, so gaps stay empty and cropped cells are retained.
  const gridCandidates = moduleRegions
    .filter((region) => !region.isFallback)
    .map((region) => ({
      region,
      grid: estimateSolarCellGrid(
        cellDirectionalEdges.vertical,
        cellDirectionalEdges.horizontal,
        whiteGridPixel, sw, sh, imgWidth, imgHeight, region.pixelBox, rawCellPixel
      ),
    }))
    .filter((item): item is { region: ModuleRegion; grid: SolarCellGrid } => Boolean(item.grid));
  const pitchReference = [...gridCandidates]
    .filter((item) => item.grid.confidence >= 0.35 && item.grid.cells.length >= 8)
    .sort((a, b) => (b.grid.confidence * Math.sqrt(b.grid.cells.length)) - (a.grid.confidence * Math.sqrt(a.grid.cells.length)))[0]?.grid;

  for (const region of moduleRegions) {
    if (region.isFallback) continue;
    const measuredGrid = pitchReference
      ? estimateSolarCellGrid(
          cellDirectionalEdges.vertical,
          cellDirectionalEdges.horizontal,
          whiteGridPixel, sw, sh, imgWidth, imgHeight, region.pixelBox, rawCellPixel,
          { pitchX: pitchReference.pitchX, pitchY: pitchReference.pitchY, confidence: pitchReference.confidence }
        )
      : gridCandidates.find((candidate) => candidate.region.id === region.id)?.grid;
    // Never show a guessed lattice: a region must have strong local edge periodicity
    // and at least four individually PV-validated cell rectangles.
    region.cellGrid = measuredGrid && measuredGrid.confidence >= 0.32 && measuredGrid.cells.length >= 4
      ? measuredGrid
      : undefined;
  }
  const cellBoundaryMapUrl = await renderCellBoundaryMap(
    cellDirectionalEdges.vertical,
    cellDirectionalEdges.horizontal,
    sw,
    sh,
    imgWidth,
    imgHeight,
    moduleRegions
  );

  // Orthogonal Row+Column Median Grid-Removed Residual + Diagonal Hairline Energy
  const offAxisEnergy = new Float32Array(sw * sh);
  const impactEnergy = new Float32Array(sw * sh);
  const afterRowMed = new Float32Array(sw * sh);
  const winBuf = new Float32Array(15);

  for (let y = 2; y < sh - 2; y++) {
    for (let x = 6; x < sw - 6; x++) {
      const idx = y * sw + x;
      if (!pvMask[idx]) continue;
      const mc = gray[idx];
      if (mc > 186) continue;
      let n = 0;
      for (let dx = -5; dx <= 5; dx++) {
        const g = gray[y * sw + (x + dx)];
        if (g < 186) winBuf[n++] = g;
      }
      if (n >= 3) {
        const sub = winBuf.subarray(0, n).sort();
        afterRowMed[idx] = Math.max(0, mc - sub[n >> 1]);
      }
    }
  }

  for (let y = 6; y < sh - 6; y++) {
    for (let x = 6; x < sw - 6; x++) {
      const idx = y * sw + x;
      if (!pvMask[idx]) continue;

      const tl = gray[(y - 1) * sw + (x - 1)];
      const tc = gray[(y - 1) * sw + x];
      const tr = gray[(y - 1) * sw + (x + 1)];
      const ml = gray[y * sw + (x - 1)];
      const mc = gray[idx];
      const mr = gray[y * sw + (x + 1)];
      const bl = gray[(y + 1) * sw + (x - 1)];
      const bc = gray[(y + 1) * sw + x];
      const br = gray[(y + 1) * sw + (x + 1)];

      const gx = Math.abs(-tl + tr - 2 * ml + 2 * mr - bl + br);
      const gy = Math.abs(-tl - 2 * tc - tr + bl + 2 * bc + br);
      const harmonic = (2 * gx * gy) / (gx + gy + 6.0);

      let n = 0;
      for (let dy = -5; dy <= 5; dy++) {
        const ni = (y + dy) * sw + x;
        if (gray[ni] < 186) winBuf[n++] = afterRowMed[ni];
      }
      let orthoResidual = 0;
      if (n >= 3) {
        const sub = winBuf.subarray(0, n).sort();
        orthoResidual = Math.max(0, afterRowMed[idx] - sub[n >> 1]);
      }

      const diag1 = mc - 0.5 * (tr + bl);
      const diag2 = mc - 0.5 * (tl + br);
      const diagRidge = mc < 182 ? Math.max(0, Math.max(diag1, diag2) - 3) * 2.8 : 0;

      const gridCornerPenalty =
        mc > 186 || tc > 192 || bc > 192 || ml > 192 || mr > 192 ? 0.12 : 1.0;

      offAxisEnergy[idx] = (harmonic + diagRidge + orthoResidual * 3.6) * gridCornerPenalty;

      const ringMean =
        (gray[(y - 2) * sw + x] +
          gray[(y + 2) * sw + x] +
          gray[y * sw + (x - 2)] +
          gray[y * sw + (x + 2)] +
          gray[(y - 2) * sw + (x - 2)] +
          gray[(y - 2) * sw + (x + 2)] +
          gray[(y + 2) * sw + (x - 2)] +
          gray[(y + 2) * sw + (x + 2)]) *
        0.125;
      if (ringMean < 145 && mc - ringMean > 16) {
        impactEnergy[idx] = mc - ringMean;
      }
    }
  }

  const heatmapRgba = Buffer.alloc(sw * sh * 4);
  let maxVal = 1;
  const smoothed = new Float32Array(sw * sh);
  for (let y = 2; y < sh - 2; y++) {
    for (let x = 2; x < sw - 2; x++) {
      if (!pvMask[y * sw + x]) continue;
      let sum = 0;
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const k = (y + dy) * sw + (x + dx);
          sum += offAxisEnergy[k] + impactEnergy[k] * 1.3;
        }
      }
      const v = sum / 25;
      smoothed[y * sw + x] = v;
      if (v > maxVal) maxVal = v;
    }
  }

  for (let i = 0; i < sw * sh; i++) {
    if (!pvMask[i]) {
      heatmapRgba[i * 4 + 3] = 0;
      continue;
    }
    const norm = Math.min(1, smoothed[i] / (maxVal * 0.65 + 1e-5));
    if (norm < 0.2) {
      heatmapRgba[i * 4 + 3] = 0;
      continue;
    }
    let r = 0;
    let g = 0;
    let b = 0;
    if (norm < 0.55) {
      const t = (norm - 0.2) / 0.35;
      r = Math.round(6 + t * (245 - 6));
      g = Math.round(182 + t * (158 - 182));
      b = Math.round(212 - t * 195);
    } else {
      const t = (norm - 0.55) / 0.45;
      r = Math.round(245 + t * 10);
      g = Math.round(158 - t * 115);
      b = Math.round(17 + t * 50);
    }
    heatmapRgba[i * 4] = r;
    heatmapRgba[i * 4 + 1] = g;
    heatmapRgba[i * 4 + 2] = b;
    heatmapRgba[i * 4 + 3] = Math.round(50 + norm * 175);
  }

  const heatmapPng = await sharp(heatmapRgba, {
    raw: { width: sw, height: sh, channels: 4 },
  })
    .png()
    .toBuffer();

  return {
    scaleWidth: sw,
    scaleHeight: sh,
    pvMask,
    offAxisEnergy,
    impactEnergy,
    moduleRegions,
    fractureHeatmapUrl: `data:image/png;base64,${heatmapPng.toString('base64')}`,
    cellBoundaryMapUrl,
    activeBounds,
    siliconMask: rawPvPixel,
    soilMask,
  };
}

/**
 * Exposure-Invariant Adaptive Silicon Normalization + Selective Off-Axis Fracture Highlighting.
 * Works identically on pitch-dark cells (median lum ~24) AND sunlit sky-reflected cells (median lum ~88),
 * anchoring clean silicon dark while amplifying off-axis curved/diagonal spider-web filaments!
 */
function enhancePvTilePixels(
  tileRaw: Buffer,
  width: number,
  height: number
): { enhanced: Buffer; gray: Float32Array } {
  const pixelCount = width * height;
  const cleanRgb = Buffer.alloc(pixelCount * 3);
  const gray = new Float32Array(pixelCount);
  const nearWhiteOrSoil = new Uint8Array(pixelCount);
  const silSamples: number[] = [];

  for (let i = 0; i < pixelCount; i++) {
    const { r, g, b, wasMarkup } = sanitizeUserMarkupRgb(
      tileRaw[i * 3],
      tileRaw[i * 3 + 1],
      tileRaw[i * 3 + 2]
    );
    cleanRgb[i * 3] = r;
    cleanRgb[i * 3 + 1] = g;
    cleanRgb[i * 3 + 2] = b;
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    gray[i] = lum;

    const isWarmSoil = r > b + 8 && lum > 68;
    if (lum > 186 || wasMarkup || isWarmSoil) {
      nearWhiteOrSoil[i] = 1;
    } else if (lum >= 8 && b >= r - 6 && i % 3 === 0) {
      silSamples.push(lum);
    }
  }

  // Compute adaptive silicon luminance percentiles so sunlit cells are NEVER bleached white
  let p10 = 18;
  let p48 = 36;
  let p95 = 115;
  if (silSamples.length >= 120) {
    silSamples.sort((a, b) => a - b);
    p10 = silSamples[Math.floor(silSamples.length * 0.1)];
    p48 = Math.max(p10 + 6, silSamples[Math.floor(silSamples.length * 0.48)]);
    p95 = Math.max(p48 + 38, silSamples[Math.floor(silSamples.length * 0.95)]);
  }
  const hiCeil = Math.min(186, Math.max(p95, p48 + 52));

  // Dilate white-frame mask by 2px so cell-edge borders don't trigger off-axis boost
  const frameGuard = new Uint8Array(pixelCount);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (nearWhiteOrSoil[y * width + x]) {
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            const ny = y + dy;
            const nx = x + dx;
            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
              frameGuard[ny * width + nx] = 1;
            }
          }
        }
      }
    }
  }

  // Compute 2D Orthogonal Row+Column Median Off-Axis Residual (isolates curved/diagonal cracks from 0° fingers & 90° busbars)
  const afterRow = new Float32Array(pixelCount);
  const orthoRes = new Float32Array(pixelCount);
  const medBuf = new Float32Array(7);

  for (let y = 2; y < height - 2; y++) {
    for (let x = 6; x < width - 6; x++) {
      const idx = y * width + x;
      if (frameGuard[idx]) continue;
      let n = 0;
      for (let dx = -6; dx <= 6; dx += 3) {
        const ni = y * width + (x + dx);
        if (!nearWhiteOrSoil[ni]) medBuf[n++] = gray[ni];
      }
      if (n >= 3) {
        medBuf.subarray(0, n).sort();
        afterRow[idx] = Math.max(0, gray[idx] - medBuf[n >> 1]);
      }
    }
  }

  for (let y = 6; y < height - 6; y++) {
    for (let x = 6; x < width - 6; x++) {
      const idx = y * width + x;
      if (frameGuard[idx] || afterRow[idx] <= 1.5) continue;
      let n = 0;
      for (let dy = -6; dy <= 6; dy += 3) {
        const ni = (y + dy) * width + x;
        if (!nearWhiteOrSoil[ni]) medBuf[n++] = afterRow[ni];
      }
      if (n >= 3) {
        medBuf.subarray(0, n).sort();
        orthoRes[idx] = Math.max(0, afterRow[idx] - medBuf[n >> 1]);
      }
    }
  }

  const enhanced = Buffer.alloc(pixelCount * 3);
  for (let i = 0; i < pixelCount; i++) {
    const r = cleanRgb[i * 3];
    const g = cleanRgb[i * 3 + 1];
    const b = cleanRgb[i * 3 + 2];
    const lum = gray[i];

    if (nearWhiteOrSoil[i] && r > b + 8) {
      // Keep warm ground grass/soil natural & slightly dimmed so attention stays on PV module
      enhanced[i * 3] = Math.round(r * 0.85);
      enhanced[i * 3 + 1] = Math.round(g * 0.85);
      enhanced[i * 3 + 2] = Math.round(b * 0.85);
      continue;
    }

    let targetLum = lum;
    if (lum <= p10) {
      targetLum = (lum / Math.max(1, p10)) * 18.0;
    } else if (lum <= p48) {
      // Anchor clean uncracked silicon (even if sunlit p48=86!) to rich dark slate [18..46]
      const t = (lum - p10) / Math.max(1, p48 - p10);
      targetLum = 18.0 + t * 28.0;
    } else if (lum < hiCeil) {
      const t = (lum - p48) / Math.max(1, hiCeil - p48);
      targetLum = 46.0 + Math.pow(t, 0.72) * 142.0;
    } else {
      targetLum = 188.0 + ((lum - hiCeil) / Math.max(1, 255.0 - hiCeil)) * 67.0;
    }

    // Selective Off-Axis Fracture Filament Boost (amplifies diagonal/curved spider-web cracks without brightening 0° fingers)
    const offAxis = orthoRes[i];
    if (!frameGuard[i] && offAxis > 2.2) {
      targetLum = Math.min(248, targetLum + (offAxis - 2.2) * 2.8);
    }

    const gain = lum > 1 ? targetLum / lum : 1;
    enhanced[i * 3] = Math.min(255, Math.max(0, Math.round(r * gain)));
    enhanced[i * 3 + 1] = Math.min(255, Math.max(0, Math.round(g * gain)));
    enhanced[i * 3 + 2] = Math.min(255, Math.max(0, Math.round(b * gain)));
  }

  return { enhanced, gray };
}

async function buildContrastBoostedTileJpeg(
  rawRgb: Buffer,
  imgWidth: number,
  imgHeight: number,
  tileBox: PixelBox,
  enableBoost: boolean
): Promise<Buffer> {
  const maxTargetDim = 1120;
  let pipeline = sharp(rawRgb, {
    raw: { width: imgWidth, height: imgHeight, channels: 3 },
  }).extract(tileBox);

  // Upscale smaller perimeter strips up to 2.2x; preserve 1:1 native detail up to 1120px on high-MP drone tiles!
  const maxSide = Math.max(tileBox.width, tileBox.height);
  const scale =
    maxSide < 960
      ? Math.min(2.2, 980 / maxSide)
      : Math.min(1, maxTargetDim / maxSide);

  const tw = Math.max(96, Math.round(tileBox.width * scale));
  const th = Math.max(96, Math.round(tileBox.height * scale));

  if (Math.abs(scale - 1) > 0.04) {
    pipeline = pipeline.resize(tw, th, { fit: 'fill', kernel: 'lanczos3' });
  }

  if (!enableBoost) {
    return pipeline.jpeg({ quality: 92 }).toBuffer();
  }

  const { data: tileRaw, info: tileInfo } = await pipeline
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { enhanced } = enhancePvTilePixels(tileRaw, tileInfo.width, tileInfo.height);

  return sharp(enhanced, {
    raw: { width: tileInfo.width, height: tileInfo.height, channels: 3 },
  })
    .sharpen({ sigma: 1.05, m1: 0.8, m2: 2.4 })
    .jpeg({ quality: 92 })
    .toBuffer();
}

async function extractForensicPatchPair(
  rawRgb: Buffer,
  imgWidth: number,
  imgHeight: number,
  pixelBox: PixelBox
): Promise<{ cropDataUrl: string; edgeMapDataUrl: string }> {
  const pad = Math.max(45, Math.round(Math.max(pixelBox.width, pixelBox.height) * 0.35));
  const cx = pixelBox.left + pixelBox.width / 2;
  const cy = pixelBox.top + pixelBox.height / 2;
  const side = Math.min(
    Math.min(imgWidth, imgHeight),
    Math.max(150, Math.max(pixelBox.width, pixelBox.height) + pad * 2)
  );

  const left = Math.max(0, Math.min(imgWidth - side, Math.round(cx - side / 2)));
  const top = Math.max(0, Math.min(imgHeight - side, Math.round(cy - side / 2)));
  const outSize = 260;

  const patchRgb = await sharp(rawRgb, {
    raw: { width: imgWidth, height: imgHeight, channels: 3 },
  })
    .extract({ left, top, width: side, height: side })
    .resize(outSize, outSize, { fit: 'fill', kernel: 'lanczos3' })
    .raw()
    .toBuffer();

  const { enhanced: enhancedCrop, gray } = enhancePvTilePixels(patchRgb, outSize, outSize);

  const cropJpeg = await sharp(enhancedCrop, {
    raw: { width: outSize, height: outSize, channels: 3 },
  })
    .sharpen({ sigma: 0.9 })
    .jpeg({ quality: 92 })
    .toBuffer();

  const edgeRgb = Buffer.alloc(outSize * outSize * 3);
  const response = new Float32Array(outSize * outSize);
  let maxResp = 10;

  for (let y = 1; y < outSize - 1; y++) {
    for (let x = 1; x < outSize - 1; x++) {
      const idx = y * outSize + x;
      const tl = gray[(y - 1) * outSize + (x - 1)];
      const tc = gray[(y - 1) * outSize + x];
      const tr = gray[(y - 1) * outSize + (x + 1)];
      const ml = gray[y * outSize + (x - 1)];
      const mc = gray[idx];
      const mr = gray[y * outSize + (x + 1)];
      const bl = gray[(y + 1) * outSize + (x - 1)];
      const bc = gray[(y + 1) * outSize + x];
      const br = gray[(y + 1) * outSize + (x + 1)];

      const gx = Math.abs(-tl + tr - 2 * ml + 2 * mr - bl + br);
      const gy = Math.abs(-tl - 2 * tc - tr + bl + 2 * bc + br);
      const offAxis = (2 * gx * gy) / (gx + gy + 5.0);

      const d1 = mc - 0.5 * (tr + bl);
      const d2 = mc - 0.5 * (tl + br);
      const ridge = mc < 182 ? Math.max(0, Math.max(d1, d2) - 3) * 2.8 : 0;

      const penalty = mc > 186 || tc > 192 || bc > 192 ? 0.15 : 1.0;
      const val = (offAxis + ridge) * penalty;
      response[idx] = val;
      if (val > maxResp) maxResp = val;
    }
  }

  for (let i = 0; i < outSize * outSize; i++) {
    const baseLum = Math.round(gray[i] * 0.14);
    const norm = Math.min(1, response[i] / (maxResp * 0.65));
    if (norm < 0.16) {
      edgeRgb[i * 3] = Math.max(7, baseLum);
      edgeRgb[i * 3 + 1] = Math.max(10, Math.round(baseLum * 1.15));
      edgeRgb[i * 3 + 2] = Math.max(18, Math.round(baseLum * 1.4));
    } else if (norm < 0.55) {
      const t = (norm - 0.16) / 0.39;
      edgeRgb[i * 3] = Math.round(8 + t * 20);
      edgeRgb[i * 3 + 1] = Math.round(45 + t * 185);
      edgeRgb[i * 3 + 2] = Math.round(90 + t * 155);
    } else {
      const t = (norm - 0.55) / 0.45;
      edgeRgb[i * 3] = Math.round(28 + t * 227);
      edgeRgb[i * 3 + 1] = Math.round(230 - t * 65);
      edgeRgb[i * 3 + 2] = Math.round(245 - t * 210);
    }
  }

  const edgePng = await sharp(edgeRgb, {
    raw: { width: outSize, height: outSize, channels: 3 },
  })
    .png()
    .toBuffer();

  return {
    cropDataUrl: `data:image/jpeg;base64,${cropJpeg.toString('base64')}`,
    edgeMapDataUrl: `data:image/png;base64,${edgePng.toString('base64')}`,
  };
}

/**
 * Stage 3B: Deterministic Native-Resolution Classical Orthogonal-Residual Spider-Web Detector.
 * Scores the automatically detected, equal-sized solar-cell grid across the complete PV array,
 * so interior cells and perimeter cells use the same geometry.
 */
async function detectClassicalModuleSpiderWebs(
  rawRgb: Buffer,
  imgWidth: number,
  imgHeight: number,
  moduleRegions: ModuleRegion[],
  gsdMmPerPx: number
): Promise<DetectedAnomaly[]> {
  const found: DetectedAnomaly[] = [];

  for (let mIdx = 0; mIdx < moduleRegions.length; mIdx++) {
    const mod = moduleRegions[mIdx];
    const mb = mod.pixelBox;
    if (mb.width < 48 || mb.height < 24 || mod.isFallback) continue; // allow partial panels and clipped cell arrays
    const cellGrid = mod.cellGrid;
    // Do not fabricate a complete 26×6 layout when a crop is partial or edge evidence is weak.
    // The crack scorer only runs when the actual cell pitch was measured from the edge map.
    if (!cellGrid || cellGrid.confidence < 0.2) continue;

    // Extract array at up to 1800px width so 1-2px fractures are crisp and computation takes <45ms
    const maxModW = 1800;
    const modScale = Math.min(1, maxModW / mb.width);
    const mw = Math.max(96, Math.round(mb.width * modScale));
    const mh = Math.max(64, Math.round(mb.height * modScale));

    const { data: modBuf } = await sharp(rawRgb, {
      raw: { width: imgWidth, height: imgHeight, channels: 3 },
    })
      .extract(mb)
      .resize(mw, mh, { fit: 'fill', kernel: 'lanczos3' })
      .raw()
      .toBuffer({ resolveWithObject: true });

    const gray = new Float32Array(mw * mh);
    const rawIgnore = new Uint8Array(mw * mh);
    const silVals: number[] = [];

    for (let i = 0; i < mw * mh; i++) {
      const { r, g, b, wasMarkup } = sanitizeUserMarkupRgb(
        modBuf[i * 3],
        modBuf[i * 3 + 1],
        modBuf[i * 3 + 2]
      );
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      gray[i] = lum;
      const isWarmGround = r > b + 7 && lum > 62;
      if (wasMarkup || lum > 184 || lum < 10 || isWarmGround || b < r - 6) {
        rawIgnore[i] = 1;
      } else if (i % 5 === 0) {
        silVals.push(lum);
      }
    }

    if (silVals.length < 300) continue;
    silVals.sort((a, b) => a - b);
    const iqr =
      silVals[Math.floor(silVals.length * 0.75)] - silVals[Math.floor(silVals.length * 0.25)];
    const orthoThreshold = Math.max(7.2, Math.min(14.5, 6.2 + iqr * 0.14));

    // Dilate ignore mask by 4px so white inter-cell gaps and aluminum frames NEVER trigger false ridges
    const ignoreMask = new Uint8Array(mw * mh);
    const marginX = Math.max(5, Math.round(mw * 0.015));
    const marginY = Math.max(5, Math.round(mh * 0.025));
    for (let y = 0; y < mh; y++) {
      for (let x = 0; x < mw; x++) {
        if (x < marginX || x >= mw - marginX || y < marginY || y >= mh - marginY) {
          ignoreMask[y * mw + x] = 1;
          continue;
        }
        if (rawIgnore[y * mw + x]) {
          for (let dy = -4; dy <= 4; dy++) {
            for (let dx = -4; dx <= 4; dx++) {
              const ny = y + dy;
              const nx = x + dx;
              if (ny >= 0 && ny < mh && nx >= 0 && nx < mw) {
                ignoreMask[ny * mw + nx] = 1;
              }
            }
          }
        }
      }
    }

    const afterRow = new Float32Array(mw * mh);
    const medBuf = new Float32Array(9);
    for (let y = 4; y < mh - 4; y++) {
      for (let x = 8; x < mw - 8; x++) {
        const idx = y * mw + x;
        if (ignoreMask[idx]) continue;
        let n = 0;
        for (let dx = -8; dx <= 8; dx += 2) {
          const ni = y * mw + (x + dx);
          if (!rawIgnore[ni]) medBuf[n++] = gray[ni];
        }
        if (n >= 4) {
          medBuf.subarray(0, n).sort();
          afterRow[idx] = Math.max(0, gray[idx] - medBuf[n >> 1]);
        }
      }
    }

    const filamentMask = new Uint8Array(mw * mh);
    for (let y = 8; y < mh - 8; y++) {
      for (let x = 8; x < mw - 8; x++) {
        const idx = y * mw + x;
        if (ignoreMask[idx] || afterRow[idx] < orthoThreshold) continue;
        let n = 0;
        for (let dy = -8; dy <= 8; dy += 2) {
          const ni = (y + dy) * mw + x;
          if (!rawIgnore[ni]) medBuf[n++] = afterRow[ni];
        }
        if (n < 4) continue;
        medBuf.subarray(0, n).sort();
        const orthoRes = afterRow[idx] - medBuf[n >> 1];
        if (orthoRes < orthoThreshold) continue;

        // Verify genuine off-axis gradient (15°..75° curved/diagonal line, NOT 0° finger or 90° busbar)
        const tl = gray[(y - 1) * mw + (x - 1)];
        const tc = gray[(y - 1) * mw + x];
        const tr = gray[(y - 1) * mw + (x + 1)];
        const ml = gray[y * mw + (x - 1)];
        const mr = gray[y * mw + (x + 1)];
        const bl = gray[(y + 1) * mw + (x - 1)];
        const bc = gray[(y + 1) * mw + x];
        const br = gray[(y + 1) * mw + (x + 1)];
        const gx = Math.abs(-tl + tr - 2 * ml + 2 * mr - bl + br);
        const gy = Math.abs(-tl - 2 * tc - tr + bl + 2 * bc + br);
        const harmonic = (2 * gx * gy) / (gx + gy + 5.0);
        if (harmonic > 15.0) {
          filamentMask[idx] = 1;
        }
      }
    }

    // Score every cell using the measured array grid (or explicitly marked low-confidence fallback).
    const nRows = cellGrid.rows;
    const nCols = cellGrid.columns;
    const cellDensity = new Float32Array(nRows * nCols);
    const cellValid = new Uint8Array(nRows * nCols);
    const cellBoxes = new Map<number, PixelBox>();
    const validDensities: number[] = [];

    // Score only actual cell rectangles verified against PV-silicon occupancy. Empty
    // lattice slots at inter-array gaps, frames or ground are never treated as cells.
    for (const cell of cellGrid.cells) {
      const cellIndex = (cell.row - 1) * nCols + (cell.column - 1);
      if (cellIndex < 0 || cellIndex >= cellDensity.length) continue;
      const cb = cell.pixelBox;
      const x0 = Math.max(0, Math.min(mw, Math.floor(((cb.left - mb.left) / mb.width) * mw)));
      const x1 = Math.max(x0, Math.min(mw, Math.ceil(((cb.left + cb.width - mb.left) / mb.width) * mw)));
      const y0 = Math.max(0, Math.min(mh, Math.floor(((cb.top - mb.top) / mb.height) * mh)));
      const y1 = Math.max(y0, Math.min(mh, Math.ceil(((cb.top + cb.height - mb.top) / mb.height) * mh)));
      const insetX = Math.max(1, Math.round((x1 - x0) * 0.1));
      const insetY = Math.max(1, Math.round((y1 - y0) * 0.1));
      const ix0 = Math.min(x1, x0 + insetX), ix1 = Math.max(ix0, x1 - insetX);
      const iy0 = Math.min(y1, y0 + insetY), iy1 = Math.max(iy0, y1 - insetY);
      let vPx = 0;
      let fPx = 0;
      for (let y = iy0; y < iy1; y++) {
        for (let x = ix0; x < ix1; x++) {
          const idx = y * mw + x;
          if (!ignoreMask[idx]) {
            vPx++;
            fPx += filamentMask[idx];
          }
        }
      }
      if (vPx >= 30) {
        const density = fPx / vPx;
        cellDensity[cellIndex] = density;
        cellValid[cellIndex] = 1;
        cellBoxes.set(cellIndex, cb);
        validDensities.push(density);
      }
    }

    if (validDensities.length < Math.min(18, Math.max(4, Math.floor(cellGrid.cells.length * 0.2)))) continue;
    validDensities.sort((a, b) => a - b);
    const medDensity = validDensities[Math.floor(validDensities.length * 0.5)];
    const anomThresh = Math.max(medDensity * 3.1, medDensity + 0.052, 0.082);

    if (process.env.CV_DEBUG) {
      const grid = Array.from({ length: nRows }, (_, r) =>
        Array.from({ length: nCols }, (_, c) => (cellValid[r * nCols + c] ? (cellDensity[r * nCols + c] * 100).toFixed(0).padStart(2) : ' x')).join(' ')
      ).join('\n');
      console.log(`[CV_DEBUG] module ${mIdx + 1} ${nRows}x${nCols} med=${medDensity.toFixed(3)} thresh=${anomThresh.toFixed(3)}\n${grid}`);
    }

    // Connected-component clustering across anomalous cells
    const visited = new Uint8Array(nRows * nCols);
    for (let r = 0; r < nRows; r++) {
      for (let c = 0; c < nCols; c++) {
        const startIdx = r * nCols + c;
        if (visited[startIdx] || !cellValid[startIdx] || cellDensity[startIdx] < anomThresh) {
          continue;
        }
        const queue: Array<[number, number]> = [[r, c]];
        visited[startIdx] = 1;
        const cluster: Array<[number, number]> = [];
        let peakDensity = 0;

        while (queue.length > 0) {
          const [cr, cc] = queue.shift()!;
          cluster.push([cr, cc]);
          const d = cellDensity[cr * nCols + cc];
          if (d > peakDensity) peakDensity = d;

          for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
              if (dr === 0 && dc === 0) continue;
              const nr = cr + dr;
              const nc = cc + dc;
              if (nr < 0 || nr >= nRows || nc < 0 || nc >= nCols) continue;
              const nIdx = nr * nCols + nc;
              if (
                !visited[nIdx] &&
                cellValid[nIdx] &&
                cellDensity[nIdx] >= anomThresh * 0.88
              ) {
                visited[nIdx] = 1;
                queue.push([nr, nc]);
              }
            }
          }
        }

        const minC = Math.min(...cluster.map((p) => p[1]));
        const maxC = Math.max(...cluster.map((p) => p[1]));
        const minR = Math.min(...cluster.map((p) => p[0]));
        const maxR = Math.max(...cluster.map((p) => p[0]));
        // Multi-cell clusters are retained at the normal threshold. A very strong single-cell
        // signal is also eligible, so one damaged cell is not hidden by a perimeter-only rule.
        const qualifies =
          cluster.length >= 3 ||
          (cluster.length >= 2 && peakDensity > Math.max(medDensity * 3.8, 0.115)) ||
          (cluster.length === 1 && peakDensity > Math.max(medDensity * 5.0, 0.16));
        if (qualifies) {
          const clusterBoxes = cluster
            .map(([cr, cc]) => cellBoxes.get(cr * nCols + cc))
            .filter((box): box is PixelBox => Boolean(box));
          if (!clusterBoxes.length) continue;
          const pLeft = Math.min(...clusterBoxes.map((box) => box.left));
          const pTop = Math.min(...clusterBoxes.map((box) => box.top));
          const pRight = Math.max(...clusterBoxes.map((box) => box.left + box.width));
          const pBottom = Math.max(...clusterBoxes.map((box) => box.top + box.height));
          const pBox: PixelBox = {
            left: Math.max(0, pLeft),
            top: Math.max(0, pTop),
            width: Math.max(1, pRight - pLeft),
            height: Math.max(1, pBottom - pTop),
          };
          const ptX = Math.round(pBox.left + pBox.width / 2);
          const ptY = Math.round(pBox.top + pBox.height / 2);
          const edgeMeta = normalizeEdgeSide('interior', false, pBox, moduleRegions);
          const loc = resolveModuleAndCell(ptX, ptY, moduleRegions, imgWidth, imgHeight);
          const conf = Number(
            Math.min(0.95, 0.74 + Math.min(0.21, (peakDensity - anomThresh) * 2.2)).toFixed(2)
          );

          found.push({
            id: `CV-M${mIdx + 1}-${found.length + 1}`,
            category: 'spider_web_crack',
            isEdgeCrack: edgeMeta.isEdgeCrack,
            edgeSide: edgeMeta.edgeSide,
            severity: cluster.length >= 3 ? 'critical' : 'moderate',
            confidence: conf,
            title: 'Solar-Cell Spider-Web Glass Fracture Network',
            description: `Multi-branch off-axis glass-fracture pattern detected across ${cluster.length} solar cell${cluster.length > 1 ? 's' : ''} (${edgeMeta.edgeSide}) using ${cellGrid.cells.length} individually validated solar cells.`,
            moduleIndex: loc.moduleIndex,
            cellLocation: loc.cellLocation,
            radialBranches: Math.min(12, Math.max(3, cluster.length + 2)),
            estimatedDiameterMm: Math.round(Math.max(pBox.width, pBox.height) * gsdMmPerPx),
            pointPx: { x: ptX, y: ptY },
            box: {
              xmin: pBox.left / imgWidth,
              ymin: pBox.top / imgHeight,
              xmax: (pBox.left + pBox.width) / imgWidth,
              ymax: (pBox.top + pBox.height) / imgHeight,
            },
            pixelBox: pBox,
            tileId: `M${mIdx + 1}-CLASSICAL`,
            offAxisEdgeScore: Number((peakDensity * 100).toFixed(2)),
          });
        }
      }
    }
  }

  return found;
}

// Ground gate thresholds (fractions of the detection box, measured on the downsampled stage-1 masks)
const GROUND_MAX_SOIL_FRAC = 0.3;
const GROUND_MIN_PV_FRAC = 0.5;
const GROUND_MIN_PV_FRAC_FALLBACK = 0.25;
// Learned detector: at most this many windows per image after merging neighbours (best probability first)
const MAX_LEARNED_PER_IMAGE = 8;

/**
 * Vetoes detections that sit on ground: centre must be inside a detected module, the box is clipped to the
 * module interior, then rejected if too much of it is soil or too little is PV silicon. Returns the
 * (possibly clipped) box, or null to reject.
 */
export function applyGroundGate(
  pBox: PixelBox,
  stage1: DownsampledAnalysis,
  imgWidth: number,
  imgHeight: number
): PixelBox | null {
  const cx = pBox.left + pBox.width / 2;
  const cy = pBox.top + pBox.height / 2;
  const mod = stage1.moduleRegions.find((m) => {
    const b = m.pixelBox;
    return cx >= b.left && cx <= b.left + b.width && cy >= b.top && cy <= b.top + b.height;
  });
  if (!mod) return null;

  let box = pBox;
  if (!mod.isFallback) {
    const mb = mod.pixelBox;
    const l = Math.max(box.left, mb.left);
    const t = Math.max(box.top, mb.top);
    const r = Math.min(box.left + box.width, mb.left + mb.width);
    const b = Math.min(box.top + box.height, mb.top + mb.height);
    if (r - l < 8 || b - t < 8) return null;
    box = { left: l, top: t, width: r - l, height: b - t };
  }

  const sx0 = Math.floor((box.left / imgWidth) * stage1.scaleWidth);
  const sx1 = Math.min(stage1.scaleWidth - 1, Math.ceil(((box.left + box.width) / imgWidth) * stage1.scaleWidth));
  const sy0 = Math.floor((box.top / imgHeight) * stage1.scaleHeight);
  const sy1 = Math.min(stage1.scaleHeight - 1, Math.ceil(((box.top + box.height) / imgHeight) * stage1.scaleHeight));
  let total = 0;
  let soil = 0;
  let pv = 0;
  for (let y = sy0; y <= sy1; y++) {
    for (let x = sx0; x <= sx1; x++) {
      const i = y * stage1.scaleWidth + x;
      total++;
      soil += stage1.soilMask[i];
      pv += stage1.siliconMask[i];
    }
  }
  if (total === 0) return null;
  if (soil / total > GROUND_MAX_SOIL_FRAC) return null;
  // Real module: >= 50% silicon. Whole-image fallback (no module localised): only accept close-up crops that are
  // clearly mostly PV (>= 25%); otherwise a ground-only frame would pass the gate and soil would be reported.
  if (pv / total < (mod.isFallback ? GROUND_MIN_PV_FRAC_FALLBACK : GROUND_MIN_PV_FRAC)) return null;
  return box;
}

/**
 * Learned detector: scores canonical-scale edge-ring windows of every detected module with the trained hairline-crack
 * model (server/models/crackModel.json). Windows already exclude ground (inner-frame box + warm/soil gate, see
 * crackModel.ts) and are ground-gated again by the caller. Returns [] when the model file is absent so the classical
 * detector keeps working unchanged.
 */
async function detectLearnedEdgeCracks(
  rawRgb: Buffer,
  imgWidth: number,
  imgHeight: number,
  moduleRegions: ModuleRegion[],
  gsdMmPerPx: number
): Promise<DetectedAnomaly[]> {
  const model = loadCrackModel();
  if (!model) return [];
  const boxes = moduleRegions
    .filter((m) => !m.isFallback && m.pixelBox.width >= 180 && m.pixelBox.height >= 110)
    .map((m) => m.pixelBox);
  if (!boxes.length) return [];
  let hits;
  try {
    ({ hits } = await scanForEdgeWebs(rawRgb, imgWidth, imgHeight, boxes, model));
  } catch (err) {
    // A scan failure must never lose the classical result for this image
    console.warn('Learned crack scan failed:', err instanceof Error ? err.message : err);
    return [];
  }
  const kept = nmsWindows(hits).slice(0, MAX_LEARNED_PER_IMAGE);
  const found: DetectedAnomaly[] = [];
  for (const h of kept) {
    const pBox: PixelBox = {
      left: Math.max(0, h.x),
      top: Math.max(0, h.y),
      width: Math.min(h.size, imgWidth - Math.max(0, h.x)),
      height: Math.min(h.size, imgHeight - Math.max(0, h.y)),
    };
    const ptX = Math.round(pBox.left + pBox.width / 2);
    const ptY = Math.round(pBox.top + pBox.height / 2);
    const edgeMeta = normalizeEdgeSide('interior', false, pBox, moduleRegions);
    const loc = resolveModuleAndCell(ptX, ptY, moduleRegions, imgWidth, imgHeight);
    if (!loc.cellLocation.startsWith('Cell R')) continue;
    found.push({
      id: `ML-${found.length + 1}`,
      category: 'spider_web_crack',
      isEdgeCrack: edgeMeta.isEdgeCrack,
      edgeSide: edgeMeta.edgeSide,
      severity: h.prob >= 0.95 ? 'critical' : 'moderate',
      confidence: Number(Math.min(0.95, 0.5 + 0.45 * h.prob).toFixed(2)),
      title: 'Edge Hairline Fracture (learned)',
      description: `Trained window classifier flagged hairline fractures at the module edge (p=${h.prob.toFixed(2)}, model ${model.version}).`,
      moduleIndex: loc.moduleIndex,
      cellLocation: loc.cellLocation,
      radialBranches: 4,
      estimatedDiameterMm: Math.round(Math.max(pBox.width, pBox.height) * gsdMmPerPx),
      pointPx: { x: ptX, y: ptY },
      box: {
        xmin: pBox.left / imgWidth,
        ymin: pBox.top / imgHeight,
        xmax: (pBox.left + pBox.width) / imgWidth,
        ymax: (pBox.top + pBox.height) / imgHeight,
      },
      pixelBox: pBox,
      tileId: 'ML-WINDOW',
      offAxisEdgeScore: Number((h.prob * 100).toFixed(2)),
    });
  }
  return found;
}

function computeIoU(a: BoundingBox, b: BoundingBox): number {
  const xA = Math.max(a.xmin, b.xmin);
  const yA = Math.max(a.ymin, b.ymin);
  const xB = Math.min(a.xmax, b.xmax);
  const yB = Math.min(a.ymax, b.ymax);
  const interW = Math.max(0, xB - xA);
  const interH = Math.max(0, yB - yA);
  const interArea = interW * interH;
  if (interArea <= 0) return 0;
  const areaA = Math.max(1e-8, (a.xmax - a.xmin) * (a.ymax - a.ymin));
  const areaB = Math.max(1e-8, (b.xmax - b.xmin) * (b.ymax - b.ymin));
  const minArea = Math.min(areaA, areaB);
  const unionArea = areaA + areaB - interArea;
  return Math.max(interArea / unionArea, (interArea / minArea) * 0.78);
}

function normalizeCategory(raw: string): DefectCategory {
  const s = (raw || '').toLowerCase().trim();
  if (
    s.includes('spider') ||
    s.includes('radial') ||
    s.includes('shatter') ||
    s.includes('arc') ||
    s.includes('web') ||
    s.includes('edge_crack')
  ) {
    return 'spider_web_crack';
  }
  if (s.includes('impact') || s.includes('pit') || s.includes('chip')) {
    return 'impact_point';
  }
  if (s.includes('linear') || s.includes('fracture') || s.includes('crack') || s.includes('scratch')) {
    return 'linear_fracture';
  }
  return 'surface_residue';
}

function normalizeEdgeSide(
  rawSide: string,
  rawIsEdge: boolean,
  pBox: PixelBox,
  moduleRegions: ModuleRegion[]
): { isEdgeCrack: boolean; edgeSide: EdgeSide } {
  for (const mod of moduleRegions) {
    const mb = mod.pixelBox;
    const cx = pBox.left + pBox.width / 2;
    const cy = pBox.top + pBox.height / 2;
    if (
      cx >= mb.left &&
      cx <= mb.left + mb.width &&
      cy >= mb.top &&
      cy <= mb.top + mb.height
    ) {
      const nearLeft = pBox.left - mb.left < mb.width * 0.16;
      const nearRight = mb.left + mb.width - (pBox.left + pBox.width) < mb.width * 0.16;
      const nearTop = pBox.top - mb.top < mb.height * 0.22;
      const nearBottom = mb.top + mb.height - (pBox.top + pBox.height) < mb.height * 0.22;
      const spansMostHeight = pBox.height > mb.height * 0.45;
      const spansMostWidth = pBox.width > mb.width * 0.45;

      if (nearLeft && spansMostHeight) return { isEdgeCrack: true, edgeSide: 'left-edge' };
      if (nearRight && spansMostHeight) return { isEdgeCrack: true, edgeSide: 'right-edge' };
      if (nearTop && spansMostWidth) return { isEdgeCrack: true, edgeSide: 'top-edge' };
      if (nearBottom && spansMostWidth) return { isEdgeCrack: true, edgeSide: 'bottom-edge' };
      if (nearBottom && nearLeft) return { isEdgeCrack: true, edgeSide: 'bottom-left' };
      if (nearTop && nearLeft) return { isEdgeCrack: true, edgeSide: 'top-left' };
      if (nearBottom && nearRight) return { isEdgeCrack: true, edgeSide: 'bottom-right' };
      if (nearTop && nearRight) return { isEdgeCrack: true, edgeSide: 'top-right' };
      if (nearLeft) return { isEdgeCrack: true, edgeSide: 'left-edge' };
      if (nearRight) return { isEdgeCrack: true, edgeSide: 'right-edge' };
      if (nearTop) return { isEdgeCrack: true, edgeSide: 'top-edge' };
      if (nearBottom) return { isEdgeCrack: true, edgeSide: 'bottom-edge' };
    }
  }

  const s = (rawSide || '').toLowerCase().trim() as EdgeSide;
  const validSides: EdgeSide[] = [
    'bottom-left',
    'top-left',
    'bottom-right',
    'top-right',
    'left-edge',
    'right-edge',
    'top-edge',
    'bottom-edge',
    'interior',
  ];
  const matched = validSides.includes(s) ? s : 'interior';
  return {
    isEdgeCrack: rawIsEdge || matched !== 'interior',
    edgeSide: matched,
  };
}

function normalizeSeverity(raw: string, category: DefectCategory): SeverityLevel {
  const s = (raw || '').toLowerCase().trim();
  if (s === 'critical') return 'critical';
  if (s === 'moderate') return 'moderate';
  if (s === 'minor') return 'minor';
  if (s === 'informational') return 'informational';
  if (category === 'spider_web_crack') return 'critical';
  if (category === 'impact_point' || category === 'linear_fracture') return 'moderate';
  return 'informational';
}

function resolveModuleAndCell(
  centerX: number,
  centerY: number,
  moduleRegions: ModuleRegion[],
  imgWidth: number,
  imgHeight: number
): { moduleIndex: string; cellLocation: string } {
  // Cell rectangles are authoritative. Do not gate the lookup on a physical-module
  // bounding box: a crack near a cropped/partial panel edge still belongs to its cell.
  for (const mod of moduleRegions) {
    const grid = mod.cellGrid;
    if (!grid || grid.confidence < 0.2) continue;
    const cell = grid.cells.find((candidate) => {
      const b = candidate.pixelBox;
      return centerX >= b.left && centerX <= b.left + b.width && centerY >= b.top && centerY <= b.top + b.height;
    });
    if (cell) {
      return {
        moduleIndex: 'Visible solar-cell field',
        cellLocation: `Cell R${cell.row} · C${cell.column}`,
      };
    }
  }
  return {
    moduleIndex: 'Surface Zone',
    cellLocation: `X ${((centerX / imgWidth) * 100).toFixed(1)}% · Y ${((centerY / imgHeight) * 100).toFixed(1)}%`,
  };
}

export function buildImageGeoJsonFeature(
  fileName: string,
  imgWidth: number,
  imgHeight: number,
  anomalies: DetectedAnomaly[],
  geoRef: GeoReferenceInfo,
  analyzedAt: string
): GeoJsonPointFeature {
  const spiderCracks = anomalies.filter((a) => a.category === 'spider_web_crack');
  const edgeSpiderCracks = spiderCracks.filter((a) => a.isEdgeCrack);
  const impactPoints = anomalies.filter((a) => a.category === 'impact_point');
  const linearFractures = anomalies.filter((a) => a.category === 'linear_fracture');
  const surfaceResidues = anomalies.filter((a) => a.category === 'surface_residue');

  const severityOrder: SeverityLevel[] = ['critical', 'moderate', 'minor', 'informational'];
  let maxSeverity: SeverityLevel | 'none' = 'none';
  for (const lvl of severityOrder) {
    if (anomalies.some((a) => a.severity === lvl)) {
      maxSeverity = lvl;
      break;
    }
  }

  const maxConfidence =
    anomalies.length > 0
      ? Number(Math.max(...anomalies.map((a) => a.confidence)).toFixed(2))
      : 0;

  const affectedModulesList = Array.from(new Set(anomalies.map((a) => a.moduleIndex)));
  const affectedCellsList = Array.from(
    new Set(anomalies.map((a) => `${a.moduleIndex} ${a.cellLocation}`))
  );
  const edgeSidesList = Array.from(
    new Set(anomalies.filter((a) => a.isEdgeCrack).map((a) => a.edgeSide))
  );

  const crackDetails: GeoJsonCrackAttributeItem[] = anomalies.map((a) => ({
    defect_id: a.id,
    category: a.category,
    is_edge_crack: a.isEdgeCrack,
    edge_side: a.edgeSide,
    severity: a.severity,
    confidence: a.confidence,
    title: a.title,
    description: a.description,
    module_index: a.moduleIndex,
    cell_location: a.cellLocation,
    point_px: [a.pointPx.x, a.pointPx.y],
    bbox_px: [a.pixelBox.left, a.pixelBox.top, a.pixelBox.width, a.pixelBox.height],
    radial_branches: a.radialBranches ?? 0,
    estimated_span_mm: a.estimatedDiameterMm ?? 0,
  }));

  const crackSummary =
    anomalies.length === 0
      ? 'No surface cracks detected'
      : anomalies
          .map(
            (a) =>
              `${a.id}: ${a.category} (${a.edgeSide}, ${a.moduleIndex} ${a.cellLocation}, px=[${a.pointPx.x},${a.pointPx.y}], conf=${Math.round(a.confidence * 100)}%)`
          )
          .join(' | ');

  const isEpsg3857 = geoRef.epsgCode === 'EPSG:3857';

  return {
    type: 'Feature',
    id: fileName,
    geometry: {
      type: 'Point',
      coordinates: geoRef.imageGeoPoint,
    },
    properties: {
      image_name: fileName,
      image_dimensions_px: [imgWidth, imgHeight],
      crs: geoRef.crsName,
      epsg_code: isEpsg3857 ? 3857 : null,
      epsg_3857_xy:
        geoRef.epsg3857X !== null && geoRef.epsg3857Y !== null
          ? [geoRef.epsg3857X, geoRef.epsg3857Y]
          : null,
      wgs84_lon_lat:
        geoRef.centerLon !== null && geoRef.centerLat !== null
          ? [Number(geoRef.centerLon.toFixed(8)), Number(geoRef.centerLat.toFixed(8))]
          : null,
      centroid_method: geoRef.centroidMethod,
      camera_wgs84_lon_lat:
        geoRef.cameraLat !== null && geoRef.cameraLon !== null
          ? [Number(geoRef.cameraLon.toFixed(8)), Number(geoRef.cameraLat.toFixed(8))]
          : null,
      crs_mode: geoRef.crsMode,
      status: anomalies.length > 0 ? 'DEFECT_DETECTED' : 'CLEAN',
      has_spider_web_crack: spiderCracks.length > 0,
      has_edge_spider_web_crack: edgeSpiderCracks.length > 0,
      spider_web_crack_count: spiderCracks.length,
      edge_spider_web_crack_count: edgeSpiderCracks.length,
      impact_point_count: impactPoints.length,
      linear_fracture_count: linearFractures.length,
      surface_residue_count: surfaceResidues.length,
      total_anomaly_count: anomalies.length,
      max_severity: maxSeverity,
      max_confidence: maxConfidence,
      affected_modules: affectedModulesList.length > 0 ? affectedModulesList.join(', ') : 'None',
      affected_cells: affectedCellsList.length > 0 ? affectedCellsList.join(', ') : 'None',
      edge_sides_detected: edgeSidesList.length > 0 ? edgeSidesList.join(', ') : 'None',
      crack_summary: crackSummary,
      altitude_m: geoRef.altitudeM,
      relative_altitude_m: geoRef.relativeAltitudeM,
      gimbal_yaw_deg: geoRef.gimbalYawDeg,
      gsd_mm_per_px: geoRef.gsdMmPerPx,
      camera_model: geoRef.cameraModel,
      captured_at: geoRef.capturedAt,
      analyzed_at: analyzedAt,
      crack_details: crackDetails,
    },
  };
}

export function buildGeoJsonFeatureCollection(
  fileName: string,
  imgWidth: number,
  imgHeight: number,
  anomalies: DetectedAnomaly[],
  geoRef: GeoReferenceInfo,
  analyzedAt: string
): GeoJsonFeatureCollection {
  const imageFeature = buildImageGeoJsonFeature(
    fileName,
    imgWidth,
    imgHeight,
    anomalies,
    geoRef,
    analyzedAt
  );

  const isEpsg3857 = geoRef.epsgCode === 'EPSG:3857';

  return {
    type: 'FeatureCollection',
    name: 'pv_image_crack_inspection',
    crs: {
      type: 'name',
      properties: {
        name: isEpsg3857 ? EPSG3857_URN : 'urn:ogc:def:crs:OGC:1.3:CRS84',
      },
    },
    metadata: {
      crs_epsg: isEpsg3857 ? 'EPSG:3857' : 'PIXEL_LOCAL',
      crs_label: geoRef.crsName,
      generated_at: analyzedAt,
      total_images: 1,
      total_image_points: 1,
      images_with_spider_web_cracks: imageFeature.properties.has_spider_web_crack ? 1 : 0,
      total_spider_web_cracks: imageFeature.properties.spider_web_crack_count,
      total_edge_spider_web_cracks: imageFeature.properties.edge_spider_web_crack_count,
      coordinate_system_note: isEpsg3857
        ? `One GeoJSON Point [x_easting_m, y_northing_m] per image in ${EPSG3857_LABEL} (${geoRef.crsMode}); all crack details stored in properties.`
        : 'One Point per image at local center [center_x, -center_y]; all crack details stored in properties.',
    },
    features: [imageFeature],
  };
}

export async function analyzeInspectionImage(
  imageBuffer: Buffer,
  fileName: string,
  config: InspectionConfig,
  includePreviewDataUrl = false
): Promise<InspectionResult> {
  const startTime = Date.now();
  const ai = getGenAIClient(); // null => offline classical-only mode
  const warnings: string[] = [];
  if (!ai) {
    warnings.push(
      'Gemini unavailable (no GEMINI_API_KEY or CV_ENGINE=classical): ran offline classical detector only; impact/linear/residue categories are not reported.'
    );
  }

  const metaProbe = await sharp(imageBuffer).rotate().metadata();
  const origW = metaProbe.width || 1920;
  const origH = metaProbe.height || 1080;
  const maxNativeDim = 8000;
  const nativeScale = Math.min(1, maxNativeDim / Math.max(origW, origH));
  const reqWidth = Math.max(32, Math.round(origW * nativeScale));
  const reqHeight = Math.max(32, Math.round(origH * nativeScale));

  let pipelineSharp = sharp(imageBuffer).rotate();
  if (nativeScale < 1) {
    pipelineSharp = pipelineSharp.resize(reqWidth, reqHeight, { fit: 'fill' });
  }
  const { data: decodedBuf, info: decodedInfo } = await pipelineSharp
    .toColorspace('srgb')
    .removeAlpha()
    .raw({ depth: 'uchar' })
    .toBuffer({ resolveWithObject: true });

  const imgWidth = decodedInfo.width;
  const imgHeight = decodedInfo.height;
  let rawRgb = decodedBuf;
  if (decodedInfo.channels === 1) {
    const expanded = Buffer.alloc(imgWidth * imgHeight * 3);
    for (let i = 0; i < imgWidth * imgHeight; i++) {
      const v = decodedBuf[i];
      expanded[i * 3] = v;
      expanded[i * 3 + 1] = v;
      expanded[i * 3 + 2] = v;
    }
    rawRgb = expanded;
  } else if (decodedInfo.channels === 4) {
    const stripped = Buffer.alloc(imgWidth * imgHeight * 3);
    for (let i = 0; i < imgWidth * imgHeight; i++) {
      stripped[i * 3] = decodedBuf[i * 4];
      stripped[i * 3 + 1] = decodedBuf[i * 4 + 1];
      stripped[i * 3 + 2] = decodedBuf[i * 4 + 2];
    }
    rawRgb = stripped;
  }

  // Stage 1: cell-bearing surface localization + edge-pitch solar-cell grid + crack-response map
  const stage1 = await runClassicalStage1(rawRgb, imgWidth, imgHeight);
  const lowConfidenceCellGrids = stage1.moduleRegions.filter(
    (region) => !region.isFallback && (region.cellGrid?.confidence ?? 0) < 0.25
  );
  if (lowConfidenceCellGrids.length > 0) {
    warnings.push(
      `${lowConfidenceCellGrids.length} solar-cell region(s) had weak edge-map periodicity. Cell coordinates in those regions are provisional; review the cell-edge overlay before relying on them.`
    );
  }
  if (process.env.CV_DEBUG) {
    for (const region of stage1.moduleRegions) {
      if (region.cellGrid) {
        console.log(
          `[CV_DEBUG] ${region.label}: ${region.cellGrid.rows}x${region.cellGrid.columns} cells, ` +
          `grid=${region.cellGrid.method}, confidence=${region.cellGrid.confidence}`
        );
      }
    }
  }

  const refModWidthPx =
    stage1.moduleRegions.length > 0
      ? stage1.moduleRegions[0].pixelBox.width
      : imgWidth * 0.45;

  const geoReference = extractGeoReference(
    imageBuffer,
    imgWidth,
    imgHeight,
    refModWidthPx,
    config,
    fileName
  );

  // Stage 2: High-Magnification Quadrant & Perimeter Slicing (Zero Downscaling on 10MP-45MP Frames!)
  const tiles: TileDiagnostic[] = [];

  // Inspection tiling is triggered by visible cell-bearing regions, including partial
  // regions clipped by the image boundary; it is not conditional on a complete module.
  const hasStructuredModules =
    config.edgeFocusMode &&
    imgWidth >= 320 &&
    imgHeight >= 160 &&
    stage1.moduleRegions.some((region) => !region.isFallback && region.cellGrid &&
      region.pixelBox.width >= 48 && region.pixelBox.height >= 24);

  if (hasStructuredModules) {
    stage1.moduleRegions.forEach((mod, mIdx) => {
      const mb = mod.pixelBox;
      if (mb.height < 24 || mb.width < 48) return;

      // 34% Left & Right perimeter strips cover Cols 1..8 and Cols 17..24 intact (never cutting edge spider-webs in half!)
      const edgeStripW = Math.max(110, Math.round(mb.width * 0.34));
      const rightEdgeLeft = Math.max(0, mb.left + mb.width - edgeStripW);
      const centerStart = mb.left + Math.round(mb.width * 0.28);
      const centerW = Math.round(mb.width * 0.44);

      if (mb.height > 740) {
        // High-resolution native drone frame (e.g. 3664x2748 or 7328x5496):
        // Split tall perimeter strips vertically with 18% overlap so every border cell stays at 1:1 native 400%-zoom detail!
        const halfH = Math.round(mb.height * 0.58);
        const botTop = mb.top + mb.height - halfH;
        const quadSpecs: Array<{
          suffix: string;
          rOff: number;
          col: number;
          isPerim: boolean;
          left: number;
          top: number;
          width: number;
          height: number;
        }> = [
          {
            suffix: 'TOP-LEFT',
            rOff: 0,
            col: 1,
            isPerim: true,
            left: mb.left,
            top: mb.top,
            width: edgeStripW,
            height: halfH,
          },
          {
            suffix: 'BOT-LEFT',
            rOff: 1,
            col: 1,
            isPerim: true,
            left: mb.left,
            top: botTop,
            width: edgeStripW,
            height: halfH,
          },
          {
            suffix: 'TOP-RIGHT',
            rOff: 0,
            col: 3,
            isPerim: true,
            left: rightEdgeLeft,
            top: mb.top,
            width: mb.left + mb.width - rightEdgeLeft,
            height: halfH,
          },
          {
            suffix: 'BOT-RIGHT',
            rOff: 1,
            col: 3,
            isPerim: true,
            left: rightEdgeLeft,
            top: botTop,
            width: mb.left + mb.width - rightEdgeLeft,
            height: halfH,
          },
          {
            suffix: 'CENTER',
            rOff: 0,
            col: 2,
            isPerim: false,
            left: centerStart,
            top: mb.top,
            width: centerW,
            height: mb.height,
          },
        ];

        for (const s of quadSpecs) {
          const safeLeft = Math.max(0, Math.min(imgWidth - 32, s.left));
          const safeTop = Math.max(0, Math.min(imgHeight - 32, s.top));
          const safeWidth = Math.max(32, Math.min(imgWidth - safeLeft, s.width));
          const safeHeight = Math.max(32, Math.min(imgHeight - safeTop, s.height));
          tiles.push(
            scoreTileWindow(
              `M${mIdx + 1}-${s.suffix}`,
              mIdx * 2 + 1 + s.rOff,
              s.col,
              s.isPerim,
              { left: safeLeft, top: safeTop, width: safeWidth, height: safeHeight },
              imgWidth,
              imgHeight,
              stage1
            )
          );
        }
      } else {
        const stripSpecs: Array<{
          suffix: string;
          col: number;
          isPerim: boolean;
          left: number;
          width: number;
        }> = [
          { suffix: 'LEFT-EDGE', col: 1, isPerim: true, left: mb.left, width: edgeStripW },
          {
            suffix: 'RIGHT-EDGE',
            col: 3,
            isPerim: true,
            left: rightEdgeLeft,
            width: mb.left + mb.width - rightEdgeLeft,
          },
          { suffix: 'CENTER', col: 2, isPerim: false, left: centerStart, width: centerW },
        ];

        for (const s of stripSpecs) {
          const safeLeft = Math.max(0, Math.min(imgWidth - 32, s.left));
          const safeTop = Math.max(0, Math.min(imgHeight - 32, mb.top));
          const safeWidth = Math.max(32, Math.min(imgWidth - safeLeft, s.width));
          const safeHeight = Math.max(32, Math.min(imgHeight - safeTop, mb.height));
          tiles.push(
            scoreTileWindow(
              `M${mIdx + 1}-${s.suffix}`,
              mIdx + 1,
              s.col,
              s.isPerim,
              { left: safeLeft, top: safeTop, width: safeWidth, height: safeHeight },
              imgWidth,
              imgHeight,
              stage1
            )
          );
        }
      }
    });
  }

  // Fallback general SAHI grid for close-up crops (e.g. zoomed sub-panel screenshots)
  if (tiles.length === 0) {
    const roi = stage1.activeBounds;
    const requestedTile = Math.max(480, Math.min(2048, config.tileSizePx || 1120));
    const effTileW = Math.min(imgWidth, Math.min(Math.max(420, roi.width), requestedTile));
    const effTileH = Math.min(imgHeight, Math.min(Math.max(420, roi.height), requestedTile));
    const overlap = Math.max(0.1, Math.min(0.4, config.overlapRatio || 0.2));
    const strideX = Math.max(180, Math.round(effTileW * (1 - overlap)));
    const strideY = Math.max(180, Math.round(effTileH * (1 - overlap)));

    const roiRight = Math.min(imgWidth, roi.left + roi.width);
    const roiBottom = Math.min(imgHeight, roi.top + roi.height);

    const xStarts: number[] = [];
    for (let x = roi.left; x < roiRight; x += strideX) {
      const clamped = Math.max(0, Math.min(x, Math.max(0, roiRight - effTileW)));
      if (xStarts.length === 0 || clamped - xStarts[xStarts.length - 1] >= 60) {
        xStarts.push(clamped);
      }
      if (clamped + effTileW >= roiRight) break;
    }

    const yStarts: number[] = [];
    for (let y = roi.top; y < roiBottom; y += strideY) {
      const clamped = Math.max(0, Math.min(y, Math.max(0, roiBottom - effTileH)));
      if (yStarts.length === 0 || clamped - yStarts[yStarts.length - 1] >= 60) {
        yStarts.push(clamped);
      }
      if (clamped + effTileH >= roiBottom) break;
    }

    for (let r = 0; r < yStarts.length; r++) {
      for (let c = 0; c < xStarts.length; c++) {
        const isPerim = r === 0 || r === yStarts.length - 1 || c === 0 || c === xStarts.length - 1;
        const safeLeft = Math.max(0, Math.min(imgWidth - 16, xStarts[c]));
        const safeTop = Math.max(0, Math.min(imgHeight - 16, yStarts[r]));
        const safeW = Math.max(16, Math.min(imgWidth - safeLeft, effTileW));
        const safeH = Math.max(16, Math.min(imgHeight - safeTop, effTileH));
        tiles.push(
          scoreTileWindow(
            `T-R${r + 1}C${c + 1}`,
            r + 1,
            c + 1,
            isPerim,
            { left: safeLeft, top: safeTop, width: safeW, height: safeH },
            imgWidth,
            imgHeight,
            stage1
          )
        );
      }
    }
  }

  const pvActiveTiles = tiles.filter((t) => t.pvCoverageRatio >= 0.1);
  const candidatePool = pvActiveTiles.length > 0 ? pvActiveTiles : tiles;

  const rankedCandidates = [...candidatePool].sort((a, b) => {
    const scoreA =
      a.offAxisEdgeScore +
      a.impactKurtosisScore * 0.65 +
      (a.isPerimeterTile ? 45 : 0) +
      a.pvCoverageRatio * 12;
    const scoreB =
      b.offAxisEdgeScore +
      b.impactKurtosisScore * 0.65 +
      (b.isPerimeterTile ? 45 : 0) +
      b.pvCoverageRatio * 12;
    return scoreB - scoreA;
  });

  // Keep central cell coverage in the VLM budget, then use the remaining slots for edge/interior strips.
  const perimCount = rankedCandidates.filter((t) => t.isPerimeterTile).length;
  const maxToInspect = Math.max(
    Math.min(14, perimCount),
    Math.max(1, Math.min(16, config.maxTilesToInspect || 12))
  );
  // Preserve at least one interior/center tile per detected array. Otherwise perimeter scoring
  // can consume the budget and leave middle solar cells unseen by the VLM.
  const requiredInteriorTiles = rankedCandidates.filter((tile) => /-CENTER$/.test(tile.id));
  const selectedTileList: TileDiagnostic[] = [];
  for (const tile of requiredInteriorTiles) {
    if (selectedTileList.length < maxToInspect) selectedTileList.push(tile);
  }
  for (const tile of rankedCandidates) {
    if (selectedTileList.length >= maxToInspect) break;
    if (!selectedTileList.some((picked) => picked.id === tile.id)) selectedTileList.push(tile);
  }
  const selectedTiles = ai ? selectedTileList : []; // offline: no VLM tiles
  const selectedIds = new Set(selectedTiles.map((t) => t.id));
  tiles.forEach((t) => {
    if (selectedIds.has(t.id)) t.inspectedByVlm = true;
  });

  const sensitivityGuidance =
    config.sensitivity === 'high'
      ? 'HIGH RECALL MODE: Inspect every detected solar cell, including central and interior cells. Report all visible glass-fracture networks, single-cell diagonal cracks, and multi-cell sweeping arcs; do not limit the search to the array perimeter.'
      : config.sensitivity === 'strict'
        ? 'PRECISION FRACTURE MODE: Inspect every solar cell. Report clear glass-fracture networks, diagonal cell fractures, and impact pits; ignore minor dust and ambiguous reflections.'
        : 'BALANCED MODE: Inspect every row and column of solar cells for localized cracks, spider-web networks, impacts, and residue. Include both interior and border cells.';

  const systemInstruction = `You are a specialized Photovoltaic Glass Fracture Computer Vision Inspector. The input tile may show any part of a photovoltaic array, including central/interior cells and perimeter cells.

CRITICAL — INSPECT EVERY VISIBLE SOLAR CELL:
1. GLASS FRACTURE NETWORKS:
   - Find thin bright silver-white curved, diagonal, criss-crossing, fan-shaped, or branching lines that cut across the regular horizontal cell fingers.
   - Fractures may be fully inside one cell, cross adjacent cells, or begin at the module frame. Do not require contact with the outer frame.
   - A localized interior-cell crack is still a valid defect; use isEdgeCrack=false and edgeSide="interior" unless it genuinely touches the outer array boundary.
2. CELL-BY-CELL COVERAGE:
   - Examine all visible cell surfaces, including center rows and columns—not just the left/right border columns.
   - Do not confuse straight cell fingers, busbars, inter-cell gaps, cell seams, or reflections with fractures.

OTHER DEFECT CATEGORIES & RULES:
- "impact_point": Small localized bright white stone/hail pit or chip inside a cell without branching arcs.
- "linear_fracture": Single isolated diagonal micro-crack inside any cell.
- "surface_residue": Opaque white sealant/bird-dropping smear on top of glass.
- Set isEdgeCrack=true only when the fracture touches the outer cell row/column or module frame; otherwise set it false and edgeSide="interior".
- GROUND RULE: never mark bare soil, grass, shadows, torque tubes, cables or anything outside the solar-cell surfaces. Every box and point MUST lie on a cell; shrink a candidate to the cell area or omit it if it touches ground.
- Ignore straight cell fingers, busbars, straight white inter-cell gaps, aluminum frames, and ground grass/soil.
${sensitivityGuidance}`;

  const rawDetections: Array<DetectedAnomaly> = [];
  let primaryModelUsed = ai ? FALLBACK_MODEL : 'classical-offline';

  const encodedTiles = await Promise.all(
    selectedTiles.map(async (tile, tIdx) => {
      const tileJpeg = await buildContrastBoostedTileJpeg(
        rawRgb,
        imgWidth,
        imgHeight,
        tile.pixelBox,
        config.contrastEnhancement
      );
      return {
        tIdx,
        tile,
        base64Tile: tileJpeg.toString('base64'),
      };
    })
  );

  type RawDetItem = {
    category: string;
    isEdgeCrack: boolean;
    edgeSide: string;
    severity: string;
    confidence: number;
    title: string;
    description: string;
    radialBranches: number;
    pointY: number;
    pointX: number;
    ymin: number;
    xmin: number;
    ymax: number;
    xmax: number;
  };

  // Focused VLM Chunking: max 4 tiles per request prevents multi-image attention dilution!
  const CHUNK_SIZE = 4;
  const tileChunks: Array<typeof encodedTiles> = [];
  for (let i = 0; i < encodedTiles.length; i += CHUNK_SIZE) {
    tileChunks.push(encodedTiles.slice(i, i + CHUNK_SIZE));
  }

  const runVlmChunk = async (
    chunk: typeof encodedTiles
  ): Promise<Array<{ globalTileIdx: number; det: RawDetItem }>> => {
    const parts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> = [];
    for (let localIdx = 0; localIdx < chunk.length; localIdx++) {
      const item = chunk[localIdx];
      parts.push({
        text: `TILE_INDEX=${localIdx} | Tile ID=${item.tile.id} (${item.tile.pixelBox.width}x${item.tile.pixelBox.height}px, offAxisFractureScore=${item.tile.offAxisEdgeScore}):`,
      });
      parts.push({
        inlineData: {
          mimeType: 'image/jpeg',
          data: item.base64Tile,
        },
      });
    }
    parts.push({
      text: `Inspect EVERY tile above (TILE_INDEX 0 through ${chunk.length - 1}) row-by-row. Pay special attention to the outermost cell columns touching the right and left white aluminum frames, as well as corners, for diagonal/curved/branching spider-web glass cracks cutting across the horizontal cell fingers!`,
    });

    resetExhaustedIfNewDay();
    const candidateModels = [PRIMARY_MODEL, FALLBACK_MODEL].filter(
      (m, i, a) => a.indexOf(m) === i && !dailyExhaustedModels.has(m)
    );
    const modelChain = candidateModels.length > 0 ? candidateModels : [FALLBACK_MODEL];

    let textOut = '{"tileResults":[]}';
    let ok = false;
    let chunkErr: unknown;

    for (let mIdx = 0; mIdx < modelChain.length && !ok; mIdx++) {
      const modelName = modelChain[mIdx];
      const maxRetries = modelName === FALLBACK_MODEL ? 3 : 1;

      for (let retry = 0; retry < maxRetries; retry++) {
        const abortController = new AbortController();
        const timer = setTimeout(() => abortController.abort(), 42000);
        try {
          const useThinking = modelName === PRIMARY_MODEL && PRIMARY_MODEL !== FALLBACK_MODEL;
          const res = await ai!.models.generateContent({
            model: modelName,
            contents: { parts },
            config: {
              abortSignal: abortController.signal,
              systemInstruction,
              ...(useThinking ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } } : {}),
              responseMimeType: 'application/json',
              responseSchema: tileDetectionSchema,
              temperature: 0.1,
            },
          });
          clearTimeout(timer);
          primaryModelUsed = modelName;
          textOut = (res.text ?? '{"tileResults":[]}').trim();
          ok = true;
          break;
        } catch (err: unknown) {
          clearTimeout(timer);
          chunkErr = err;
          const errMsg = err instanceof Error ? err.message : String(err);
          if (errMsg.includes('PerDay') || errMsg.includes('NOT_FOUND')) {
            dailyExhaustedModels.add(modelName);
            break;
          }
          await new Promise((r) => setTimeout(r, 1200 * (retry + 1)));
        }
      }
    }

    if (!ok) {
      throw chunkErr instanceof Error ? chunkErr : new Error('Tile VLM chunk inference failed');
    }

    let parsed: {
      tileResults?: Array<{
        tileIndex: number;
        detections?: RawDetItem[];
      }>;
    };
    try {
      parsed = JSON.parse(textOut);
    } catch {
      throw new Error('Gemini returned non-JSON output');
    }

    const items: Array<{ globalTileIdx: number; det: RawDetItem }> = [];
    for (const tr of parsed.tileResults || []) {
      const localIdx = tr.tileIndex ?? 0;
      if (!Number.isInteger(localIdx) || localIdx < 0 || localIdx >= chunk.length) continue; // was clamped onto wrong tile
      const globalTileIdx = chunk[localIdx].tIdx;
      for (const det of tr.detections || []) {
        items.push({ globalTileIdx, det });
      }
    }
    return items;
  };

  // Run Stage 3A (Focused VLM Chunks) + Stage 3B (Native-Resolution Classical Orthogonal-Residual Detector)
  const [chunkOutputs, classicalDetections, learnedDetections] = await Promise.all([
    (async () => {
      const allItems: Array<{ globalTileIdx: number; det: RawDetItem }> = [];
      if (!ai) return allItems; // offline: skip VLM entirely
      let failedChunks = 0;
      // Run up to 2 chunks concurrently for speed while staying gentle on rate limits
      for (let i = 0; i < tileChunks.length; i += 2) {
        const pair = tileChunks.slice(i, i + 2);
        const pairRes = await Promise.all(
          pair.map((ch) =>
            runVlmChunk(ch).catch((e: unknown) => {
              failedChunks++; // network/quota failure no longer aborts the whole image
              console.warn('VLM chunk failed:', e instanceof Error ? e.message : e);
              return [] as Array<{ globalTileIdx: number; det: RawDetItem }>;
            })
          )
        );
        for (const arr of pairRes) allItems.push(...arr);
      }
      if (failedChunks > 0) {
        warnings.push(
          `Gemini failed on ${failedChunks}/${tileChunks.length} tile batches; results rely more on the classical detector.`
        );
        if (failedChunks === tileChunks.length) primaryModelUsed = 'classical-offline (Gemini failed)';
      }
      return allItems;
    })(),
    detectClassicalModuleSpiderWebs(
      rawRgb,
      imgWidth,
      imgHeight,
      stage1.moduleRegions,
      geoReference.gsdMmPerPx
    ),
    detectLearnedEdgeCracks(rawRgb, imgWidth, imgHeight, stage1.moduleRegions, geoReference.gsdMmPerPx),
  ]);

  let groundRejected = 0;
  const minConf =
    config.sensitivity === 'strict' ? 0.58 : config.sensitivity === 'high' ? 0.46 : 0.52;

  for (let dIdx = 0; dIdx < chunkOutputs.length; dIdx++) {
    const { globalTileIdx, det: d } = chunkOutputs[dIdx];
    const tile = selectedTiles[globalTileIdx];
    if (!tile) continue;

    const cat = normalizeCategory(d.category);
    if (!config.includeSurfaceResidue && cat === 'surface_residue') {
      continue;
    }
    if ((d.confidence ?? 0) < minConf) continue;
    if (![d.ymin, d.xmin, d.ymax, d.xmax].every((v) => Number.isFinite(v))) continue; // guard NaN boxes

    const yminClamped = Math.max(0, Math.min(995, d.ymin));
    const xminClamped = Math.max(0, Math.min(995, d.xmin));
    const ymaxClamped = Math.max(yminClamped + 8, Math.min(1000, d.ymax));
    const xmaxClamped = Math.max(xminClamped + 8, Math.min(1000, d.xmax));

    const globalLeft = Math.round(
      tile.pixelBox.left + (xminClamped / 1000) * tile.pixelBox.width
    );
    const globalTop = Math.round(
      tile.pixelBox.top + (yminClamped / 1000) * tile.pixelBox.height
    );
    const globalRight = Math.round(
      tile.pixelBox.left + (xmaxClamped / 1000) * tile.pixelBox.width
    );
    const globalBottom = Math.round(
      tile.pixelBox.top + (ymaxClamped / 1000) * tile.pixelBox.height
    );

    let pBox: PixelBox = {
      left: Math.max(0, globalLeft),
      top: Math.max(0, globalTop),
      width: Math.max(16, Math.min(imgWidth - globalLeft, globalRight - globalLeft)),
      height: Math.max(16, Math.min(imgHeight - globalTop, globalBottom - globalTop)),
    };

    if (config.rejectGround) {
      const gated = applyGroundGate(pBox, stage1, imgWidth, imgHeight);
      if (!gated) {
        groundRejected++;
        continue;
      }
      pBox = gated; // clipped to module interior
    }

    const rawPtX =
      typeof d.pointX === 'number'
        ? Math.round(
            tile.pixelBox.left + (Math.max(0, Math.min(1000, d.pointX)) / 1000) * tile.pixelBox.width
          )
        : Math.round(pBox.left + pBox.width / 2);
    const rawPtY =
      typeof d.pointY === 'number'
        ? Math.round(
            tile.pixelBox.top + (Math.max(0, Math.min(1000, d.pointY)) / 1000) * tile.pixelBox.height
          )
        : Math.round(pBox.top + pBox.height / 2);

    const ptX = Math.max(pBox.left, Math.min(pBox.left + pBox.width, rawPtX));
    const ptY = Math.max(pBox.top, Math.min(pBox.top + pBox.height, rawPtY));

    const edgeMeta = normalizeEdgeSide(
      d.edgeSide,
      Boolean(d.isEdgeCrack),
      pBox,
      stage1.moduleRegions
    );

    const loc = resolveModuleAndCell(ptX, ptY, stage1.moduleRegions, imgWidth, imgHeight);
    const estMm = Math.round(Math.max(pBox.width, pBox.height) * geoReference.gsdMmPerPx);

    tile.detectionsCount += 1;

    rawDetections.push({
      id: `${tile.id}-D${dIdx + 1}`,
      category: cat,
      isEdgeCrack: edgeMeta.isEdgeCrack,
      edgeSide: edgeMeta.edgeSide,
      severity: normalizeSeverity(d.severity, cat),
      confidence: Number(Math.min(0.99, Math.max(0.5, d.confidence)).toFixed(2)),
      title:
        d.title || (cat === 'spider_web_crack' ? 'Edge Spider-Web Glass Crack' : 'Surface Anomaly'),
      description:
        d.description ||
        'Detected surface fracture pattern during high-resolution solar-cell inspection tile scan.',
      moduleIndex: loc.moduleIndex,
      cellLocation: loc.cellLocation,
      radialBranches:
        cat === 'spider_web_crack' ? Math.max(2, d.radialBranches || 4) : d.radialBranches || 0,
      estimatedDiameterMm: estMm,
      pointPx: { x: ptX, y: ptY },
      box: {
        xmin: pBox.left / imgWidth,
        ymin: pBox.top / imgHeight,
        xmax: (pBox.left + pBox.width) / imgWidth,
        ymax: (pBox.top + pBox.height) / imgHeight,
      },
      pixelBox: pBox,
      tileId: tile.id,
      offAxisEdgeScore: tile.offAxisEdgeScore,
    });
  }

  // Keep the cell-aware classical detector active even when the learned model is loaded.
  // The learned classifier intentionally scans a narrower edge-ring window set; disabling the
  // classical detector here would silently remove its broader cell-grid coverage. Merge by spatial
  // overlap, not by module identity: one VLM hit must not hide a separate crack elsewhere on that panel.
  const classicalForMerge = classicalDetections;
  for (const cvDet of classicalForMerge) {
    const alreadyCovered = rawDetections.some((detection) =>
      detection.category === 'spider_web_crack' && computeIoU(detection.box, cvDet.box) > 0.14
    );
    if (!alreadyCovered) {
      if (config.rejectGround) {
        const gated = applyGroundGate(cvDet.pixelBox, stage1, imgWidth, imgHeight);
        if (!gated) {
          groundRejected++;
          continue;
        }
        cvDet.pixelBox = gated;
        cvDet.box = {
          xmin: gated.left / imgWidth,
          ymin: gated.top / imgHeight,
          xmax: (gated.left + gated.width) / imgWidth,
          ymax: (gated.top + gated.height) / imgHeight,
        };
      }
      rawDetections.push(cvDet);
    }
  }
  // Learned detections: every window is independent evidence (unlike the classical detector they are NOT suppressed
  // because the module already has a VLM/classical hit). Skip only windows already covered by an accepted detection.
  for (const ml of learnedDetections) {
    if (rawDetections.some((d) => computeIoU(d.box, ml.box) > 0.14)) continue;
    if (config.rejectGround) {
      const gated = applyGroundGate(ml.pixelBox, stage1, imgWidth, imgHeight);
      if (!gated) {
        groundRejected++;
        continue;
      }
      ml.pixelBox = gated;
      ml.box = {
        xmin: gated.left / imgWidth,
        ymin: gated.top / imgHeight,
        xmax: (gated.left + gated.width) / imgWidth,
        ymax: (gated.top + gated.height) / imgHeight,
      };
    }
    rawDetections.push(ml);
  }
  if (groundRejected > 0) {
    warnings.push(`${groundRejected} detection(s) rejected as ground / off-module.`);
  }

  // Stage 4: Cross-Tile Non-Maximum Suppression
  const severityRank: Record<SeverityLevel, number> = {
    critical: 4,
    moderate: 3,
    minor: 2,
    informational: 1,
  };

  rawDetections.sort(
    (a, b) =>
      (b.category === 'spider_web_crack' ? 3 : 0) +
      severityRank[b.severity] * 2 +
      b.confidence -
      ((a.category === 'spider_web_crack' ? 3 : 0) + severityRank[a.severity] * 2 + a.confidence)
  );

  const deduplicated: DetectedAnomaly[] = [];
  for (const cand of rawDetections) {
    const duplicate = deduplicated.some((kept) => {
      if (kept.category !== cand.category) return false;
      if (computeIoU(kept.box, cand.box) > 0.28) return true;
      const distPx = Math.hypot(kept.pointPx.x - cand.pointPx.x, kept.pointPx.y - cand.pointPx.y);
      return distPx < Math.min(kept.pixelBox.width, cand.pixelBox.width) * 0.45;
    });
    if (!duplicate) {
      deduplicated.push(cand);
    }
  }

  const enrichedAnomalies = await Promise.all(
    deduplicated.slice(0, 28).map(async (anom, idx) => {
      const patches = await extractForensicPatchPair(rawRgb, imgWidth, imgHeight, anom.pixelBox);
      return {
        ...anom,
        id: `ANOM-${String(idx + 1).padStart(2, '0')}`,
        cropDataUrl: patches.cropDataUrl,
        edgeMapDataUrl: patches.edgeMapDataUrl,
      };
    })
  );

  let previewImageUrl: string | undefined;
  if (includePreviewDataUrl) {
    const previewJpeg = await sharp(rawRgb, {
      raw: { width: imgWidth, height: imgHeight, channels: 3 },
    })
      .resize(2200, 2200, { fit: 'inside', withoutEnlargement: true })
      .jpeg({ quality: 86 })
      .toBuffer();
    previewImageUrl = `data:image/jpeg;base64,${previewJpeg.toString('base64')}`;
  }

  const analyzedAt = new Date().toISOString();
  const geoJson = buildGeoJsonFeatureCollection(
    fileName,
    imgWidth,
    imgHeight,
    enrichedAnomalies,
    geoReference,
    analyzedAt
  );

  const spiderWebCracksCount = enrichedAnomalies.filter(
    (a) => a.category === 'spider_web_crack'
  ).length;
  const edgeSpiderWebCracksCount = enrichedAnomalies.filter(
    (a) => a.category === 'spider_web_crack' && a.isEdgeCrack
  ).length;
  const impactPointsCount = enrichedAnomalies.filter((a) => a.category === 'impact_point').length;
  const linearFracturesCount = enrichedAnomalies.filter(
    (a) => a.category === 'linear_fracture'
  ).length;
  const surfaceResidueCount = enrichedAnomalies.filter(
    (a) => a.category === 'surface_residue'
  ).length;

  return {
    imageId: `${fileName}-${Date.now()}`,
    fileName,
    imageWidth: imgWidth,
    imageHeight: imgHeight,
    megapixels: Number(((imgWidth * imgHeight) / 1_000_000).toFixed(1)),
    analyzedAt,
    processingTimeMs: Date.now() - startTime,
    modelUsed: `Adaptive-Silicon + Ortho-Ridge + ${primaryModelUsed}`,
    warnings: warnings.length > 0 ? warnings : undefined,
    geoReference,
    pipelineSummary: {
      totalGridTiles: tiles.length,
      pvActiveTiles: candidatePool.length,
      vlmInspectedTiles: selectedTiles.length,
      spiderWebCracksCount,
      edgeSpiderWebCracksCount,
      impactPointsCount,
      linearFracturesCount,
      surfaceResidueCount,
    },
    moduleRegions: stage1.moduleRegions,
    tiles,
    anomalies: enrichedAnomalies,
    geoJson,
    previewImageUrl,
    fractureHeatmapUrl: stage1.fractureHeatmapUrl,
    cellBoundaryMapUrl: stage1.cellBoundaryMapUrl || undefined,
  };
}

function scoreTileWindow(
  id: string,
  row: number,
  col: number,
  isPerimeterTile: boolean,
  pixelBox: PixelBox,
  imgWidth: number,
  imgHeight: number,
  stage1: DownsampledAnalysis
): TileDiagnostic {
  const sx0 = Math.floor((pixelBox.left / imgWidth) * stage1.scaleWidth);
  const sx1 = Math.min(
    stage1.scaleWidth - 1,
    Math.ceil(((pixelBox.left + pixelBox.width) / imgWidth) * stage1.scaleWidth)
  );
  const sy0 = Math.floor((pixelBox.top / imgHeight) * stage1.scaleHeight);
  const sy1 = Math.min(
    stage1.scaleHeight - 1,
    Math.ceil(((pixelBox.top + pixelBox.height) / imgHeight) * stage1.scaleHeight)
  );

  let totalPx = 0;
  let pvPx = 0;
  let edgeSum = 0;
  let impactPeak = 0;
  const topEnergies: number[] = [];

  for (let sy = sy0; sy <= sy1; sy++) {
    for (let sx = sx0; sx <= sx1; sx++) {
      totalPx++;
      const idx = sy * stage1.scaleWidth + sx;
      if (stage1.pvMask[idx]) {
        pvPx++;
        const e = stage1.offAxisEnergy[idx];
        const imp = stage1.impactEnergy[idx];
        edgeSum += e;
        if (imp > impactPeak) impactPeak = imp;
        if (e + imp > 16) {
          topEnergies.push(e + imp * 1.4);
        }
      }
    }
  }

  topEnergies.sort((a, b) => b - a);
  const topKCount = Math.min(35, topEnergies.length);
  let topKMean = 0;
  for (let k = 0; k < topKCount; k++) topKMean += topEnergies[k];
  if (topKCount > 0) topKMean /= topKCount;

  const pvCoverageRatio = totalPx > 0 ? pvPx / totalPx : 0;
  const meanEdge = pvPx > 0 ? edgeSum / pvPx : 0;
  const offAxisScore = Number((meanEdge * 0.4 + topKMean * 0.6).toFixed(2));
  const impactScore = Number(impactPeak.toFixed(2));

  return {
    id,
    row,
    col,
    isPerimeterTile,
    pixelBox,
    box: {
      xmin: pixelBox.left / imgWidth,
      ymin: pixelBox.top / imgHeight,
      xmax: (pixelBox.left + pixelBox.width) / imgWidth,
      ymax: (pixelBox.top + pixelBox.height) / imgHeight,
    },
    pvCoverageRatio: Number(pvCoverageRatio.toFixed(3)),
    offAxisEdgeScore: offAxisScore,
    impactKurtosisScore: impactScore,
    inspectedByVlm: false,
    detectionsCount: 0,
  };
}
