import JSZip from 'jszip';
import type {
  GeoJsonFeatureCollection,
  GeoJsonPointFeature,
  InspectionResult,
} from '../types/inspection';

const IMAGE_EXT_REGEX = /\.(jpe?g|jfif|png|webp|tiff?)$/i;

export function isSupportedImageFileName(name: string): boolean {
  const lower = name.toLowerCase();
  if (lower.includes('__macosx/') || lower.split('/').pop()?.startsWith('.')) {
    return false;
  }
  return IMAGE_EXT_REGEX.test(lower);
}

export function inferMimeTypeFromName(name: string): string {
  const lower = name.toLowerCase();
  if (lower.endsWith('.png')) return 'image/png';
  if (lower.endsWith('.webp')) return 'image/webp';
  if (lower.endsWith('.tif') || lower.endsWith('.tiff')) return 'image/tiff';
  return 'image/jpeg';
}

/**
 * Extracts all supported image files from one or more .zip archives on the client
 * so batch processing can stream each frame with real-time per-image progress.
 */
export async function extractImagesFromZipFile(zipFile: File): Promise<File[]> {
  const zip = await JSZip.loadAsync(zipFile);
  const extracted: File[] = [];
  const entries = Object.values(zip.files).filter(
    (entry) => !entry.dir && isSupportedImageEntry(entry.name)
  );

  for (const entry of entries) {
    const blob = await entry.async('blob');
    const cleanRelName = entry.name.replace(/^\/+/, '');
    const mime = inferMimeTypeFromName(cleanRelName);
    const fileObj = new File([blob], `${zipFile.name.replace(/\.zip$/i, '')}/${cleanRelName}`, {
      type: mime,
    });
    extracted.push(fileObj);
  }
  return extracted;
}

function isSupportedImageEntry(name: string): boolean {
  return isSupportedImageFileName(name);
}

/**
 * Recursively traverses dropped folders & files (including nested subdirectories and .zip files).
 */
export async function collectFilesFromDataTransfer(
  dataTransfer: DataTransfer
): Promise<File[]> {
  const collected: File[] = [];

  async function traverseEntry(entry: FileSystemEntry, pathPrefix = ''): Promise<void> {
    if (entry.isFile) {
      const fileEntry = entry as FileSystemFileEntry;
      const file = await new Promise<File>((resolve, reject) => {
        fileEntry.file(resolve, reject);
      });
      const fullPath = pathPrefix ? `${pathPrefix}/${file.name}` : file.name;
      if (file.name.toLowerCase().endsWith('.zip')) {
        const zipImages = await extractImagesFromZipFile(file);
        collected.push(...zipImages);
      } else if (isSupportedImageFileName(fullPath)) {
        // Preserve folder path in label if nested
        const namedFile =
          pathPrefix.length > 0
            ? new File([file], fullPath, { type: file.type || inferMimeTypeFromName(fullPath) })
            : file;
        collected.push(namedFile);
      }
    } else if (entry.isDirectory) {
      const dirEntry = entry as FileSystemDirectoryEntry;
      const reader = dirEntry.createReader();
      const nextPrefix = pathPrefix ? `${pathPrefix}/${dirEntry.name}` : dirEntry.name;

      const readAllEntries = async (): Promise<FileSystemEntry[]> => {
        const all: FileSystemEntry[] = [];
        while (true) {
          const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => {
            reader.readEntries(resolve, reject);
          });
          if (batch.length === 0) break;
          all.push(...batch);
        }
        return all;
      };

      const children = await readAllEntries();
      for (const child of children) {
        await traverseEntry(child, nextPrefix);
      }
    }
  }

  const items = Array.from(dataTransfer.items || []);
  const entries = items
    .map((item) => (typeof item.webkitGetAsEntry === 'function' ? item.webkitGetAsEntry() : null))
    .filter((e): e is FileSystemEntry => e !== null);

  if (entries.length > 0) {
    for (const entry of entries) {
      await traverseEntry(entry);
    }
    return collected;
  }

  return normalizeSelectedFileList(dataTransfer.files);
}

/**
 * Normalizes a FileList (from multi-file input, webkitdirectory folder input, or .zip input)
 * into flat supported image File objects.
 */
export async function normalizeSelectedFileList(fileList: FileList | null): Promise<File[]> {
  if (!fileList || fileList.length === 0) return [];
  const out: File[] = [];

  for (const file of Array.from(fileList)) {
    const relPath =
      (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
    if (file.name.toLowerCase().endsWith('.zip')) {
      const zipImages = await extractImagesFromZipFile(file);
      out.push(...zipImages);
    } else if (isSupportedImageFileName(relPath)) {
      if (relPath !== file.name) {
        out.push(
          new File([file], relPath, { type: file.type || inferMimeTypeFromName(relPath) })
        );
      } else {
        out.push(file);
      }
    }
  }
  return out;
}

/**
 * Builds an RFC 7946 GeoJSON FeatureCollection with ONE Point geometry per image,
 * placing all detected spider-web crack & anomaly details inside each image feature's properties.
 */
export function buildMergedGeoJsonPoints(
  results: InspectionResult[],
  onlyImagesWithCracks = false
): GeoJsonFeatureCollection {
  const features: GeoJsonPointFeature[] = [];
  let skippedNoGeo = 0;

  for (const res of results) {
    const imgFeature = res.geoJson?.features?.[0];
    if (!imgFeature) continue;
    if (imgFeature.properties.crs_mode === 'PIXEL_LOCAL') {
      skippedNoGeo++; // pixel coords don't belong in an EPSG:3857 layer
      continue;
    }
    if (onlyImagesWithCracks && imgFeature.properties.total_anomaly_count === 0) {
      continue;
    }
    features.push(imgFeature);
  }

  const imagesWithSpiderCount = features.filter(
    (f) => f.properties.has_spider_web_crack
  ).length;
  const totalSpiderCount = features.reduce(
    (acc, f) => acc + f.properties.spider_web_crack_count,
    0
  );
  const totalEdgeSpiderCount = features.reduce(
    (acc, f) => acc + f.properties.edge_spider_web_crack_count,
    0
  );

  return {
    type: 'FeatureCollection',
    name: 'pv_image_crack_inspection',
    crs: {
      type: 'name',
      properties: {
        name: 'urn:ogc:def:crs:EPSG::3857',
      },
    },
    metadata: {
      crs_epsg: 'EPSG:3857',
      crs_label: 'EPSG:3857 - WGS 84 / Pseudo-Mercator',
      generated_at: new Date().toISOString(),
      total_images: results.length,
      total_image_points: features.length,
      images_with_spider_web_cracks: imagesWithSpiderCount,
      total_spider_web_cracks: totalSpiderCount,
      total_edge_spider_web_cracks: totalEdgeSpiderCount,
      skipped_without_georeference: skippedNoGeo,
      coordinate_system_note:
        'One GeoJSON Point [x_easting_m, y_northing_m] per image in EPSG:3857 - WGS 84 / Pseudo-Mercator (from GeoTIFF / DJI RTK XMP / EXIF GPS or configured Site Reference), with all crack detections stored in properties.',
    },
    features,
  };
}

export function triggerFileDownload(filename: string, content: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
