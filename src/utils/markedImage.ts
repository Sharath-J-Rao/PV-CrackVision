import type { InspectionResult } from '../types/inspection';

/** v1.4.9: every marked issue box is red with a thin border (user-added and detected alike). */
export const MARK_COLOR = '#ff0000';
/** v1.4.12.4: red box line width in output pixels (was 2 in v1.4.11.7, 1 before). */
export const MARK_LINE_PX = 12;
/** Crosshair at the crack origin stays thin so it does not hide the crack. */
export const MARK_CROSS_PX = 2;
/** v1.4.9: mark text is Arial 10 px in the exported image, with no background behind it. */
export const MARK_FONT_PX = 10;

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Could not load the image to mark.'));
    img.src = src;
  });
}

/**
 * Draws the detections onto the original image and returns a JPEG.
 * maxSide limits the longest edge (previews); omit it for a full-resolution marked image.
 * Issue boxes are red 1 px lines and labels are Arial 10 px with no background at every export size (v1.4.9); the original
 * EXIF / XMP / IPTC metadata is carried over into the marked JPEG.
 */
export async function renderMarkedImage(
  imageSrc: string,
  result: InspectionResult,
  opts: { maxSide?: number; quality?: number; showModules?: boolean } = {}
): Promise<Blob> {
  const img = await loadImage(imageSrc);
  const W0 = result.imageWidth || img.naturalWidth, H0 = result.imageHeight || img.naturalHeight;
  const k = opts.maxSide ? Math.min(1, opts.maxSide / Math.max(W0, H0)) : 1;
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(W0 * k));
  canvas.height = Math.max(1, Math.round(H0 * k));
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas is not available in this browser.');
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  ctx.scale(k, k); // everything below is in original-image pixels

  const u = Math.max(W0, H0) / 900; // same base unit as the on-screen overlay (cell grid only)
  const px = 1 / k; // one OUTPUT pixel expressed in original-image pixels, so a thin line stays 1 px and Arial 10 stays 10 px

  if (opts.showModules !== false) {
    ctx.save();
    ctx.lineWidth = Math.max(0.7, u * 0.9);
    for (const region of result.moduleRegions) {
      for (const cell of region.cellGrid?.cells ?? []) {
        ctx.strokeStyle = cell.partial ? 'rgba(250, 204, 21, 0.8)' : 'rgba(34, 211, 238, 0.62)';
        ctx.strokeRect(cell.pixelBox.left, cell.pixelBox.top, cell.pixelBox.width, cell.pixelBox.height);
      }
    }
    ctx.restore();
  }

  ctx.textBaseline = 'alphabetic';
  for (const a of result.anomalies) {
    const pad = 2 * px;
    const { left, top, width, height } = a.pixelBox;
    ctx.save();
    // thin red box, no fill
    ctx.lineWidth = MARK_LINE_PX * px; // v1.4.11.7: thicker red line (was 1 px)
    ctx.strokeStyle = MARK_COLOR;
    ctx.strokeRect(left - pad, top - pad, width + pad * 2, height + pad * 2);

    // thin crosshair on the fracture origin
    ctx.lineWidth = MARK_CROSS_PX * px;
    const cl = 7 * px;
    ctx.beginPath();
    ctx.moveTo(a.pointPx.x - cl, a.pointPx.y); ctx.lineTo(a.pointPx.x + cl, a.pointPx.y);
    ctx.moveTo(a.pointPx.x, a.pointPx.y - cl); ctx.lineTo(a.pointPx.x, a.pointPx.y + cl);
    ctx.stroke();

    // label: Arial 10, red, no background; above the box, or below it when there is no room above
    const label = a.source === 'user' ? `${a.id} user-marked` : `${a.id} ${Math.round(a.confidence * 100)}%`;
    ctx.font = `${MARK_FONT_PX * px}px Arial`;
    ctx.fillStyle = MARK_COLOR;
    const tw = ctx.measureText(label).width;
    const tx = Math.max(0, Math.min(W0 - tw, left - pad));
    const above = top - pad - 3 * px >= MARK_FONT_PX * px;
    const ty = above ? top - pad - 3 * px : Math.min(H0 - 2 * px, top + height + pad + (MARK_FONT_PX + 1) * px);
    ctx.fillText(label, tx, ty);
    ctx.restore();
  }

  // header text: file name + verdict (Arial 10, no background)
  const n = result.anomalies.length;
  const head = `${result.fileName}  |  ${n ? `${n} issue${n === 1 ? '' : 's'} marked` : 'no issues detected'}`;
  ctx.font = `${MARK_FONT_PX * px}px Arial`;
  ctx.fillStyle = MARK_COLOR;
  ctx.fillText(head, 6 * px, 14 * px, Math.max(1, W0 - 12 * px));

  return encodeMarkedJpeg(canvas, imageSrc, opts.quality);
}

/** JPEG-encode a finished marked canvas and carry the source EXIF / XMP / IPTC across (shared by every marked-image renderer). */
async function encodeMarkedJpeg(canvas: HTMLCanvasElement, imageSrc: string, quality = 0.92): Promise<Blob> {
  const jpeg = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not encode the marked image.'))), 'image/jpeg', quality)
  );
  // v1.4.9: canvas re-encoding drops all metadata, so carry the original EXIF / XMP / IPTC across to the marked JPEG.
  try {
    const segments = await sourceMetadataSegments(imageSrc);
    if (segments.length) return await injectJpegMetadata(jpeg, segments, canvas.width, canvas.height);
  } catch {
    /* metadata is best-effort; never block the marked image */
  }
  return jpeg;
}

/** Cell-stage marking colours: model output vs. operator-reviewed boxes are visually distinct. */
export const CELL_DETECTED_COLOR = '#facc15';
export const CELL_REVIEWED_COLOR = '#22d3ee';

/**
 * v1.4.9: Marked image for the CELL stage (Cell Annotation): the cell boxes only, no crack findings.
 * `boxes` are in the cell-pass frame (`size`); `kind` picks the colour and header wording.
 */
export async function renderCellBoxesImage(
  imageSrc: string,
  size: { width: number; height: number },
  boxes: ReadonlyArray<{ left: number; top: number; width: number; height: number }>,
  fileName: string,
  kind: 'detected' | 'reviewed',
  opts: { maxSide?: number; quality?: number } = {}
): Promise<Blob> {
  const img = await loadImage(imageSrc);
  const W0 = size.width || img.naturalWidth, H0 = size.height || img.naturalHeight;
  const k = opts.maxSide ? Math.min(1, opts.maxSide / Math.max(W0, H0)) : 1;
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(W0 * k));
  canvas.height = Math.max(1, Math.round(H0 * k));
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas is not available in this browser.');
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  ctx.scale(k, k); // everything below is in cell-pass pixels
  const px = 1 / k;
  const color = kind === 'detected' ? CELL_DETECTED_COLOR : CELL_REVIEWED_COLOR;
  ctx.lineWidth = Math.max(px, (Math.max(W0, H0) / 900) * 0.6);
  ctx.strokeStyle = color;
  for (const b of boxes) ctx.strokeRect(b.left, b.top, b.width, b.height);
  ctx.font = `${MARK_FONT_PX * px}px Arial`;
  ctx.fillStyle = color;
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(`${fileName}  |  ${boxes.length} cell box${boxes.length === 1 ? '' : 'es'} (${kind === 'detected' ? 'model detections' : 'operator-reviewed'})`, 6 * px, 14 * px, Math.max(1, W0 - 12 * px));
  return encodeMarkedJpeg(canvas, imageSrc, opts.quality);
}

// ---------------------------------------------------------------------------------------------------------------------
// EXIF persistence (v1.4.9)
// ---------------------------------------------------------------------------------------------------------------------

const ascii = (bytes: Uint8Array, start: number, text: string): boolean => {
  if (start + text.length > bytes.length) return false;
  for (let i = 0; i < text.length; i++) if (bytes[start + i] !== text.charCodeAt(i)) return false;
  return true;
};
const EXIF_ID = 'Exif\0\0';
const XMP_ID = 'http://ns.adobe.com/xap/1.0/\0';
const PS_ID = 'Photoshop 3.0\0';

/** Metadata-bearing JPEG segments (marker + length + payload) of a source JPEG: Exif APP1, XMP APP1, IPTC APP13. */
export function readJpegMetadataSegments(bytes: Uint8Array): Uint8Array[] {
  const out: Uint8Array[] = [];
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) return out;
  let i = 2;
  while (i + 4 <= bytes.length && bytes[i] === 0xff) {
    const marker = bytes[i + 1];
    if (marker === 0xff) { i++; continue; }
    if (marker === 0xda || marker === 0xd9) break; // start of scan / end of image
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) { i += 2; continue; }
    const len = (bytes[i + 2] << 8) | bytes[i + 3];
    if (len < 2 || i + 2 + len > bytes.length) break;
    const payload = i + 4;
    // v1.4.12.2: copy EVERY metadata segment byte-for-byte (all APP1 incl. Exif/XMP/extended XMP, APP3-APP13, APP15).
    // Left out on purpose: APP0 (JFIF, written by the encoder), APP2 (MPF offsets / ICC), APP14 (Adobe colour flag) because they
    // would be wrong or change decoding of the re-encoded pixels.
    if ((marker === 0xe1 || (marker >= 0xe3 && marker <= 0xed) || marker === 0xef)) out.push(bytes.slice(i, i + 2 + len));
    i += 2 + len;
  }
  return out;
}

/** A PNG eXIf chunk is a bare TIFF block; wrap it as a JPEG Exif APP1 segment. */
export function readPngExifSegment(bytes: Uint8Array): Uint8Array | null {
  if (bytes.length < 16 || bytes[0] !== 0x89 || bytes[1] !== 0x50 || bytes[2] !== 0x4e || bytes[3] !== 0x47) return null;
  let i = 8;
  while (i + 12 <= bytes.length) {
    const len = ((bytes[i] << 24) | (bytes[i + 1] << 16) | (bytes[i + 2] << 8) | bytes[i + 3]) >>> 0;
    const type = String.fromCharCode(bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]);
    if (type === 'eXIf' && len > 0 && len + 8 <= 0xffff && i + 8 + len <= bytes.length) {
      const seg = new Uint8Array(4 + 6 + len);
      seg[0] = 0xff; seg[1] = 0xe1;
      const total = 2 + 6 + len;
      seg[2] = total >> 8; seg[3] = total & 0xff;
      for (let c = 0; c < 6; c++) seg[4 + c] = EXIF_ID.charCodeAt(c);
      seg.set(bytes.subarray(i + 8, i + 8 + len), 10);
      return seg;
    }
    if (type === 'IEND') return null;
    i += 12 + len;
  }
  return null;
}

/**
 * The canvas already holds upright pixels (the browser applies the EXIF orientation when drawing), so the copied EXIF must say
 * "Orientation = 1" and its pixel dimensions must match the marked image; otherwise viewers would rotate it a second time.
 * Returns an edited copy; non-Exif segments are returned unchanged.
 */
export function normaliseExifSegment(segment: Uint8Array, width: number, height: number): Uint8Array<ArrayBuffer> {
  const seg = new Uint8Array(segment);
  if (!(seg[0] === 0xff && seg[1] === 0xe1 && ascii(seg, 4, EXIF_ID))) return seg;
  const tiff = 10; // offset of the TIFF header inside the segment
  const little = seg[tiff] === 0x49 && seg[tiff + 1] === 0x49;
  if (!little && !(seg[tiff] === 0x4d && seg[tiff + 1] === 0x4d)) return seg;
  const u16 = (o: number) => (little ? seg[o] | (seg[o + 1] << 8) : (seg[o] << 8) | seg[o + 1]);
  const u32 = (o: number) => (little
    ? (seg[o] | (seg[o + 1] << 8) | (seg[o + 2] << 16) | (seg[o + 3] << 24)) >>> 0
    : ((seg[o] << 24) | (seg[o + 1] << 16) | (seg[o + 2] << 8) | seg[o + 3]) >>> 0);
  const put16 = (o: number, v: number) => { if (little) { seg[o] = v & 0xff; seg[o + 1] = (v >> 8) & 0xff; } else { seg[o] = (v >> 8) & 0xff; seg[o + 1] = v & 0xff; } };
  const put32 = (o: number, v: number) => {
    if (little) { seg[o] = v & 0xff; seg[o + 1] = (v >>> 8) & 0xff; seg[o + 2] = (v >>> 16) & 0xff; seg[o + 3] = (v >>> 24) & 0xff; }
    else { seg[o] = (v >>> 24) & 0xff; seg[o + 1] = (v >>> 16) & 0xff; seg[o + 2] = (v >>> 8) & 0xff; seg[o + 3] = v & 0xff; }
  };
  const entries = (ifd: number, visit: (entry: number, tag: number) => void) => {
    const start = tiff + ifd;
    if (ifd < 8 || start + 2 > seg.length) return;
    const count = u16(start);
    for (let e = 0; e < count; e++) {
      const entry = start + 2 + e * 12;
      if (entry + 12 > seg.length) return;
      visit(entry, u16(entry));
    }
  };
  const setNumber = (entry: number, value: number) => { // type 3 = SHORT, 4 = LONG; a single value is stored inline
    const type = u16(entry + 2);
    if (type === 3) put16(entry + 8, value);
    else if (type === 4) put32(entry + 8, value);
  };
  let exifIfd = 0;
  entries(u32(tiff + 4), (entry, tag) => {
    if (tag === 0x0112) setNumber(entry, 1);
    else if (tag === 0x8769) exifIfd = u32(entry + 8);
  });
  if (exifIfd) entries(exifIfd, (entry, tag) => {
    if (tag === 0xa002) setNumber(entry, width);
    else if (tag === 0xa003) setNumber(entry, height);
  });
  return seg;
}

const metadataCache = new Map<string, Promise<Uint8Array[]>>();
function sourceMetadataSegments(imageSrc: string): Promise<Uint8Array[]> {
  let found = metadataCache.get(imageSrc);
  if (!found) {
    found = fetch(imageSrc).then((r) => r.arrayBuffer()).then((buf) => {
      const bytes = new Uint8Array(buf);
      const jpeg = readJpegMetadataSegments(bytes);
      if (jpeg.length) return jpeg;
      const png = readPngExifSegment(bytes);
      return png ? [png] : [];
    }).catch(() => [] as Uint8Array[]);
    metadataCache.set(imageSrc, found);
    if (metadataCache.size > 200) metadataCache.delete(metadataCache.keys().next().value as string);
  }
  return found;
}

/** Insert the source metadata segments straight after SOI (and after a JFIF APP0 if the encoder wrote one). */
export async function injectJpegMetadata(jpeg: Blob, segments: Uint8Array[], width: number, height: number): Promise<Blob> {
  const bytes = new Uint8Array(await jpeg.arrayBuffer());
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) return jpeg;
  let at = 2;
  if (bytes[2] === 0xff && bytes[3] === 0xe0) at = 4 + ((bytes[4] << 8) | bytes[5]); // JFIF APP0: marker (2) + length field value (includes its own 2 bytes)
  const parts: BlobPart[] = [new Uint8Array(bytes.subarray(0, at))];
  for (const segment of segments) parts.push(new Uint8Array(segment)); // v1.4.12.2: straight copy, no edits (width/height kept for API compatibility)
  parts.push(new Uint8Array(bytes.subarray(at)));
  return new Blob(parts, { type: 'image/jpeg' });
}

export const markedFileName = (label: string): string =>
  `${label.replace(/\.[a-z0-9]+$/i, '').replace(/[^a-zA-Z0-9_-]+/g, '_')}_marked.jpg`;

export function downloadBlob(name: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
