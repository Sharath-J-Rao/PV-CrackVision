import type { DefectCategory, InspectionResult } from '../types/inspection';

const COLORS: Record<DefectCategory, string> = {
  spider_web_crack: '#ef4444',
  impact_point: '#f59e0b',
  linear_fracture: '#fb923c',
  surface_residue: '#38bdf8',
};
const USER_COLOR = '#22c55e';

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
 * Line widths / fonts scale with the image so marks look the same on a 5 MP and a 45 MP frame.
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

  const u = Math.max(W0, H0) / 900; // same base unit as the on-screen overlay
  const font = (px: number, bold = true) => `${bold ? '600 ' : ''}${Math.round(px)}px ui-monospace, Menlo, Consolas, monospace`;

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

  ctx.textBaseline = 'middle';
  for (const a of result.anomalies) {
    const color = a.source === 'user' ? USER_COLOR : COLORS[a.category];
    const pad = u * 4;
    const { left, top, width, height } = a.pixelBox;
    ctx.save();
    ctx.lineWidth = u * 2.4;
    ctx.strokeStyle = color;
    ctx.fillStyle = color + '24';
    ctx.fillRect(left - pad, top - pad, width + pad * 2, height + pad * 2);
    ctx.strokeRect(left - pad, top - pad, width + pad * 2, height + pad * 2);

    // crosshair on the fracture origin
    const cl = u * 14;
    ctx.lineWidth = u * 1.8;
    ctx.beginPath();
    ctx.moveTo(a.pointPx.x - cl, a.pointPx.y); ctx.lineTo(a.pointPx.x + cl, a.pointPx.y);
    ctx.moveTo(a.pointPx.x, a.pointPx.y - cl); ctx.lineTo(a.pointPx.x, a.pointPx.y + cl);
    ctx.stroke();

    // label tag above the box
    const label = a.source === 'user' ? `${a.id} user-marked` : `${a.id} ${Math.round(a.confidence * 100)}%`;
    ctx.font = font(u * 12);
    const tw = ctx.measureText(label).width + u * 10, th = u * 18;
    const tx = Math.max(0, Math.min(W0 - tw, left - pad));
    const ty = top - pad - th >= 0 ? top - pad - th : top + height + pad;
    ctx.fillStyle = color;
    ctx.fillRect(tx, ty, tw, th);
    ctx.fillStyle = '#0b0e14';
    ctx.fillText(label, tx + u * 5, ty + th / 2);
    ctx.restore();
  }

  // header banner: file name + verdict
  const n = result.anomalies.length;
  const head = `${result.fileName}  |  ${n ? `${n} issue${n === 1 ? '' : 's'} marked` : 'no issues detected'}`;
  ctx.font = font(u * 14);
  const bw = Math.min(W0, ctx.measureText(head).width + u * 20), bh = u * 26;
  ctx.fillStyle = n ? 'rgba(127, 29, 29, 0.88)' : 'rgba(6, 78, 59, 0.88)';
  ctx.fillRect(0, 0, bw, bh);
  ctx.fillStyle = '#ffffff';
  ctx.fillText(head, u * 10, bh / 2, bw - u * 14);

  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not encode the marked image.'))), 'image/jpeg', opts.quality ?? 0.92)
  );
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
