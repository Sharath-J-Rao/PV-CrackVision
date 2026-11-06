/**
 * Training-data extractor. Uses the PRODUCTION code path so train and inference features are identical:
 *   image -> runClassicalStage1 (module boxes) -> per-module crop resized to a canonical width ->
 *   210px edge-ring windows -> windowFeatures() from server/crackModel.ts.
 * Usage: tsx extract.mts <projRoot> <outDir> <img1> <img2> ...
 * Writes <outDir>/windows.json (+ canonical module JPEGs used for labelling montages).
 */
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';

const [root, outDir, ...imgs] = process.argv.slice(2);
const { runClassicalStage1 } = await import(path.join(root, 'server/cvPipeline.ts'));
const { windowFeatures, edgeRingWindows, CANON_W } = await import(path.join(root, 'server/crackModel.ts'));

const D = 210;
const STRIDE = 105;
fs.mkdirSync(outDir, { recursive: true });
const out: any[] = [];

for (const f of imgs) {
  const key = path.basename(path.dirname(f)) + '/' + path.basename(f);
  const { data, info } = await sharp(fs.readFileSync(f)).rotate().toColorspace('srgb').removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const s1 = await runClassicalStage1(data, info.width, info.height);
  let mi = 0;
  for (const m of s1.moduleRegions) {
    const b = m.pixelBox;
    if (m.isFallback || b.width < 180 || b.height < 110) { mi++; continue; }
    const sc = CANON_W / b.width;
    const cw = CANON_W, ch = Math.max(D, Math.round(b.height * sc));
    const { data: mod } = await sharp(data, { raw: { width: info.width, height: info.height, channels: 3 } })
      .extract({ left: b.left, top: b.top, width: b.width, height: b.height })
      .resize(cw, ch, { fit: 'fill', kernel: 'lanczos3' }).raw().toBuffer({ resolveWithObject: true });
    const tag = key.replace(/[\/.]/g, '_') + '_m' + mi;
    await sharp(mod, { raw: { width: cw, height: ch, channels: 3 } }).jpeg({ quality: 95 }).toFile(path.join(outDir, tag + '.jpg'));
    const win = Buffer.allocUnsafe(D * D * 3);
    for (const { x, y } of edgeRingWindows(mod, cw, ch, D, STRIDE)) {
      for (let r = 0; r < D; r++) mod.copy(win, r * D * 3, ((y + r) * cw + x) * 3, ((y + r) * cw + x + D) * 3);
      out.push({ img: key, mi, tag, x, y, cw, ch, scale: sc, f: windowFeatures(win, D, 8) });
    }
    mi++;
  }
  console.log(key, 'modules', s1.moduleRegions.length, 'windows so far', out.length);
}
fs.writeFileSync(path.join(outDir, 'windows.json'), JSON.stringify(out));
