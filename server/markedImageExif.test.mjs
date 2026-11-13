import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import ExifReader from 'exifreader';
import { readJpegMetadataSegments, normaliseExifSegment, injectJpegMetadata, readPngExifSegment, MARK_COLOR, MARK_FONT_PX, markedAnomalyStyle } from '../src/utils/markedImage.ts';

async function sourceJpeg() {
  // 60x40 source stored sideways with EXIF Orientation 6 plus camera tags and GPS.
  return sharp({ create: { width: 60, height: 40, channels: 3, background: { r: 90, g: 90, b: 90 } } })
    .withExif({ IFD0: { Make: 'DJI', Model: 'M3T', Orientation: '6' }, IFD3: { GPSLatitudeRef: 'N', GPSLatitude: '12/1 30/1 0/1', GPSLongitudeRef: 'E', GPSLongitude: '77/1 36/1 0/1' } })
    .jpeg().toBuffer();
}

test('marked image colours match the inspection canvas and use the configured low-confidence threshold', () => {
  assert.equal(MARK_COLOR, '#ef4444');
  assert.equal(MARK_FONT_PX, 10);
  assert.equal(markedAnomalyStyle({ category: 'spider_web_crack', outerRing: true, confidence: 0.99 }).stroke, '#ef4444');
  assert.equal(markedAnomalyStyle({ category: 'spider_web_crack', confidence: 0.89 }).stroke, '#facc15');
  assert.equal(markedAnomalyStyle({ category: 'spider_web_crack', confidence: 0.95 }).stroke, '#e879f9');
  assert.equal(markedAnomalyStyle({ category: 'spider_web_crack', confidence: 0.5 }, 40).stroke, '#e879f9');
  assert.equal(markedAnomalyStyle({ category: 'spider_web_crack', source: 'user', outerRing: true, confidence: 0.2 }).stroke, '#22c55e');
  assert.equal(markedAnomalyStyle({ category: 'impact_point', confidence: 0.95 }).stroke, '#f59e0b');
  assert.equal(markedAnomalyStyle({ category: 'linear_fracture', confidence: 0.95 }).stroke, '#fb923c');
  assert.equal(markedAnomalyStyle({ category: 'surface_residue', confidence: 0.95 }).stroke, '#38bdf8');
});

test('EXIF segment is found, orientation reset to 1 and dimensions patched', async () => {
  const src = await sourceJpeg();
  const segments = readJpegMetadataSegments(new Uint8Array(src));
  assert.ok(segments.length >= 1);
  const fixed = normaliseExifSegment(segments[0], 40, 60);
  const tags = ExifReader.load(Buffer.concat([Buffer.from([0xff, 0xd8]), Buffer.from(fixed), Buffer.from([0xff, 0xd9])]));
  assert.equal(tags.Orientation.value, 1);
  assert.equal(tags.Make.description, 'DJI');
  assert.equal(tags.Model.description, 'M3T');
});

const JFIF = Buffer.from('ffe000104a46494600010100000100010000', 'hex'); // APP0 JFIF, as browser canvas encoders write it

for (const withJfif of [true, false]) {
  test(`metadata persists through a re-encoded marked JPEG (${withJfif ? 'with' : 'without'} JFIF APP0)`, async () => {
    const src = await sourceJpeg();
    let marked = await sharp({ create: { width: 40, height: 60, channels: 3, background: { r: 10, g: 10, b: 10 } } }).jpeg().toBuffer(); // no EXIF
    if (withJfif) marked = Buffer.concat([marked.subarray(0, 2), JFIF, marked.subarray(2)]);
    const out = Buffer.from(await (await injectJpegMetadata(new Blob([marked], { type: 'image/jpeg' }), readJpegMetadataSegments(new Uint8Array(src)), 40, 60)).arrayBuffer());
    if (withJfif) assert.equal(out[2], 0xff), assert.equal(out[3], 0xe0); // JFIF stays first
    const tags = ExifReader.load(out, { expanded: true });
    assert.equal(tags.exif.Make.description, 'DJI');
    const srcTags = ExifReader.load(src, { expanded: true });
    assert.equal(tags.exif.Orientation.value, srcTags.exif.Orientation.value, 'EXIF is a straight copy of the original (orientation untouched)');
    const srcSeg = readJpegMetadataSegments(new Uint8Array(src))[0];
    assert.ok(Buffer.from(out).includes(Buffer.from(srcSeg)), 'original Exif segment bytes appear unchanged in the marked JPEG');
    assert.ok(tags.gps && Math.abs(tags.gps.Latitude - 12.5) < 1e-6 && Math.abs(tags.gps.Longitude - 77.6) < 1e-6, 'GPS persists');
    const info = await sharp(out).metadata(); // still a valid, decodable JPEG, not rotated again
    assert.equal(info.width, 40); assert.equal(info.height, 60);
    await sharp(out).raw().toBuffer();
  });
}

test('JPEG without metadata is left untouched', async () => {
  const plain = await sharp({ create: { width: 8, height: 8, channels: 3, background: '#fff' } }).jpeg().toBuffer();
  assert.deepEqual(readJpegMetadataSegments(new Uint8Array(plain)), []);
});

test('PNG eXIf chunk is converted to a JPEG Exif segment', () => {
  const tiff = Buffer.from('49492a0008000000' + '0100' + '0f01' + '0200' + '04000000' + '44004a49' + '00000000', 'hex'); // II, IFD0 with Make="DJI\0"
  const chunk = (type, data) => { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); return Buffer.concat([len, Buffer.from(type), data, Buffer.alloc(4)]); };
  const png = Buffer.concat([Buffer.from('89504e470d0a1a0a', 'hex'), chunk('IHDR', Buffer.alloc(13)), chunk('eXIf', tiff), chunk('IEND', Buffer.alloc(0))]);
  const seg = readPngExifSegment(new Uint8Array(png));
  assert.ok(seg && seg[0] === 0xff && seg[1] === 0xe1);
  assert.equal(Buffer.from(seg.subarray(4, 10)).toString('latin1'), 'Exif\0\0');
});
