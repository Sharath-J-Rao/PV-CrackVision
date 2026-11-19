import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const isolatedData = fs.mkdtempSync(path.join(os.tmpdir(), 'pv-review-reset-'));
process.env.PV_CRACKVISION_DATA_DIR = isolatedData;
const { saveReview, loadReview, clearSavedCellReviewData } = await import('./reviewStore.ts');

try {
  test('cell reset removes saved cell boxes but keeps crack history marked stale/unconfirmed', () => {
    const withBoth = 'a'.repeat(40);
    const cellOnly = 'b'.repeat(40);
    saveReview(withBoth, { fileName: 'both.jpg', cell: { boxes: [{ left: 1, top: 1, width: 10, height: 10 }] }, crack: { result: { anomalies: [] }, crackReviewed: true, crackStale: false } });
    saveReview(cellOnly, { fileName: 'cells.jpg', cell: { boxes: [{ left: 2, top: 2, width: 10, height: 10 }] } });

    assert.equal(clearSavedCellReviewData(), 2);
    const retained = loadReview(withBoth);
    assert.equal(retained.cell, undefined);
    assert.equal(retained.crack.crackReviewed, false);
    assert.equal(retained.crack.crackStale, true);
    assert.equal(loadReview(cellOnly), null);
  });
} finally {
  fs.rmSync(isolatedData, { recursive: true, force: true });
}
