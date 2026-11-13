import test from 'node:test';
import assert from 'node:assert/strict';
import { markedAnomalyStyle } from '../src/utils/markedImage.ts';

test('marked-image export colours match inspection overlays', () => {
  assert.equal(markedAnomalyStyle({ category: 'spider_web_crack', outerRing: true, confidence: 0.99 }).stroke, '#ef4444');
  assert.equal(markedAnomalyStyle({ category: 'spider_web_crack', confidence: 0.89 }).stroke, '#facc15');
  assert.equal(markedAnomalyStyle({ category: 'spider_web_crack', confidence: 0.95 }).stroke, '#e879f9');
  assert.equal(markedAnomalyStyle({ category: 'spider_web_crack', confidence: 0.5 }, 40).stroke, '#e879f9');
  assert.equal(markedAnomalyStyle({ category: 'spider_web_crack', source: 'user', outerRing: true, confidence: 0.2 }).stroke, '#22c55e');
  assert.equal(markedAnomalyStyle({ category: 'impact_point', confidence: 0.95 }).stroke, '#f59e0b');
  assert.equal(markedAnomalyStyle({ category: 'linear_fracture', confidence: 0.95 }).stroke, '#fb923c');
  assert.equal(markedAnomalyStyle({ category: 'surface_residue', confidence: 0.95 }).stroke, '#38bdf8');
});
