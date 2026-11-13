/** Select a deterministic held-out image subset whose box count approximates the requested ratio. */
export type BoxCountRecord = { imageFile: string; boxes: readonly unknown[] };

export function chooseValidationImageFiles<T extends BoxCountRecord>(records: readonly T[], validationRatio = 0.2): Set<string> {
  if (records.length < 2) throw new Error('At least two distinct source images are required for a train/validation split.');
  const ratio = Math.min(0.5, Math.max(0.05, validationRatio));
  const validationCount = Math.max(1, Math.min(records.length - 1, Math.ceil(records.length * ratio)));
  const totalBoxes = records.reduce((sum, record) => sum + record.boxes.length, 0);
  const targetBoxes = totalBoxes * ratio;
  const remaining = [...records].sort((a, b) => a.imageFile.localeCompare(b.imageFile));
  const selected = new Set<string>();
  let selectedBoxes = 0;
  while (selected.size < validationCount) {
    const slotsLeft = validationCount - selected.size;
    const targetForNext = (targetBoxes - selectedBoxes) / slotsLeft;
    let bestIndex = 0;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (let index = 0; index < remaining.length; index++) {
      const count = remaining[index].boxes.length;
      const distance = Math.abs(count - targetForNext);
      if (distance < bestDistance || (distance === bestDistance && remaining[index].imageFile < remaining[bestIndex].imageFile)) {
        bestDistance = distance;
        bestIndex = index;
      }
    }
    const [chosen] = remaining.splice(bestIndex, 1);
    selected.add(chosen.imageFile);
    selectedBoxes += chosen.boxes.length;
  }
  return selected;
}
