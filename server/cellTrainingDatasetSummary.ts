/** Typed, testable summary returned by the YOLO dataset builder. */
export type CellTrainingDatasetSummaryInput = {
  yamlPath: string;
  images: number;
  boxes: number;
  trainImages: number;
  validationImages: number;
  trainBoxes: number;
  validationBoxes: number;
};

export function buildCellTrainingDatasetSummary(input: CellTrainingDatasetSummaryInput): CellTrainingDatasetSummaryInput {
  return {
    yamlPath: input.yamlPath,
    images: input.images,
    boxes: input.boxes,
    trainImages: input.trainImages,
    validationImages: input.validationImages,
    trainBoxes: input.trainBoxes,
    validationBoxes: input.validationBoxes,
  };
}
