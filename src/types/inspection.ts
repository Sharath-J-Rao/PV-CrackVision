export type DefectCategory =
  | 'spider_web_crack'
  | 'impact_point'
  | 'linear_fracture'
  | 'surface_residue';

export type SeverityLevel = 'critical' | 'moderate' | 'minor' | 'informational';

export type EdgeSide =
  | 'bottom-left'
  | 'top-left'
  | 'bottom-right'
  | 'top-right'
  | 'left-edge'
  | 'right-edge'
  | 'top-edge'
  | 'bottom-edge'
  | 'interior';

export type CrsMode =
  | 'EPSG3857_RTK'
  | 'EPSG3857_EXIF'
  | 'EPSG3857_GEOTIFF'
  | 'EPSG3857_METADATA'
  | 'EPSG3857_SITE_REF'
  | 'PIXEL_LOCAL';

export interface BoundingBox {
  /** Normalized 0..1 coordinates relative to full image */
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
}

export interface PixelBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** How the image centroid (the GeoJSON Point) was derived */
export type CentroidMethod =
  | 'geotiff_center'
  | 'lrf_target'
  | 'gimbal_offset'
  | 'camera_nadir'
  | 'metadata_json_nadir'
  | 'site_reference'
  | 'none';

export interface GeoReferenceInfo {
  /** Where the point came from; see CentroidMethod */
  centroidMethod: CentroidMethod;
  /** Raw camera (drone) position, kept for transparency when it differs from the centroid */
  cameraLat: number | null;
  cameraLon: number | null;
  hasEmbeddedGps: boolean;
  crsMode: CrsMode;
  epsgCode: 'EPSG:3857' | 'PIXEL_LOCAL';
  crsName: string;
  /** Single GeoJSON Point coordinate [x_easting_m, y_northing_m] in EPSG:3857 (or [center_px_x, -center_px_y]) for the image */
  imageGeoPoint: [number, number];
  epsg3857X: number | null;
  epsg3857Y: number | null;
  centerLat: number | null;
  centerLon: number | null;
  altitudeM: number | null;
  relativeAltitudeM: number | null;
  gimbalYawDeg: number;
  cameraModel: string | null;
  capturedAt: string | null;
  gsdMmPerPx: number;
}

export interface GeoJsonCrackAttributeItem {
  defect_id: string;
  category: DefectCategory;
  is_edge_crack: boolean;
  edge_side: EdgeSide;
  severity: SeverityLevel;
  confidence: number;
  title: string;
  description: string;
  module_index: string;
  cell_location: string;
  point_px: [number, number];
  bbox_px: [number, number, number, number];
  radial_branches: number;
  estimated_span_mm: number;
}

export interface GeoJsonPointProperties {
  image_name: string;
  image_dimensions_px: [number, number];
  crs: string;
  epsg_code: number | null;
  epsg_3857_xy: [number, number] | null;
  wgs84_lon_lat: [number, number] | null;
  centroid_method: CentroidMethod;
  camera_wgs84_lon_lat: [number, number] | null;
  crs_mode: CrsMode;
  status: 'DEFECT_DETECTED' | 'CLEAN';
  has_spider_web_crack: boolean;
  has_edge_spider_web_crack: boolean;
  spider_web_crack_count: number;
  edge_spider_web_crack_count: number;
  impact_point_count: number;
  linear_fracture_count: number;
  surface_residue_count: number;
  total_anomaly_count: number;
  max_severity: SeverityLevel | 'none';
  max_confidence: number;
  affected_modules: string;
  affected_cells: string;
  edge_sides_detected: string;
  crack_summary: string;
  altitude_m: number | null;
  relative_altitude_m: number | null;
  gimbal_yaw_deg: number;
  gsd_mm_per_px: number;
  camera_model: string | null;
  captured_at: string | null;
  analyzed_at: string;
  crack_details: GeoJsonCrackAttributeItem[];
}

export interface GeoJsonPointFeature {
  type: 'Feature';
  id: string;
  geometry: {
    type: 'Point';
    /** Single [x_3857, y_3857] Point in EPSG:3857 - WGS 84 / Pseudo-Mercator meters (or [center_x, -center_y] when PIXEL_LOCAL) */
    coordinates: [number, number];
  };
  properties: GeoJsonPointProperties;
}

export interface GeoJsonFeatureCollection {
  type: 'FeatureCollection';
  name: string;
  crs: {
    type: 'name';
    properties: {
      name: string;
    };
  };
  metadata: {
    crs_epsg: string;
    crs_label: string;
    generated_at: string;
    total_images: number;
    total_image_points: number;
    images_with_spider_web_cracks: number;
    total_spider_web_cracks: number;
    total_edge_spider_web_cracks: number;
    coordinate_system_note: string;
    /** Images left out of the points layer because they carry no real georeference (PIXEL_LOCAL) */
    skipped_without_georeference?: number;
  };
  features: GeoJsonPointFeature[];
}

export interface DetectedAnomaly {
  id: string;
  category: DefectCategory;
  isEdgeCrack: boolean;
  edgeSide: EdgeSide;
  severity: SeverityLevel;
  confidence: number;
  title: string;
  description: string;
  moduleIndex: string;
  cellLocation: string;
  radialBranches?: number;
  estimatedDiameterMm?: number;
  /** Pinpoint fracture origin/epicenter in native image pixels */
  pointPx: { x: number; y: number };
  box: BoundingBox;
  pixelBox: PixelBox;
  cropDataUrl?: string;
  edgeMapDataUrl?: string;
  tileId: string;
  offAxisEdgeScore: number;
  /** 'user' = issue added manually in the UI (also stored as a training positive) */
  source?: 'model' | 'user';
  /** v1.4.13.1: id of the training label stored for a manually added crack, so removing the crack also removes its label. */
  feedbackId?: string;
  /** v1.4.14: cell sits on the outer ring of its module's detected cell grid (drawn red; the usual false-positive zone). */
  outerRing?: boolean;
  /** v1.4.14: option-B shadow detector flags this cell too. Absent when the shadow model did not run. Informational only. */
  shadowAgree?: boolean;
  /** v1.4.14: which engine scored this cell ('classifier' = pretrained crop classifier). */
  engine?: 'classifier';
}

export interface TileDiagnostic {
  id: string;
  row: number;
  col: number;
  isPerimeterTile: boolean;
  pixelBox: PixelBox;
  box: BoundingBox;
  pvCoverageRatio: number;
  offAxisEdgeScore: number;
  impactKurtosisScore: number;
  inspectedByVlm: boolean;
  detectionsCount: number;
}

export interface SolarCellBox {
  /** Lattice coordinates are only local identifiers; the pixel box is the actual detected cell. */
  row: number;
  column: number;
  pixelBox: PixelBox;
  confidence: number;
  /** True when the image frame clips this cell. */
  partial: boolean;
}

export interface SolarCellGrid {
  /** Lattice dimensions are retained for stable row/column identifiers, not used to fabricate boxes. */
  rows: number;
  columns: number;
  /** Overall lattice bounds; cell geometry is the validated `cells` collection below. */
  pixelBox: PixelBox;
  /** Only individually validated solar-cell rectangles are included. */
  cells: SolarCellBox[];
  /** Shared cell pitch in original-image pixels, calibrated once per frame. */
  pitchX: number;
  pitchY: number;
  /** Confidence from edge periodicity and per-cell PV occupancy (0..1). */
  confidence: number;
  method: 'edge-map-periodicity' | 'gridline-periodicity' | 'legacy-fallback' | 'fixed-module-layout' | 'yolo-cell-detection';
}

export interface ModuleRegion {
  id: string;
  label: string;
  box: BoundingBox;
  pixelBox: PixelBox;
  /** Automatically detected solar-cell grid used by the annotation UI and all detection stages. */
  cellGrid?: SolarCellGrid;
  /** True for the whole-image placeholder used when no PV array was localized */
  isFallback?: boolean;
}

export interface InspectionConfig {
  tileSizePx: number;
  overlapRatio: number;
  sensitivity: 'high' | 'balanced' | 'strict';
  maxTilesToInspect: number;
  includeSurfaceResidue: boolean;
  contrastEnhancement: boolean;
  edgeFocusMode: boolean;
  /** Merged GeoJSON keeps only images where at least one issue was detected */
  onlyImagesWithIssuesInGeoJson: boolean;
  /** Drop/clip detections that fall on soil/grass outside the PV module */
  rejectGround: boolean;
  useSiteReferenceForNonGps: boolean;
  siteReferenceLat: number;
  siteReferenceLon: number;
}

export interface InspectionResult {
  imageId: string;
  fileName: string;
  imageWidth: number;
  imageHeight: number;
  megapixels: number;
  analyzedAt: string;
  processingTimeMs: number;
  modelUsed: string;
  /** Non-fatal notes, e.g. a bundled model failed its integrity check */
  warnings?: string[];
  geoReference: GeoReferenceInfo;
  pipelineSummary: {
    totalGridTiles: number;
    pvActiveTiles: number;
    vlmInspectedTiles: number;
    spiderWebCracksCount: number;
    edgeSpiderWebCracksCount: number;
    impactPointsCount: number;
    linearFracturesCount: number;
    surfaceResidueCount: number;
  };
  moduleRegions: ModuleRegion[];
  /** Direct YOLO cell boxes, when an in-app trained cell model is active. */
  cellDetectionBoxes?: PixelBox[];
  cellDetectionSource?: 'yolo11n' | 'classical';
  /**
   * Which cells the crack classifier inspected: the operator's Cell Annotation boxes, direct YOLO boxes from this pass,
   * the classical edge-periodicity grid (fallback), or none (no reviewed cells and fallback off).
   */
  cellBasis?: 'reviewed-cells' | 'yolo-cells' | 'classical-grid' | 'none';
  cellDetectionDiagnostics?: {
    imageWidth: number; imageHeight: number; modelInputWidth: number; modelInputHeight: number;
    inferencePasses: number; tilePasses: number; candidateCount: number; candidatesAboveThreshold: number;
    validBoxesBeforeMerge: number; boxesKept: number; maximumScore: number; confidenceThreshold: number;
    modelSizeBytes: number; modelModifiedAt: string; outputShape: string; candidatesLimited: number;
  };
  tiles: TileDiagnostic[];
  anomalies: DetectedAnomaly[];
  /** Single-Point GeoJSON FeatureCollection representing this image and its crack attributes */
  geoJson: GeoJsonFeatureCollection;
  previewImageUrl?: string;
  fractureHeatmapUrl?: string;
  /** Directional edge map used by cell-grid estimation (orange vertical seams, cyan horizontal seams). */
  cellBoundaryMapUrl?: string;
}
