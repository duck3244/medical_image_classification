// 백엔드 응답 스키마 — backend/api/routes 와 동기 유지.

export interface HealthResponse {
  status: string;
  disclaimer: string;
  tensorflow: string;
  gpu_count: number;
  gpu_names: string[];
  current_task: string;
  current_backbone: string;
  loaded_model_key: [string, string, string] | null;
  default_weights: string | null;
}

export interface TaskInfo {
  id: string;
  label: string;
  classes: string[];
  img_size: number;
  license: string;
}

export interface TasksResponse {
  tasks: TaskInfo[];
  current: string;
  available_backbones: string[];
}

export interface WeightsFile {
  stage: "stage1" | "stage2" | "final";
  path: string;
  size_mb: number;
}

export interface ModelRun {
  run_dir: string;
  weights: WeightsFile[];
  modified: string;
  has_run_config: boolean;
}

export interface ModelsResponse {
  models_dir: string;
  default_weights: string | null;
  runs: ModelRun[];
}

export interface PredictResult {
  predicted_class: string | null;
  predicted_index: number | null;
  confidence: number | null;
  probability_positive?: number | null;
  class_probabilities?: Record<string, number> | null;
  gradcam_url: string | null;
  gradcam_error: string | null;
}

export interface PredictResponse {
  disclaimer: string;
  task: string;
  backbone: string;
  weights: string;
  uploaded_filename: string;
  result: PredictResult;
  run_dir: string;
}
