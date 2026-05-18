// Fetch 기반 API 클라이언트.
// Vite dev 서버가 /api 와 /static 을 백엔드(8000)로 프록시하므로 same-origin 호출만 사용.

import type {
  HealthResponse,
  ModelsResponse,
  PredictResponse,
  TasksResponse,
} from "./types";

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    // 백엔드의 detail 메시지를 가능한 한 그대로 전달
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      // ignore non-JSON error body
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return asJson<HealthResponse>(await fetch("/api/health"));
}

export async function fetchTasks(): Promise<TasksResponse> {
  return asJson<TasksResponse>(await fetch("/api/tasks"));
}

export async function fetchModels(): Promise<ModelsResponse> {
  return asJson<ModelsResponse>(await fetch("/api/models"));
}

export interface PredictInput {
  file: File;
  task?: string;
  backbone?: string;
  weights?: string;
}

export async function postPredict(input: PredictInput): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", input.file);
  if (input.task) form.append("task", input.task);
  if (input.backbone) form.append("backbone", input.backbone);
  if (input.weights) form.append("weights", input.weights);
  return asJson<PredictResponse>(
    await fetch("/api/predict", { method: "POST", body: form }),
  );
}
