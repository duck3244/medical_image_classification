import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import { fetchModels, fetchTasks, postPredict } from "../api/client";
import type { PredictResponse } from "../api/types";

function formatPercent(p: number | null | undefined): string {
  if (p == null || Number.isNaN(p)) return "—";
  return `${(p * 100).toFixed(1)}%`;
}

export default function PredictPage() {
  const tasksQ = useQuery({ queryKey: ["tasks"], queryFn: fetchTasks });
  const modelsQ = useQuery({ queryKey: ["models"], queryFn: fetchModels });

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [task, setTask] = useState<string>("");
  const [backbone, setBackbone] = useState<string>("");
  const [weights, setWeights] = useState<string>("");

  const predictM = useMutation({
    mutationFn: postPredict,
  });

  // 사용 가능한 weights 옵션 펼치기
  const weightOptions = useMemo(() => {
    const runs = modelsQ.data?.runs ?? [];
    return runs.flatMap((r) =>
      r.weights.map((w) => ({
        label: `${r.run_dir} / ${w.stage} (${w.size_mb} MB)`,
        value: w.path,
      })),
    );
  }, [modelsQ.data]);

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(f ? URL.createObjectURL(f) : null);
    predictM.reset();
  }

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    predictM.mutate({
      file,
      task: task || undefined,
      backbone: backbone || undefined,
      weights: weights || undefined,
    });
  }

  const result: PredictResponse | undefined = predictM.data;

  return (
    <div className="grid md:grid-cols-2 gap-6">
      <section className="bg-white rounded-lg shadow-sm border p-5">
        <h2 className="font-semibold text-slate-800">이미지 업로드 & 예측</h2>
        <form className="mt-4 space-y-4" onSubmit={onSubmit}>
          <div>
            <label className="block text-sm text-slate-600 mb-1">이미지 (PNG/JPG, ≤10 MB)</label>
            <input
              type="file"
              accept="image/png,image/jpeg"
              onChange={onFileChange}
              className="block w-full text-sm file:mr-3 file:py-2 file:px-3 file:rounded-md file:border file:border-slate-300 file:bg-slate-50 file:text-slate-700 hover:file:bg-slate-100"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm text-slate-600 mb-1">Task (override)</label>
              <select
                value={task}
                onChange={(e) => setTask(e.target.value)}
                className="w-full text-sm border rounded-md px-2 py-1.5 bg-white"
              >
                <option value="">(현재 설정 사용)</option>
                {tasksQ.data?.tasks.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm text-slate-600 mb-1">Backbone (override)</label>
              <select
                value={backbone}
                onChange={(e) => setBackbone(e.target.value)}
                className="w-full text-sm border rounded-md px-2 py-1.5 bg-white"
              >
                <option value="">(현재 설정 사용)</option>
                {tasksQ.data?.available_backbones.map((b) => (
                  <option key={b} value={b}>
                    {b}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm text-slate-600 mb-1">Weights (override)</label>
            <select
              value={weights}
              onChange={(e) => setWeights(e.target.value)}
              className="w-full text-sm border rounded-md px-2 py-1.5 bg-white"
            >
              <option value="">
                (기본: {modelsQ.data?.default_weights ? "lifespan 자동 선택" : "없음"})
              </option>
              {weightOptions.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={!file || predictM.isPending}
              className="px-4 py-2 rounded-md bg-slate-900 text-white text-sm disabled:bg-slate-300 hover:bg-slate-700"
            >
              {predictM.isPending ? "예측 중..." : "예측"}
            </button>
            {predictM.isError && (
              <span className="text-rose-600 text-sm">
                {(predictM.error as Error).message}
              </span>
            )}
          </div>
        </form>

        {previewUrl && (
          <div className="mt-5">
            <div className="text-xs text-slate-500 mb-1">선택한 이미지 미리보기</div>
            <img
              src={previewUrl}
              alt="upload preview"
              className="max-h-72 rounded-md border"
            />
          </div>
        )}
      </section>

      <section className="bg-white rounded-lg shadow-sm border p-5">
        <h2 className="font-semibold text-slate-800">결과</h2>
        {!result && !predictM.isPending && (
          <p className="mt-4 text-sm text-slate-500">
            이미지를 업로드하고 예측 버튼을 누르면 결과와 Grad-CAM이 표시됩니다.
          </p>
        )}
        {result && (
          <div className="mt-4 space-y-4 text-sm">
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-slate-50 rounded-md p-3">
                <div className="text-xs text-slate-500">예측 클래스</div>
                <div className="font-semibold text-slate-800">
                  {result.result.predicted_class ?? "—"}
                </div>
              </div>
              <div className="bg-slate-50 rounded-md p-3">
                <div className="text-xs text-slate-500">신뢰도</div>
                <div className="font-semibold text-slate-800">
                  {formatPercent(result.result.confidence)}
                </div>
              </div>
              <div className="bg-slate-50 rounded-md p-3">
                <div className="text-xs text-slate-500">Task / Backbone</div>
                <div className="text-slate-800">
                  {result.task} / {result.backbone}
                </div>
              </div>
            </div>

            {result.result.class_probabilities && (
              <div>
                <div className="text-xs text-slate-500 mb-1">클래스 확률</div>
                <ul className="space-y-1">
                  {Object.entries(result.result.class_probabilities).map(([k, v]) => (
                    <li key={k} className="flex items-center gap-2">
                      <span className="w-20 text-slate-600">{k}</span>
                      <span className="flex-1 bg-slate-100 rounded h-2 overflow-hidden">
                        <span
                          className="block h-full bg-slate-700"
                          style={{ width: `${Math.max(0, Math.min(100, v * 100))}%` }}
                        />
                      </span>
                      <span className="w-14 text-right tabular-nums">
                        {formatPercent(v)}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {result.result.gradcam_url && (
              <div>
                <div className="text-xs text-slate-500 mb-1">Grad-CAM</div>
                <img
                  src={result.result.gradcam_url}
                  alt="Grad-CAM overlay"
                  className="max-h-80 rounded-md border"
                />
              </div>
            )}
            {result.result.gradcam_error && (
              <div className="text-xs text-amber-700">
                Grad-CAM 생성 실패: {result.result.gradcam_error}
              </div>
            )}

            <div className="text-[11px] text-slate-500 border-t pt-3">
              {result.disclaimer}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
