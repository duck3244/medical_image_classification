import { useQuery } from "@tanstack/react-query";

import { fetchModels } from "../api/client";

export default function ModelsPage() {
  const { data, isLoading, isError, error, refetch, isFetching } = useQuery({
    queryKey: ["models"],
    queryFn: fetchModels,
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h2 className="font-semibold text-slate-800">학습된 가중치</h2>
        <button
          onClick={() => refetch()}
          className="text-xs px-2 py-1 border rounded-md text-slate-600 hover:bg-slate-100"
        >
          {isFetching ? "새로고침..." : "새로고침"}
        </button>
      </div>

      {isLoading && <div className="text-sm text-slate-500">로드 중...</div>}
      {isError && (
        <div className="text-sm text-rose-600">
          {(error as Error).message}
        </div>
      )}

      {data && (
        <>
          <div className="text-xs text-slate-500">
            scan: <code>{data.models_dir}</code>
            <br />
            default: <code>{data.default_weights ?? "(없음)"}</code>
          </div>

          {data.runs.length === 0 ? (
            <div className="text-sm text-slate-600 bg-amber-50 border border-amber-200 rounded-md p-3">
              학습된 가중치가 없습니다. <code>backend</code>에서 다음을 실행하세요:
              <pre className="mt-2 text-xs bg-white border rounded p-2 overflow-x-auto">
                python main.py --mode train --task nih_cxr_binary --backbone EfficientNetB0
              </pre>
            </div>
          ) : (
            <table className="w-full text-sm border rounded-md overflow-hidden bg-white">
              <thead className="bg-slate-50 text-left text-xs text-slate-500">
                <tr>
                  <th className="p-2">Run</th>
                  <th className="p-2">Modified</th>
                  <th className="p-2">Weights</th>
                </tr>
              </thead>
              <tbody>
                {data.runs.map((r) => (
                  <tr key={r.run_dir} className="border-t align-top">
                    <td className="p-2 font-mono text-xs">{r.run_dir}</td>
                    <td className="p-2 text-xs text-slate-500">{r.modified}</td>
                    <td className="p-2">
                      <ul className="space-y-1">
                        {r.weights.map((w) => (
                          <li key={w.path} className="text-xs">
                            <span className="inline-block w-14 text-slate-500">{w.stage}</span>
                            <span className="text-slate-700">{w.size_mb} MB</span>
                            <div className="font-mono text-[10px] text-slate-400 break-all">
                              {w.path}
                            </div>
                          </li>
                        ))}
                      </ul>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  );
}
