// 최초 방문 시 표시되는 동의 모달. localStorage 플래그로 재방문 시 생략.

import { useEffect, useState } from "react";

const STORAGE_KEY = "medimg.disclaimer.acknowledged.v1";

export default function DisclaimerModal() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const ack = localStorage.getItem(STORAGE_KEY);
    if (!ack) setOpen(true);
  }, []);

  if (!open) return null;

  function acknowledge() {
    localStorage.setItem(STORAGE_KEY, new Date().toISOString());
    setOpen(false);
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4"
    >
      <div className="bg-white rounded-lg shadow-xl max-w-lg w-full p-6">
        <h2 className="text-lg font-semibold text-warning">⚠️ 사용 전 반드시 확인</h2>
        <div className="mt-3 space-y-2 text-sm text-slate-700">
          <p>
            본 서비스는 <strong>개인 학습·연구 목적</strong>으로만 제작되었습니다.
          </p>
          <ul className="list-disc pl-5 space-y-1">
            <li>실제 의료 진단, 임상 의사결정, 환자 관리에 사용할 수 없습니다.</li>
            <li>의료기기(MFDS/FDA 등)로 승인받지 않았습니다.</li>
            <li>예측 결과는 의학적 소견이 아닙니다.</li>
            <li>증상이 있는 경우 반드시 의료 전문가의 진료를 받으십시오.</li>
          </ul>
          <p className="text-xs text-slate-500 pt-2">
            This project is for personal educational and research purposes only.
            It is NOT a medical device and MUST NOT be used for diagnosis or
            clinical decisions.
          </p>
        </div>
        <div className="mt-5 flex justify-end">
          <button
            type="button"
            onClick={acknowledge}
            className="px-4 py-2 rounded-md bg-slate-900 text-white text-sm hover:bg-slate-700"
          >
            확인하고 계속
          </button>
        </div>
      </div>
    </div>
  );
}
