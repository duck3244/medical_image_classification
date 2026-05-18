// 모든 페이지 상단에 노출되는 영구 고지 배너.

export default function DisclaimerBanner() {
  return (
    <div className="bg-warning-bg border-y border-warning-border text-warning text-sm">
      <div className="mx-auto max-w-5xl px-4 py-2 leading-snug">
        <strong>⚠️ DISCLAIMER:</strong> 본 서비스는 <strong>교육·연구 목적</strong>이며,
        실제 의료 진단이나 임상 의사결정에 사용할 수 없습니다.
        의료기기로 승인받지 않았으며 NOT a medical device.
      </div>
    </div>
  );
}
