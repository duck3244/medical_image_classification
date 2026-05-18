import { NavLink, Outlet } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import DisclaimerBanner from "./DisclaimerBanner";
import DisclaimerModal from "./DisclaimerModal";
import { fetchHealth } from "../api/client";

function HealthDot() {
  const { data, isError, isLoading } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 15_000,
  });

  let color = "bg-slate-300";
  let label = "checking...";
  if (isLoading) {
    color = "bg-slate-300";
    label = "checking...";
  } else if (isError || !data) {
    color = "bg-rose-500";
    label = "offline";
  } else {
    color = data.loaded_model_key ? "bg-emerald-500" : "bg-amber-400";
    label = data.loaded_model_key
      ? `model loaded (${data.current_backbone})`
      : "no model";
  }

  return (
    <span className="inline-flex items-center gap-2 text-xs text-slate-500">
      <span className={`w-2 h-2 rounded-full ${color}`} />
      {label}
    </span>
  );
}

const navItemCls = ({ isActive }: { isActive: boolean }) =>
  [
    "px-3 py-1.5 rounded-md text-sm",
    isActive
      ? "bg-slate-900 text-white"
      : "text-slate-600 hover:bg-slate-100",
  ].join(" ");

export default function Layout() {
  return (
    <div className="min-h-full flex flex-col">
      <DisclaimerBanner />
      <header className="border-b bg-white">
        <div className="mx-auto max-w-5xl px-4 h-14 flex items-center gap-4">
          <div className="font-semibold">
            🩻 Medical Image Classification{" "}
            <span className="text-slate-400 text-sm font-normal">(Learning)</span>
          </div>
          <nav className="flex items-center gap-1 ml-4">
            <NavLink to="/predict" className={navItemCls}>
              Predict
            </NavLink>
            <NavLink to="/models" className={navItemCls}>
              Models
            </NavLink>
          </nav>
          <div className="ml-auto">
            <HealthDot />
          </div>
        </div>
      </header>
      <main className="flex-1">
        <div className="mx-auto max-w-5xl px-4 py-6">
          <Outlet />
        </div>
      </main>
      <footer className="border-t bg-white">
        <div className="mx-auto max-w-5xl px-4 py-3 text-xs text-slate-500">
          NOT a medical device. NOT for diagnosis. — © Learning Project
        </div>
      </footer>
      <DisclaimerModal />
    </div>
  );
}
