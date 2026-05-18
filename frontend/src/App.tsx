import { Navigate, Route, Routes } from "react-router-dom";

import Layout from "./components/Layout";
import PredictPage from "./pages/PredictPage";
import ModelsPage from "./pages/ModelsPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Navigate to="/predict" replace />} />
        <Route path="/predict" element={<PredictPage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="*" element={<Navigate to="/predict" replace />} />
      </Route>
    </Routes>
  );
}
