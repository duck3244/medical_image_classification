import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
// VITE_API_TARGET 환경변수로 백엔드 주소를 덮어쓸 수 있다 (기본: localhost:8000).
// dev 서버는 /api 와 /static 을 백엔드로 프록시하여 CORS 회피 + 동일 출처 효과.
export default defineConfig(function (_a) {
    var mode = _a.mode;
    var env = loadEnv(mode, process.cwd(), "VITE_");
    var apiTarget = env.VITE_API_TARGET || "http://localhost:8000";
    return {
        plugins: [react()],
        server: {
            port: 5173,
            strictPort: false,
            proxy: {
                "/api": { target: apiTarget, changeOrigin: true },
                "/static": { target: apiTarget, changeOrigin: true },
            },
        },
        preview: {
            port: 5173,
        },
        build: {
            outDir: "dist",
            sourcemap: true,
        },
    };
});
