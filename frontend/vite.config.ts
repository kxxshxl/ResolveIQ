import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api to the backend so the browser stays same-origin (no CORS needed).
const backend = process.env.BACKEND_URL ?? "http://127.0.0.1:8100";

export default defineConfig({
  plugins: [react()],
  server: { port: 5174, proxy: { "/api": backend, "/health": backend } },
});
