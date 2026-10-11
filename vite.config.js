import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  // GitHub Pages serves the site under /inann/; local dev and preview stay at /
  base: process.env.PAGES_BASE ?? '/',
  plugins: [react(), tailwindcss()],
  // onnxruntime-web loads its wasm next to itself; pre-bundling breaks that
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  worker: { format: 'es' },
});
