import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // onnxruntime-web loads its wasm next to itself; pre-bundling breaks that
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  worker: { format: 'es' },
});
