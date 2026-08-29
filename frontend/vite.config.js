import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In dev (`npm run dev`) Vite proxies the same paths nginx serves in production,
// so the app code never needs to know where it is running.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 47300,
    proxy: {
      '/api': {
        target: process.env.AI_SERVICE_URL || 'http://localhost:47800',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
      '/webhook': {
        target: process.env.N8N_URL || 'http://localhost:47678',
        changeOrigin: true,
      },
    },
  },
  build: { outDir: 'dist', sourcemap: false },
});
