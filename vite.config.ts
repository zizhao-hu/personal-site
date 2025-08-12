import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'url'
import { dirname, resolve } from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 8501,
    watch: {
      usePolling: false,
      interval: 1000
    }
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src')
    }
  },
  build: {
    // Ensure WebLLM and WebAssembly files are handled properly
    rollupOptions: {
      output: {
        manualChunks: {
          'webllm': ['@mlc-ai/web-llm'],
          'vendor': ['react', 'react-dom']
        }
      }
    },
    // Use esbuild minification which is more reliable with WebLLM
    minify: 'esbuild',
    target: 'es2020',
    // Increase chunk size warning limit for WebLLM
    chunkSizeWarningLimit: 1000
  },
  optimizeDeps: {
    // Exclude WebLLM from dependency optimization
    exclude: ['@mlc-ai/web-llm']
  }
})
