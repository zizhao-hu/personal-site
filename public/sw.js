/**
 * Service Worker for WebLLM Model Caching
 * 
 * Caches model weights and WASM files for faster subsequent loads.
 * Models are stored in the Cache API for persistence across sessions.
 */

const CACHE_NAME = 'webllm-model-cache-v1';

// URLs that should be cached (model files from HuggingFace)
const MODEL_URL_PATTERNS = [
  'huggingface.co',
  'cdn-lfs.huggingface.co',
  'raw.githubusercontent.com/mlc-ai',
  '.wasm',
  '.bin',
  'ndarray-cache.json',
  'mlc-chat-config.json',
  'tokenizer.json',
  'tokenizer.model'
];

// Check if a URL should be cached
function shouldCache(url) {
  return MODEL_URL_PATTERNS.some(pattern => url.includes(pattern));
}

// Install event - pre-cache essential files
self.addEventListener('install', (event) => {
  console.log('[SW] Service Worker installing...');
  // Skip waiting to activate immediately
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[SW] Service Worker activating...');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          // Delete old versions of our cache
          if (cacheName.startsWith('webllm-model-cache-') && cacheName !== CACHE_NAME) {
            console.log('[SW] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      // Take control of all clients immediately
      return self.clients.claim();
    })
  );
});

// Fetch event - intercept requests and serve from cache when possible
self.addEventListener('fetch', (event) => {
  const url = event.request.url;
  
  // Only handle GET requests for model files
  if (event.request.method !== 'GET' || !shouldCache(url)) {
    return;
  }

  event.respondWith(
    caches.open(CACHE_NAME).then(async (cache) => {
      // Try to get from cache first
      const cachedResponse = await cache.match(event.request);
      
      if (cachedResponse) {
        console.log('[SW] Serving from cache:', url.substring(0, 80) + '...');
        return cachedResponse;
      }

      // Not in cache, fetch from network
      console.log('[SW] Fetching from network:', url.substring(0, 80) + '...');
      
      try {
        const networkResponse = await fetch(event.request);
        
        // Only cache successful responses
        if (networkResponse.ok) {
          // Clone the response since it can only be consumed once
          const responseToCache = networkResponse.clone();
          
          // Cache the response (don't await, let it happen in background)
          cache.put(event.request, responseToCache).then(() => {
            console.log('[SW] Cached:', url.substring(0, 80) + '...');
          }).catch((err) => {
            console.warn('[SW] Failed to cache:', err);
          });
        }
        
        return networkResponse;
      } catch (error) {
        console.error('[SW] Fetch failed:', error);
        throw error;
      }
    })
  );
});

// Message event - handle cache management commands
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'CLEAR_MODEL_CACHE') {
    event.waitUntil(
      caches.delete(CACHE_NAME).then(() => {
        console.log('[SW] Model cache cleared');
        event.ports[0]?.postMessage({ success: true });
      }).catch((err) => {
        console.error('[SW] Failed to clear cache:', err);
        event.ports[0]?.postMessage({ success: false, error: err.message });
      })
    );
  }
  
  if (event.data && event.data.type === 'GET_CACHE_SIZE') {
    event.waitUntil(
      caches.open(CACHE_NAME).then(async (cache) => {
        const keys = await cache.keys();
        let totalSize = 0;
        
        for (const request of keys) {
          const response = await cache.match(request);
          if (response) {
            const blob = await response.clone().blob();
            totalSize += blob.size;
          }
        }
        
        event.ports[0]?.postMessage({ 
          success: true, 
          size: totalSize,
          count: keys.length 
        });
      }).catch((err) => {
        event.ports[0]?.postMessage({ success: false, error: err.message });
      })
    );
  }
});
