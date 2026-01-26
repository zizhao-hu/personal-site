/**
 * WebLLM Web Worker
 * 
 * This worker handles all WebLLM computation off the main thread,
 * preventing UI freezes during model loading and inference.
 */

import { WebWorkerMLCEngineHandler } from "@mlc-ai/web-llm";

// Create the handler that manages WebLLM engine in this worker
const handler = new WebWorkerMLCEngineHandler();

// Forward all messages to the handler
self.onmessage = (msg: MessageEvent) => {
  handler.onmessage(msg);
};
