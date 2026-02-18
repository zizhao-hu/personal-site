import { CreateWebWorkerMLCEngine, WebWorkerMLCEngine, prebuiltAppConfig } from "@mlc-ai/web-llm";
import { getAnswerByQuestion } from '@/data/predefined-qa';
import { ZIZHAO_CONTEXT } from '@/data/zizhao-context';

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export class WebLLMService {
  private engine: WebWorkerMLCEngine | null = null;
  private worker: Worker | null = null;
  private isInitialized = false;
  private isInitializing = false;
  private progressCallback: ((progress: string) => void) | null = null;
  private loadingProgress: number = 0;
  private currentModel: string = "SmolLM2-360M-Instruct-q4f16_1-MLC";
  private initializationStartTime: number = 0;
  private estimatedTotalTime: number = 30000;

  // Use centralized context file as system prompt
  private systemPrompt = ZIZHAO_CONTEXT;

  async initialize(progressCallback?: (progress: string) => void, modelId?: string): Promise<void> {
    if (this.isInitializing) {
      return;
    }

    // If model is changing, reset the engine
    if (modelId && modelId !== this.currentModel) {
      this.isInitialized = false;
      await this.cleanup();
      this.currentModel = modelId;
    } else if (this.isInitialized) {
      return;
    }

    this.isInitializing = true;
    this.progressCallback = progressCallback || null;
    this.loadingProgress = 0;
    this.initializationStartTime = Date.now();

    // Set estimated time based on model size
    this.estimatedTotalTime = this.getEstimatedLoadTime(modelId || this.currentModel);

    // Check for mobile device
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

    // Set timeout based on environment and device
    const timeout = import.meta.env.PROD
      ? (isMobile ? 120000 : 90000)  // 2 min mobile, 1.5 min desktop in prod
      : (isMobile ? 180000 : 120000); // 3 min mobile, 2 min desktop in dev

    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => {
        reject(new Error("WebLLM initialization timeout"));
      }, timeout);
    });

    try {
      await Promise.race([
        this.attemptWebLLMInitialization(),
        timeoutPromise
      ]);
    } catch (error) {
      console.error("WebLLM failed:", error);
      await this.cleanup();

      if (import.meta.env.PROD || isMobile) {
        console.warn("WebLLM failed - falling back to demo mode", { isMobile, isProd: import.meta.env.PROD });
        throw new Error("WebLLM not available - using demo mode");
      } else {
        throw error;
      }
    }
  }

  private getEstimatedLoadTime(modelId: string): number {
    // Estimated load times in milliseconds based on model size
    if (modelId.includes('360M') || modelId.includes('135M')) return 20000;
    if (modelId.includes('0.5B') || modelId.includes('0_5B')) return 30000;
    if (modelId.includes('1B') || modelId.includes('1.5B') || modelId.includes('1_5B')) return 45000;
    if (modelId.includes('2B') || modelId.includes('2b')) return 60000;
    if (modelId.includes('3B') || modelId.includes('3b')) return 90000;
    if (modelId.includes('7B') || modelId.includes('7b')) return 150000;
    return 60000; // Default
  }

  private async cleanup(): Promise<void> {
    if (this.engine) {
      try {
        await this.engine.unload();
      } catch (e) {
        console.warn("Error unloading engine:", e);
      }
      this.engine = null;
    }
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
  }

  private async attemptWebLLMInitialization(): Promise<void> {
    try {
      this.updateProgress("Checking browser compatibility...");

      if (typeof WebAssembly === 'undefined') {
        throw new Error("WebAssembly is not supported in this browser");
      }

      const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

      if (typeof WebGPU === 'undefined') {
        console.warn("WebGPU is not available, WebLLM may fall back to CPU");
        if (isMobile) {
          console.warn("Mobile device detected - WebGPU fallback may be slower");
        }
      }

      this.updateProgress("Loading model configurations...");

      // Verify model exists in prebuilt config
      const modelList = prebuiltAppConfig?.model_list || [];
      console.log("Available models:", modelList.length);

      // Find the requested model
      let selectedModel = this.currentModel;
      const modelExists = modelList.some((m: { model_id: string }) => m.model_id === selectedModel);

      if (!modelExists) {
        console.log(`Model ${selectedModel} not found, searching for alternatives...`);
        // Try to find a similar model
        const fallbackModel = this.findFallbackModel(modelList);
        if (fallbackModel) {
          selectedModel = fallbackModel;
          console.log(`Using fallback model: ${selectedModel}`);
        } else {
          throw new Error(`Model ${this.currentModel} not found and no fallback available`);
        }
      }

      this.updateProgress(`Initializing Web Worker...`);

      // Create Web Worker for non-blocking inference
      this.worker = new Worker(
        new URL("../workers/webllm-worker.ts", import.meta.url),
        { type: "module" }
      );

      this.updateProgress(`Loading ${selectedModel}...`);

      // Create engine using Web Worker
      this.engine = await CreateWebWorkerMLCEngine(
        this.worker,
        selectedModel,
        {
          initProgressCallback: (progress) => {
            console.log("WebLLM progress:", progress);

            if (progress && typeof progress === 'object') {
              if (progress.progress !== undefined) {
                const progressPercentage = Math.round(progress.progress * 100);
                this.loadingProgress = progressPercentage;
                this.updateProgress(progress.text || `Loading model... ${progressPercentage}%`);
              }
            } else if (typeof progress === 'string') {
              this.updateProgress(progress);
            }
          }
        }
      );

      this.updateProgress(`Model loaded successfully! Ready to chat.`);
      console.log(`WebLLM engine initialized successfully with ${selectedModel}`);
      this.isInitialized = true;
      this.isInitializing = false;

    } catch (error) {
      console.error("Failed to initialize WebLLM:", error);
      this.isInitializing = false;
      throw error;
    }
  }

  private findFallbackModel(modelList: Array<{ model_id: string }>): string | null {
    // Priority list of fallback models (smallest/fastest first)
    const fallbacks = [
      "SmolLM2-360M-Instruct-q4f16_1-MLC",
      "SmolLM2-135M-Instruct-q4f16_1-MLC",
      "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
      "Qwen2-0.5B-Instruct-q4f16_1-MLC",
      "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    ];

    for (const fallback of fallbacks) {
      if (modelList.some(m => m.model_id === fallback)) {
        return fallback;
      }
    }

    // Return first available model if no fallback matches
    return modelList[0]?.model_id || null;
  }

  async generateResponse(messages: ChatMessage[]): Promise<string> {
    if (!this.engine) {
      throw new Error("WebLLM engine not initialized. Call initialize() first.");
    }

    try {
      const lastUserMessage = messages.filter(msg => msg.role === "user").pop();
      if (lastUserMessage) {
        const predefinedAnswer = getAnswerByQuestion(lastUserMessage.content);
        if (predefinedAnswer) {
          console.log("Using predefined answer for:", lastUserMessage.content);
          return predefinedAnswer;
        }
      }

      const formattedMessages = this.formatMessages(messages);

      console.log("Generating response with WebLLM...");

      const response = await this.engine.chat.completions.create({
        messages: formattedMessages,
        stream: false,
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 256,
        presence_penalty: 0.1,
        frequency_penalty: 0.1,
      });

      const content = response.choices[0]?.message?.content;
      if (!content) {
        throw new Error("No response generated");
      }

      return content;
    } catch (error) {
      console.error("Error generating response:", error);
      throw error;
    }
  }

  /**
   * Generate a streaming response for real-time token display
   */
  async *generateStreamingResponse(messages: ChatMessage[]): AsyncGenerator<string, void, unknown> {
    if (!this.engine) {
      throw new Error("WebLLM engine not initialized. Call initialize() first.");
    }

    try {
      // Check for predefined answers first
      const lastUserMessage = messages.filter(msg => msg.role === "user").pop();
      if (lastUserMessage) {
        const predefinedAnswer = getAnswerByQuestion(lastUserMessage.content);
        if (predefinedAnswer) {
          console.log("Using predefined answer (streaming):", lastUserMessage.content);
          // Simulate streaming for predefined answers
          const words = predefinedAnswer.split(' ');
          for (const word of words) {
            yield word + ' ';
            await new Promise(resolve => setTimeout(resolve, 30));
          }
          return;
        }
      }

      const formattedMessages = this.formatMessages(messages);

      console.log("Generating streaming response with WebLLM...");

      const chunks = await this.engine.chat.completions.create({
        messages: formattedMessages,
        stream: true,
        stream_options: { include_usage: true },
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 256,
        presence_penalty: 0.1,
        frequency_penalty: 0.1,
      });

      for await (const chunk of chunks) {
        const content = chunk.choices[0]?.delta?.content;
        if (content) {
          yield content;
        }
      }
    } catch (error) {
      console.error("Error generating streaming response:", error);
      throw error;
    }
  }

  private formatMessages(messages: ChatMessage[]) {
    const formattedMessages: Array<{ role: "system" | "user" | "assistant", content: string }> = [
      {
        role: "system",
        content: this.systemPrompt
      }
    ];

    messages.forEach(msg => {
      formattedMessages.push({
        role: msg.role,
        content: msg.content
      });
    });

    return formattedMessages;
  }

  async reset(): Promise<void> {
    if (this.engine) {
      await this.engine.resetChat();
    }
  }

  isReady(): boolean {
    return this.isInitialized;
  }

  isLoading(): boolean {
    return this.isInitializing;
  }

  getCurrentModel(): string {
    return this.currentModel;
  }

  getLoadingProgress(): number {
    return this.loadingProgress;
  }

  private updateProgress(message: string): void {
    const elapsedTime = Date.now() - this.initializationStartTime;
    const estimatedTimeRemaining = Math.max(0, this.estimatedTotalTime - elapsedTime);
    const timeRemainingSeconds = Math.ceil(estimatedTimeRemaining / 1000);

    let progressMessage = message;
    if (timeRemainingSeconds > 0 && !message.includes('%')) {
      progressMessage += ` (est. ${timeRemainingSeconds}s remaining)`;
    }

    console.log(`[Progress] ${progressMessage}`);
    if (this.progressCallback) {
      this.progressCallback(progressMessage);
    }
  }

  getProgressMessage(): string {
    const elapsedTime = Date.now() - this.initializationStartTime;
    const estimatedTimeRemaining = Math.max(0, this.estimatedTotalTime - elapsedTime);
    const timeRemainingSeconds = Math.ceil(estimatedTimeRemaining / 1000);

    return `Loading model... (est. ${timeRemainingSeconds}s remaining)`;
  }
}

// Export a singleton instance
export const webLLMService = new WebLLMService();
