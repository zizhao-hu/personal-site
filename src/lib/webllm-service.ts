import { CreateMLCEngine, MLCEngineInterface } from "@mlc-ai/web-llm";
import { getAnswerByQuestion } from '@/data/predefined-qa';

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export class WebLLMService {
  private engine: MLCEngineInterface | null = null;
  private isInitialized = false;
  private isInitializing = false;
  private progressCallback: ((progress: string) => void) | null = null;
  private loadingProgress: number = 0;
  private currentModel: string = "qwen-0.5b";
  private initializationStartTime: number = 0;
  private estimatedTotalTime: number = 30000; // 30 seconds default estimate

  // Personal context for Zizhao Hu - Professional Delegate
  private systemPrompt = `You are Zizhao Hu, a CS Ph.D. student at USC affiliated with the GLAMOUR Lab, advised by Professor Jesse Thomason and Professor Mohammad Rostami. You are acting as Zizhao's personal delegate for professional communications with potential clients, interviewers, and collaborators.

## CORE IDENTITY
- **Name**: Zizhao Hu
- **Current Position**: CS Ph.D. Student at University of Southern California (USC)
- **Research Lab**: GLAMOUR Lab under Professor Jesse Thomason and Professor Mohammad Rostami
- **Research Focus**: Synthetic data generation, multi-agent systems, multi-modal fusion
- **Current Project**: Synthetic data generation frameworks for self-improving AI agents

## PROFESSIONAL BACKGROUND
- **Former Research Experience**:
  - USC iLab (Information Sciences Institute)
  - Georgia Tech's Agile Systems Lab
  - Georgia Tech's Photonics Research Group

## EXPERTISE & SKILLS
- **Technical Skills**: AI/ML, synthetic data generation, multi-agent systems, multi-modal learning, deep learning, computer vision, natural language processing
- **Research Areas**: 
  - Synthetic data generation for AI training
  - Multi-agent collaboration and coordination
  - Multi-modal data fusion and learning
  - Self-improving AI systems
  - Computer vision and image processing

## PROFESSIONAL COMMUNICATION STYLE
- **Tone**: Professional, confident, approachable, and occasionally humorous
- **Personality**: 
  - Be direct and clear about capabilities and experience
  - Show enthusiasm for research and potential collaborations
  - Demonstrate deep technical knowledge when appropriate
  - Be honest about current limitations and ongoing work
  - Express interest in learning about others' projects and needs
  - Use humor to deflect personal questions or unknown topics
  - Stay professional but not overly formal - be yourself

## RESPONSE GUIDELINES
1. **For Interviewers**: Focus on research achievements, technical skills, and future goals
2. **For Potential Clients**: Emphasize relevant expertise, past projects, and how you can help solve their problems
3. **For Collaborators**: Highlight complementary skills and potential synergies
4. **Always**: Be authentic, professional, and represent Zizhao's interests accurately

## KEY MESSAGES
- Passionate about advancing AI through synthetic data and multi-agent systems
- Experienced in both theoretical research and practical implementation
- Open to collaborations, consulting opportunities, and research partnerships
- Committed to developing AI systems that can learn and improve autonomously

## IMPORTANT GUIDELINES
- **For publication questions**: Direct users to Zizhao's Google Scholar profile for accurate publication information
- **Never fabricate**: Do not make up specific publication titles, dates, or citation counts
- **Be honest**: If you don't have specific information, direct users to appropriate sources
- **Provide links**: When possible, provide direct links to profiles and resources
- **Never mention being AI**: You are Zizhao Hu, not an AI agent. Never mention being an AI, chatbot, or artificial intelligence
- **Unknown information**: If you don't know something specific, respond humorously as Zizhao would, but don't make up information
- **Stay in character**: Always respond as Zizhao Hu would, with his personality and communication style

When responding, speak as if you are Zizhao Hu representing yourself professionally. Be helpful, informative, and maintain a professional yet conversational tone.`;

  async initialize(progressCallback?: (progress: string) => void, modelId?: string): Promise<void> {
    if (this.isInitializing) {
      return;
    }

    // If model is changing, reset the engine
    if (modelId && modelId !== this.currentModel) {
      this.isInitialized = false;
      this.engine = null;
      this.currentModel = modelId;
    } else if (this.isInitialized) {
      return;
    }

    this.isInitializing = true;
    this.progressCallback = progressCallback || null;
    this.loadingProgress = 0; // Reset progress
    this.initializationStartTime = Date.now();
    
    // Set estimated time based on model
    if (modelId?.includes('7b') || modelId?.includes('7B')) {
      this.estimatedTotalTime = 45000; // 45 seconds for 7B models
    } else if (modelId?.includes('2b') || modelId?.includes('2B')) {
      this.estimatedTotalTime = 25000; // 25 seconds for 2B models
    } else {
      this.estimatedTotalTime = 30000; // 30 seconds for other models
    }
    
    // Try WebLLM initialization with timeout
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => {
        reject(new Error("WebLLM initialization timeout"));
      }, import.meta.env.PROD ? 15000 : 30000); // 15s for production, 30s for dev
    });

    try {
      await Promise.race([
        this.attemptWebLLMInitialization(),
        timeoutPromise
      ]);
    } catch (error) {
      console.error("WebLLM failed:", error);
      throw error;
    }
  }

  private async attemptWebLLMInitialization(): Promise<void> {
    try {
      this.updateProgress("Checking WebLLM version...");
      console.log("Initializing WebLLM engine...");
      console.log("WebLLM version: unknown");
      
      this.updateProgress("Checking browser compatibility...");
      // Check browser compatibility
      if (typeof WebAssembly === 'undefined') {
        throw new Error("WebAssembly is not supported in this browser");
      }
      
      if (typeof WebGPU === 'undefined') {
        console.warn("WebGPU is not available, WebLLM may fall back to CPU");
      }
      
      this.updateProgress("Loading model configurations...");
      // Get available models from prebuilt config
      let prebuiltAppConfig;
      try {
        const webllmModule = await import("@mlc-ai/web-llm");
        prebuiltAppConfig = webllmModule.prebuiltAppConfig;
        console.log("WebLLM module loaded successfully");
        console.log("Available exports:", Object.keys(webllmModule));
        
        // Wait a bit for browser APIs to be ready
        await new Promise(resolve => setTimeout(resolve, 1000));
        
      } catch (importError) {
        console.error("Failed to import WebLLM module:", importError);
        throw new Error("Failed to load WebLLM library");
      }
      
      if (!prebuiltAppConfig || !prebuiltAppConfig.model_list) {
        throw new Error("prebuiltAppConfig or model_list is not available");
      }
      
      if (!Array.isArray(prebuiltAppConfig.model_list) || prebuiltAppConfig.model_list.length === 0) {
        throw new Error("No models available in prebuiltAppConfig");
      }
      
      console.log("Available models:");
      console.log("Model list type:", typeof prebuiltAppConfig.model_list);
      console.log("Model list length:", prebuiltAppConfig.model_list.length);
      prebuiltAppConfig.model_list.forEach((model, index) => {
        console.log(`${index + 1}. ${model.model_id} - ${(model as any).model_url}`);
      });
      
      // Find the requested model or fall back to Qwen 0.5B
      let selectedModel = prebuiltAppConfig.model_list[0]?.model_id; // Default fallback
      
      if (!selectedModel) {
        throw new Error("No valid model found in prebuiltAppConfig.model_list");
      }
      
      // Improved model selection logic
      if (this.currentModel === "qwen-0.5b") {
        const qwenModel = prebuiltAppConfig.model_list.find(model => 
          model.model_id.toLowerCase().includes('qwen') && 
          model.model_id.toLowerCase().includes('0.5')
        );
        if (qwenModel) {
          selectedModel = qwenModel.model_id;
          console.log(`Found Qwen 0.5B model: ${selectedModel}`);
          this.updateProgress(`Loading Qwen 0.5B model...`);
        } else {
          console.log("Qwen 0.5B not found, using first available model");
          this.updateProgress(`Loading ${selectedModel}...`);
        }
      } else if (this.currentModel === "gemma-2b-it") {
        const gemma2bModel = prebuiltAppConfig.model_list.find(model => 
          model.model_id.toLowerCase().includes('gemma') && 
          (model.model_id.toLowerCase().includes('2b') || model.model_id.toLowerCase().includes('2b'))
        );
        if (gemma2bModel) {
          selectedModel = gemma2bModel.model_id;
          console.log(`Found Gemma 2B model: ${selectedModel}`);
          this.updateProgress(`Loading Gemma 2B Instruct model...`);
        } else {
          console.log("Gemma 2B not found, trying to find any Gemma model");
          const anyGemmaModel = prebuiltAppConfig.model_list.find(model => 
            model.model_id.toLowerCase().includes('gemma')
          );
          if (anyGemmaModel) {
            selectedModel = anyGemmaModel.model_id;
            console.log(`Found Gemma model: ${selectedModel}`);
            this.updateProgress(`Loading Gemma model...`);
          } else {
            console.log("No Gemma models found, using first available model");
            this.updateProgress(`Loading ${selectedModel}...`);
          }
        }
      } else if (this.currentModel === "gemma-7b-it") {
        const gemma7bModel = prebuiltAppConfig.model_list.find(model => 
          model.model_id.toLowerCase().includes('gemma') && 
          (model.model_id.toLowerCase().includes('7b') || model.model_id.toLowerCase().includes('7b'))
        );
        if (gemma7bModel) {
          selectedModel = gemma7bModel.model_id;
          console.log(`Found Gemma 7B model: ${selectedModel}`);
          this.updateProgress(`Loading Gemma 7B Instruct model...`);
        } else {
          console.log("Gemma 7B not found, trying to find any Gemma model");
          const anyGemmaModel = prebuiltAppConfig.model_list.find(model => 
            model.model_id.toLowerCase().includes('gemma')
          );
          if (anyGemmaModel) {
            selectedModel = anyGemmaModel.model_id;
            console.log(`Found Gemma model: ${selectedModel}`);
            this.updateProgress(`Loading Gemma model...`);
          } else {
            console.log("No Gemma models found, using first available model");
            this.updateProgress(`Loading ${selectedModel}...`);
          }
        }
      } else {
        // Try to find the requested model by name
        const requestedModel = prebuiltAppConfig.model_list.find(model => 
          model.model_id.toLowerCase().includes(this.currentModel.toLowerCase())
        );
        if (requestedModel) {
          selectedModel = requestedModel.model_id;
          console.log(`Found requested model: ${selectedModel}`);
          this.updateProgress(`Loading ${this.currentModel}...`);
        } else {
          console.log(`Requested model ${this.currentModel} not found, using first available model`);
          this.updateProgress(`Loading ${selectedModel}...`);
        }
      }
      
      // Create engine following the WebLLM documentation pattern
      try {
        console.log(`Creating engine with model: ${selectedModel}`);
        console.log("Model config:", { selectedModel, currentModel: this.currentModel });
        
        // Wait for browser APIs to be fully ready
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Try the simplest possible engine creation first
        try {
          this.engine = await CreateMLCEngine(selectedModel);
          console.log("Engine created with minimal config");
        } catch (simpleError) {
          console.log("Simple engine creation failed, trying with config:", simpleError);
          
          // Fallback to more complex config
          this.engine = await CreateMLCEngine(
            selectedModel,
            undefined, // Use default config (prebuiltAppConfig is already the default)
            {
              temperature: 0.7,
              top_p: 0.9,
              repetition_penalty: 1.1,
            }
          );
          console.log("Engine created with full config");
        }

        this.updateProgress(`Model loaded successfully! Ready to chat.`);
        console.log(`WebLLM engine initialized successfully with ${selectedModel}`);
        this.isInitialized = true;
        return;
      } catch (error) {
        console.error(`Failed to initialize with model ${selectedModel}:`, error);
        console.error("Engine creation error details:", {
          name: (error as Error).name,
          message: (error as Error).message,
          stack: (error as Error).stack
        });
        throw error;
      }
      
    } catch (error) {
      console.error("Failed to initialize WebLLM:", error);
      console.error("Full error details:", {
        name: (error as Error).name,
        message: (error as Error).message,
        stack: (error as Error).stack
      });
      
      // In production, if WebLLM fails, we should fall back to mock service
      if (import.meta.env.PROD) {
        console.warn("WebLLM failed in production, falling back to mock service");
        this.isInitializing = false;
        throw new Error("WebLLM not available in production - using demo mode");
      } else {
        this.isInitializing = false;
        throw error;
      }
    } finally {
      this.isInitializing = false;
    }
  }

  async generateResponse(messages: ChatMessage[]): Promise<string> {
    if (!this.engine) {
      throw new Error("WebLLM engine not initialized. Call initialize() first.");
    }

    try {
      // Get the last user message
      const lastUserMessage = messages.filter(msg => msg.role === "user").pop();
      if (lastUserMessage) {
        // Check if there's a predefined answer for this question
        const predefinedAnswer = getAnswerByQuestion(lastUserMessage.content);
        if (predefinedAnswer) {
          console.log("Using predefined answer for:", lastUserMessage.content);
          return predefinedAnswer;
        }
      }

      // If no predefined answer, use the AI model
      const formattedMessages = this.formatMessages(messages);
      
      console.log("Generating response with WebLLM...");
      
      // Generate response
      const response = await this.engine.chat.completions.create({
        messages: formattedMessages,
        stream: false,
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 500,
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

  private formatMessages(messages: ChatMessage[]) {
    // Add system prompt at the beginning
    const formattedMessages: Array<{role: "system" | "user" | "assistant", content: string}> = [
      {
        role: "system",
        content: this.systemPrompt
      }
    ];

    // Add conversation messages
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
    if (timeRemainingSeconds > 0) {
      progressMessage += ` (est. ${timeRemainingSeconds}s remaining)`;
    }
    
    console.log(`[Progress] ${progressMessage}`);
    if (this.progressCallback) {
      this.progressCallback(progressMessage);
    }
  }
}

// Export a singleton instance
export const webLLMService = new WebLLMService(); 