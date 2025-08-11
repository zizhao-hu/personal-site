import { CreateMLCEngine, prebuiltAppConfig } from "@mlc-ai/web-llm";

export async function debugWebLLM() {
  console.log("=== WebLLM Debug Information ===");
  
  try {
    // Check if WebLLM is properly imported
    console.log("WebLLM import successful");
    
    // Check available models in prebuilt config
    console.log("Available models in prebuilt config:", Object.keys(prebuiltAppConfig.model_list));
    
    // Check browser capabilities
    console.log("WebAssembly available:", typeof WebAssembly !== 'undefined');
    console.log("WebGPU available:", typeof WebGPU !== 'undefined');
    
    // Try to get WebLLM version
    try {
      const webllmModule = await import("@mlc-ai/web-llm");
      console.log("WebLLM module exports:", Object.keys(webllmModule));
    } catch (e) {
      console.error("Failed to get WebLLM module info:", e);
    }
    
    // Test with a simple model
    console.log("Testing with Llama-2-7b-chat-q4f32_1...");
    const engine = await CreateMLCEngine("Llama-2-7b-chat-q4f32_1");
    console.log("Engine created successfully:", engine);
    
    return true;
  } catch (error) {
    console.error("WebLLM debug failed:", error);
    console.error("Error details:", {
      name: error.name,
      message: error.message,
      stack: error.stack
    });
    return false;
  }
} 