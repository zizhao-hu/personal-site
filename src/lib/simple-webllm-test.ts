import { CreateMLCEngine } from "@mlc-ai/web-llm";

export async function simpleWebLLMTest() {
  console.log("=== Simple WebLLM Test ===");
  
  try {
    // Test 1: Basic import
    console.log("✓ WebLLM import successful");
    
    // Test 2: Check browser support
    console.log("✓ WebAssembly available:", typeof WebAssembly !== 'undefined');
    console.log("✓ WebGPU available:", typeof WebGPU !== 'undefined');
    
    // Test 3: Try to create engine with minimal config
    console.log("Attempting to create WebLLM engine...");
    
    const engine = await CreateMLCEngine(
      "Llama-2-7b-chat-q4f32_1",
      undefined, // Use default config
      undefined  // Use default chat options
    );
    
    console.log("✓ WebLLM engine created successfully");
    console.log("Engine object:", engine);
    
    return { success: true, engine };
    
  } catch (error) {
    console.error("✗ WebLLM test failed:", error);
    console.error("Error type:", (error as Error).constructor.name);
    console.error("Error message:", (error as Error).message);
    console.error("Error stack:", (error as Error).stack);
    
    return { success: false, error };
  }
} 