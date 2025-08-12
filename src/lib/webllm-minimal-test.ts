import { CreateMLCEngine } from "@mlc-ai/web-llm";

export async function minimalWebLLMTest() {
  console.log("=== Minimal WebLLM Test ===");
  
  try {
    // Test 1: Check if CreateMLCEngine is available
    console.log("✓ CreateMLCEngine available:", typeof CreateMLCEngine);
    
    // Test 2: Check browser support
    console.log("✓ WebAssembly available:", typeof WebAssembly !== 'undefined');
    console.log("✓ WebGPU available:", typeof WebGPU !== 'undefined');
    
    // Test 3: Try to get prebuiltAppConfig
    try {
      const { prebuiltAppConfig } = await import("@mlc-ai/web-llm");
      console.log("✓ prebuiltAppConfig available:", !!prebuiltAppConfig);
      if (prebuiltAppConfig && prebuiltAppConfig.model_list) {
        console.log("✓ model_list available, length:", prebuiltAppConfig.model_list.length);
        if (prebuiltAppConfig.model_list.length > 0) {
          const firstModel = prebuiltAppConfig.model_list[0];
          console.log("✓ First model:", firstModel.model_id);
        }
      } else {
        console.log("✗ model_list not available");
      }
    } catch (configError) {
      console.error("✗ Failed to get prebuiltAppConfig:", configError);
    }
    
    // Test 4: Try to create engine with the first available model
    try {
      const { prebuiltAppConfig } = await import("@mlc-ai/web-llm");
      if (prebuiltAppConfig && prebuiltAppConfig.model_list && prebuiltAppConfig.model_list.length > 0) {
        const firstModel = prebuiltAppConfig.model_list[0].model_id;
        console.log(`Attempting to create engine with: ${firstModel}`);
        
        const engine = await CreateMLCEngine(firstModel);
        console.log("✓ Engine created successfully");
        console.log("Engine object:", engine);
        
        return { success: true, engine, model: firstModel };
      } else {
        console.log("✗ No models available for testing");
        return { success: false, error: "No models available" };
      }
    } catch (engineError) {
      console.error("✗ Engine creation failed:", engineError);
      return { success: false, error: engineError };
    }
    
  } catch (error) {
    console.error("✗ Minimal test failed:", error);
    return { success: false, error };
  }
}
