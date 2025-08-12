export async function testWebLLMImport() {
  console.log("=== Testing WebLLM Import ===");
  
  try {
    // Test 1: Dynamic import
    console.log("Testing dynamic import...");
    const webllm = await import("@mlc-ai/web-llm");
    console.log("✓ Dynamic import successful");
    console.log("Available exports:", Object.keys(webllm));
    
    // Test 2: Check specific exports
    if (typeof webllm.CreateMLCEngine === 'function') {
      console.log("✓ CreateMLCEngine available");
    } else {
      console.log("✗ CreateMLCEngine not found");
    }
    
    if (webllm.prebuiltAppConfig) {
      console.log("✓ prebuiltAppConfig available");
      const modelList = webllm.prebuiltAppConfig.model_list || {};
      const availableModels = Object.keys(modelList);
      console.log("Available models:", availableModels);
      
      // Show details of each available model
      availableModels.forEach(modelId => {
        const modelConfig = modelList[modelId as keyof typeof modelList];
        if (modelConfig) {
          console.log(`  - ${modelId}:`, {
            model_url: (modelConfig as any).model_url,
            local_id: (modelConfig as any).local_id,
            model_lib_url: (modelConfig as any).model_lib_url
          });
        }
      });
    } else {
      console.log("✗ prebuiltAppConfig not found");
    }
    
    // Test 3: Check version (removed as it doesn't exist in the module)
    console.log("✗ Version not available");
    
    return { success: true, exports: Object.keys(webllm) };
    
  } catch (error) {
    console.error("✗ WebLLM import failed:", error);
    console.error("Error type:", (error as Error).constructor.name);
    console.error("Error message:", (error as Error).message);
    console.error("Error stack:", (error as Error).stack);
    
    return { success: false, error };
  }
} 