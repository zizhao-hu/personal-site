export async function testWebLLMImport() {
  console.log("=== Testing WebLLM Import ===");
  
  try {
    // Test 1: Dynamic import
    console.log("Testing dynamic import...");
    const webllm = await import("@mlc-ai/web-llm");
    console.log("✓ Dynamic import successful");
    console.log("Available exports:", Object.keys(webllm));
    
    // Test 2: Check specific exports
    if (webllm.CreateMLCEngine) {
      console.log("✓ CreateMLCEngine available");
    } else {
      console.log("✗ CreateMLCEngine not found");
    }
    
    if (webllm.prebuiltAppConfig) {
      console.log("✓ prebuiltAppConfig available");
      const availableModels = Object.keys(webllm.prebuiltAppConfig.model_list || {});
      console.log("Available models:", availableModels);
      
      // Show details of each available model
      availableModels.forEach(modelId => {
        const modelConfig = webllm.prebuiltAppConfig.model_list[modelId];
        console.log(`  - ${modelId}:`, {
          model_url: modelConfig.model_url,
          local_id: modelConfig.local_id,
          model_lib_url: modelConfig.model_lib_url
        });
      });
    } else {
      console.log("✗ prebuiltAppConfig not found");
    }
    
    // Test 3: Check version
    if (webllm.version) {
      console.log("✓ WebLLM version:", webllm.version);
    } else {
      console.log("✗ Version not available");
    }
    
    return { success: true, exports: Object.keys(webllm) };
    
  } catch (error) {
    console.error("✗ WebLLM import failed:", error);
    console.error("Error type:", error.constructor.name);
    console.error("Error message:", error.message);
    console.error("Error stack:", error.stack);
    
    return { success: false, error };
  }
} 