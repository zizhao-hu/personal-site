import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Header } from '@/components/custom/header';
import { Overview } from '@/components/custom/overview';
import { ConversationStarters } from '@/components/custom/conversation-starters';
import { ChatInput } from '@/components/custom/chatinput';

import { webLLMService } from '@/lib/webllm-service';
import { mockLLMService } from '@/lib/mock-llm-service';
import { testWebLLMImport } from '@/lib/webllm-import-test';

export const Home = () => {
  const navigate = useNavigate();
  
  // Chat state for initialization and direct chat
  const [question, setQuestion] = useState("");
  const [isLoading] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [useMockService, setUseMockService] = useState(false);
  const [initializationError, setInitializationError] = useState("");
  const [progressMessage, setProgressMessage] = useState("");
  const [progressPercentage, setProgressPercentage] = useState<number | undefined>(undefined);
  const [selectedModel, setSelectedModel] = useState<string>("gemma-2b-it");

  // Initialize services on component mount
  useEffect(() => {
    const initializeServices = async () => {
      setIsInitializing(true);
      setInitializationError("");
      
      try {
        // Test import first
        console.log("Testing WebLLM import...");
        const importResult = await testWebLLMImport();
        
        if (!importResult.success) {
          console.error("WebLLM import failed:", importResult.error);
          throw new Error(`WebLLM import failed: ${(importResult.error as Error).message}`);
        }
        
        // Try WebLLM first
        try {
          setProgressMessage("Initializing AI model...");
          await webLLMService.initialize((progress) => {
            setProgressMessage(progress);
            // Extract percentage from progress message if it contains one
            const percentageMatch = progress.match(/(\d+)%/);
            if (percentageMatch) {
              setProgressPercentage(parseInt(percentageMatch[1]));
            } else {
              // Fallback: try to get progress directly from service
              const directProgress = webLLMService.getLoadingProgress();
              if (directProgress > 0) {
                setProgressPercentage(Math.round(directProgress * 100));
              } else {
                setProgressPercentage(undefined);
              }
            }
          }, selectedModel);
          console.log("WebLLM initialized successfully");
          setUseMockService(false);
        } catch (webllmError) {
          console.warn("WebLLM initialization failed, falling back to mock service:", webllmError);
          // Show appropriate message based on environment
          if (import.meta.env.PROD) {
            setProgressMessage("Switching to demo mode (WebLLM not available in production)...");
          } else {
            setProgressMessage("Switching to demo mode...");
          }
        }
      } catch (error) {
        console.error("Failed to initialize WebLLM, falling back to mock service:", error);
        
        try {
          // Fallback to mock service with timeout
          const mockTimeout = new Promise((_, reject) => {
            setTimeout(() => reject(new Error("Mock service timeout")), 5000);
          });
          
          await Promise.race([
            mockLLMService.initialize(),
            mockTimeout
          ]);
          
          console.log("Mock LLM service initialized successfully");
          console.log("Mock service ready:", mockLLMService.isReady());
          setUseMockService(true);
          if (import.meta.env.PROD) {
            setInitializationError("Demo Mode - WebLLM is not available in production environments. Using predefined responses.");
          } else {
            setInitializationError("Using demo mode - WebLLM failed to load. Some features may be limited.");
          }
        } catch (mockError) {
          console.error("Failed to initialize mock service:", mockError);
          setInitializationError("Failed to initialize AI model. Please refresh the page and try again.");
        }
      } finally {
        setIsInitializing(false);
        console.log("Initialization complete. Mock service ready:", mockLLMService.isReady(), "WebLLM service ready:", webLLMService.isReady());
      }
    };

    initializeServices();
  }, [selectedModel]);

  // Monitor progress during initialization
  useEffect(() => {
    const updateProgress = () => {
      if (isInitializing) {
        const directProgress = webLLMService.getLoadingProgress();
        if (directProgress > 0) {
          const percentage = Math.round(directProgress * 100);
          setProgressPercentage(percentage);
          console.log("Home progress updated:", percentage + "%");
        }
        
        // Update progress message from service
        const progressMessage = webLLMService.getProgressMessage();
        setProgressMessage(progressMessage);
      }
    };

    // Update progress every 500ms during initialization
    const intervalId = setInterval(updateProgress, 500);
    return () => clearInterval(intervalId);
  }, [isInitializing]);

  // Handle model selection
  const handleModelSelect = async (modelId: string) => {
    if (modelId === selectedModel) return;
    
    setSelectedModel(modelId);
    setIsInitializing(true);
    setInitializationError("");
    setProgressMessage("");
    setProgressPercentage(undefined);
    
    try {
      await webLLMService.initialize((progress) => {
        setProgressMessage(progress);
        const percentageMatch = progress.match(/(\d+)%/);
        if (percentageMatch) {
          setProgressPercentage(parseInt(percentageMatch[1]));
        } else {
          setProgressPercentage(undefined);
        }
      }, modelId);
      setUseMockService(false);
    } catch (error) {
      console.error("Failed to initialize new model:", error);
      setInitializationError("Failed to load selected model. Please try again.");
    } finally {
      setIsInitializing(false);
    }
  };

  async function handleSubmit(text?: string) {
    console.log("handleSubmit called with:", text);
    
    const currentService = useMockService ? mockLLMService : webLLMService;
    
    // Check if service is ready, but don't block if it's not - just show a message
    if (!currentService.isReady()) {
      console.log("LLM service not ready");
      console.log("Service ready:", currentService.isReady(), "Loading:", isLoading);
      // Navigate to chat page with a message about initialization
      navigate('/chat', { 
        state: { 
          initialQuestion: "The AI model is still initializing. Please wait a moment and try again." 
        } 
      });
      return;
    }
    
    if (isLoading) {
      console.log("Already loading, ignoring click");
      return;
    }

    const messageText = text || question;
    if (!messageText.trim()) {
      console.log("No message text to submit");
      return;
    }

    // Navigate to chat page with the question
    navigate('/chat', { 
      state: { 
        initialQuestion: messageText 
      } 
    });
  }

  const handleStarterClick = (question: string) => {
    // Navigate to chat page with the question as a parameter
    navigate('/chat', { 
      state: { 
        initialQuestion: question 
      } 
    });
  };

  return (
    <div className="flex flex-col h-dvh bg-background">
      {/* Fixed Header */}
      <Header />
      
      {/* Fixed Info Card - Overview */}
      <div className="flex-shrink-0">
        <Overview />
      </div>
      
      {/* Scrollable Content Area */}
      <div className="flex flex-col flex-1 min-h-0">
        <div className="flex flex-col flex-1 overflow-y-auto">
          {/* Show conversation starters only when services are ready and not initializing */}
          {!isInitializing && (useMockService ? mockLLMService.isReady() : webLLMService.isReady()) && !initializationError && (
            <div className="px-4 md:px-6 py-8">
              <ConversationStarters onStarterClick={handleStarterClick} />
            </div>
          )}
          

          
          {/* Error State */}
          {initializationError && !isInitializing && (
            <div className="px-4 md:px-6 py-8">
              <div className="text-center">
                <p className={`mb-2 ${useMockService ? 'text-yellow-600 dark:text-yellow-400' : 'text-red-600 dark:text-red-400'}`}>
                  {useMockService ? 'Demo Mode' : 'Initialization Error'}
                </p>
                <p className="text-gray-600 dark:text-gray-400 text-sm">{initializationError}</p>
                {useMockService && (
                  <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
                    The chatbot is working in demo mode with predefined responses.
                  </p>
                )}
              </div>
            </div>
          )}
          
          {/* Additional content or spacing */}
          <div className="flex-1" />
        </div>
        
        {/* Chat Input - Always visible */}
        <div className="flex mx-auto px-4 bg-background pb-4 md:pb-6 gap-2 w-full md:max-w-3xl">
          <ChatInput  
            question={question}
            setQuestion={setQuestion}
            onSubmit={handleSubmit}
            isLoading={isLoading}
            disabled={!(useMockService ? mockLLMService.isReady() : webLLMService.isReady())}
            selectedModel={selectedModel}
            onModelSelect={handleModelSelect}
            isModelLoading={isInitializing}
            modelProgressPercentage={progressPercentage}
          />
        </div>
      </div>
    </div>
  );
};
