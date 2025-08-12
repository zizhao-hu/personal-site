import { ChatInput } from "@/components/custom/chatinput";
import { PreviewMessage, ThinkingMessage } from "../../components/custom/message";
import { WaveLoading } from "@/components/custom/wave-loading";
import { useScrollToBottom } from '@/components/custom/use-scroll-to-bottom';
import { useState, useEffect } from "react";
import { message } from "../../interfaces/interfaces"
import { Overview } from "@/components/custom/overview";
import { ConversationStarters } from "@/components/custom/conversation-starters";
import { Header } from "@/components/custom/header";
import {v4 as uuidv4} from 'uuid';
import { webLLMService, ChatMessage } from "@/lib/webllm-service";
import { mockLLMService } from "@/lib/mock-llm-service";
import { testWebLLMImport } from "@/lib/webllm-import-test";

export function Chat() {
  const [messagesContainerRef, messagesEndRef] = useScrollToBottom<HTMLDivElement>();
  const [messages, setMessages] = useState<message[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isInitializing, setIsInitializing] = useState<boolean>(false);
  const [initializationError, setInitializationError] = useState<string>("");
  const [useMockService, setUseMockService] = useState<boolean>(false);
  const [progressMessage, setProgressMessage] = useState<string>("");
  const [progressPercentage, setProgressPercentage] = useState<number | undefined>(undefined);
  const [selectedModel, setSelectedModel] = useState<string>("qwen-0.5b");
  
  // Poll for progress updates when initializing
  useEffect(() => {
    if (!isInitializing) return;
    
    const progressInterval = setInterval(() => {
      const directProgress = webLLMService.getLoadingProgress();
      if (directProgress > 0) {
        const percentage = Math.round(directProgress * 100);
        setProgressPercentage(percentage);
        console.log("Progress polled:", percentage + "%");
      }
    }, 500); // Check every 500ms
    
    return () => clearInterval(progressInterval);
  }, [isInitializing]);

  // Initialize WebLLM on component mount
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
          // Don't re-throw, just continue to mock service
          setProgressMessage("Switching to demo mode...");
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
          setUseMockService(true);
          setInitializationError("Using demo mode - WebLLM failed to load. Some features may be limited.");
        } catch (mockError) {
          console.error("Failed to initialize mock service:", mockError);
          setInitializationError("Failed to initialize AI model. Please refresh the page and try again.");
        }
      } finally {
        setIsInitializing(false);
      }
    };

    initializeServices();
  }, [selectedModel]);

  // Scroll to top of chat area when component mounts
  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = 0;
    }
  }, []);

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
  const currentService = useMockService ? mockLLMService : webLLMService;
  
  if (!currentService.isReady() || isLoading) {
    console.log("LLM service not ready or already loading");
    return;
  }

  const messageText = text || question;
  if (!messageText.trim()) return;

  setIsLoading(true);
  
  const traceId = uuidv4();
  setMessages(prev => [...prev, { content: messageText, role: "user", id: traceId }]);
  setQuestion("");

  try {
    // Convert messages to ChatMessage format
    const chatMessages: ChatMessage[] = messages.map(msg => ({
      role: msg.role as "user" | "assistant",
      content: msg.content
    }));

    // Add the current user message
    chatMessages.push({
      role: "user",
      content: messageText
    });

    // Generate response using the current service
    const response = await currentService.generateResponse(chatMessages);
    
    // Add the assistant response
    setMessages(prev => [...prev, { content: response, role: "assistant", id: uuidv4() }]);
    
  } catch (error) {
    console.error("LLM service error:", error);
    setMessages(prev => [...prev, { 
      content: "Sorry, I encountered an error while processing your request. Please try again.", 
      role: "assistant", 
      id: uuidv4() 
    }]);
  } finally {
    setIsLoading(false);
  }
}

  return (
    <div className="flex flex-col h-dvh bg-background">
      {/* Fixed Header */}
      <Header />
      
      {/* Fixed Info Card - Overview */}
      <div className="flex-shrink-0">
        <Overview />
      </div>
      
      {/* Scrollable Chat Area */}
      <div className="flex flex-col flex-1 min-h-0">
        {/* Chat Messages Area */}
        <div className="flex flex-col flex-1 overflow-y-auto" ref={messagesContainerRef}>
          {/* Show conversation starters only when no messages and not initializing */}
          {messages.length === 0 && !isInitializing && (
            <div className="px-4 md:px-6 py-4">
              <ConversationStarters onStarterClick={handleSubmit} />
            </div>
          )}
          
          {/* Chat Messages */}
          {messages.map((message, index) => (
            <PreviewMessage key={index} message={message} />
          ))}
          
          {/* Loading State - Only in chat area */}
          {isLoading && <ThinkingMessage />}
          
          {/* Initialization Loading - Only in chat area */}
          {isInitializing && (
            <WaveLoading 
              message={progressMessage || "Initializing AI model..."} 
              progress={progressPercentage}
              estimatedTime={progressMessage?.includes('remaining') ? progressMessage.split('(')[1]?.split(')')[0] : undefined}
            />
          )}
          
          {/* Error State - Only in chat area */}
          {initializationError && !isInitializing && (
            <div className="flex items-center justify-center p-8">
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
          
          <div ref={messagesEndRef} className="shrink-0 min-w-[24px] min-h-[24px]"/>
        </div>
        
        {/* Always available conversation starters */}
        {messages.length > 0 && (
          <div className="px-4 md:px-6 pb-4">
            <ConversationStarters onStarterClick={handleSubmit} showTitle={false} compact={true} />
          </div>
        )}
        
        {/* Chat Input */}
        <div className="flex mx-auto px-4 bg-background pb-4 md:pb-6 gap-2 w-full md:max-w-3xl">
          <ChatInput  
            question={question}
            setQuestion={setQuestion}
            onSubmit={handleSubmit}
            isLoading={isLoading}
            disabled={!(useMockService ? mockLLMService.isReady() : webLLMService.isReady()) || isInitializing}
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