import { useState, useEffect, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { v4 as uuidv4 } from "uuid";
import { Header } from "@/components/custom/header";
import { PreviewMessage, ThinkingMessage } from "@/components/custom/message";
import { ChatInput } from "@/components/custom/chatinput";

import { useScrollToBottom } from "@/components/custom/use-scroll-to-bottom";
import { webLLMService } from "@/lib/webllm-service";
import { mockLLMService } from "@/lib/mock-llm-service";
import { message } from "@/interfaces/interfaces";
import { ChatMessage } from "@/lib/webllm-service";

export function Chat() {
  const [messagesContainerRef] = useScrollToBottom<HTMLDivElement>();
  const [messages, setMessages] = useState<message[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isInitializing, setIsInitializing] = useState<boolean>(false);
  const [initializationError, setInitializationError] = useState<string>("");
  const [useMockService, setUseMockService] = useState<boolean>(false);
  const [progressMessage, setProgressMessage] = useState<string>("");
  const [progressPercentage, setProgressPercentage] = useState<number | undefined>(undefined);
  const [selectedModel, setSelectedModel] = useState<string>("gemma-2b-it");
  
  const navigate = useNavigate();
  const location = useLocation();

  // Handle initial question from navigation
  const hasProcessedInitialQuestionRef = useRef(false);
  
  useEffect(() => {
    if (location.state?.initialQuestion && !hasProcessedInitialQuestionRef.current) {
      console.log("Received initial question from navigation:", location.state.initialQuestion);
      hasProcessedInitialQuestionRef.current = true;
      // Clear the navigation state to prevent re-triggering
      navigate(location.pathname, { replace: true });
      // Submit the initial question after a short delay to ensure services are ready
      setTimeout(() => {
        handleSubmit(location.state.initialQuestion);
      }, 1000);
    }
  }, [location.state]);

  // Check if services are already initialized from Home component
  const hasCheckedServicesRef = useRef(false);
  
  useEffect(() => {
    if (hasCheckedServicesRef.current) return;
    
    const checkServices = () => {
      const webLLMReady = webLLMService.isReady();
      const mockReady = mockLLMService.isReady();
      
      if (webLLMReady) {
        console.log("WebLLM service already ready from Home component");
        setUseMockService(false);
        setIsInitializing(false);
        setInitializationError("");
      } else if (mockReady) {
        console.log("Mock service already ready from Home component");
        setUseMockService(true);
        setIsInitializing(false);
        setInitializationError("Demo Mode - Using predefined responses.");
      } else {
        console.log("No services ready, will initialize in Chat component");
        // Only initialize if no services are ready
        initializeServices();
      }
      hasCheckedServicesRef.current = true;
    };

    // Check after a short delay to allow Home component to finish initialization
    const timeoutId = setTimeout(checkServices, 500);
    return () => clearTimeout(timeoutId);
  }, []);

  // Monitor service readiness and transition from loading to chat
  useEffect(() => {
    const checkServiceReady = () => {
      const currentService = useMockService ? mockLLMService : webLLMService;
      
      // Continue updating progress even while checking readiness
      if (isInitializing) {
        const directProgress = webLLMService.getLoadingProgress();
        if (directProgress > 0) {
          const percentage = Math.round(directProgress * 100);
          setProgressPercentage(percentage);
          console.log("Progress updated:", percentage + "%");
        }
        
        // Update progress message from service
        const progressMessage = webLLMService.getProgressMessage();
        setProgressMessage(progressMessage);
      }
      
      if (currentService.isReady() && isInitializing) {
        console.log("Service is now ready, transitioning to chat interface");
        // Add a small delay to ensure progress is complete
        setTimeout(() => {
          setIsInitializing(false);
          setInitializationError("");
        }, 1000);
      }
    };

    // Check every 500ms if service becomes ready
    const intervalId = setInterval(checkServiceReady, 500);
    return () => clearInterval(intervalId);
  }, [useMockService, isInitializing]);

  // Initialize WebLLM only if not already initialized
  const initializeServices = async () => {
    setIsInitializing(true);
    setInitializationError("");
    
    try {
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
        console.log("WebLLM initialized successfully in Chat component");
        setUseMockService(false);
        // Don't set isInitializing to false here - let the monitoring effect handle it
      } catch (webllmError) {
        console.warn("WebLLM initialization failed in Chat component, falling back to mock service:", webllmError);
        // Show appropriate message based on environment
        if (import.meta.env.PROD) {
          setProgressMessage("Switching to demo mode (WebLLM not available in production)...");
        } else {
          setProgressMessage("Switching to demo mode...");
        }
      }
    } catch (error) {
      console.error("Failed to initialize WebLLM in Chat component, falling back to mock service:", error);
      
      try {
        // Fallback to mock service with timeout
        const mockTimeout = new Promise((_, reject) => {
          setTimeout(() => reject(new Error("Mock service timeout")), 5000);
        });
        
        await Promise.race([
          mockLLMService.initialize(),
          mockTimeout
        ]);
        
        console.log("Mock LLM service initialized successfully in Chat component");
        console.log("Mock service ready:", mockLLMService.isReady());
        setUseMockService(true);
        // Don't set isInitializing to false here - let the monitoring effect handle it
        if (import.meta.env.PROD) {
          setInitializationError("Demo Mode - WebLLM is not available in production environments. Using predefined responses.");
        } else {
          setInitializationError("Using demo mode - WebLLM failed to load. Some features may be limited.");
        }
      } catch (mockError) {
        console.error("Failed to initialize mock service in Chat component:", mockError);
        setInitializationError("Failed to initialize AI model. Please refresh the page and try again.");
      }
    } finally {
      // Don't set isInitializing to false here - let the monitoring effect handle it
      console.log("Initialization complete in Chat component. Mock service ready:", mockLLMService.isReady(), "WebLLM service ready:", webLLMService.isReady());
    }
  };

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
    console.log("Chat handleSubmit called with:", text);
    
    const currentService = useMockService ? mockLLMService : webLLMService;
    
    // Check if service is ready
    if (!currentService.isReady()) {
      console.log("LLM service not ready in Chat component");
      console.log("Service ready:", currentService.isReady(), "Loading:", isLoading);
      // Add a message to the chat indicating the service is not ready
      const errorMessage: message = {
        id: uuidv4(),
        role: "assistant",
        content: "The AI model is still initializing. Please wait a moment and try again."
      };
      setMessages(prev => [...prev, errorMessage]);
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

    // Add user message to chat
    const userMessage: message = {
      id: uuidv4(),
      role: "user",
      content: messageText
    };
    
    setMessages(prev => [...prev, userMessage]);
    setQuestion("");
    setIsLoading(true);

    try {
      console.log("Submitting message:", messageText);
      
             // Prepare chat messages for the service (filter out system messages)
       const chatMessages: ChatMessage[] = messages
         .filter(msg => msg.role !== "system")
         .map(msg => ({
           role: msg.role as "user" | "assistant",
           content: msg.content
         }));
      
      // Add the new user message
      chatMessages.push({
        role: "user",
        content: messageText
      });
      
      console.log("Generating response for messages:", chatMessages.length);
      
      // Generate response
      const response = await currentService.generateResponse(chatMessages);
      console.log("Received response:", response);
      
      // Add assistant response to chat
      const assistantMessage: message = {
        id: uuidv4(),
        role: "assistant",
        content: response
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      
    } catch (error) {
      console.error("Error generating response:", error);
      
      // Add error message to chat
      const errorMessage: message = {
        id: uuidv4(),
        role: "assistant",
        content: "Sorry, I encountered an error while processing your request. Please try again."
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-dvh bg-background">
      {/* Fixed Header */}
      <Header />
      
      {/* Back Button */}
      <div className="flex-shrink-0 px-4 py-2">
        <button
          onClick={() => navigate('/')}
          className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
        >
          ← Back to Home
        </button>
      </div>
      
      {/* Chat Messages Area */}
      <div className="flex flex-col flex-1 min-h-0">
        <div 
          className="flex flex-col flex-1 overflow-y-auto" 
          ref={messagesContainerRef}
        >

          
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
          
          {/* Chat Messages */}
          {messages.map((message, index) => (
            <PreviewMessage key={index} message={message} />
          ))}
          
          {/* Loading Message */}
          {isLoading && (
            <ThinkingMessage />
          )}
        </div>
        
        {/* Chat Input */}
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
}