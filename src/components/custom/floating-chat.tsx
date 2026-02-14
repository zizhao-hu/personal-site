import { useState, useEffect, useRef } from "react";
import { v4 as uuidv4 } from "uuid";
import { Textarea } from "../ui/textarea";
import { Button } from "../ui/button";
import { ArrowUpIcon } from "./icons";
import { ModelSelector } from "./model-selector";
import { PreviewMessage, ThinkingMessage } from "./message";
import { webLLMService } from "@/lib/webllm-service";
import { mockLLMService } from "@/lib/mock-llm-service";
import { testWebLLMImport } from "@/lib/webllm-import-test";
import { message } from "@/interfaces/interfaces";
import { ChatMessage } from "@/lib/webllm-service";
import { motion, AnimatePresence } from "framer-motion";
import { X, MessageSquare, ChevronDown, AlertTriangle } from "lucide-react";
import { toast } from "sonner";

export const FloatingChat = () => {
  const [messages, setMessages] = useState<message[]>([]);
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [useMockService, setUseMockService] = useState(false);
  const [progressPercentage, setProgressPercentage] = useState<number | undefined>(undefined);
  const [selectedModel, setSelectedModel] = useState("Qwen2.5-0.5B-Instruct-q4f16_1-MLC");
  const [showOverlay, setShowOverlay] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Listen for questions from conversation starters
  useEffect(() => {
    const handleChatQuestion = (event: CustomEvent<string>) => {
      setQuestion(event.detail);
      setShowOverlay(true);
      // Auto-submit after a short delay
      setTimeout(() => {
        textareaRef.current?.focus();
      }, 100);
    };

    window.addEventListener('chat-question', handleChatQuestion as EventListener);
    return () => {
      window.removeEventListener('chat-question', handleChatQuestion as EventListener);
    };
  }, []);

  // Initialize services
  useEffect(() => {
    const initializeServices = async () => {
      // Check if already initialized
      if (webLLMService.isReady() || mockLLMService.isReady()) {
        setUseMockService(!webLLMService.isReady());
        return;
      }

      setIsInitializing(true);

      try {
        const importResult = await testWebLLMImport();
        if (!importResult.success) throw new Error("WebLLM import failed");

        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

        if (!isMobile) {
          try {
            await webLLMService.initialize((progress) => {
              const percentageMatch = progress.match(/(\d+)%/);
              if (percentageMatch) {
                setProgressPercentage(parseInt(percentageMatch[1]));
              }
            }, selectedModel);
            setUseMockService(false);
          } catch {
            setUseMockService(true);
          }
        } else {
          setUseMockService(true);
        }
      } catch {
        try {
          await mockLLMService.initialize();
          setUseMockService(true);
        } catch {
          console.error("Failed to initialize any service");
        }
      } finally {
        setIsInitializing(false);
      }
    };

    initializeServices();
  }, [selectedModel]);

  // Handle model selection
  const handleModelSelect = async (modelId: string) => {
    if (modelId === selectedModel) return;

    setSelectedModel(modelId);
    setIsInitializing(true);
    setProgressPercentage(undefined);

    try {
      await webLLMService.initialize((progress) => {
        const percentageMatch = progress.match(/(\d+)%/);
        if (percentageMatch) {
          setProgressPercentage(parseInt(percentageMatch[1]));
        }
      }, modelId);
      setUseMockService(false);
    } catch {
      console.error("Failed to initialize model");
    } finally {
      setIsInitializing(false);
    }
  };

  const handleSubmit = async () => {
    const currentService = useMockService ? mockLLMService : webLLMService;

    if (!currentService.isReady()) {
      toast.error("AI model is still loading...");
      return;
    }

    if (isLoading || isStreaming || !question.trim()) return;

    const userMessage: message = {
      id: uuidv4(),
      role: "user",
      content: question
    };

    setMessages(prev => [...prev, userMessage]);
    setQuestion("");
    setIsLoading(true);
    setShowOverlay(true);

    try {
      const chatMessages: ChatMessage[] = messages
        .filter(msg => msg.role !== "system")
        .map(msg => ({
          role: msg.role as "user" | "assistant",
          content: msg.content
        }));

      chatMessages.push({ role: "user", content: userMessage.content });

      const assistantMessageId = uuidv4();
      const assistantMessage: message = {
        id: assistantMessageId,
        role: "assistant",
        content: ""
      };

      setMessages(prev => [...prev, assistantMessage]);
      setIsLoading(false);
      setIsStreaming(true);

      if (!useMockService && webLLMService.isReady()) {
        let fullContent = "";
        for await (const chunk of webLLMService.generateStreamingResponse(chatMessages)) {
          fullContent += chunk;
          setMessages(prev =>
            prev.map(msg =>
              msg.id === assistantMessageId
                ? { ...msg, content: fullContent }
                : msg
            )
          );
        }
      } else {
        const response = await currentService.generateResponse(chatMessages);
        setMessages(prev =>
          prev.map(msg =>
            msg.id === assistantMessageId
              ? { ...msg, content: response }
              : msg
          )
        );
      }
    } catch (error) {
      console.error("Error:", error);
      setMessages(prev => [...prev, {
        id: uuidv4(),
        role: "assistant",
        content: "Sorry, I encountered an error. Please try again."
      }]);
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
    }
  };

  const isServiceReady = useMockService ? mockLLMService.isReady() : webLLMService.isReady();
  const isDisabled = !isServiceReady || isInitializing;

  return (
    <>
      {/* Overlay Response Panel */}
      <AnimatePresence>
        {showOverlay && messages.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="fixed bottom-20 md:bottom-24 left-2 right-2 md:left-auto md:right-4 md:w-[420px] max-h-[60vh] bg-white dark:bg-gray-900 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 z-40 overflow-hidden"
          >
            {/* Overlay Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
              <div className="flex items-center gap-2">
                <MessageSquare className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Chat</span>
              </div>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setShowOverlay(false)}
                  className="p-1.5 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
                >
                  <ChevronDown className="w-4 h-4 text-gray-500" />
                </button>
                <button
                  onClick={() => {
                    setMessages([]);
                    setShowOverlay(false);
                  }}
                  className="p-1.5 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
                >
                  <X className="w-4 h-4 text-gray-500" />
                </button>
              </div>
            </div>

            {/* Messages */}
            <div className="overflow-y-auto max-h-[calc(60vh-56px)] p-2">
              {messages.map((message, index) => (
                <PreviewMessage
                  key={message.id || index}
                  message={message}
                  isStreaming={isStreaming && index === messages.length - 1 && message.role === 'assistant'}
                />
              ))}
              {isLoading && !isStreaming && <ThinkingMessage />}
              <div ref={messagesEndRef} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Minimized indicator when overlay is hidden but has messages */}
      <AnimatePresence>
        {!showOverlay && messages.length > 0 && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            onClick={() => setShowOverlay(true)}
            className="fixed bottom-20 md:bottom-24 right-4 p-3 bg-blue-600 hover:bg-blue-700 text-white rounded-full shadow-lg z-40 flex items-center gap-2"
          >
            <MessageSquare className="w-5 h-5" />
            <span className="text-xs font-medium pr-1">{messages.length}</span>
          </motion.button>
        )}
      </AnimatePresence>

      {/* Floating Input Bar — Compact */}
      <div className="fixed bottom-0 left-0 right-0 z-50 px-2 pb-2 md:px-4 md:pb-3 bg-gradient-to-t from-background via-background/80 to-transparent pt-3">
        <div className="max-w-xl mx-auto">
          <div className="relative bg-card rounded-lg shadow-md border border-border">
            {/* Loading Overlay */}
            {isDisabled && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="absolute inset-0 bg-brand-light/90 dark:bg-brand-dark/80 backdrop-blur-sm rounded-lg z-20 flex items-center justify-center"
              >
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground font-medium font-heading">
                    Loading model...
                  </span>
                  <div className="w-16 bg-border rounded-full h-1">
                    <motion.div
                      className="bg-brand-orange h-1 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${progressPercentage || 0}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-brand-orange font-medium">
                    {progressPercentage || 0}%
                  </span>
                </div>
              </motion.div>
            )}

            {/* Single Row — Model + Input + Send */}
            <div className="flex items-center gap-1.5 p-1">
              <ModelSelector
                selectedModel={selectedModel}
                onModelSelect={handleModelSelect}
                isLoading={isInitializing}
                progressPercentage={progressPercentage}
              />
              <div className="relative group">
                <button className="p-0.5 text-amber-500 hover:text-amber-600 dark:text-amber-400">
                  <AlertTriangle className="w-3 h-3" />
                </button>
                <div className="absolute left-0 bottom-full mb-1 w-44 p-1.5 bg-gray-900 dark:bg-gray-700 text-white text-[10px] rounded opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                  Small models may produce inaccurate information.
                </div>
              </div>
              <div className="flex-1">
                <Textarea
                  ref={textareaRef}
                  placeholder="Ask me anything..."
                  className="min-h-[28px] max-h-[60px] py-1.5 px-2.5 resize-none rounded-md text-xs border-0 bg-muted focus:ring-1 focus:ring-brand-orange font-sans"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault();
                      handleSubmit();
                    }
                  }}
                  rows={1}
                  disabled={isDisabled}
                />
              </div>
              <Button
                className="rounded-md p-1 h-7 w-7 flex-shrink-0"
                onClick={handleSubmit}
                disabled={!question.trim() || isDisabled || isLoading || isStreaming}
              >
                <ArrowUpIcon size={11} />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};
