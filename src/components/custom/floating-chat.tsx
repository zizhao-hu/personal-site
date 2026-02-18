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
import { X, MessageSquare, ChevronDown, AlertTriangle, Minus } from "lucide-react";
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
  const [drawerOpen, setDrawerOpen] = useState(false);

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
      setDrawerOpen(true);
      setTimeout(() => textareaRef.current?.focus(), 100);
    };

    window.addEventListener('chat-question', handleChatQuestion as EventListener);
    return () => window.removeEventListener('chat-question', handleChatQuestion as EventListener);
  }, []);

  // Initialize services
  useEffect(() => {
    const initializeServices = async () => {
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
              if (percentageMatch) setProgressPercentage(parseInt(percentageMatch[1]));
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
        if (percentageMatch) setProgressPercentage(parseInt(percentageMatch[1]));
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
    setDrawerOpen(true);

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
  const hasMessages = messages.length > 0;

  return (
    <>
      {/* ─── Backdrop ───────────────────────────────────────────── */}
      <AnimatePresence>
        {drawerOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-40 bg-brand-dark/20 dark:bg-brand-dark/40 backdrop-blur-[2px]"
            onClick={() => setDrawerOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* ─── Floating Chat Container ─────────────────────────────── */}
      <div className="fixed bottom-0 left-0 right-0 z-50 flex flex-col items-center pointer-events-none">
        <div className="w-full max-w-xl px-2 md:px-4 pb-2 md:pb-3 pointer-events-auto">

          {/* ─ Drawer (messages) ─ appears right above the input ── */}
          <AnimatePresence>
            {drawerOpen && hasMessages && (
              <motion.div
                initial={{ opacity: 0, y: 30, scaleY: 0.92 }}
                animate={{ opacity: 1, y: 0, scaleY: 1 }}
                exit={{ opacity: 0, y: 30, scaleY: 0.92 }}
                transition={{
                  type: "spring",
                  stiffness: 380,
                  damping: 28,
                  mass: 0.7
                }}
                style={{ transformOrigin: "bottom center" }}
                className="mb-2 relative overflow-hidden rounded-2xl"
              >
                {/* Futuristic panel border effect */}
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-b from-brand-orange/20 via-transparent to-brand-orange/10 pointer-events-none z-10" />
                <div className="absolute inset-[1px] rounded-[15px] bg-background z-0" />

                {/* Scan-line overlay for futuristic feel */}
                <div
                  className="absolute inset-0 rounded-2xl pointer-events-none z-20 opacity-[0.03]"
                  style={{
                    backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, currentColor 2px, currentColor 3px)",
                  }}
                />

                {/* Actual content */}
                <div className="relative z-10 border border-brand-orange/20 dark:border-brand-orange/15 rounded-2xl shadow-elevation-4 dark:shadow-elevation-4-dark overflow-hidden bg-background">
                  {/* Drawer Header */}
                  <div className="flex items-center justify-between px-4 py-2.5 border-b border-border bg-muted/50">
                    <div className="flex items-center gap-2">
                      {/* Pulsing status dot */}
                      <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-orange opacity-50" />
                        <span className="relative inline-flex h-2 w-2 rounded-full bg-brand-orange" />
                      </span>
                      <MessageSquare className="w-3.5 h-3.5 text-brand-orange" />
                      <span className="text-xs font-medium font-heading text-foreground tracking-wide">
                        AI Chat
                      </span>
                      <span className="text-[10px] text-muted-foreground font-heading">
                        ({messages.length} msg{messages.length !== 1 ? "s" : ""})
                      </span>
                    </div>
                    <div className="flex items-center gap-0.5">
                      <button
                        onClick={() => setDrawerOpen(false)}
                        className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                        title="Collapse"
                      >
                        <Minus className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => {
                          setMessages([]);
                          setDrawerOpen(false);
                        }}
                        className="p-1.5 rounded-lg text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                        title="Clear chat"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  {/* Messages area */}
                  <div className="overflow-y-auto max-h-[50vh] p-3 space-y-1 scroll-smooth">
                    {messages.map((msg, index) => (
                      <PreviewMessage
                        key={msg.id || index}
                        message={msg}
                        isStreaming={isStreaming && index === messages.length - 1 && msg.role === 'assistant'}
                      />
                    ))}
                    {isLoading && !isStreaming && <ThinkingMessage />}
                    <div ref={messagesEndRef} />
                  </div>

                  {/* Bottom edge accent glow */}
                  <div className="h-px bg-gradient-to-r from-transparent via-brand-orange/30 to-transparent" />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* ─ Minimised badge — tap to reopen ───────────────────── */}
          <AnimatePresence>
            {!drawerOpen && hasMessages && (
              <motion.button
                initial={{ opacity: 0, scale: 0.85, y: 8 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.85, y: 8 }}
                transition={{ type: "spring", stiffness: 400, damping: 22 }}
                onClick={() => setDrawerOpen(true)}
                className="mb-2 mx-auto flex items-center gap-2 px-3 py-1.5 rounded-full bg-brand-orange text-white shadow-lg hover:bg-brand-orange/90 transition-colors font-heading text-xs"
              >
                <MessageSquare className="w-3.5 h-3.5" />
                <span>{messages.length} message{messages.length !== 1 ? "s" : ""}</span>
                <ChevronDown className="w-3 h-3 rotate-180" />
              </motion.button>
            )}
          </AnimatePresence>

          {/* ─ Input Bar ─────────────────────────────────────────── */}
          <div className="relative bg-card rounded-xl shadow-elevation-3 dark:shadow-elevation-3-dark border border-border overflow-hidden">
            {/* Loading Overlay */}
            {isDisabled && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="absolute inset-0 bg-brand-light/90 dark:bg-brand-dark/80 backdrop-blur-sm rounded-xl z-20 flex items-center justify-center"
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
                  <span className="text-[10px] text-brand-orange font-medium font-heading">
                    {progressPercentage || 0}%
                  </span>
                </div>
              </motion.div>
            )}

            {/* Top edge accent */}
            <div className="h-[2px] bg-gradient-to-r from-transparent via-brand-orange/40 to-transparent" />

            {/* Single Row — Model + Input + Send */}
            <div className="flex items-center gap-1.5 p-1.5">
              <ModelSelector
                selectedModel={selectedModel}
                onModelSelect={handleModelSelect}
                isLoading={isInitializing}
                progressPercentage={progressPercentage}
              />
              <div className="relative group">
                <button className="p-0.5 text-brand-orange/60 hover:text-brand-orange transition-colors">
                  <AlertTriangle className="w-3 h-3" />
                </button>
                <div className="absolute left-0 bottom-full mb-1 w-44 p-1.5 bg-brand-dark text-brand-light text-[10px] rounded-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 font-heading">
                  Small models may produce inaccurate information.
                </div>
              </div>
              <div className="flex-1">
                <Textarea
                  ref={textareaRef}
                  placeholder="Ask me anything..."
                  className="min-h-[28px] max-h-[60px] py-1.5 px-2.5 resize-none rounded-lg text-xs border-0 bg-muted focus:ring-1 focus:ring-brand-orange/50 font-sans"
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

              {/* Chat toggle when messages exist */}
              {hasMessages && !drawerOpen && (
                <button
                  onClick={() => setDrawerOpen(true)}
                  className="relative p-1.5 rounded-lg text-brand-orange hover:bg-brand-orange/10 transition-colors"
                  title="Show chat"
                >
                  <MessageSquare className="w-4 h-4" />
                  <span className="absolute -top-0.5 -right-0.5 w-3.5 h-3.5 bg-brand-orange text-white text-[8px] rounded-full flex items-center justify-center font-heading">
                    {messages.length}
                  </span>
                </button>
              )}

              <Button
                className="rounded-lg p-1 h-7 w-7 flex-shrink-0 bg-brand-orange hover:bg-brand-orange/85 text-white"
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
