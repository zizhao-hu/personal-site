import { useState, useEffect, useRef, useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import { Textarea } from "../ui/textarea";
import { ArrowUpIcon, SparklesIcon } from "./icons";
import { ModelSelector } from "./model-selector";
import { Markdown } from "./markdown";
import { prismService, type ChatMessage } from "@/lib/prism-service";
import { webLLMService } from "@/lib/webllm-service";
import { message } from "@/interfaces/interfaces";
import { motion, AnimatePresence } from "framer-motion";
import { X, MessageSquare, ChevronDown, Minus, Zap, Brain } from "lucide-react";

// ── Mode types ──────────────────────────────────────────────────
type ChatMode = "prism" | "ai";
type AILoadState = "idle" | "loading" | "ready" | "failed";

export const FloatingChat = () => {
  // ── Core chat state ───────────────────────────────────────────
  const [messages, setMessages] = useState<message[]>([]);
  const [question, setQuestion] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);

  // ── Mode state ────────────────────────────────────────────────
  const [activeMode, setActiveMode] = useState<ChatMode>("prism");
  const [aiLoadState, setAILoadState] = useState<AILoadState>("idle");
  const [aiProgress, setAIProgress] = useState(0);
  const [selectedModel, setSelectedModel] = useState("Qwen2.5-0.5B-Instruct-q4f16_1-MLC");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const aiLoadAttempted = useRef(false);

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

  // ── Background AI Loading ─────────────────────────────────────
  const loadAIInBackground = useCallback(async (modelId: string) => {
    // Skip on GPU-heavy pages
    if (window.location.pathname.includes('starship-sim')) return;

    // Skip on mobile
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    if (isMobile) return;

    setAILoadState("loading");
    setAIProgress(0);

    try {
      await webLLMService.initialize((progress) => {
        const percentageMatch = progress.match(/(\d+)%/);
        if (percentageMatch) {
          setAIProgress(parseInt(percentageMatch[1]));
        }
      }, modelId);

      setAILoadState("ready");
      setActiveMode("ai");
      console.log("✅ AI model loaded — switching to AI mode");
    } catch (err) {
      console.warn("⚠️ AI model failed to load, staying in PRISM mode:", err);
      setAILoadState("failed");
    }
  }, []);

  // Trigger background AI load on mount (with delay)
  useEffect(() => {
    if (aiLoadAttempted.current) return;
    aiLoadAttempted.current = true;

    const timer = setTimeout(() => {
      loadAIInBackground(selectedModel);
    }, 2000);

    return () => clearTimeout(timer);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Model Switching ───────────────────────────────────────────
  const handleModelSelect = async (modelId: string) => {
    if (modelId === selectedModel && aiLoadState === 'ready') return;
    setSelectedModel(modelId);
    setActiveMode("prism"); // Fall back while loading
    await loadAIInBackground(modelId);
  };

  // ── Submit Handler ────────────────────────────────────────────
  const handleSubmit = async () => {
    if (isStreaming || !question.trim()) return;

    const userMessage: message = {
      id: uuidv4(),
      role: "user",
      content: question,
    };

    setMessages(prev => [...prev, userMessage]);
    setQuestion("");
    setDrawerOpen(true);

    const chatHistory: ChatMessage[] = messages
      .filter(msg => msg.role !== "system")
      .map(msg => ({
        role: msg.role as "user" | "assistant",
        content: msg.content,
      }));
    chatHistory.push({ role: "user", content: userMessage.content });

    const assistantMessageId = uuidv4();
    const assistantMessage: message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
    };
    setMessages(prev => [...prev, assistantMessage]);

    try {
      if (activeMode === "ai" && webLLMService.isReady()) {
        // ── AI Mode: stream tokens ────────────────────────────
        setIsStreaming(true);
        let fullContent = "";
        for await (const chunk of webLLMService.generateStreamingResponse(chatHistory)) {
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
        // ── PRISM Mode: instant response ──────────────────────
        const response = prismService.generateResponse(chatHistory);
        const words = response.split(" ");
        let built = "";
        for (let i = 0; i < words.length; i += 3) {
          built += (built ? " " : "") + words.slice(i, i + 3).join(" ");
          const snapshot = built;
          setMessages(prev =>
            prev.map(msg =>
              msg.id === assistantMessageId
                ? { ...msg, content: snapshot }
                : msg
            )
          );
          await new Promise(r => setTimeout(r, 20));
        }
      }
    } catch (error) {
      console.error("Error generating response:", error);
      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantMessageId
            ? { ...msg, content: "Sorry, I encountered an error. Please try again." }
            : msg
        )
      );
    } finally {
      setIsStreaming(false);
    }
  };

  // ── Derived state ─────────────────────────────────────────────
  const hasMessages = messages.length > 0;
  const ModeIcon = activeMode === "ai" ? Brain : Zap;

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

          {/* ─ Drawer (messages) ─ appears above the input ── */}
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
                className="mb-2 relative rounded-2xl"
              >
                {/* Panel border effect */}
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-b from-brand-orange/20 via-transparent to-brand-orange/10 pointer-events-none z-[1]" />
                <div className="absolute inset-[1px] rounded-[15px] bg-background z-0" />

                {/* Scan-line overlay */}
                <div
                  className="absolute inset-0 rounded-2xl pointer-events-none z-[2] opacity-[0.03]"
                  style={{
                    backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, currentColor 2px, currentColor 3px)",
                  }}
                />

                {/* Content */}
                <div className="relative z-10 border border-brand-orange/20 dark:border-brand-orange/15 rounded-2xl shadow-elevation-4 dark:shadow-elevation-4-dark bg-background">
                  {/* Drawer Header */}
                  <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-muted/50">
                    <div className="flex items-center gap-2">
                      {/* Pulsing status dot */}
                      <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-orange opacity-50" />
                        <span className="relative inline-flex h-2 w-2 rounded-full bg-brand-orange" />
                      </span>

                      {/* Mode label */}
                      <div className="flex items-center gap-1">
                        <ModeIcon className="w-3 h-3 text-brand-orange" />
                        <span className="text-[10px] font-medium font-heading text-foreground tracking-wide">
                          {activeMode === "ai" ? "AI" : "PRISM"}
                        </span>
                      </div>

                      <span className="text-[10px] text-muted-foreground font-heading">
                        · {messages.length} msg{messages.length !== 1 ? "s" : ""}
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
                  <div className="overflow-y-auto max-h-[50vh] p-2 space-y-2 scroll-smooth">
                    {messages.map((msg, index) => (
                      <motion.div
                        key={msg.id || index}
                        initial={{ y: 4, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        className={`flex gap-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        {msg.role === 'assistant' && (
                          <div className="w-5 h-5 flex items-center justify-center rounded-full ring-1 ring-border shrink-0 mt-0.5">
                            <SparklesIcon size={10} />
                          </div>
                        )}
                        <div
                          className={`max-w-[85%] text-[13px] leading-relaxed ${msg.role === 'user'
                            ? 'bg-zinc-700 dark:bg-muted text-white px-2.5 py-1.5 rounded-xl rounded-br-md'
                            : 'text-foreground'
                            }`}
                        >
                          {msg.role === 'assistant' ? (
                            <div className="chat-compact-md">
                              <Markdown>{msg.content}</Markdown>
                              {isStreaming && index === messages.length - 1 && (
                                <span className="inline-block w-1.5 h-3.5 bg-current animate-pulse ml-0.5" />
                              )}
                            </div>
                          ) : (
                            msg.content
                          )}
                        </div>
                      </motion.div>
                    ))}
                    <div ref={messagesEndRef} />
                  </div>

                  {/* Bottom edge accent */}
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
            {/* Top edge accent */}
            <div className="h-[2px] bg-gradient-to-r from-transparent via-brand-orange/40 to-transparent" />

            <div className="flex items-center gap-1.5 p-1.5 relative z-30">
              {/* Gemini-style model selector chip — contains everything */}
              <ModelSelector
                selectedModel={selectedModel}
                onModelSelect={handleModelSelect}
                aiLoadState={aiLoadState}
                aiProgress={aiProgress}
                activeMode={activeMode}
              />

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

              <button
                className="rounded-lg h-7 w-7 flex-shrink-0 bg-brand-orange hover:bg-brand-orange/85 text-white flex items-center justify-center transition-colors disabled:opacity-40 disabled:pointer-events-none"
                onClick={handleSubmit}
                disabled={!question.trim() || isStreaming}
              >
                <ArrowUpIcon size={12} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};
