import { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { ChevronDown, Check, Sparkles, Zap, Brain, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export interface ModelInfo {
  id: string;
  name: string;
  size: string;
  description: string;
}

interface ModelSelectorProps {
  selectedModel: string;
  onModelSelect: (modelId: string) => void;
  aiLoadState: 'idle' | 'loading' | 'ready' | 'failed';
  aiProgress: number;
  activeMode: 'prism' | 'ai';
}

const AVAILABLE_MODELS: ModelInfo[] = [
  {
    id: "SmolLM2-360M-Instruct-q4f16_1-MLC",
    name: "SmolLM2 360M",
    size: "~200MB",
    description: "Fastest, great for mobile"
  },
  {
    id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
    name: "Qwen 2.5 0.5B",
    size: "~350MB",
    description: "Fast & capable, default"
  },
  {
    id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
    name: "Qwen 2.5 1.5B",
    size: "~900MB",
    description: "Better quality, still quick"
  },
  {
    id: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    name: "Llama 3.2 1B",
    size: "~700MB",
    description: "Meta's latest small model"
  },
  {
    id: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    name: "Llama 3.2 3B",
    size: "~2GB",
    description: "Excellent quality"
  },
  {
    id: "Phi-3.5-mini-instruct-q4f16_1-MLC",
    name: "Phi 3.5 Mini",
    size: "~2GB",
    description: "Superior reasoning"
  },
  {
    id: "gemma-2-2b-it-q4f16_1-MLC",
    name: "Gemma 2 2B",
    size: "~1.5GB",
    description: "Google's efficient model"
  }
];

export const ModelSelector = ({
  selectedModel,
  onModelSelect,
  aiLoadState,
  aiProgress,
  activeMode,
}: ModelSelectorProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [dropdownPos, setDropdownPos] = useState<{ top: number; left: number; width: number } | null>(null);
  const btnRef = useRef<HTMLButtonElement>(null);

  const currentModel = AVAILABLE_MODELS.find(m => m.id === selectedModel) || AVAILABLE_MODELS[1];
  const isLoading = aiLoadState === 'loading';
  const isReady = aiLoadState === 'ready';

  // Close dropdown on outside click
  useEffect(() => {
    if (!isOpen) return;
    const handle = (e: MouseEvent) => {
      const t = e.target as Element;
      if (!t.closest('.gemini-selector') && !t.closest('.gemini-dropdown')) setIsOpen(false);
    };
    document.addEventListener('mousedown', handle);
    return () => document.removeEventListener('mousedown', handle);
  }, [isOpen]);

  // Compute dropdown position
  useEffect(() => {
    if (isOpen && btnRef.current) {
      const r = btnRef.current.getBoundingClientRect();
      setDropdownPos({ top: r.top, left: r.left, width: Math.max(r.width, 260) });
    }
  }, [isOpen]);

  const handleSelect = (id: string) => {
    onModelSelect(id);
    setIsOpen(false);
  };



  // ── Derive the chip display ────────────────────────────────────
  let chipLabel: string;
  let ChipIcon: typeof Zap;
  let chipStyle: string;

  if (isReady && activeMode === 'ai') {
    chipLabel = currentModel.name;
    ChipIcon = Sparkles;
    chipStyle = 'bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/40 dark:to-indigo-950/40 border-blue-200/60 dark:border-blue-700/40 text-blue-700 dark:text-blue-300';
  } else if (isLoading) {
    chipLabel = currentModel.name;
    ChipIcon = Loader2;
    chipStyle = 'bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950/30 dark:to-orange-950/30 border-amber-200/60 dark:border-amber-700/40 text-amber-700 dark:text-amber-400';
  } else {
    chipLabel = 'PRISM';
    ChipIcon = Zap;
    chipStyle = 'bg-muted border-border text-muted-foreground';
  }

  return (
    <>
      {/* ─── Gemini-style Chip Button ─────────────────────────── */}
      <button
        ref={btnRef}
        onClick={() => setIsOpen(!isOpen)}
        className={`gemini-selector relative flex items-center gap-1.5 pl-2 pr-1.5 py-1 rounded-full border transition-all duration-300 hover:shadow-sm group ${chipStyle}`}
      >
        {/* Loading shimmer overlay */}
        {isLoading && (
          <div
            className="absolute inset-0 rounded-full overflow-hidden pointer-events-none"
            style={{ zIndex: 0 }}
          >
            <div
              className="absolute inset-0 opacity-20"
              style={{
                background: 'linear-gradient(90deg, transparent 0%, rgba(245,158,11,0.4) 50%, transparent 100%)',
                backgroundSize: '200% 100%',
                animation: 'gemini-shimmer 2s ease-in-out infinite',
              }}
            />
          </div>
        )}

        {/* Chip content */}
        <div className="relative z-10 flex items-center gap-1.5">
          <ChipIcon className={`w-3 h-3 ${isLoading ? 'animate-spin' : ''}`} />
          <span className="text-[11px] font-medium font-heading whitespace-nowrap">
            {chipLabel}
          </span>

          {/* Inline progress — Gemini loading dot style */}
          {isLoading && (
            <span className="text-[9px] font-heading font-semibold tabular-nums opacity-70">
              {aiProgress}%
            </span>
          )}

          {/* Ready checkmark */}
          {isReady && activeMode === 'ai' && (
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
          )}
        </div>

        <ChevronDown className={`relative z-10 w-3 h-3 transition-transform duration-200 opacity-50 group-hover:opacity-80 ${isOpen ? 'rotate-180' : ''}`} />

        {/* Loading progress bar — fills bottom edge of chip */}
        {isLoading && (
          <motion.div
            className="absolute bottom-0 left-0 h-[2px] rounded-full bg-gradient-to-r from-amber-400 via-orange-400 to-amber-400"
            initial={{ width: '0%' }}
            animate={{ width: `${aiProgress}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
            style={{ zIndex: 10 }}
          />
        )}
      </button>

      {/* ─── Dropdown Portal ──────────────────────────────────── */}
      {createPortal(
        <AnimatePresence>
          {isOpen && dropdownPos && (
            <motion.div
              initial={{ opacity: 0, y: 8, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.96 }}
              transition={{ duration: 0.15, ease: 'easeOut' }}
              className="gemini-dropdown fixed rounded-xl shadow-2xl border border-border bg-card/95 backdrop-blur-xl overflow-hidden"
              style={{
                zIndex: 99999,
                bottom: `${window.innerHeight - dropdownPos.top + 8}px`,
                left: `${Math.min(dropdownPos.left, window.innerWidth - 280)}px`,
                width: '268px',
              }}
            >
              {/* Header */}
              <div className="px-3 py-2.5 border-b border-border bg-muted/30">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-3.5 h-3.5 text-brand-orange" />
                  <span className="text-xs font-semibold font-heading text-foreground">Model</span>
                </div>
                <p className="text-[10px] text-muted-foreground mt-0.5 font-heading">
                  Runs locally in your browser
                </p>
              </div>

              {/* PRISM option (always available) */}
              <button
                onClick={() => handleSelect(selectedModel)}
                className={`w-full px-3 py-2 flex items-center gap-2.5 text-left transition-colors hover:bg-muted/50 ${activeMode === 'prism' && !isLoading ? 'bg-amber-50/50 dark:bg-amber-900/10' : ''
                  }`}
                disabled={activeMode === 'prism' && !isLoading}
              >
                <div className="w-6 h-6 rounded-full bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center shrink-0">
                  <Zap className="w-3 h-3 text-amber-600 dark:text-amber-400" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[11px] font-medium text-foreground">PRISM</span>
                    <span className="text-[9px] px-1 py-0.5 rounded bg-amber-100 dark:bg-amber-900/20 text-amber-600 dark:text-amber-400 font-heading font-medium">INSTANT</span>
                  </div>
                  <p className="text-[10px] text-muted-foreground truncate">Vector-matched responses, no loading</p>
                </div>
                {activeMode === 'prism' && !isLoading && (
                  <Check className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400 shrink-0" />
                )}
              </button>

              <div className="h-px bg-border mx-3" />

              {/* AI Models */}
              <div className="max-h-52 overflow-y-auto py-0.5">
                {AVAILABLE_MODELS.map((model) => {
                  const isSelected = model.id === selectedModel && (isReady || isLoading);
                  const isCurrentlyLoading = isLoading && model.id === selectedModel;

                  return (
                    <button
                      key={model.id}
                      onClick={() => handleSelect(model.id)}
                      disabled={isCurrentlyLoading}
                      className={`w-full px-3 py-2 flex items-center gap-2.5 text-left transition-colors hover:bg-muted/50 ${isSelected ? 'bg-blue-50/50 dark:bg-blue-900/10' : ''
                        }`}
                    >
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 ${isSelected
                        ? 'bg-blue-100 dark:bg-blue-900/30'
                        : 'bg-muted'
                        }`}>
                        {isCurrentlyLoading ? (
                          <Loader2 className="w-3 h-3 text-blue-500 animate-spin" />
                        ) : (
                          <Brain className={`w-3 h-3 ${isSelected ? 'text-blue-600 dark:text-blue-400' : 'text-muted-foreground'}`} />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className={`text-[11px] font-medium ${isSelected ? 'text-foreground' : 'text-foreground/80'}`}>
                            {model.name}
                          </span>
                          <span className="text-[9px] text-muted-foreground">{model.size}</span>
                        </div>
                        {isCurrentlyLoading ? (
                          <div className="flex items-center gap-1.5 mt-0.5">
                            <div className="flex-1 h-1 bg-border rounded-full overflow-hidden">
                              <motion.div
                                className="h-full rounded-full bg-gradient-to-r from-blue-400 to-indigo-400"
                                initial={{ width: 0 }}
                                animate={{ width: `${aiProgress}%` }}
                                transition={{ duration: 0.4 }}
                              />
                            </div>
                            <span className="text-[9px] text-blue-500 font-heading font-semibold tabular-nums">
                              {aiProgress}%
                            </span>
                          </div>
                        ) : (
                          <p className="text-[10px] text-muted-foreground truncate">{model.description}</p>
                        )}
                      </div>
                      {isReady && isSelected && (
                        <Check className="w-3.5 h-3.5 text-blue-500 shrink-0" />
                      )}
                    </button>
                  );
                })}
              </div>


              {/* Footer */}
              <div className="px-3 py-2 border-t border-border bg-muted/20">
                <p className="text-[9px] text-muted-foreground font-heading">
                  Private · No data leaves your device
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}

      {/* ─── Shimmer keyframes ────────────────────────────────── */}
      <style>{`
        @keyframes gemini-shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
      `}</style>
    </>
  );
};