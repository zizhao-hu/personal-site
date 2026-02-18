import { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { ChevronDown, Brain, Clock } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export interface ModelInfo {
  id: string;
  name: string;
  size: string;
  estimatedLoadTime: string;
  description: string;
}

interface ModelSelectorProps {
  selectedModel: string;
  onModelSelect: (modelId: string) => void;
  isLoading: boolean;
  progressPercentage?: number;
}

const AVAILABLE_MODELS: ModelInfo[] = [
  {
    id: "SmolLM2-360M-Instruct-q4f16_1-MLC",
    name: "SmolLM2 360M",
    size: "360M parameters (~200MB)",
    estimatedLoadTime: "10-20 seconds",
    description: "Fastest model, great for mobile & quick responses"
  },
  {
    id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
    name: "Qwen 2.5 0.5B",
    size: "0.5B parameters (~350MB)",
    estimatedLoadTime: "15-30 seconds",
    description: "Fast and capable, excellent default choice"
  },
  {
    id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
    name: "Qwen 2.5 1.5B",
    size: "1.5B parameters (~900MB)",
    estimatedLoadTime: "30-60 seconds",
    description: "Better quality, still fast"
  },
  {
    id: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    name: "Llama 3.2 1B",
    size: "1B parameters (~700MB)",
    estimatedLoadTime: "25-45 seconds",
    description: "Meta's latest small model, great quality"
  },
  {
    id: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    name: "Llama 3.2 3B",
    size: "3B parameters (~2GB)",
    estimatedLoadTime: "60-120 seconds",
    description: "Excellent quality, recommended for detailed responses"
  },
  {
    id: "Phi-3.5-mini-instruct-q4f16_1-MLC",
    name: "Phi 3.5 Mini",
    size: "3.8B parameters (~2GB)",
    estimatedLoadTime: "60-120 seconds",
    description: "Microsoft's best small model, superior reasoning"
  },
  {
    id: "gemma-2-2b-it-q4f16_1-MLC",
    name: "Gemma 2 2B",
    size: "2B parameters (~1.5GB)",
    estimatedLoadTime: "45-90 seconds",
    description: "Google's efficient instruction-tuned model"
  }
];

export const ModelSelector = ({ selectedModel, onModelSelect, isLoading, progressPercentage }: ModelSelectorProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [currentModel, setCurrentModel] = useState<ModelInfo | null>(null);
  const [dropdownPos, setDropdownPos] = useState<{ top: number; left: number } | null>(null);
  const btnRef = useRef<HTMLButtonElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Element;
      if (!target.closest('.model-selector') && !target.closest('.model-selector-portal')) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  useEffect(() => {
    const model = AVAILABLE_MODELS.find(m => m.id === selectedModel);
    setCurrentModel(model || AVAILABLE_MODELS[0]);
  }, [selectedModel]);

  // Calculate position when opening
  useEffect(() => {
    if (isOpen && btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect();
      setDropdownPos({
        top: rect.top, // dropdown will be positioned bottom-anchored to this point
        left: rect.left,
      });
    }
  }, [isOpen]);

  const handleModelSelect = (modelId: string) => {
    onModelSelect(modelId);
    setIsOpen(false);
  };

  if (!currentModel) return null;

  return (
    <div className="relative model-selector">
      <button
        ref={btnRef}
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-1.5 px-2 py-1.5 backdrop-blur-sm rounded-lg transition-all duration-200 border shadow-sm ${isLoading
          ? 'bg-blue-100/90 dark:bg-blue-900/50 border-blue-300 dark:border-blue-600'
          : 'bg-white/80 dark:bg-gray-800/80 hover:bg-white dark:hover:bg-gray-800 border-gray-200 dark:border-gray-600'
          }`}
      >
        <Brain className={`w-3.5 h-3.5 ${isLoading ? 'text-blue-700 dark:text-blue-300' : 'text-blue-600 dark:text-blue-400'}`} />
        <span className={`text-xs font-medium ${isLoading ? 'text-blue-700 dark:text-blue-300' : 'text-gray-700 dark:text-gray-300'}`}>
          {isLoading ? 'Loading...' : currentModel.name}
        </span>
        {isLoading && (
          <div className="w-2.5 h-2.5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
        )}
        <ChevronDown className={`w-3 h-3 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''} ${isLoading ? 'text-blue-600 dark:text-blue-400' : 'text-gray-500'}`} />
      </button>

      {createPortal(
        <AnimatePresence>
          {isOpen && dropdownPos && (
            <motion.div
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.95 }}
              transition={{ duration: 0.15 }}
              className="model-selector-portal fixed w-80 bg-white dark:bg-gray-800 rounded-lg shadow-2xl border border-gray-200 dark:border-gray-700"
              style={{
                zIndex: 99999,
                bottom: `${window.innerHeight - dropdownPos.top + 8}px`,
                left: `${Math.min(dropdownPos.left, window.innerWidth - 330)}px`,
              }}
            >
              <div className="p-3 border-b border-gray-200 dark:border-gray-700">
                <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200 mb-1">
                  Select AI Model
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Choose a model based on your needs
                </p>
              </div>

              <div className="max-h-64 overflow-y-auto">
                {AVAILABLE_MODELS.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => handleModelSelect(model.id)}
                    className={`w-full p-3 text-left hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors duration-150 ${model.id === selectedModel ? 'bg-blue-50 dark:bg-blue-900/20 border-r-2 border-blue-500' : ''
                      }`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex-1">
                        <h4 className="text-sm font-medium text-gray-800 dark:text-gray-200">
                          {model.name}
                        </h4>
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          {model.description}
                        </p>
                      </div>
                      {model.id === selectedModel && (
                        <div className="w-2 h-2 bg-blue-500 rounded-full ml-2 mt-1" />
                      )}
                    </div>

                    <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                      <span>{model.size}</span>
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        <span>{model.estimatedLoadTime}</span>
                      </div>
                    </div>
                  </button>
                ))}
              </div>

              {isLoading && progressPercentage !== undefined && (
                <div className="p-3 border-t border-gray-200 dark:border-gray-700 bg-blue-50 dark:bg-blue-900/20">
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-xs font-medium text-blue-700 dark:text-blue-300">
                      Loading {currentModel?.name}...
                    </p>
                    <p className="text-xs font-medium text-blue-600 dark:text-blue-400">
                      {progressPercentage}%
                    </p>
                  </div>
                  <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-2">
                    <motion.div
                      className="bg-blue-600 h-2 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${progressPercentage}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                </div>
              )}
              <div className="p-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700/50 rounded-b-lg">
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Models are loaded in your browser for privacy and speed
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}
    </div>
  );
}; 