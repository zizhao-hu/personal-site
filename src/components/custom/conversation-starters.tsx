import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { getConversationStarters } from '@/data/predefined-qa';

interface ConversationStartersProps {
  onStarterClick: (question: string) => void;
  showTitle?: boolean;
  compact?: boolean;
}

export const ConversationStarters = ({ onStarterClick, showTitle = true, compact = false }: ConversationStartersProps) => {
  // Get conversation starters from predefined questions
  const startersData = getConversationStarters();
  
  // State to track current question index
  const [currentIndex, setCurrentIndex] = useState(0);
  
  const handlePrevious = () => {
    setCurrentIndex((prev) => (prev === 0 ? startersData.length - 1 : prev - 1));
  };
  
  const handleNext = () => {
    setCurrentIndex((prev) => (prev === startersData.length - 1 ? 0 : prev + 1));
  };
  
  const handleStarterClick = () => {
    const currentQuestion = startersData[currentIndex].question;
    onStarterClick(currentQuestion);
  };

  return (
    <div className="rounded-xl p-4 flex flex-col gap-3">
      {showTitle && (
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
          Ask me about:
        </h3>
      )}
      
      {/* Single Question Selector */}
      <div className="flex items-center justify-center gap-3">
        {/* Left Arrow */}
        <button
          onClick={handlePrevious}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          aria-label="Previous question"
        >
          <ChevronLeft className="w-5 h-5 text-gray-600 dark:text-gray-400" />
        </button>
        
        {/* Question Button with Smooth Animation */}
        <div className="flex-1 max-w-md overflow-hidden">
          <AnimatePresence mode="wait">
            <motion.button
              key={currentIndex}
              className={`w-full p-4 text-left rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600 hover:bg-blue-50 dark:hover:bg-gray-800 transition-colors duration-200 ${compact ? 'text-xs' : 'text-sm'}`}
              onClick={handleStarterClick}
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              transition={{ 
                duration: 0.3,
                ease: "easeInOut"
              }}
            >
              <span className="text-gray-700 dark:text-gray-300">
                {startersData[currentIndex].question}
              </span>
            </motion.button>
          </AnimatePresence>
        </div>
        
        {/* Right Arrow */}
        <button
          onClick={handleNext}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          aria-label="Next question"
        >
          <ChevronRight className="w-5 h-5 text-gray-600 dark:text-gray-400" />
        </button>
      </div>
      
      {/* Dots Indicator */}
      <div className="flex justify-center gap-2">
        {startersData.map((_, index) => (
          <button
            key={index}
            onClick={() => setCurrentIndex(index)}
            className={`w-2 h-2 rounded-full transition-all duration-200 ${
              index === currentIndex 
                ? 'bg-blue-600 dark:bg-blue-400 scale-125' 
                : 'bg-gray-300 dark:bg-gray-600 hover:bg-gray-400 dark:hover:bg-gray-500'
            }`}
            aria-label={`Go to question ${index + 1}`}
          />
        ))}
      </div>
    </div>
  );
}; 