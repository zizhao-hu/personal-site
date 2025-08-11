import { motion } from 'framer-motion';
import { MessageCircle, BotIcon, GraduationCap, Code, Brain, Rocket } from 'lucide-react';
import { getConversationStarters } from '@/data/predefined-qa';

interface ConversationStartersProps {
  onStarterClick: (question: string) => void;
  showTitle?: boolean;
  compact?: boolean;
}

export const ConversationStarters = ({ onStarterClick, showTitle = true, compact = false }: ConversationStartersProps) => {
  // Get conversation starters from predefined questions
  const startersData = getConversationStarters();
  
  // Map icon names to actual icon components
  const iconMap = {
    Brain: <Brain className="w-4 h-4" />,
    Code: <Code className="w-4 h-4" />,
    Rocket: <Rocket className="w-4 h-4" />,
    GraduationCap: <GraduationCap className="w-4 h-4" />,
    MessageCircle: <MessageCircle className="w-4 h-4" />,
    BotIcon: <BotIcon className="w-4 h-4" />
  };

  const conversationStarters = startersData.map(starter => ({
    question: starter.question,
    icon: iconMap[starter.icon as keyof typeof iconMap] || <MessageCircle className="w-4 h-4" />
  }));

  return (
    <div className="rounded-xl p-4 flex flex-col gap-3">
      {showTitle && (
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
          Ask me about:
        </h3>
      )}
      <div className={`grid gap-2 ${compact ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1 md:grid-cols-2'}`}>
        {conversationStarters.map((starter, index) => (
          <motion.button
            key={index}
            className={`flex items-center gap-3 p-3 text-left rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600 hover:bg-blue-50 dark:hover:bg-gray-800 transition-all duration-200 ${compact ? 'text-xs' : 'text-sm'}`}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => onStarterClick(starter.question)}
          >
            <div className="text-blue-600 dark:text-blue-400">
              {starter.icon}
            </div>
            <span className="text-gray-700 dark:text-gray-300">
              {starter.question}
            </span>
          </motion.button>
        ))}
      </div>
    </div>
  );
}; 