import { motion } from 'framer-motion';

interface LoadingProps {
  message?: string;
  subMessage?: string;
  progress?: number; // Progress as a percentage (0-100)
}

export const Loading = ({ message = "Loading...", subMessage, progress }: LoadingProps) => {
  return (
    <div className="flex items-center justify-center p-8">
      <div className="text-center">
        <motion.div 
          className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        />
        <p className="text-gray-600 dark:text-gray-400">{message}</p>
        {subMessage && (
          <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">{subMessage}</p>
        )}
        {progress !== undefined && (
          <div className="mt-4 w-80 mx-auto">
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
              <motion.div 
                className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full shadow-sm"
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.3, ease: "easeOut" }}
              />
            </div>
            <div className="flex justify-between items-center mt-2">
              <p className="text-xs text-gray-500 dark:text-gray-400">Loading model...</p>
              <p className="text-xs font-medium text-blue-600 dark:text-blue-400">{progress}%</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 