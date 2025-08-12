

interface WaveLoadingProps {
  message?: string;
  progress?: number;
  estimatedTime?: string;
}

export const WaveLoading = ({ message = "Initializing AI model...", progress, estimatedTime }: WaveLoadingProps) => {
  return (
    <div className="flex items-center justify-center p-8 min-h-[200px]">
      <div className="text-center">
        {/* Wave Animation */}
        <div className="flex justify-center mb-6">
          <div className="flex space-x-1">
            {[...Array(7)].map((_, i) => (
              <div
                key={i}
                className="w-2 h-12 bg-gradient-to-t from-blue-600 to-blue-400 dark:from-blue-500 dark:to-blue-300 rounded-full animate-wave"
                style={{
                  animationDelay: `${i * 0.1}s`,
                  transformOrigin: 'bottom'
                }}
              />
            ))}
          </div>
        </div>
        
        {/* Progress Text */}
        <div className="text-gray-700 dark:text-gray-300 text-sm font-medium mb-2">
          {message}
        </div>
        
        {/* Progress Bar */}
        {progress !== undefined && (
          <div className="w-64 mx-auto mb-3">
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-blue-500 to-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              {progress}% complete
            </div>
          </div>
        )}
        
        {/* Estimated Time */}
        {estimatedTime && (
          <div className="text-xs text-gray-500 dark:text-gray-400">
            {estimatedTime}
          </div>
        )}
        
        {/* Subtitle */}
        <div className="text-xs text-gray-500 dark:text-gray-400 mt-2">
          This may take a few moments on first load
        </div>
      </div>
    </div>
  );
};
