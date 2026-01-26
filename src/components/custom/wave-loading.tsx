

interface WaveLoadingProps {
  message?: string;
  progress?: number;
  estimatedTime?: string;
}

export const WaveLoading = ({ message = "Initializing AI model...", progress, estimatedTime }: WaveLoadingProps) => {
  console.log("WaveLoading rendered with:", { message, progress, estimatedTime });
  
  return (
    <div className="flex items-center justify-center p-8 min-h-[200px]">
      <div className="text-center">
        {/* Wave Animation */}
        <div className="flex justify-center mb-8">
          <div className="flex space-x-2">
            {[...Array(7)].map((_, i) => (
              <div
                key={i}
                className="w-3 h-16 bg-gradient-to-t from-blue-600 to-blue-400 dark:from-blue-500 dark:to-blue-300 rounded-full animate-wave shadow-lg"
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
        <div className="w-80 mx-auto mb-4">
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 shadow-inner">
            <div 
              className="bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 h-3 rounded-full transition-all duration-500 ease-out shadow-sm"
              style={{ width: `${Math.max(progress || 0, 5)}%` }}
            />
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400 mt-2 font-medium">
            {Math.round(progress || 0)}% complete
          </div>
        </div>
        
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
