import { Textarea } from "../ui/textarea";
import { cx } from 'classix';
import { Button } from "../ui/button";
import { ArrowUpIcon } from "./icons"
import { toast } from 'sonner';
import { motion } from 'framer-motion';
import { ModelSelector } from "./model-selector";

interface ChatInputProps {
    question: string;
    setQuestion: (question: string) => void;
    onSubmit: (text?: string) => void;
    isLoading: boolean;
    disabled?: boolean;
    selectedModel: string;
    onModelSelect: (modelId: string) => void;
    isModelLoading: boolean;
    modelProgressPercentage?: number;
}



export const ChatInput = ({ question, setQuestion, onSubmit, isLoading, disabled = false, selectedModel, onModelSelect, isModelLoading, modelProgressPercentage }: ChatInputProps) => {
    return(
    <div className="relative w-full flex flex-col gap-4">
        <input
        type="file"
        className="fixed -top-4 -left-4 size-0.5 opacity-0 pointer-events-none"
        multiple
        tabIndex={-1}
        />

        <div className="relative">
            {/* Loading Overlay - Show when model is not ready */}
            {disabled && (
                <motion.div 
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ duration: 0.2 }}
                    className="absolute inset-0 bg-blue-50/80 dark:bg-blue-900/20 backdrop-blur-sm rounded-xl border-2 border-blue-200 dark:border-blue-700 z-20 flex items-center justify-center"
                >
                    <div className="text-center">
                        <motion.div 
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2 }}
                            className="w-48 mx-auto"
                        >
                            <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-2 mb-2">
                                <motion.div 
                                    className="bg-blue-600 h-2 rounded-full"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${modelProgressPercentage || 0}%` }}
                                    transition={{ duration: 0.3 }}
                                />
                            </div>
                            <p className="text-xs text-blue-600 dark:text-blue-400 font-medium">
                                {modelProgressPercentage || 0}% Complete
                            </p>
                        </motion.div>
                    </div>
                </motion.div>
            )}

            <Textarea
            placeholder="Send a message..."
            className={cx(
                'min-h-[24px] max-h-[calc(75dvh)] overflow-hidden resize-none rounded-xl text-base pr-16 pb-12 relative'
            )}
            style={isModelLoading ? {
                background: 'linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4, #3b82f6, #8b5cf6, #06b6d4, #3b82f6)',
                backgroundSize: '400% 100%',
                animation: 'gradientMove 3s linear infinite'
            } : {}}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();

                    if (isLoading || disabled) {
                        toast.error('Please wait for the model to finish its response!');
                    } else {
                        onSubmit();
                    }
                }
            }}
            rows={3}
            autoFocus
            disabled={disabled}
            />

            {/* Model Selector - Bottom Left */}
            <div className="absolute bottom-2 left-2 z-10">
                <ModelSelector 
                    selectedModel={selectedModel}
                    onModelSelect={onModelSelect}
                    isLoading={isModelLoading}
                    progressPercentage={modelProgressPercentage}
                />
            </div>

            {/* Send Button - Bottom Right */}
            <Button 
                className="rounded-full p-1.5 h-fit absolute bottom-2 right-2 m-0.5 border dark:border-zinc-600"
                onClick={() => onSubmit(question)}
                disabled={question.length === 0 || disabled}
            >
                <ArrowUpIcon size={14} />
            </Button>
        </div>
    </div>
    );
}