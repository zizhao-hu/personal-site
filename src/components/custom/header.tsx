import { ThemeToggle } from "./theme-toggle";
import { GraduationCap } from "lucide-react";
import { useNavigate } from "react-router-dom";

interface HeaderProps {
  onHomeClick?: () => void;
}

export const Header = ({ onHomeClick }: HeaderProps) => {
  const navigate = useNavigate();
  return (
    <>
      <header className="flex items-center justify-between px-2 sm:px-4 py-2 bg-background text-black dark:text-white w-full border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-3">
          <button 
            onClick={() => {
              navigate('/');
              if (onHomeClick) {
                onHomeClick();
              }
            }}
            className="flex items-center gap-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg px-2 py-1 transition-colors duration-200"
          >
            <GraduationCap className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <span className="font-semibold text-gray-800 dark:text-gray-200 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200">
              Zizhao Hu
            </span>
            <span className="text-sm text-gray-500 dark:text-gray-400">• CS Ph.D. Student at USC</span>
          </button>
        </div>
        <div className="flex items-center space-x-1 sm:space-x-2">
          <ThemeToggle />
        </div>
      </header>
    </>
  );
};