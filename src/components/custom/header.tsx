import { ThemeToggle } from "./theme-toggle";
import { Menu, X } from "lucide-react";
import { useNavigate, useLocation } from "react-router-dom";
import { useState } from "react";

interface HeaderProps {
  onHomeClick?: () => void;
}

const navItems = [
  { path: "/", label: "About" },
  { path: "/research", label: "Research" },
  { path: "/projects", label: "Projects" },
  { path: "/blogs", label: "Blogs" },
  { path: "/tutorials", label: "Tutorials" },
];

export const Header = ({ onHomeClick }: HeaderProps) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const isActive = (path: string) => {
    if (path === "/") return location.pathname === "/";
    return location.pathname.startsWith(path);
  };

  return (
    <>
      <header className="sticky top-0 z-50 flex items-center justify-between px-4 sm:px-6 py-0 bg-background/80 backdrop-blur-md text-black dark:text-white w-full border-b border-gray-200 dark:border-gray-700 transition-all duration-300">
        {/* Logo */}
        <button
          onClick={() => {
            navigate('/');
            if (onHomeClick) onHomeClick();
          }}
          className="flex items-center gap-1.5 hover:opacity-80 transition-opacity py-2"
        >
          <div className="w-6 h-6 rounded-md bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center">
            <span className="text-white font-bold text-xs">ZH</span>
          </div>
          <div className="hidden sm:block">
            <span className="text-sm font-semibold text-gray-800 dark:text-gray-200">Zizhao Hu</span>
          </div>
        </button>

        {/* Desktop Navigation - Google News Style Flat Tabs */}
        <nav className="hidden md:flex items-center h-full">
          {navItems.map((item) => {
            const active = isActive(item.path);
            return (
              <button
                key={item.path}
                onClick={() => navigate(item.path)}
                className={`relative px-3 py-2.5 text-xs font-medium transition-colors duration-200 ${active
                    ? "text-blue-600 dark:text-blue-400"
                    : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
                  }`}
              >
                {item.label}
                {/* Active underline indicator */}
                {active && (
                  <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600 dark:bg-blue-400" />
                )}
              </button>
            );
          })}
        </nav>

        {/* Right Side */}
        <div className="flex items-center gap-2">
          <ThemeToggle />

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800"
          >
            {mobileMenuOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
          </button>
        </div>
      </header>

      {/* Mobile Navigation */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-background border-b border-gray-200 dark:border-gray-700 px-2 py-1">
          {navItems.map((item) => {
            const active = isActive(item.path);
            return (
              <button
                key={item.path}
                onClick={() => {
                  navigate(item.path);
                  setMobileMenuOpen(false);
                }}
                className={`relative w-full text-left px-3 py-2 text-xs font-medium transition-colors duration-200 ${active
                    ? "text-blue-600 dark:text-blue-400 border-l-2 border-blue-600 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/20"
                    : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-800"
                  }`}
              >
                {item.label}
              </button>
            );
          })}
        </div>
      )}
    </>
  );
};
