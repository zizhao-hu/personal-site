import { ThemeToggle } from "./theme-toggle";
import { Menu, X, ChevronDown } from "lucide-react";
import { useNavigate, useLocation } from "react-router-dom";
import { useState, useRef, useEffect } from "react";

interface HeaderProps {
  onHomeClick?: () => void;
}

const navItems = [
  { path: "/", label: "About" },
  { path: "/projects", label: "Projects" },
  { path: "/blogs", label: "Blogs" },
  { path: "/tutorials", label: "Tutorials" },
];

const researchSubItems = [
  { path: "/research", label: "Overview" },
  { path: "/research/llm-vlm", label: "LLM / VLM" },
  { path: "/research/architecture", label: "Architecture" },
  { path: "/research/continual-learning", label: "Continual Learning" },
  { path: "/research/synthetic-data", label: "Synthetic Data" },
];

export const Header = ({ onHomeClick }: HeaderProps) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [researchDropdownOpen, setResearchDropdownOpen] = useState(false);
  const [mobileResearchOpen, setMobileResearchOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const isActive = (path: string) => {
    if (path === "/") return location.pathname === "/";
    if (path === "/research") return location.pathname === "/research";
    return location.pathname.startsWith(path);
  };

  const isResearchActive = location.pathname.startsWith("/research");

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setResearchDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

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

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center h-full">
          {/* About tab */}
          <button
            onClick={() => navigate("/")}
            className={`relative px-3 py-2.5 text-xs font-medium transition-colors duration-200 ${isActive("/")
                ? "text-blue-600 dark:text-blue-400"
                : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
              }`}
          >
            About
            {isActive("/") && (
              <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600 dark:bg-blue-400" />
            )}
          </button>

          {/* Research dropdown */}
          <div
            ref={dropdownRef}
            className="relative"
            onMouseEnter={() => setResearchDropdownOpen(true)}
            onMouseLeave={() => setResearchDropdownOpen(false)}
          >
            <button
              onClick={() => {
                navigate("/research");
                setResearchDropdownOpen(false);
              }}
              className={`relative px-3 py-2.5 text-xs font-medium transition-colors duration-200 flex items-center gap-0.5 ${isResearchActive
                  ? "text-blue-600 dark:text-blue-400"
                  : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
                }`}
            >
              Research
              <ChevronDown className={`w-3 h-3 transition-transform duration-200 ${researchDropdownOpen ? "rotate-180" : ""}`} />
              {isResearchActive && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600 dark:bg-blue-400" />
              )}
            </button>

            {/* Dropdown panel */}
            {researchDropdownOpen && (
              <div className="absolute top-full left-0 mt-0 w-52 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl py-1 z-50 animate-in fade-in slide-in-from-top-1 duration-150">
                {researchSubItems.map((item) => {
                  const active = isActive(item.path);
                  return (
                    <button
                      key={item.path}
                      onClick={() => {
                        navigate(item.path);
                        setResearchDropdownOpen(false);
                      }}
                      className={`w-full text-left px-3 py-2 text-xs font-medium transition-colors duration-150 ${active
                          ? "text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30"
                          : "text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-white"
                        }`}
                    >
                      {item.label}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Remaining nav items */}
          {navItems.filter(item => item.path !== "/").map((item) => {
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
        <div className="md:hidden bg-background/95 backdrop-blur-md border-b border-gray-200 dark:border-gray-700 px-2 py-1 sticky top-[41px] z-40">
          {/* About */}
          <button
            onClick={() => {
              navigate("/");
              setMobileMenuOpen(false);
            }}
            className={`relative w-full text-left px-3 py-2 text-xs font-medium transition-colors duration-200 ${isActive("/")
                ? "text-blue-600 dark:text-blue-400 border-l-2 border-blue-600 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/20"
                : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-800"
              }`}
          >
            About
          </button>

          {/* Research collapsible group */}
          <button
            onClick={() => setMobileResearchOpen(!mobileResearchOpen)}
            className={`relative w-full text-left px-3 py-2 text-xs font-medium transition-colors duration-200 flex items-center justify-between ${isResearchActive
                ? "text-blue-600 dark:text-blue-400 border-l-2 border-blue-600 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/20"
                : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-800"
              }`}
          >
            Research
            <ChevronDown className={`w-3 h-3 transition-transform duration-200 ${mobileResearchOpen ? "rotate-180" : ""}`} />
          </button>

          {mobileResearchOpen && (
            <div className="ml-3 border-l border-gray-200 dark:border-gray-700">
              {researchSubItems.map((item) => {
                const active = isActive(item.path);
                return (
                  <button
                    key={item.path}
                    onClick={() => {
                      navigate(item.path);
                      setMobileMenuOpen(false);
                    }}
                    className={`relative w-full text-left pl-4 pr-3 py-1.5 text-xs font-medium transition-colors duration-200 ${active
                        ? "text-blue-600 dark:text-blue-400 bg-blue-50/50 dark:bg-blue-900/10"
                        : "text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
                      }`}
                  >
                    {item.label}
                  </button>
                );
              })}
            </div>
          )}

          {/* Remaining items */}
          {navItems.filter(item => item.path !== "/").map((item) => {
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

