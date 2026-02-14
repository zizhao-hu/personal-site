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
      <header className="sticky top-0 z-50 flex items-center justify-between px-4 sm:px-6 py-0 bg-background/80 backdrop-blur-md text-foreground w-full border-b border-border transition-all duration-300">
        {/* Logo */}
        <button
          onClick={() => {
            navigate('/');
            if (onHomeClick) onHomeClick();
          }}
          className="flex items-center gap-1.5 hover:opacity-80 transition-opacity py-2"
        >
          <div className="w-6 h-6 rounded-md bg-gradient-to-br from-brand-orange to-brand-blue flex items-center justify-center">
            <span className="text-white font-bold text-xs font-heading">ZH</span>
          </div>
          <div className="hidden sm:block">
            <span className="text-sm font-semibold text-foreground font-heading">Zizhao Hu</span>
          </div>
        </button>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center h-full">
          {/* About tab */}
          <button
            onClick={() => navigate("/")}
            className={`relative px-3 py-2.5 text-xs font-medium transition-colors duration-200 font-heading ${isActive("/")
              ? "text-brand-orange"
              : "text-muted-foreground hover:text-foreground"
              }`}
          >
            About
            {isActive("/") && (
              <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-brand-orange" />
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
              className={`relative px-3 py-2.5 text-xs font-medium transition-colors duration-200 flex items-center gap-0.5 font-heading ${isResearchActive
                ? "text-brand-orange"
                : "text-muted-foreground hover:text-foreground"
                }`}
            >
              Research
              <ChevronDown className={`w-3 h-3 transition-transform duration-200 ${researchDropdownOpen ? "rotate-180" : ""}`} />
              {isResearchActive && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-brand-orange" />
              )}
            </button>

            {/* Dropdown panel */}
            {researchDropdownOpen && (
              <div className="absolute top-full left-0 mt-0 w-52 bg-card border border-border rounded-lg shadow-xl py-1 z-50 animate-in fade-in slide-in-from-top-1 duration-150">
                {researchSubItems.map((item) => {
                  const active = isActive(item.path);
                  return (
                    <button
                      key={item.path}
                      onClick={() => {
                        navigate(item.path);
                        setResearchDropdownOpen(false);
                      }}
                      className={`w-full text-left px-3 py-2 text-xs font-medium transition-colors duration-150 font-heading ${active
                        ? "text-brand-orange bg-accent"
                        : "text-foreground/70 hover:bg-muted hover:text-foreground"
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
                className={`relative px-3 py-2.5 text-xs font-medium transition-colors duration-200 font-heading ${active
                  ? "text-brand-orange"
                  : "text-muted-foreground hover:text-foreground"
                  }`}
              >
                {item.label}
                {active && (
                  <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-brand-orange" />
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
            className="md:hidden p-1.5 rounded-md hover:bg-muted"
          >
            {mobileMenuOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
          </button>
        </div>
      </header>

      {/* Mobile Navigation */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-background/95 backdrop-blur-md border-b border-border px-2 py-1 sticky top-[41px] z-40">
          {/* About */}
          <button
            onClick={() => {
              navigate("/");
              setMobileMenuOpen(false);
            }}
            className={`relative w-full text-left px-3 py-2 text-xs font-medium transition-colors duration-200 font-heading ${isActive("/")
              ? "text-brand-orange border-l-2 border-brand-orange bg-accent"
              : "text-muted-foreground hover:text-foreground hover:bg-muted"
              }`}
          >
            About
          </button>

          {/* Research collapsible group */}
          <button
            onClick={() => setMobileResearchOpen(!mobileResearchOpen)}
            className={`relative w-full text-left px-3 py-2 text-xs font-medium transition-colors duration-200 flex items-center justify-between font-heading ${isResearchActive
              ? "text-brand-orange border-l-2 border-brand-orange bg-accent"
              : "text-muted-foreground hover:text-foreground hover:bg-muted"
              }`}
          >
            Research
            <ChevronDown className={`w-3 h-3 transition-transform duration-200 ${mobileResearchOpen ? "rotate-180" : ""}`} />
          </button>

          {mobileResearchOpen && (
            <div className="ml-3 border-l border-border">
              {researchSubItems.map((item) => {
                const active = isActive(item.path);
                return (
                  <button
                    key={item.path}
                    onClick={() => {
                      navigate(item.path);
                      setMobileMenuOpen(false);
                    }}
                    className={`relative w-full text-left pl-4 pr-3 py-1.5 text-xs font-medium transition-colors duration-200 font-heading ${active
                      ? "text-brand-orange bg-accent/50"
                      : "text-muted-foreground hover:text-foreground"
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
                className={`relative w-full text-left px-3 py-2 text-xs font-medium transition-colors duration-200 font-heading ${active
                  ? "text-brand-orange border-l-2 border-brand-orange bg-accent"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
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

