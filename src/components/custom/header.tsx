'use client';

import { ThemeToggle } from "./theme-toggle";
import { Menu, X, ChevronDown, FileText } from "lucide-react";
import { useRouter, usePathname } from 'next/navigation';

import { useState, useRef, useEffect } from "react";

interface HeaderProps {
  onHomeClick?: () => void;
}

const navItems = [
  { path: "/", label: "about" },
  { path: "/projects", label: "projects" },
  { path: "/blogs", label: "blogs" },
  { path: "/tools", label: "tools.md" },
];

const researchSubItems = [
  { path: "/research", label: "overview.md" },
  { path: "/research/llm-vlm", label: "memorization.md" },
  { path: "/research/synthetic-data", label: "synthetic-data.md" },
  { path: "/research/architecture", label: "architecture.md" },
  { path: "/research/continual-learning", label: "multi-agent.md" },
];

export const Header = ({ onHomeClick }: HeaderProps) => {
  const router = useRouter();
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [researchDropdownOpen, setResearchDropdownOpen] = useState(false);
  const [mobileResearchOpen, setMobileResearchOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const isActive = (path: string) => {
    if (path === "/") return pathname === "/";
    if (path === "/research") return pathname === "/research";
    return pathname.startsWith(path);
  };

  const isResearchActive = pathname.startsWith("/research");

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
      <header className="sticky top-0 z-50 w-full bg-background/90 backdrop-blur-md border-b border-border">
        {/* Tab bar */}
        <div className="flex items-stretch justify-between px-1 sm:px-2 h-9">
          {/* Logo + tabs */}
          <div className="flex items-stretch min-w-0">
            <button
              onClick={() => {
                router.push('/');
                if (onHomeClick) onHomeClick();
              }}
              className="flex items-center gap-2 px-2 sm:px-3 hover:bg-muted/60 transition-colors"
              aria-label="Home"
            >
              <div className="w-5 h-5 rounded-md bg-foreground text-background flex items-center justify-center font-mono font-bold text-[10px]">
                ZH
              </div>
            </button>

            {/* Desktop tabs */}
            <nav className="hidden md:flex items-stretch">
              {/* Home tab */}
              <FileTab
                label="about"
                active={isActive("/")}
                onClick={() => router.push("/")}
              />

              {/* Research tab + dropdown */}
              <div
                ref={dropdownRef}
                className="relative flex items-stretch"
                onMouseEnter={() => setResearchDropdownOpen(true)}
                onMouseLeave={() => setResearchDropdownOpen(false)}
              >
                <FileTab
                  label="research/"
                  active={isResearchActive}
                  trailing={<ChevronDown className={`w-3 h-3 transition-transform duration-200 ${researchDropdownOpen ? "rotate-180" : ""}`} />}
                  onClick={() => {
                    router.push("/research");
                    setResearchDropdownOpen(false);
                  }}
                />
                {researchDropdownOpen && (
                  <div className="absolute top-full left-0 mt-0 w-56 bg-background border border-border shadow-xl z-50 animate-in fade-in slide-in-from-top-1 duration-150 font-mono">
                    <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider text-muted-foreground border-b border-border bg-muted/40">
                      research/
                    </div>
                    {researchSubItems.map((item) => {
                      const active = isActive(item.path);
                      return (
                        <button
                          key={item.path}
                          onClick={() => {
                            router.push(item.path);
                            setResearchDropdownOpen(false);
                          }}
                          className={`w-full text-left px-3 py-2 text-xs transition-colors duration-150 flex items-center gap-2 ${active
                            ? "text-foreground bg-muted/70"
                            : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                            }`}
                        >
                          <FileText className="w-3 h-3 opacity-60 flex-shrink-0" />
                          <span className="truncate">{item.label}</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Other tabs */}
              {navItems.filter((i) => i.path !== "/").map((item) => (
                <FileTab
                  key={item.path}
                  label={item.label}
                  active={isActive(item.path)}
                  onClick={() => router.push(item.path)}
                />
              ))}
            </nav>
          </div>

          {/* Right side */}
          <div className="flex items-center gap-1 px-1">
            <ThemeToggle />
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-1.5 rounded-md hover:bg-muted text-muted-foreground"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </header>

      {/* Mobile drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-background border-b border-border font-mono sticky top-[36px] z-40">
          {navItems.map((item) => (
            <button
              key={item.path}
              onClick={() => {
                router.push(item.path);
                setMobileMenuOpen(false);
              }}
              className={`w-full text-left px-4 py-2 text-xs flex items-center gap-2 transition-colors ${isActive(item.path)
                ? "text-foreground bg-muted/70 border-l-2 border-brand-orange"
                : "text-muted-foreground hover:bg-muted/40"
                }`}
            >
              <FileText className="w-3 h-3 opacity-60" />
              {item.label}
            </button>
          ))}

          <button
            onClick={() => setMobileResearchOpen(!mobileResearchOpen)}
            className={`w-full text-left px-4 py-2 text-xs flex items-center justify-between transition-colors ${isResearchActive
              ? "text-foreground bg-muted/70 border-l-2 border-brand-orange"
              : "text-muted-foreground hover:bg-muted/40"
              }`}
          >
            <span className="flex items-center gap-2">
              <ChevronDown className={`w-3 h-3 transition-transform ${mobileResearchOpen ? "rotate-0" : "-rotate-90"}`} />
              research/
            </span>
          </button>
          {mobileResearchOpen && (
            <div className="border-l border-border ml-4">
              {researchSubItems.map((item) => (
                <button
                  key={item.path}
                  onClick={() => {
                    router.push(item.path);
                    setMobileMenuOpen(false);
                  }}
                  className={`w-full text-left pl-6 pr-4 py-1.5 text-[11px] flex items-center gap-2 transition-colors ${isActive(item.path)
                    ? "text-foreground bg-muted/50"
                    : "text-muted-foreground hover:bg-muted/30"
                    }`}
                >
                  <FileText className="w-3 h-3 opacity-60" />
                  {item.label}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </>
  );
};

interface FileTabProps {
  label: string;
  active: boolean;
  onClick: () => void;
  trailing?: React.ReactNode;
}

const FileTab = ({ label, active, onClick, trailing }: FileTabProps) => {
  return (
    <button
      onClick={onClick}
      className={`relative flex items-center gap-1.5 px-3 h-full font-mono text-[12px] border-r border-border transition-colors duration-150 group ${
        active
          ? "bg-background text-foreground"
          : "bg-muted/40 text-muted-foreground hover:text-foreground hover:bg-muted/60"
      }`}
    >
      <FileText className={`w-3 h-3 ${active ? "text-brand-orange" : "opacity-50"}`} />
      <span className="truncate">{label}</span>
      {trailing}
      {active && (
        <span className="absolute top-0 left-0 right-0 h-[2px] bg-brand-orange" />
      )}
    </button>
  );
};
