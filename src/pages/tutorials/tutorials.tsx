import { Header } from "../../components/custom/header";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Clock, BookOpen, Code, ChevronRight, LayoutGrid, List } from "lucide-react";
import { tutorials } from "../../data/tutorials";
import { tagPillClass } from "../../lib/tag-colors";

const difficultyConfig = {
  beginner: {
    label: "Beginner",
    color: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
    borderColor: "border-l-green-500",
    dot: "bg-green-500",
  },
  intermediate: {
    label: "Intermediate",
    color: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
    borderColor: "border-l-yellow-500",
    dot: "bg-yellow-500",
  },
  advanced: {
    label: "Advanced",
    color: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
    borderColor: "border-l-red-500",
    dot: "bg-red-500",
  },
};

const difficultyFilters = [
  { id: "all", label: "All Levels", count: tutorials.length },
  { id: "beginner", label: "Beginner", count: tutorials.filter(t => t.difficulty === "beginner").length },
  { id: "intermediate", label: "Intermediate", count: tutorials.filter(t => t.difficulty === "intermediate").length },
  { id: "advanced", label: "Advanced", count: tutorials.filter(t => t.difficulty === "advanced").length },
];

const allTopics = [...new Set(tutorials.flatMap((t) => t.topics))];
const topicCounts = allTopics.reduce((acc, topic) => {
  acc[topic] = tutorials.filter(t => t.topics.includes(topic)).length;
  return acc;
}, {} as Record<string, number>);

const allSeries = [...new Set(tutorials.filter(t => t.series).map(t => t.series!))];

export const Tutorials = () => {
  const navigate = useNavigate();
  const [activeFilter, setActiveFilter] = useState<string>("all");
  const [selectedTopics, setSelectedTopics] = useState<string[]>([]);
  const [activeSeries, setActiveSeries] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"list" | "grid">("list");

  const filteredTutorials = tutorials.filter((tutorial) => {
    const difficultyMatch = activeFilter === "all" || tutorial.difficulty === activeFilter;
    const topicMatch = selectedTopics.length === 0 || selectedTopics.some(t => tutorial.topics.includes(t));
    const seriesMatch = !activeSeries || tutorial.series === activeSeries;
    return difficultyMatch && topicMatch && seriesMatch;
  });

  const toggleTopic = (topic: string) => {
    setSelectedTopics((prev) =>
      prev.includes(topic) ? prev.filter((t) => t !== topic) : [...prev, topic]
    );
  };

  return (
    <div className="h-full flex flex-col">
      <Header />
      <main className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
          {/* Page Header */}
          <div className="mb-6">
            <h1 className="text-2xl font-bold font-heading text-foreground mb-1">
              Tutorials
            </h1>
            <p className="text-sm text-muted-foreground">
              Step-by-step guides for AI, machine learning, and web development
            </p>
          </div>

          {/* Main Layout: Sidebar + Content */}
          <div className="flex gap-6">

            {/* ── Left Sidebar ── */}
            <aside className="hidden lg:block w-56 flex-shrink-0">
              <div className="sticky top-6 space-y-6">

                {/* Difficulty Levels */}
                <div>
                  <h3 className="text-[10px] uppercase tracking-wider font-heading text-muted-foreground mb-2 px-2">
                    Difficulty
                  </h3>
                  <nav className="space-y-0.5">
                    {difficultyFilters.map((filter) => (
                      <button
                        key={filter.id}
                        onClick={() => setActiveFilter(filter.id)}
                        className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs transition-all duration-150 ${activeFilter === filter.id
                          ? "bg-brand-orange/10 text-brand-orange font-medium"
                          : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                          }`}
                      >
                        <span className="flex items-center gap-2 font-heading">
                          {filter.id !== "all" && (
                            <span className={`w-2 h-2 rounded-full ${difficultyConfig[filter.id as keyof typeof difficultyConfig]?.dot}`} />
                          )}
                          {filter.label}
                        </span>
                        <span className={`text-[10px] tabular-nums ${activeFilter === filter.id ? "text-brand-orange" : "text-muted-foreground/60"
                          }`}>
                          {filter.count}
                        </span>
                      </button>
                    ))}
                  </nav>
                </div>

                <div className="h-px bg-border" />

                {/* Series */}
                {allSeries.length > 0 && (
                  <>
                    <div>
                      <h3 className="text-[10px] uppercase tracking-wider font-heading text-muted-foreground mb-2 px-2">
                        Series
                      </h3>
                      <nav className="space-y-0.5">
                        <button
                          onClick={() => setActiveSeries(null)}
                          className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs transition-all duration-150 ${!activeSeries
                            ? "bg-brand-orange/10 text-brand-orange font-medium"
                            : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                            }`}
                        >
                          <span className="font-heading">All</span>
                        </button>
                        {allSeries.map((series) => {
                          const count = tutorials.filter(t => t.series === series).length;
                          return (
                            <button
                              key={series}
                              onClick={() => setActiveSeries(activeSeries === series ? null : series)}
                              className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs transition-all duration-150 ${activeSeries === series
                                ? "bg-brand-orange/10 text-brand-orange font-medium"
                                : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                                }`}
                            >
                              <span className="flex items-center gap-2 font-heading">
                                <BookOpen className="w-3 h-3" />
                                {series}
                              </span>
                              <span className={`text-[10px] tabular-nums ${activeSeries === series ? "text-brand-orange" : "text-muted-foreground/60"
                                }`}>
                                {count}
                              </span>
                            </button>
                          );
                        })}
                      </nav>
                    </div>
                    <div className="h-px bg-border" />
                  </>
                )}

                {/* Topics */}
                <div>
                  <div className="flex items-center justify-between mb-2 px-2">
                    <h3 className="text-[10px] uppercase tracking-wider font-heading text-muted-foreground">
                      Topics
                    </h3>
                    {selectedTopics.length > 0 && (
                      <button
                        onClick={() => setSelectedTopics([])}
                        className="text-[10px] text-brand-orange hover:underline font-heading"
                      >
                        Clear
                      </button>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-1.5 px-1">
                    {allTopics.map((topic) => (
                      <button
                        key={topic}
                        onClick={() => toggleTopic(topic)}
                        className={`px-2 py-0.5 text-[10px] rounded-full transition-all duration-150 font-heading ${tagPillClass(topic, selectedTopics.includes(topic))}`}
                      >
                        {topic}
                        <span className="ml-1 opacity-60">{topicCounts[topic]}</span>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="h-px bg-border" />

                {/* Stats */}
                <div className="px-2">
                  <h3 className="text-[10px] uppercase tracking-wider font-heading text-muted-foreground mb-2">
                    Stats
                  </h3>
                  <div className="space-y-1.5 text-xs text-muted-foreground">
                    <div className="flex justify-between">
                      <span>Total tutorials</span>
                      <span className="font-medium text-foreground">{tutorials.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Series</span>
                      <span className="font-medium text-foreground">{allSeries.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Topics</span>
                      <span className="font-medium text-foreground">{allTopics.length}</span>
                    </div>
                  </div>
                </div>
              </div>
            </aside>

            {/* ── Main Content ── */}
            <div className="flex-1 min-w-0">

              {/* Mobile tabs + view toggle */}
              <div className="flex items-center justify-between mb-4">
                <div className="lg:hidden flex gap-0 border-b border-border overflow-x-auto">
                  {difficultyFilters.map((filter) => (
                    <button
                      key={filter.id}
                      onClick={() => setActiveFilter(filter.id)}
                      className={`relative px-3 py-2 text-xs font-heading whitespace-nowrap transition-colors ${activeFilter === filter.id
                        ? "text-brand-orange"
                        : "text-muted-foreground hover:text-foreground"
                        }`}
                    >
                      {filter.label}
                      {activeFilter === filter.id && (
                        <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-brand-orange" />
                      )}
                    </button>
                  ))}
                </div>

                <div className="flex items-center gap-0.5 bg-muted rounded-lg p-0.5">
                  <button
                    onClick={() => setViewMode("list")}
                    className={`p-1.5 rounded-md transition-colors ${viewMode === "list" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"
                      }`}
                  >
                    <List className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => setViewMode("grid")}
                    className={`p-1.5 rounded-md transition-colors ${viewMode === "grid" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"
                      }`}
                  >
                    <LayoutGrid className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Active filters */}
              {(activeFilter !== "all" || selectedTopics.length > 0 || activeSeries) && (
                <div className="mb-3 flex items-center gap-2 text-xs text-muted-foreground flex-wrap">
                  <span>Showing {filteredTutorials.length} of {tutorials.length}</span>
                  {activeSeries && (
                    <span className="px-1.5 py-0.5 bg-blue-500/10 text-blue-500 rounded text-[10px] font-heading">
                      {activeSeries}
                    </span>
                  )}
                  {selectedTopics.map(topic => (
                    <span key={topic} className="px-1.5 py-0.5 bg-brand-orange/10 text-brand-orange rounded text-[10px] font-heading">
                      {topic}
                    </span>
                  ))}
                </div>
              )}

              {/* Tutorial Cards */}
              <div className={viewMode === "grid" ? "grid grid-cols-1 sm:grid-cols-2 gap-4" : "flex flex-col gap-3"}>
                {filteredTutorials.map((tutorial) => {
                  const config = difficultyConfig[tutorial.difficulty];
                  return viewMode === "list" ? (
                    <article
                      key={tutorial.id}
                      onClick={() => navigate(`/tutorials/${tutorial.slug}`)}
                      className={`group bg-card border border-border border-l-3 ${config.borderColor} rounded-xl p-4 hover:shadow-elevation-2 dark:hover:shadow-elevation-2-dark hover:border-brand-orange/20 transition-all cursor-pointer`}
                    >
                      <div className="flex flex-col sm:flex-row sm:items-start gap-3">
                        <div className="flex-1 min-w-0">
                          <div className="flex flex-wrap items-center gap-1.5 mb-1">
                            <h2 className="text-sm font-semibold font-heading text-foreground group-hover:text-brand-orange transition-colors">
                              {tutorial.title}
                            </h2>
                            <span className={`px-1.5 py-0.5 text-[10px] font-medium rounded ${config.color}`}>
                              {config.label}
                            </span>
                            {tutorial.series && (
                              <span className="px-1.5 py-0.5 text-[10px] bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded font-heading">
                                {tutorial.series}
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground mb-2">{tutorial.description}</p>
                          <div className="flex flex-wrap items-center gap-3">
                            <div className="flex items-center gap-1 text-xs text-muted-foreground">
                              <Clock className="w-3 h-3" />
                              <span>{tutorial.estimatedTime}</span>
                            </div>
                            <div className="flex items-center gap-1 text-xs text-muted-foreground">
                              <Code className="w-3 h-3" />
                              <span>{tutorial.topics.join(", ")}</span>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center text-brand-orange">
                          <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                        </div>
                      </div>
                    </article>
                  ) : (
                    <article
                      key={tutorial.id}
                      onClick={() => navigate(`/tutorials/${tutorial.slug}`)}
                      className={`group bg-card border border-border border-l-3 ${config.borderColor} rounded-xl p-4 hover:shadow-elevation-2 dark:hover:shadow-elevation-2-dark hover:border-brand-orange/20 transition-all cursor-pointer flex flex-col`}
                    >
                      <div className="flex flex-wrap items-center gap-1.5 mb-2">
                        <span className={`px-1.5 py-0.5 text-[10px] font-medium rounded ${config.color}`}>
                          {config.label}
                        </span>
                        {tutorial.series && (
                          <span className="px-1.5 py-0.5 text-[10px] bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded font-heading">
                            {tutorial.series}
                          </span>
                        )}
                      </div>
                      <h2 className="text-sm font-semibold font-heading text-foreground group-hover:text-brand-orange transition-colors mb-1.5">
                        {tutorial.title}
                      </h2>
                      <p className="text-xs text-muted-foreground mb-3 flex-1 line-clamp-2">{tutorial.description}</p>
                      <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                        <div className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          <span>{tutorial.estimatedTime}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Code className="w-3 h-3" />
                          <span>{tutorial.topics.slice(0, 2).join(", ")}</span>
                        </div>
                      </div>
                    </article>
                  );
                })}
              </div>

              {filteredTutorials.length === 0 && (
                <div className="text-center py-16">
                  <p className="text-muted-foreground text-sm mb-2">No tutorials found matching your filters.</p>
                  <button
                    onClick={() => { setActiveFilter("all"); setSelectedTopics([]); setActiveSeries(null); }}
                    className="text-brand-orange hover:underline text-xs font-heading"
                  >
                    Clear all filters
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};
