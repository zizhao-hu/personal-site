import { Header } from "../../components/custom/header";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Clock, BookOpen, Code, ChevronRight } from "lucide-react";
import { tutorials } from "../../data/tutorials";

const difficultyConfig = {
  beginner: {
    label: "Beginner",
    color: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
    borderColor: "border-l-green-500",
  },
  intermediate: {
    label: "Intermediate",
    color: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
    borderColor: "border-l-yellow-500",
  },
  advanced: {
    label: "Advanced",
    color: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
    borderColor: "border-l-red-500",
  },
};

const difficultyFilters = [
  { id: "all", label: "All Levels" },
  { id: "beginner", label: "Beginner" },
  { id: "intermediate", label: "Intermediate" },
  { id: "advanced", label: "Advanced" },
];

export const Tutorials = () => {
  const navigate = useNavigate();
  const [activeFilter, setActiveFilter] = useState<string>("all");

  const filteredTutorials = tutorials.filter(
    (tutorial) => activeFilter === "all" || tutorial.difficulty === activeFilter
  );

  const series = [...new Set(tutorials.filter((t) => t.series).map((t) => t.series))];

  return (
    <div className="h-full flex flex-col">
      <Header />
      <main className="flex-1 overflow-auto">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6">
          {/* Page Header */}
          <div className="mb-5">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
              Tutorials
            </h1>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Step-by-step guides for AI, machine learning, and web development
            </p>
          </div>

          {/* Difficulty Filter Tabs */}
          <div className="border-b border-gray-200 dark:border-gray-700 mb-4">
            <nav className="flex gap-0">
              {difficultyFilters.map((filter) => (
                <button
                  key={filter.id}
                  onClick={() => setActiveFilter(filter.id)}
                  className={`relative px-4 py-2 text-xs font-medium transition-colors duration-200 ${
                    activeFilter === filter.id
                      ? "text-blue-600 dark:text-blue-400"
                      : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
                  }`}
                >
                  {filter.label}
                  {activeFilter === filter.id && (
                    <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600 dark:bg-blue-400" />
                  )}
                </button>
              ))}
            </nav>
          </div>

          {/* Series Section */}
          {activeFilter === "all" && series.length > 0 && (
            <div className="mb-5">
              <h2 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
                Tutorial Series
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {series.map((seriesName) => {
                  const seriesTutorials = tutorials.filter((t) => t.series === seriesName);
                  return (
                    <div
                      key={seriesName}
                      className="group bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-800/50 border border-gray-200 dark:border-gray-700 rounded-md p-3 hover:shadow-sm transition-all duration-200"
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <BookOpen className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
                        <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                          {seriesName}
                        </h3>
                      </div>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {seriesTutorials.length} tutorials
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Tutorial List */}
          <div className="space-y-3">
            {filteredTutorials.map((tutorial) => {
              const config = difficultyConfig[tutorial.difficulty];
              return (
                <article
                  key={tutorial.id}
                  onClick={() => navigate(`/tutorials/${tutorial.slug}`)}
                  className={`group bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 border-l-3 ${config.borderColor} rounded-md p-4 hover:shadow-md transition-shadow cursor-pointer`}
                >
                  <div className="flex flex-col sm:flex-row sm:items-start gap-3">
                    <div className="flex-1">
                      <div className="flex flex-wrap items-center gap-1.5 mb-1">
                        <h2 className="text-sm font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                          {tutorial.title}
                        </h2>
                        <span className={`px-1.5 py-0.5 text-xs font-medium rounded ${config.color}`}>
                          {config.label}
                        </span>
                        {tutorial.series && (
                          <span className="px-1.5 py-0.5 text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded">
                            {tutorial.series}
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                        {tutorial.description}
                      </p>
                      <div className="flex flex-wrap items-center gap-3">
                        <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
                          <Clock className="w-3 h-3" />
                          <span>{tutorial.estimatedTime}</span>
                        </div>
                        <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
                          <Code className="w-3 h-3" />
                          <span>{tutorial.topics.join(", ")}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center text-blue-600 dark:text-blue-400">
                      <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                    </div>
                  </div>
                </article>
              );
            })}
          </div>

          {filteredTutorials.length === 0 && (
            <div className="text-center py-12">
              <p className="text-gray-500 dark:text-gray-400">
                No tutorials found for this difficulty level.
              </p>
              <button
                onClick={() => setActiveFilter("all")}
                className="mt-4 text-blue-600 dark:text-blue-400 hover:underline"
              >
                Show all tutorials
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};
