import { Header } from "../../components/custom/header";
import { useState } from "react";
import { Clock, BookOpen, Code, ChevronRight } from "lucide-react";

interface Tutorial {
  id: string;
  title: string;
  description: string;
  difficulty: "beginner" | "intermediate" | "advanced";
  estimatedTime: string;
  topics: string[];
  slug: string;
  series?: string;
}

const tutorials: Tutorial[] = [
  {
    id: "1",
    title: "Getting Started with WebLLM",
    description: "Learn how to run large language models directly in the browser using WebGPU. No server required!",
    difficulty: "beginner",
    estimatedTime: "15 min",
    topics: ["WebLLM", "WebGPU", "JavaScript"],
    slug: "getting-started-webllm",
    series: "WebLLM Fundamentals",
  },
  {
    id: "2",
    title: "Building a Chat Interface with React",
    description: "Create a modern chat UI component with message streaming, typing indicators, and markdown support.",
    difficulty: "intermediate",
    estimatedTime: "30 min",
    topics: ["React", "TypeScript", "Tailwind CSS"],
    slug: "react-chat-interface",
  },
  {
    id: "3",
    title: "Understanding Transformer Architecture",
    description: "A visual guide to the transformer architecture that powers modern LLMs like GPT and LLaMA.",
    difficulty: "intermediate",
    estimatedTime: "25 min",
    topics: ["Deep Learning", "NLP", "Attention"],
    slug: "transformer-architecture",
    series: "ML Fundamentals",
  },
  {
    id: "4",
    title: "Implementing Continual Learning with DREAM",
    description: "Build a neural network that can learn new tasks without forgetting old ones using the DREAM algorithm.",
    difficulty: "advanced",
    estimatedTime: "45 min",
    topics: ["PyTorch", "Continual Learning", "Research"],
    slug: "dream-continual-learning",
  },
  {
    id: "5",
    title: "Web Workers for ML Inference",
    description: "Offload heavy ML computations to Web Workers for a smooth user experience in browser-based AI apps.",
    difficulty: "intermediate",
    estimatedTime: "20 min",
    topics: ["Web Workers", "JavaScript", "Performance"],
    slug: "web-workers-ml",
    series: "WebLLM Fundamentals",
  },
  {
    id: "6",
    title: "Fine-tuning LLMs with LoRA",
    description: "Learn low-rank adaptation techniques for efficiently fine-tuning large language models on custom data.",
    difficulty: "advanced",
    estimatedTime: "60 min",
    topics: ["LoRA", "Fine-tuning", "PyTorch"],
    slug: "lora-finetuning",
  },
];

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
  const [activeFilter, setActiveFilter] = useState<string>("all");

  const filteredTutorials = tutorials.filter(
    (tutorial) => activeFilter === "all" || tutorial.difficulty === activeFilter
  );

  const series = [...new Set(tutorials.filter((t) => t.series).map((t) => t.series))];

  return (
    <div className="h-full flex flex-col">
      <Header />
      <main className="flex-1 overflow-auto">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              Tutorials
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Step-by-step guides for AI, machine learning, and web development
            </p>
          </div>

          {/* Difficulty Filter Tabs */}
          <div className="border-b border-gray-200 dark:border-gray-700 mb-6">
            <nav className="flex gap-0">
              {difficultyFilters.map((filter) => (
                <button
                  key={filter.id}
                  onClick={() => setActiveFilter(filter.id)}
                  className={`relative px-6 py-3 text-sm font-medium transition-colors duration-200 ${
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
            <div className="mb-8">
              <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-4">
                Tutorial Series
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {series.map((seriesName) => {
                  const seriesTutorials = tutorials.filter((t) => t.series === seriesName);
                  return (
                    <div
                      key={seriesName}
                      className="group bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-800/50 border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-all duration-200 cursor-pointer"
                    >
                      <div className="flex items-center gap-3 mb-2">
                        <BookOpen className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                        <h3 className="font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                          {seriesName}
                        </h3>
                      </div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {seriesTutorials.length} tutorials
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Tutorial List */}
          <div className="space-y-4">
            {filteredTutorials.map((tutorial) => {
              const config = difficultyConfig[tutorial.difficulty];
              return (
                <article
                  key={tutorial.id}
                  className={`group bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 border-l-4 ${config.borderColor} rounded-lg p-5 hover:shadow-elevation-3 dark:hover:shadow-elevation-3-dark transition-micro cursor-pointer`}
                >
                  <div className="flex flex-col sm:flex-row sm:items-start gap-4">
                    <div className="flex-1">
                      <div className="flex flex-wrap items-center gap-2 mb-2">
                        <h2 className="text-lg font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                          {tutorial.title}
                        </h2>
                        <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${config.color}`}>
                          {config.label}
                        </span>
                        {tutorial.series && (
                          <span className="px-2 py-0.5 text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full">
                            {tutorial.series}
                          </span>
                        )}
                      </div>
                      <p className="text-gray-600 dark:text-gray-400 mb-3">
                        {tutorial.description}
                      </p>
                      <div className="flex flex-wrap items-center gap-4">
                        <div className="flex items-center gap-1 text-sm text-gray-500 dark:text-gray-400">
                          <Clock className="w-4 h-4" />
                          <span>{tutorial.estimatedTime}</span>
                        </div>
                        <div className="flex items-center gap-1 text-sm text-gray-500 dark:text-gray-400">
                          <Code className="w-4 h-4" />
                          <span>{tutorial.topics.join(", ")}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center text-blue-600 dark:text-blue-400">
                      <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
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
