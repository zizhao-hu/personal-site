import { Header } from "../../components/custom/header";
import { ExternalLink, MessageSquare, Brain, Sparkles, Zap, Globe } from "lucide-react";

interface WebApp {
  id: string;
  title: string;
  description: string;
  icon: React.ElementType;
  tags: string[];
  demoUrl: string;
  featured?: boolean;
  status: "live" | "beta" | "coming-soon";
}

const webapps: WebApp[] = [
  {
    id: "webllm-chat",
    title: "WebLLM Chatbot",
    description: "Experience AI chat directly in your browser with no server required. Powered by WebLLM, this chatbot runs entirely client-side using WebGPU acceleration.",
    icon: MessageSquare,
    tags: ["AI", "WebGPU", "Privacy-First"],
    demoUrl: "/chat",
    featured: true,
    status: "live",
  },
  {
    id: "research-assistant",
    title: "Research Paper Assistant",
    description: "Upload and analyze research papers with AI-powered summarization, key insights extraction, and citation management.",
    icon: Brain,
    tags: ["AI", "Research", "NLP"],
    demoUrl: "#",
    status: "coming-soon",
  },
  {
    id: "code-playground",
    title: "AI Code Playground",
    description: "Interactive coding environment with AI assistance for debugging, refactoring, and learning new programming concepts.",
    icon: Sparkles,
    tags: ["Coding", "AI", "Education"],
    demoUrl: "#",
    status: "coming-soon",
  },
  {
    id: "data-viz",
    title: "Data Visualization Studio",
    description: "Create beautiful, interactive data visualizations from your datasets with natural language commands.",
    icon: Zap,
    tags: ["Data", "Visualization", "Analytics"],
    demoUrl: "#",
    status: "coming-soon",
  },
  {
    id: "ml-explorer",
    title: "ML Model Explorer",
    description: "Visualize and understand machine learning models interactively. Explore decision boundaries, feature importance, and model behavior.",
    icon: Globe,
    tags: ["ML", "Visualization", "Education"],
    demoUrl: "#",
    status: "coming-soon",
  },
];

const statusColors = {
  live: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
  beta: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
  "coming-soon": "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
};

const statusLabels = {
  live: "Live",
  beta: "Beta",
  "coming-soon": "Coming Soon",
};

export const Webapps = () => {
  const featuredApp = webapps.find((app) => app.featured);
  const otherApps = webapps.filter((app) => !app.featured);

  return (
    <div className="h-full flex flex-col">
      <Header />
      <main className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              Web Apps
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Interactive demos and tools showcasing AI and web technologies
            </p>
          </div>

          {/* Featured App */}
          {featuredApp && (
            <div className="mb-8">
              <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-4">
                Featured
              </h2>
              <a
                href={featuredApp.demoUrl}
                className="group block bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-6 hover:shadow-lg transition-all duration-300"
              >
                <div className="flex flex-col md:flex-row md:items-center gap-6">
                  <div className="w-16 h-16 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center flex-shrink-0">
                    <featuredApp.icon className="w-8 h-8 text-white" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="text-xl font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                        {featuredApp.title}
                      </h3>
                      <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${statusColors[featuredApp.status]}`}>
                        {statusLabels[featuredApp.status]}
                      </span>
                    </div>
                    <p className="text-gray-600 dark:text-gray-400 mb-4">
                      {featuredApp.description}
                    </p>
                    <div className="flex flex-wrap items-center gap-4">
                      <div className="flex gap-2">
                        {featuredApp.tags.map((tag) => (
                          <span
                            key={tag}
                            className="px-2 py-0.5 text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                      <div className="flex items-center gap-1 text-blue-600 dark:text-blue-400 text-sm font-medium">
                        Try it now
                        <ExternalLink className="w-4 h-4" />
                      </div>
                    </div>
                  </div>
                </div>
              </a>
            </div>
          )}

          {/* Other Apps Grid */}
          <div>
            <h2 className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-4">
              More Apps
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {otherApps.map((app) => {
                const Icon = app.icon;
                const isClickable = app.status === "live" || app.status === "beta";

                return (
                  <div
                    key={app.id}
                    className={`group bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-5 transition-all duration-200 ${
                      isClickable
                        ? "hover:shadow-md dark:hover:shadow-gray-900/50 cursor-pointer"
                        : "opacity-75"
                    }`}
                  >
                    <div className="flex items-start gap-4">
                      <div className="w-12 h-12 rounded-lg bg-gray-100 dark:bg-gray-700 flex items-center justify-center flex-shrink-0">
                        <Icon className="w-6 h-6 text-gray-600 dark:text-gray-400" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className={`font-semibold text-gray-900 dark:text-white truncate ${
                            isClickable ? "group-hover:text-blue-600 dark:group-hover:text-blue-400" : ""
                          } transition-colors`}>
                            {app.title}
                          </h3>
                          <span className={`px-2 py-0.5 text-xs font-medium rounded-full flex-shrink-0 ${statusColors[app.status]}`}>
                            {statusLabels[app.status]}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2 mb-3">
                          {app.description}
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {app.tags.map((tag) => (
                            <span
                              key={tag}
                              className="px-2 py-0.5 text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};
