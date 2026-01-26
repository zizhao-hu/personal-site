import { Header } from "../../components/custom/header";
import { useParams, useNavigate } from "react-router-dom";
import { getTutorialBySlug } from "../../data/tutorials";
import { ArrowLeft, Clock, BookOpen, Code, CheckCircle } from "lucide-react";
import ReactMarkdown from "react-markdown";

const difficultyConfig = {
  beginner: {
    label: "Beginner",
    color: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
  },
  intermediate: {
    label: "Intermediate",
    color: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
  },
  advanced: {
    label: "Advanced",
    color: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
  },
};

export const TutorialDetail = () => {
  const { slug } = useParams<{ slug: string }>();
  const navigate = useNavigate();
  const tutorial = slug ? getTutorialBySlug(slug) : undefined;

  if (!tutorial) {
    return (
      <div className="h-full flex flex-col">
        <Header />
        <main className="flex-1 overflow-auto flex items-center justify-center">
          <div className="text-center">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              Tutorial Not Found
            </h1>
            <button
              onClick={() => navigate("/tutorials")}
              className="text-blue-600 dark:text-blue-400 hover:underline"
            >
              Back to Tutorials
            </button>
          </div>
        </main>
      </div>
    );
  }

  const config = difficultyConfig[tutorial.difficulty];

  return (
    <div className="h-full flex flex-col">
      <Header />
      <main className="flex-1 overflow-auto">
        <article className="max-w-4xl mx-auto px-4 sm:px-6 py-8">
          {/* Back Button */}
          <button
            onClick={() => navigate("/tutorials")}
            className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white mb-6 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Tutorials
          </button>

          {/* Tutorial Header */}
          <header className="mb-8">
            <div className="flex flex-wrap items-center gap-3 mb-4">
              <span className={`px-3 py-1 text-sm font-medium rounded-full ${config.color}`}>
                {config.label}
              </span>
              {tutorial.series && (
                <span className="px-3 py-1 text-sm bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full">
                  {tutorial.series}
                </span>
              )}
            </div>

            <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-4 leading-tight">
              {tutorial.title}
            </h1>

            <p className="text-lg text-gray-600 dark:text-gray-400 mb-4">
              {tutorial.description}
            </p>

            <div className="flex flex-wrap items-center gap-6 text-sm text-gray-500 dark:text-gray-400">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4" />
                <span>{tutorial.estimatedTime}</span>
              </div>
              <div className="flex items-center gap-2">
                <Code className="w-4 h-4" />
                <span>{tutorial.topics.join(", ")}</span>
              </div>
            </div>
          </header>

          {/* Prerequisites */}
          {tutorial.prerequisites && tutorial.prerequisites.length > 0 && (
            <div className="mb-8 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg">
              <h2 className="flex items-center gap-2 font-semibold text-amber-800 dark:text-amber-300 mb-2">
                <BookOpen className="w-4 h-4" />
                Prerequisites
              </h2>
              <ul className="space-y-1">
                {tutorial.prerequisites.map((prereq, index) => (
                  <li key={index} className="flex items-start gap-2 text-sm text-amber-700 dark:text-amber-400">
                    <CheckCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                    {prereq}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Tutorial Content */}
          <div className="prose prose-lg dark:prose-invert max-w-none prose-headings:text-gray-900 dark:prose-headings:text-white prose-p:text-gray-600 dark:prose-p:text-gray-300 prose-a:text-blue-600 dark:prose-a:text-blue-400 prose-code:text-pink-600 dark:prose-code:text-pink-400 prose-code:bg-gray-100 dark:prose-code:bg-gray-800 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-pre:bg-gray-900 dark:prose-pre:bg-gray-950 prose-pre:text-gray-100 prose-strong:text-gray-900 dark:prose-strong:text-white prose-blockquote:border-blue-500 prose-blockquote:text-gray-600 dark:prose-blockquote:text-gray-400 prose-li:text-gray-600 dark:prose-li:text-gray-300 prose-table:border-gray-200 dark:prose-table:border-gray-700 prose-th:bg-gray-50 dark:prose-th:bg-gray-800 prose-th:border-gray-200 dark:prose-th:border-gray-700 prose-td:border-gray-200 dark:prose-td:border-gray-700">
            <ReactMarkdown
              components={{
                code({ node, className, children, ...props }) {
                  const isInline = !className;
                  if (isInline) {
                    return (
                      <code className="text-pink-600 dark:text-pink-400 bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded text-sm" {...props}>
                        {children}
                      </code>
                    );
                  }
                  return (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                },
                pre({ children }) {
                  return (
                    <pre className="bg-gray-900 dark:bg-gray-950 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
                      {children}
                    </pre>
                  );
                },
                table({ children }) {
                  return (
                    <div className="overflow-x-auto my-6">
                      <table className="min-w-full border border-gray-200 dark:border-gray-700">
                        {children}
                      </table>
                    </div>
                  );
                },
                th({ children }) {
                  return (
                    <th className="px-4 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-left font-semibold">
                      {children}
                    </th>
                  );
                },
                td({ children }) {
                  return (
                    <td className="px-4 py-2 border border-gray-200 dark:border-gray-700">
                      {children}
                    </td>
                  );
                },
              }}
            >
              {tutorial.content}
            </ReactMarkdown>
          </div>

          {/* Footer */}
          <div className="mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <button
                onClick={() => navigate("/tutorials")}
                className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                Back to all tutorials
              </button>
              {tutorial.series && (
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  Part of the <span className="font-medium text-blue-600 dark:text-blue-400">{tutorial.series}</span> series
                </span>
              )}
            </div>
          </div>
        </article>
      </main>
    </div>
  );
};
