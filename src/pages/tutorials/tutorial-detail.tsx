import { Header } from "@/components/custom/header";
import { markdownCodeComponents } from "@/components/custom/code-block";
import { useParams, useNavigate } from "react-router-dom";
import { getTutorialBySlug } from "@/data/tutorials";
import { ArrowLeft, Clock, Code, BookOpen, CheckCircle, ExternalLink } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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
      <div className="flex flex-col min-h-dvh bg-background">
        <Header />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <h1 className="text-2xl font-bold font-heading text-foreground mb-4">
              Tutorial Not Found
            </h1>
            <button
              onClick={() => navigate("/tutorials")}
              className="text-brand-orange hover:underline font-heading text-sm"
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
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      <main className="flex-1 overflow-y-auto pb-24">
        <article className="max-w-4xl mx-auto px-4 sm:px-6 py-6">
          {/* Back Button */}
          <button
            onClick={() => navigate("/tutorials")}
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground mb-4 transition-colors font-heading"
          >
            <ArrowLeft className="w-3 h-3" />
            Back to Tutorials
          </button>

          {/* Tutorial Header */}
          <header className="mb-6">
            <div className="flex flex-wrap items-center gap-2 mb-2">
              <span className={`px-2 py-0.5 text-xs font-medium font-heading rounded ${config.color}`}>
                {config.label}
              </span>
              {tutorial.series && (
                <span className="px-2 py-0.5 text-xs bg-brand-orange/10 text-brand-orange rounded font-heading">
                  {tutorial.series}
                </span>
              )}
            </div>

            <h1 className="text-2xl sm:text-3xl font-bold font-heading text-foreground mb-2 leading-tight">
              {tutorial.title}
            </h1>

            <p className="text-sm text-muted-foreground mb-3">
              {tutorial.description}
            </p>

            <div className="flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
              <div className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                <span>{tutorial.estimatedTime}</span>
              </div>
              <div className="flex items-center gap-1">
                <Code className="w-3 h-3" />
                <span>{tutorial.topics.join(", ")}</span>
              </div>
            </div>
          </header>

          {/* Prerequisites */}
          {tutorial.prerequisites && tutorial.prerequisites.length > 0 && (
            <div className="mb-6 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-md">
              <h2 className="flex items-center gap-1.5 text-sm font-semibold font-heading text-amber-800 dark:text-amber-300 mb-1.5">
                <BookOpen className="w-3.5 h-3.5" />
                Prerequisites
              </h2>
              <ul className="space-y-0.5">
                {tutorial.prerequisites.map((prereq, index) => (
                  <li key={index} className="flex items-start gap-1.5 text-xs text-amber-700 dark:text-amber-400">
                    <CheckCircle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                    {prereq}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Tutorial Content */}
          <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:font-heading prose-headings:text-foreground prose-p:text-muted-foreground prose-a:text-brand-orange prose-strong:text-foreground prose-blockquote:border-brand-orange prose-blockquote:text-muted-foreground prose-li:text-muted-foreground prose-table:border-border prose-th:bg-muted prose-th:border-border prose-td:border-border">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                ...markdownCodeComponents,
                table({ children }) {
                  return (
                    <div className="overflow-x-auto my-6 rounded-lg border border-border shadow-sm">
                      <table className="min-w-full divide-y divide-border">
                        {children}
                      </table>
                    </div>
                  );
                },
                thead({ children }) {
                  return (
                    <thead className="bg-muted/70">
                      {children}
                    </thead>
                  );
                },
                tbody({ children }) {
                  return (
                    <tbody className="divide-y divide-border [&>tr:nth-child(even)]:bg-muted/30 [&>tr:hover]:bg-muted/50 transition-colors">
                      {children}
                    </tbody>
                  );
                },
                th({ children }) {
                  return (
                    <th className="px-4 py-2.5 text-left text-xs font-semibold font-heading text-foreground uppercase tracking-wider border-b border-border">
                      {children}
                    </th>
                  );
                },
                td({ children }) {
                  return (
                    <td className="px-4 py-2.5 text-sm text-muted-foreground whitespace-normal">
                      {children}
                    </td>
                  );
                },
                a({ href, children }) {
                  const isExternal = href?.startsWith('http');
                  return (
                    <a
                      href={href}
                      target={isExternal ? "_blank" : undefined}
                      rel={isExternal ? "noopener noreferrer" : undefined}
                      className="text-brand-orange hover:text-brand-orange/80 underline decoration-brand-orange/30 hover:decoration-brand-orange transition-colors inline-flex items-center gap-0.5"
                    >
                      {children}
                      {isExternal && <ExternalLink className="w-3 h-3 inline-block" />}
                    </a>
                  );
                },
              }}
            >
              {tutorial.content}
            </ReactMarkdown>
          </div>

          {/* Footer */}
          <div className="mt-8 pt-4 border-t border-border">
            <div className="flex items-center justify-between">
              <button
                onClick={() => navigate("/tutorials")}
                className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors font-heading"
              >
                <ArrowLeft className="w-3 h-3" />
                Back to all tutorials
              </button>
              {tutorial.series && (
                <span className="text-xs text-muted-foreground">
                  Part of <span className="font-medium text-brand-orange">{tutorial.series}</span>
                </span>
              )}
            </div>
          </div>
        </article>
      </main>
    </div>
  );
};
