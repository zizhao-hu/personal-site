import { Header } from "@/components/custom/header";
import { markdownCodeComponents } from "@/components/custom/code-block";
import { useParams, useNavigate } from "react-router-dom";
import { getProjectBySlug } from "@/data/projects";
import { ArrowLeft, Github, ExternalLink } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { tagBadgeClass } from "@/lib/tag-colors";

const statusConfig: Record<string, { label: string; class: string }> = {
    active: {
        label: "Active",
        class: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
    },
    research: {
        label: "Research",
        class: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400",
    },
    prototype: {
        label: "Prototype",
        class: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
    },
    completed: {
        label: "Completed",
        class: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
    },
    concept: {
        label: "Concept",
        class: "bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400",
    },
};

export const ProjectDetail = () => {
    const { slug } = useParams<{ slug: string }>();
    const navigate = useNavigate();
    const project = slug ? getProjectBySlug(slug) : undefined;

    if (!project) {
        return (
            <div className="flex flex-col min-h-dvh bg-background">
                <Header />
                <main className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <h1 className="text-2xl font-bold font-heading text-foreground mb-4">
                            Project Not Found
                        </h1>
                        <button
                            onClick={() => navigate("/projects")}
                            className="text-brand-orange hover:underline font-heading text-sm"
                        >
                            Back to Projects
                        </button>
                    </div>
                </main>
            </div>
        );
    }

    const status = statusConfig[project.status] || statusConfig.concept;
    const Icon = project.icon;

    return (
        <div className="flex flex-col min-h-dvh bg-background">
            <Header />
            <main className="flex-1 overflow-y-auto pb-24">
                <div className="max-w-4xl mx-auto px-4 sm:px-6 py-6">
                    {/* Back button */}
                    <button
                        onClick={() => navigate("/projects")}
                        className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6 font-heading"
                    >
                        <ArrowLeft className="w-3 h-3" />
                        Back to Projects
                    </button>

                    {/* Hero */}
                    <div className="mb-8">
                        <div className="flex items-center gap-3 mb-3">
                            <div className="w-10 h-10 rounded-xl bg-muted flex items-center justify-center">
                                <Icon className="w-5 h-5 text-brand-orange" />
                            </div>
                            <div>
                                <div className="flex items-center gap-2 flex-wrap">
                                    <h1 className="text-xl sm:text-2xl font-bold font-heading text-foreground">
                                        {project.title}
                                    </h1>
                                    <span className={`px-2 py-0.5 text-[10px] font-medium rounded ${status.class}`}>
                                        {status.label}
                                    </span>
                                </div>
                            </div>
                        </div>

                        <p className="text-sm text-muted-foreground leading-relaxed mb-4">
                            {project.description}
                        </p>

                        {/* Tags */}
                        <div className="flex flex-wrap gap-1.5 mb-4">
                            {project.tags.map((tag) => (
                                <span
                                    key={tag}
                                    className={`px-2 py-0.5 text-[10px] rounded font-heading ${tagBadgeClass(tag)}`}
                                >
                                    {tag}
                                </span>
                            ))}
                        </div>

                        {/* Links */}
                        <div className="flex gap-4">
                            {project.github && (
                                <a
                                    href={project.github}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors font-heading"
                                >
                                    <Github className="w-3.5 h-3.5" />
                                    View on GitHub
                                </a>
                            )}
                            {project.link && (
                                <a
                                    href={project.link}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1.5 text-xs text-brand-orange hover:underline font-heading"
                                >
                                    <ExternalLink className="w-3.5 h-3.5" />
                                    Live Demo
                                </a>
                            )}
                        </div>
                    </div>

                    {/* Divider */}
                    <div className="h-px bg-border mb-8" />

                    {/* Article content */}
                    {project.content ? (
                        <article className="prose prose-sm dark:prose-invert max-w-none
              prose-headings:font-heading prose-headings:text-foreground
              prose-p:text-muted-foreground prose-p:leading-relaxed
              prose-a:text-brand-orange prose-a:no-underline hover:prose-a:underline
              prose-strong:text-foreground
              prose-code:text-brand-orange prose-code:bg-muted prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs
              prose-pre:bg-muted prose-pre:border prose-pre:border-border prose-pre:rounded-xl
              prose-img:rounded-xl prose-img:border prose-img:border-border
              prose-table:text-xs
              prose-th:text-foreground prose-th:font-heading prose-th:text-left
              prose-td:text-muted-foreground
              prose-li:text-muted-foreground
              prose-blockquote:border-brand-orange prose-blockquote:text-muted-foreground
            ">
                            <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                rehypePlugins={[rehypeRaw]}
                                components={markdownCodeComponents}
                            >
                                {project.content}
                            </ReactMarkdown>
                        </article>
                    ) : (
                        <div className="text-center py-16">
                            <p className="text-muted-foreground text-sm">
                                Full article coming soon.
                            </p>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
};
