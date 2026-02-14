import { Header } from "@/components/custom/header";
import { ExternalLink, FileText, Calendar, MessageSquare, Eye } from "lucide-react";

const publications = [
    {
        title: "Multimodal Synthetic Data Finetuning and Model Collapse",
        authors: "Zizhao Hu, et al.",
        venue: "ACM International Conference on Multimodal Interaction (ICMI)",
        year: 2025,
        type: "conference" as const,
        link: "https://scholar.google.com/citations?user=A8J42tQAAAAJ",
        highlight: true,
    },
];

const keyTopics = [
    {
        title: "Multi-Agent LLM Systems",
        description: "Designing orchestration frameworks where specialized LLM agents collaborate on complex reasoning tasks. One agent plans, another executes, and a third verifies—creating self-correcting workflows that scale beyond single-model capabilities.",
        icon: MessageSquare,
    },
    {
        title: "Vision-Language Model Training",
        description: "Building VLMs that seamlessly integrate visual and textual understanding. Research on contrastive pre-training, cross-modal attention, and efficient fine-tuning strategies that reduce compute while maintaining performance across multimodal benchmarks.",
        icon: Eye,
    },
    {
        title: "Test-Time Compute & Reasoning",
        description: "Scaling inference over training. Exploring how to make LLMs 'think longer' before responding, using chain-of-thought verification, confidence calibration, and dynamic compute allocation based on problem complexity.",
        icon: MessageSquare,
    },
    {
        title: "Safety & Alignment",
        description: "Ensuring LLMs remain reliable and aligned through reasoning refinement, jailbreak detection, and safety injections. Contributing to frontier model training as a MOVE Fellow at Handshake AI.",
        icon: Eye,
    },
];

export const LlmVlmResearch = () => {
    return (
        <div className="flex flex-col min-h-dvh bg-background">
            <Header />

            <main className="flex-1 overflow-y-auto pb-24">
                <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6">
                    {/* Hero */}
                    <div className="mb-6">
                        <div className="flex items-center gap-2 mb-2">
                            <div className="px-2 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-[10px] font-semibold uppercase tracking-tight">
                                Research Direction
                            </div>
                        </div>
                        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                            LLM / VLM
                        </h1>
                        <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl">
                            Large Language Models and Vision-Language Models are at the core of modern AI.
                            My work in this area spans <strong>multi-agent LLM orchestration</strong>,
                            <strong> vision-language pre-training</strong>, <strong>test-time reasoning</strong>,
                            and <strong>AI safety & alignment</strong>—building systems that can reason across
                            modalities while remaining reliable and controllable.
                        </p>
                    </div>

                    {/* Key Topics */}
                    <section className="mb-8">
                        <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-4">
                            Key Research Topics
                        </h2>
                        <div className="grid md:grid-cols-2 gap-3">
                            {keyTopics.map((topic) => {
                                const Icon = topic.icon;
                                return (
                                    <div
                                        key={topic.title}
                                        className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50 hover:shadow-md transition-shadow"
                                    >
                                        <div className="flex items-center gap-2 mb-2">
                                            <div className="w-7 h-7 rounded-md bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                                                <Icon className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
                                            </div>
                                            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">{topic.title}</h3>
                                        </div>
                                        <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed">{topic.description}</p>
                                    </div>
                                );
                            })}
                        </div>
                    </section>

                    {/* Related Publications */}
                    <section className="mb-6">
                        <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-1.5">
                            <FileText className="w-4 h-4 text-purple-600" />
                            Related Publications
                        </h2>
                        <div className="space-y-3">
                            {publications.map((pub, index) => (
                                <div
                                    key={index}
                                    className={`p-3 rounded-lg border ${pub.highlight
                                            ? "border-blue-200 dark:border-blue-800 bg-blue-50/50 dark:bg-blue-900/20"
                                            : "border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50"
                                        }`}
                                >
                                    {pub.highlight && (
                                        <span className="inline-block px-1.5 py-0.5 text-xs font-medium bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded mb-1.5">
                                            Featured
                                        </span>
                                    )}
                                    <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-0.5">
                                        {pub.title}
                                    </h3>
                                    <p className="text-xs text-gray-600 dark:text-gray-400 mb-1.5">
                                        {pub.authors}
                                    </p>
                                    <div className="flex items-center gap-3 text-xs">
                                        <span className="flex items-center gap-1 text-gray-500 dark:text-gray-500">
                                            <Calendar className="w-2.5 h-2.5" />
                                            {pub.year}
                                        </span>
                                        <span className="text-purple-600 dark:text-purple-400 font-medium">
                                            {pub.venue}
                                        </span>
                                    </div>
                                    {pub.link && (
                                        <a
                                            href={pub.link}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="inline-flex items-center gap-1 mt-2 text-xs text-blue-600 dark:text-blue-400 hover:underline"
                                        >
                                            View Paper <ExternalLink className="w-2.5 h-2.5" />
                                        </a>
                                    )}
                                </div>
                            ))}
                        </div>
                    </section>
                </div>
            </main>
        </div>
    );
};
