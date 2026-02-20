import { Header } from "@/components/custom/header";
import { ExternalLink, FileText, Calendar, BookOpen, Sparkles, RefreshCw, Users, Bot, Orbit } from "lucide-react";

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
        title: "Multi-Agent Interaction",
        description: "How multiple LLM/VLM agents collaborate, debate, verify, and refine each other's outputs. Research on agent orchestration, role specialization, emergent communication protocols, and multi-agent self-play for improving reasoning and task decomposition.",
        icon: Users,
    },
    {
        title: "Self-Improving AI",
        description: "Systems that generate their own training signal through synthetic data, self-reflection, and iterative refinement. Investigating feedback loops where agents evaluate their own outputs, generate preference pairs, and continuously improve without human annotation.",
        icon: RefreshCw,
    },
    {
        title: "Vision-Language-Action (VLA)",
        description: "Unified models that perceive (vision), reason (language), and act (control). Research on grounding language in embodied environments, action prediction from multimodal inputs, and bridging the sim-to-real gap for robotic and interactive agents.",
        icon: Bot,
    },
    {
        title: "Pretraining & Post-Training",
        description: "Full model lifecycle from large-scale pretraining, through supervised fine-tuning (SFT), to post-training alignment with RLHF/DPO. Focus on how each stage contributes to multi-agent capability and self-improvement potential.",
        icon: Sparkles,
    },
    {
        title: "Agent Orchestration Frameworks",
        description: "Building scalable frameworks for multi-agent pipelines — task routing, tool use, memory systems, and self-correction loops. How to design agent architectures that are reliable, composable, and can scale from single tasks to complex workflows.",
        icon: Orbit,
    },
    {
        title: "Continual Learning for Agents",
        description: "Enabling LLM/VLM/VLA agents to learn new knowledge, skills, and domains over time without catastrophic forgetting. Developing replay-free and parameter-efficient continual learning methods tailored for large-scale agentic systems.",
        icon: BookOpen,
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
                            LLM / VLM / VLA
                        </h1>
                        <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl">
                            My primary research focus is on <strong>multi-agent interaction</strong> and{" "}
                            <strong>self-improving AI</strong> across language, vision-language, and vision-language-action
                            models. I study how multiple agents collaborate, generate synthetic experience, and
                            continuously refine each other—creating systems that get smarter through interaction
                            rather than just larger datasets.
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
