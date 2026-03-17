import { Header } from "@/components/custom/header";
import { ExternalLink, FileText, Calendar, Brain, HardDrive, Search, BookOpen, Sparkles, Eraser, Zap } from "lucide-react";

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
        title: "In-Parameter Memory",
        description: "Knowledge stored directly in model weights during pretraining and fine-tuning. Research on how facts, skills, and biases are encoded across layers, how weight updates create or destroy memories, and the interplay between parameter count and memorization capacity.",
        icon: Brain,
    },
    {
        title: "In-Context Memory (KV Cache)",
        description: "How models leverage the key-value cache to hold and reason over information within a single context window. Research on KV-cache compression, eviction policies, attention sink heads, sparse retrieval from long contexts, and memory-efficient serving.",
        icon: HardDrive,
    },
    {
        title: "External Retrieval Mechanisms",
        description: "Augmenting LLMs with retrieval-augmented generation (RAG), tool use, and episodic memory stores. Research on when to retrieve vs. recall from parameters, retrieval quality's impact on generation, and hybrid architectures that blend parametric and non-parametric memory.",
        icon: Search,
    },
    {
        title: "Forgetting & Unlearning",
        description: "Controlled removal of memorized information — from mitigating catastrophic forgetting in continual learning, to targeted unlearning of private or copyrighted data. Research on self-distillation, gradient-based erasure, and benchmarking what models truly forget.",
        icon: Eraser,
    },
    {
        title: "Inference Optimization",
        description: "Making LLM serving faster and cheaper through efficient attention, KV-cache compression, sparse and low-rank approximations, speculative decoding, and quantization — all grounded in understanding which memories the model actually needs at inference time.",
        icon: Zap,
    },
    {
        title: "Reasoning Under Memory Constraints",
        description: "How memory limitations shape reasoning quality. Research on chain-of-thought as working memory, the relationship between context length and reasoning depth, and how models degrade gracefully (or don't) when memory is constrained.",
        icon: Sparkles,
    },
    {
        title: "Continual Learning",
        description: "Enabling deployed models to absorb new knowledge over time without catastrophic forgetting. Research on replay-based, regularization-based, and architecture-based strategies for lifelong learning in LLMs — connecting memorization theory to practical model updates.",
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
                                Primary Focus
                            </div>
                        </div>
                        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                            AI Memorization
                        </h1>
                        <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl">
                            LLMs memorize through three mechanisms: <strong>in-parameter</strong> (knowledge baked into weights),{" "}
                            <strong>in-context</strong> (information held in the KV cache during inference), and{" "}
                            <strong>external retrieval</strong> (augmenting generation with retrieved documents or tools).
                            My research studies how these mechanisms interact — and how to make models that remember what matters,
                            forget what they should, and reason efficiently within real-world memory budgets.
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
