import { Header } from "@/components/custom/header";
import { ExternalLink, FileText, Calendar, Cpu, Zap } from "lucide-react";

const publications = [
    {
        title: "Static Key Attention in Vision",
        authors: "Zizhao Hu, et al.",
        venue: "Preprint",
        year: 2024,
        type: "preprint" as const,
        link: "https://scholar.google.com/citations?user=A8J42tQAAAAJ",
        highlight: true,
    },
    {
        title: "Lateralization MLP: A Simple Brain-inspired Architecture for Diffusion",
        authors: "Zizhao Hu, et al.",
        venue: "Preprint",
        year: 2024,
        type: "preprint" as const,
        link: "https://scholar.google.com/citations?user=A8J42tQAAAAJ",
    },
];

const keyTopics = [
    {
        title: "Static Key Attention",
        description: "Rethinking the attention mechanism by pre-computing certain attention patterns. Static keys reduce the computational cost of self-attention while maintaining model performance, enabling more efficient vision transformers.",
        icon: Zap,
    },
    {
        title: "Brain-Inspired Architectures",
        description: "Drawing inspiration from the lateralization of biological brains to design asymmetric neural network architectures. The Lateralization MLP introduces hemisphere-specific processing for diffusion models.",
        icon: Cpu,
    },
    {
        title: "Efficient Transformers",
        description: "Developing attention variants and architectural modifications that reduce the quadratic cost of self-attention. Exploring sparse attention, linear attention, and hybrid approaches for scalable deep learning.",
        icon: Zap,
    },
    {
        title: "Mixture of Experts (MoE)",
        description: "Scaling model capacity without proportionally scaling compute. Research on expert routing, load balancing, and specialized expert architectures for conditional computation in large-scale models.",
        icon: Cpu,
    },
];

export const ArchitectureResearch = () => {
    return (
        <div className="flex flex-col min-h-dvh bg-background">
            <Header />

            <main className="flex-1 overflow-y-auto pb-24">
                <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6">
                    {/* Hero */}
                    <div className="mb-6">
                        <div className="flex items-center gap-2 mb-2">
                            <div className="px-2 py-0.5 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 text-[10px] font-semibold uppercase tracking-tight">
                                Research Direction
                            </div>
                        </div>
                        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                            Architecture
                        </h1>
                        <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl">
                            Novel neural network architectures that push the boundaries of efficiency and capability.
                            My work explores <strong>static key attention mechanisms</strong>,
                            <strong> brain-inspired lateralization</strong>, <strong>mixture of experts</strong>,
                            and <strong>efficient transformer variants</strong>—making deep learning more powerful
                            while reducing computational overhead.
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
                                            <div className="w-7 h-7 rounded-md bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center">
                                                <Icon className="w-3.5 h-3.5 text-purple-600 dark:text-purple-400" />
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
                                            ? "border-purple-200 dark:border-purple-800 bg-purple-50/50 dark:bg-purple-900/20"
                                            : "border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50"
                                        }`}
                                >
                                    {pub.highlight && (
                                        <span className="inline-block px-1.5 py-0.5 text-xs font-medium bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300 rounded mb-1.5">
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
