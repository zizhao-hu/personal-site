import { Header } from "@/components/custom/header";
import { ExternalLink, FileText, Calendar, BookOpen, Sparkles, Settings, RefreshCw } from "lucide-react";

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
        title: "Pretraining",
        description: "Large-scale pretraining of language and vision-language models from scratch. Research on data mixing strategies, tokenization, training stability, scaling laws, and contrastive objectives that produce strong foundation models across modalities.",
        icon: BookOpen,
    },
    {
        title: "Supervised Fine-Tuning (SFT)",
        description: "Adapting pretrained models to follow instructions and perform domain-specific tasks through curated demonstration data. Investigating efficient fine-tuning methods (LoRA, QLoRA), data quality vs. quantity trade-offs, and multi-task instruction tuning.",
        icon: Sparkles,
    },
    {
        title: "Post-Training",
        description: "Alignment and preference optimization after SFT—including RLHF, DPO, and iterative self-play. Research on reward modeling, safety tuning, red-teaming, and how post-training shapes model behavior, helpfulness, and refusal boundaries.",
        icon: Settings,
    },
    {
        title: "Continual Learning",
        description: "Enabling LLMs and VLMs to learn new knowledge and capabilities over time without catastrophic forgetting. Developing replay-free and parameter-efficient continual learning methods tailored for large-scale generative models.",
        icon: RefreshCw,
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
                            My work in this area spans the full model lifecycle—from <strong>pretraining</strong> foundation
                            models, through <strong>supervised fine-tuning (SFT)</strong>, to <strong>post-training</strong> alignment
                            and safety—while also investigating how to enable <strong>continual learning</strong> in
                            large-scale generative systems.
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
