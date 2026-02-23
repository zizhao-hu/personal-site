import { Header } from "@/components/custom/header";
import { ExternalLink, FileText, Calendar, Database, ShieldAlert, RefreshCw } from "lucide-react";

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
        title: "Generate-Validate Loop",
        description: "Self-improving data pipelines where models generate candidate outputs, then validate them against criteria (correctness, safety, diversity) before accepting them as training data. This closed-loop approach produces higher-quality data than one-shot generation and enables models to bootstrap their own improvement without human annotation.",
        icon: RefreshCw,
    },
    {
        title: "Synthetic Data Generation",
        description: "Creating high-quality synthetic training data using generative models. Developing pipelines that produce diverse, balanced datasets without the privacy concerns or biases of web-scraped data, while ensuring the generated data is genuinely useful for model training.",
        icon: Database,
    },
    {
        title: "Model Collapse Prevention",
        description: "Understanding and preventing the phenomenon where models trained on synthetic data from previous model generations progressively degrade. Researching the feedback loops that cause model collapse and developing mitigation strategies.",
        icon: ShieldAlert,
    },
    {
        title: "Data Quality & Curation",
        description: "Developing automated methods to assess, filter, and curate training data. Research on quality metrics, deduplication, bias detection, and data mixing strategies that optimize model performance per training dollar.",
        icon: Database,
    },
    {
        title: "Safety Through Data",
        description: "Using synthetic data generation as a lever for AI safety. Creating targeted safety training examples, red-teaming datasets, and alignment data that help models learn to refuse harmful requests while remaining helpful.",
        icon: ShieldAlert,
    },
];

export const SyntheticDataResearch = () => {
    return (
        <div className="flex flex-col min-h-dvh bg-background">
            <Header />

            <main className="flex-1 overflow-y-auto pb-24">
                <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6">
                    {/* Hero */}
                    <div className="mb-6">
                        <div className="flex items-center gap-2 mb-2">
                            <div className="px-2 py-0.5 rounded-full bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 text-[10px] font-semibold uppercase tracking-tight">
                                Research Direction
                            </div>
                        </div>
                        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                            Synthetic Data
                        </h1>
                        <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl">
                            High-quality data is the foundation of modern AI. My research investigates
                            <strong> generate-validate loops</strong> for self-improving data pipelines,
                            <strong> synthetic data generation</strong>, <strong>model collapse dynamics</strong>,
                            <strong> data curation methods</strong>, and <strong>safety-oriented data pipelines</strong>—enabling
                            AI systems to bootstrap their own improvement while maintaining quality and safety.
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
                                            <div className="w-7 h-7 rounded-md bg-orange-100 dark:bg-orange-900/30 flex items-center justify-center">
                                                <Icon className="w-3.5 h-3.5 text-orange-600 dark:text-orange-400" />
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
                            <FileText className="w-4 h-4 text-orange-600" />
                            Related Publications
                        </h2>
                        <div className="space-y-3">
                            {publications.map((pub, index) => (
                                <div
                                    key={index}
                                    className={`p-3 rounded-lg border ${pub.highlight
                                        ? "border-orange-200 dark:border-orange-800 bg-orange-50/50 dark:bg-orange-900/20"
                                        : "border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50"
                                        }`}
                                >
                                    {pub.highlight && (
                                        <span className="inline-block px-1.5 py-0.5 text-xs font-medium bg-orange-100 dark:bg-orange-900/50 text-orange-700 dark:text-orange-300 rounded mb-1.5">
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
