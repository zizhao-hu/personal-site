import { Header } from "@/components/custom/header";
import { ExternalLink, FileText, Calendar, Layers, RefreshCw } from "lucide-react";

const keyTopics = [
    {
        title: "Catastrophic Forgetting Mitigation",
        description: "Developing methods that enable neural networks to learn new tasks without destroying performance on previously learned ones. Combining replay-based, regularization-based, and architecture-based strategies for robust knowledge retention.",
        icon: Layers,
    },
    {
        title: "Difficulty-Aware Replay (DREAM)",
        description: "Prioritizing difficult, boundary-adjacent samples in experience replay buffers. By focusing on the most informative examples, DREAM achieves better performance with smaller memory footprints compared to random replay strategies.",
        icon: RefreshCw,
    },
    {
        title: "Curriculum Continual Learning",
        description: "Ordering training tasks and samples intelligently to maximize positive transfer and minimize interference. Research on how task sequencing and difficulty progression affect continual learning outcomes.",
        icon: Layers,
    },
    {
        title: "Evaluation & Benchmarking",
        description: "Developing comprehensive evaluation frameworks for continual learning that go beyond simple accuracy metrics. Measuring forward transfer, backward transfer, forgetting rates, and computational efficiency.",
        icon: RefreshCw,
    },
];

export const ContinualLearningResearch = () => {
    return (
        <div className="flex flex-col min-h-dvh bg-background">
            <Header />

            <main className="flex-1 overflow-y-auto pb-24">
                <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6">
                    {/* Hero */}
                    <div className="mb-6">
                        <div className="flex items-center gap-2 mb-2">
                            <div className="px-2 py-0.5 rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 text-[10px] font-semibold uppercase tracking-tight">
                                Research Direction
                            </div>
                        </div>
                        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                            Continual Learning
                        </h1>
                        <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl">
                            Enabling AI systems to learn continuously without catastrophic forgetting.
                            My research develops methods for <strong>lifelong learning</strong>,
                            <strong> difficulty-aware replay</strong>, <strong>curriculum strategies</strong>,
                            and <strong>evaluation frameworks</strong> that allow models to evolve with new data
                            while preserving prior knowledge—a critical step toward truly adaptive AI.
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
                                            <div className="w-7 h-7 rounded-md bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                                                <Icon className="w-3.5 h-3.5 text-green-600 dark:text-green-400" />
                                            </div>
                                            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">{topic.title}</h3>
                                        </div>
                                        <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed">{topic.description}</p>
                                    </div>
                                );
                            })}
                        </div>
                    </section>

                    {/* Related Work */}
                    <section className="mb-6">
                        <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-1.5">
                            <FileText className="w-4 h-4 text-green-600" />
                            Related Work
                        </h2>
                        <div className="space-y-3">
                            <div className="p-3 rounded-lg border border-green-200 dark:border-green-800 bg-green-50/50 dark:bg-green-900/20">
                                <span className="inline-block px-1.5 py-0.5 text-xs font-medium bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300 rounded mb-1.5">
                                    Active Project
                                </span>
                                <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-0.5">
                                    DREAM-C2L: Continual Learning Framework
                                </h3>
                                <p className="text-xs text-gray-600 dark:text-gray-400 mb-1.5">
                                    Open-source framework for continual learning research with difficulty-aware sample ordering,
                                    replay-based retention methods, and reproducible HPC experiment pipelines.
                                </p>
                                <div className="flex items-center gap-3 text-xs">
                                    <span className="flex items-center gap-1 text-gray-500 dark:text-gray-500">
                                        <Calendar className="w-2.5 h-2.5" />
                                        Ongoing
                                    </span>
                                    <a
                                        href="https://github.com/zizhao-hu/dream-c2l"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center gap-1 text-xs text-blue-600 dark:text-blue-400 hover:underline"
                                    >
                                        GitHub <ExternalLink className="w-2.5 h-2.5" />
                                    </a>
                                </div>
                            </div>
                        </div>
                    </section>
                </div>
            </main>
        </div>
    );
};
