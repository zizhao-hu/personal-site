import { Header } from "@/components/custom/header";
import { ExternalLink, FileText, Calendar, Users, MessageSquare, Orbit, Bot, Sparkles, BookOpen } from "lucide-react";

const keyTopics = [
    {
        title: "Multi-Agent Collaboration",
        description: "How multiple LLM/VLM agents collaborate, debate, verify, and refine each other's outputs. Research on emergent communication protocols, consensus-building, and multi-agent self-play for improving reasoning and task completion quality.",
        icon: Users,
    },
    {
        title: "Agent Orchestration",
        description: "Building scalable frameworks for multi-agent pipelines — task routing, tool use, memory systems, and self-correction loops. How to design agent architectures that are reliable, composable, and can scale from single tasks to complex workflows.",
        icon: Orbit,
    },
    {
        title: "Role Specialization",
        description: "Training and prompting agents for distinct roles — critic, coder, researcher, planner — and studying how role assignment affects team performance. Research on when specialization outperforms generalist agents and how to dynamically allocate roles.",
        icon: Bot,
    },
    {
        title: "Debate & Verification",
        description: "Using adversarial debate and cross-agent verification to improve output quality. Research on how agents can catch each other's mistakes, reduce hallucination through mutual critique, and produce more reliable final outputs.",
        icon: MessageSquare,
    },
    {
        title: "Self-Improving Agents",
        description: "Systems that generate their own training signal through synthetic data, self-reflection, and iterative refinement. Investigating feedback loops where agents evaluate their own outputs, generate preference pairs, and continuously improve without human annotation.",
        icon: Sparkles,
    },
    {
        title: "Evaluation & Benchmarking",
        description: "Developing evaluation frameworks for multi-agent systems — measuring coordination efficiency, task decomposition quality, communication overhead, and emergent capabilities that arise from agent interaction at scale.",
        icon: BookOpen,
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
                            Multi-Agent Systems
                        </h1>
                        <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl">
                            How multiple LLM and VLM agents work together to solve problems no single model can handle alone.
                            My research explores <strong>agent collaboration</strong>, <strong>role specialization</strong>,{" "}
                            <strong>debate-based verification</strong>, and <strong>orchestration frameworks</strong> that
                            enable reliable, scalable multi-agent workflows — from simple pipelines to complex reasoning chains.
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
                                    PRISM: Multi-Agent Synthetic Data Pipeline
                                </h3>
                                <p className="text-xs text-gray-600 dark:text-gray-400 mb-1.5">
                                    A multi-agent pipeline for generating persona-diverse synthetic data, combining
                                    intent-based routing with role-specialized agents for high-quality data curation.
                                </p>
                                <div className="flex items-center gap-3 text-xs">
                                    <span className="flex items-center gap-1 text-gray-500 dark:text-gray-500">
                                        <Calendar className="w-2.5 h-2.5" />
                                        Ongoing
                                    </span>
                                </div>
                            </div>
                        </div>
                    </section>
                </div>
            </main>
        </div>
    );
};
