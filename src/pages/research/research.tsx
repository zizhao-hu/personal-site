import { Header } from "@/components/custom/header";
import { BookOpen, ExternalLink, FileText, Users, Calendar, Award, Eye, Brain, Layers } from "lucide-react";

interface Publication {
  title: string;
  authors: string;
  venue: string;
  year: number;
  type: "conference" | "journal" | "preprint" | "workshop";
  link?: string;
  highlight?: boolean;
}

const publications: Publication[] = [
  {
    title: "Multimodal Synthetic Data Finetuning and Model Collapse",
    authors: "Zizhao Hu, et al.",
    venue: "ACM International Conference on Multimodal Interaction (ICMI)",
    year: 2025,
    type: "conference",
    link: "https://scholar.google.com/citations?user=A8J42tQAAAAJ",
    highlight: true,
  },
  {
    title: "Static Key Attention in Vision",
    authors: "Zizhao Hu, et al.",
    venue: "Preprint",
    year: 2024,
    type: "preprint",
    link: "https://scholar.google.com/citations?user=A8J42tQAAAAJ",
  },
  {
    title: "Lateralization MLP: A Simple Brain-inspired Architecture for Diffusion",
    authors: "Zizhao Hu, et al.",
    venue: "Preprint",
    year: 2024,
    type: "preprint",
    link: "https://scholar.google.com/citations?user=A8J42tQAAAAJ",
  },
];

const researchAreas = [
  {
    title: "Multi-Agent Systems & Self-Improving AI",
    description: "Developing autonomous agents that collaborate, compete, and improve through interaction. My core focus is on systems that can bootstrap their own capabilities—agents that generate training data, evaluate their own outputs, and evolve without constant human supervision.",
    icon: Users,
    color: "blue",
    highlight: true,
  },
  {
    title: "Vision-Language Architectures",
    description: "Building models that seamlessly integrate visual and textual understanding. Research on attention mechanisms, multimodal fusion, and efficient architectures that can reason across modalities.",
    icon: Eye,
    color: "purple",
  },
  {
    title: "Continual Learning & Data Curation",
    description: "Enabling AI systems to learn continuously without forgetting. Developing methods for lifelong learning, knowledge retention, and adaptive training that allow models to evolve with new data while preserving prior capabilities.",
    icon: Layers,
    color: "green",
  },
  {
    title: "Synthetic Data & Model Safety",
    description: "Researching how to generate high-quality synthetic data while avoiding model collapse and ensuring AI systems remain safe and reliable through iterative training cycles.",
    icon: Award,
    color: "orange",
  },
];

export const Research = () => {
  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      
      <main className="flex-1 overflow-y-auto pb-24">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6">
          {/* Hero Section */}
          <div className="mb-6">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Research
            </h1>
            <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
              Building <strong>AI systems that improve themselves while remaining under control</strong>.
              Working at the intersection of multi-agent collaboration, vision-language models,
              and continual learning.
            </p>
          </div>

          {/* Research Areas */}
          <section className="mb-6">
            <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-1.5">
              <Brain className="w-4 h-4 text-blue-600" />
              Research Areas
            </h2>
            <div className="grid md:grid-cols-2 gap-3">
              {researchAreas.map((area) => {
                const Icon = area.icon;
                return (
                  <div
                    key={area.title}
                    className={`p-3 rounded-lg border ${
                      area.highlight
                        ? "border-blue-200 dark:border-blue-800 bg-blue-50/50 dark:bg-blue-900/20"
                        : "border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50"
                    } hover:shadow-md transition-shadow`}
                  >
                    {area.highlight && (
                      <span className="inline-block px-1.5 py-0.5 text-xs font-medium bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded mb-1.5">
                        Primary Focus
                      </span>
                    )}
                    <div className="flex items-center gap-2 mb-1.5">
                      <div className={`w-7 h-7 rounded-md bg-${area.color}-100 dark:bg-${area.color}-900/30 flex items-center justify-center`}>
                        <Icon className={`w-3.5 h-3.5 text-${area.color}-600 dark:text-${area.color}-400`} />
                      </div>
                      <h3 className="text-sm font-semibold text-gray-900 dark:text-white">{area.title}</h3>
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">{area.description}</p>
                  </div>
                );
              })}
            </div>
          </section>

          {/* Publications */}
          <section className="mb-6">
            <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-1.5">
              <FileText className="w-4 h-4 text-purple-600" />
              Recent Publications
            </h2>
            <div className="space-y-3">
              {publications.map((pub, index) => (
                <div
                  key={index}
                  className={`p-3 rounded-lg border ${
                    pub.highlight
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
                    <span className={`px-1.5 py-0.5 text-xs rounded ${
                      pub.type === 'conference' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' :
                      pub.type === 'workshop' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400' :
                      'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                    }`}>
                      {pub.type}
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

            <a
              href="https://scholar.google.com/citations?user=A8J42tQAAAAJ"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 mt-4 px-3 py-1.5 bg-gray-100 dark:bg-gray-800 rounded-md text-xs font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            >
              View all on Google Scholar
              <ExternalLink className="w-3 h-3" />
            </a>
          </section>

          {/* Academic Service */}
          <section className="mb-6">
            <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-1.5">
              <BookOpen className="w-4 h-4 text-green-600" />
              Academic Service
            </h2>
            <div className="p-3 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50">
              <p className="text-xs text-gray-700 dark:text-gray-300 mb-1">
                <strong>Reviewer:</strong> NeurIPS 2024, ICLR 2024-2025, ICML 2024-2025
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                Active contributor through peer review at top-tier venues.
              </p>
            </div>
          </section>

          {/* Current Position */}
          <section className="p-4 rounded-lg bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 border border-indigo-100 dark:border-indigo-800">
            <h2 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
              Current Position
            </h2>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-gray-700 dark:text-gray-300">
              <p><strong>PhD Student</strong> • USC</p>
              <p><strong>Lab:</strong> GLAMOUR Lab</p>
              <p><strong>Fellowship:</strong> MOVE @ Handshake AI</p>
              <p><strong>Advisors:</strong> J. Thomason, M. Rostami</p>
              <p><strong>Affiliation:</strong> GLAMOUR Lab, USC ISI</p>
              <p><strong>Graduation:</strong> 2027</p>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};
