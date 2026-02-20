import { Header } from "@/components/custom/header";
import { BookOpen, ExternalLink, FileText, Users, Calendar, Award, Eye, Brain, Layers } from "lucide-react";
import { useNavigate } from "react-router-dom";

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
    title: "LLM / VLM / VLA",
    description: "Multi-agent interaction, self-improving AI, continual learning, and efficient model memory. Primary focus on how LLM/VLM/VLA agents collaborate, self-improve through generate-validate loops, and maintain knowledge efficiently over time.",
    icon: Users,
    color: "blue",
    highlight: true,
    path: "/research/llm-vlm",
  },
  {
    title: "Synthetic Data",
    description: "Synthetic data generation, model collapse dynamics, data curation methods, and safety-oriented data pipelines for self-improving AI.",
    icon: Award,
    color: "orange",
    highlight: true,
    path: "/research/synthetic-data",
  },
  {
    title: "Architecture",
    description: "Transformer memory mechanisms, efficient architectures, multimodal architectures, and scalable designs. Research on how models store, retrieve, and reason over information at scale.",
    icon: Eye,
    color: "purple",
    path: "/research/architecture",
  },
  {
    title: "Continual Learning",
    description: "Catastrophic forgetting mitigation, difficulty-aware replay, curriculum strategies, and evaluation frameworks for lifelong learning.",
    icon: Layers,
    color: "green",
    path: "/research/continual-learning",
  },
];


export const Research = () => {
  const navigate = useNavigate();

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
              Focused on multi-agent systems and self-improving AI through synthetic data,
              brain-inspired neural architectures, and continual learning.
            </p>
          </div>

          {/* Research Areas */}
          <section className="mb-6">
            <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-1.5">
              <Brain className="w-4 h-4 text-blue-600" />
              Research Directions
            </h2>
            <div className="grid md:grid-cols-2 gap-3">
              {researchAreas.map((area) => {
                const Icon = area.icon;
                return (
                  <div
                    key={area.title}
                    onClick={() => navigate(area.path)}
                    className={`p-3 rounded-lg border cursor-pointer ${area.highlight
                      ? "border-blue-200 dark:border-blue-800 bg-blue-50/50 dark:bg-blue-900/20"
                      : "border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50"
                      } hover:shadow-md transition-all hover:scale-[1.01] group`}
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
                      <h3 className="text-sm font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">{area.title}</h3>
                      <ExternalLink className="w-3 h-3 text-gray-400 ml-auto opacity-0 group-hover:opacity-100 transition-opacity" />
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
                    <span className={`px-1.5 py-0.5 text-xs rounded ${pub.type === 'conference' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' :
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
              <p><strong>Fellowship:</strong> MOVE @ Handshake AI (Alumni)</p>
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
