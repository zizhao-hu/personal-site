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
    venue: "NeurIPS Workshop",
    year: 2024,
    type: "workshop",
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
    title: "Curriculum Learning & Data Curation",
    description: "Creating intelligent training curricula that optimize how models learn. Developing methods to order, filter, and synthesize training data for maximum learning efficiency and model robustness.",
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
        <div className="max-w-4xl mx-auto px-4 py-8">
          {/* Hero Section */}
          <div className="mb-12">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
              Research
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed">
              My research vision is building <strong>AI systems that improve themselves</strong>. 
              I work at the intersection of multi-agent collaboration, vision-language models, 
              and curriculum learning—creating agents that can generate their own training data, 
              evaluate their outputs, and evolve autonomously while remaining safe and reliable.
            </p>
          </div>

          {/* Research Areas */}
          <section className="mb-12">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
              <Brain className="w-5 h-5 text-blue-600" />
              Research Areas
            </h2>
            <div className="grid md:grid-cols-2 gap-4">
              {researchAreas.map((area) => {
                const Icon = area.icon;
                return (
                  <div
                    key={area.title}
                    className={`p-5 rounded-xl border ${
                      area.highlight 
                        ? "border-blue-200 dark:border-blue-800 bg-blue-50/50 dark:bg-blue-900/20" 
                        : "border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50"
                    } hover:shadow-lg transition-shadow`}
                  >
                    {area.highlight && (
                      <span className="inline-block px-2 py-1 text-xs font-medium bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded mb-2">
                        Primary Focus
                      </span>
                    )}
                    <div className="flex items-center gap-3 mb-2">
                      <div className={`w-10 h-10 rounded-lg bg-${area.color}-100 dark:bg-${area.color}-900/30 flex items-center justify-center`}>
                        <Icon className={`w-5 h-5 text-${area.color}-600 dark:text-${area.color}-400`} />
                      </div>
                      <h3 className="font-semibold text-gray-900 dark:text-white">{area.title}</h3>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{area.description}</p>
                  </div>
                );
              })}
            </div>
          </section>

          {/* Publications */}
          <section className="mb-12">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
              <FileText className="w-5 h-5 text-purple-600" />
              Recent Publications
            </h2>
            <div className="space-y-4">
              {publications.map((pub, index) => (
                <div
                  key={index}
                  className={`p-5 rounded-xl border ${
                    pub.highlight 
                      ? "border-blue-200 dark:border-blue-800 bg-blue-50/50 dark:bg-blue-900/20" 
                      : "border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50"
                  }`}
                >
                  {pub.highlight && (
                    <span className="inline-block px-2 py-1 text-xs font-medium bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded mb-2">
                      Featured
                    </span>
                  )}
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
                    {pub.title}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    {pub.authors}
                  </p>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="flex items-center gap-1 text-gray-500 dark:text-gray-500">
                      <Calendar className="w-3 h-3" />
                      {pub.year}
                    </span>
                    <span className="text-purple-600 dark:text-purple-400 font-medium">
                      {pub.venue}
                    </span>
                    <span className={`px-2 py-0.5 text-xs rounded ${
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
                      className="inline-flex items-center gap-1 mt-3 text-sm text-blue-600 dark:text-blue-400 hover:underline"
                    >
                      View Paper <ExternalLink className="w-3 h-3" />
                    </a>
                  )}
                </div>
              ))}
            </div>
            
            <a
              href="https://scholar.google.com/citations?user=A8J42tQAAAAJ"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 mt-6 px-4 py-2 bg-gray-100 dark:bg-gray-800 rounded-lg text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            >
              View all publications on Google Scholar
              <ExternalLink className="w-4 h-4" />
            </a>
          </section>

          {/* Academic Service */}
          <section className="mb-12">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-green-600" />
              Academic Service
            </h2>
            <div className="p-5 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50">
              <p className="text-gray-700 dark:text-gray-300 mb-3">
                <strong>Reviewer:</strong> NeurIPS 2024, ICLR 2024-2025, ICML 2024-2025
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Active contributor to the research community through peer review at top-tier venues.
              </p>
            </div>
          </section>

          {/* Current Position */}
          <section className="p-6 rounded-xl bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 border border-indigo-100 dark:border-indigo-800">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
              Current Position
            </h2>
            <div className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <p><strong>PhD Student</strong> • University of Southern California (USC)</p>
              <p><strong>Lab:</strong> MINDS Group / GLAMOUR Lab</p>
              <p><strong>Advisors:</strong> Prof. Jesse Thomason, Prof. Mohammad Rostami</p>
              <p><strong>Affiliation:</strong> Information Sciences Institute (ISI)</p>
              <p><strong>Expected Graduation:</strong> 2027</p>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};
