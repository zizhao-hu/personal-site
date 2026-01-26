import { Header } from "@/components/custom/header";
import { BookOpen, ExternalLink, FileText, Users, Calendar, Award } from "lucide-react";

interface Publication {
  title: string;
  authors: string;
  venue: string;
  year: number;
  type: "conference" | "journal" | "preprint";
  link?: string;
  pdf?: string;
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
];

const researchAreas = [
  {
    title: "Multi-Agent Systems",
    description: "Developing frameworks for autonomous agents that can collaborate, negotiate, and achieve complex goals through emergent behavior.",
    icon: Users,
    color: "blue",
  },
  {
    title: "Synthetic Data Generation",
    description: "Creating high-quality synthetic datasets for training AI models while ensuring safety, privacy, and avoiding model collapse.",
    icon: FileText,
    color: "purple",
  },
  {
    title: "LLM Safety & Alignment",
    description: "Researching methods to ensure large language models remain safe, reliable, and aligned with human values during training and deployment.",
    icon: Award,
    color: "green",
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
              My research focuses on the intersection of multi-agent AI systems, synthetic data generation, 
              and model safety. I'm particularly interested in how autonomous agents can learn to collaborate 
              and how we can ensure AI systems remain reliable as they scale.
            </p>
          </div>

          {/* Research Areas */}
          <section className="mb-12">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-blue-600" />
              Research Areas
            </h2>
            <div className="grid md:grid-cols-3 gap-4">
              {researchAreas.map((area) => {
                const Icon = area.icon;
                return (
                  <div
                    key={area.title}
                    className="p-5 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50 hover:shadow-lg transition-shadow"
                  >
                    <div className={`w-10 h-10 rounded-lg bg-${area.color}-100 dark:bg-${area.color}-900/30 flex items-center justify-center mb-3`}>
                      <Icon className={`w-5 h-5 text-${area.color}-600 dark:text-${area.color}-400`} />
                    </div>
                    <h3 className="font-semibold text-gray-900 dark:text-white mb-2">{area.title}</h3>
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
              Publications
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
                  </div>
                  {pub.link && (
                    <a
                      href={pub.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 mt-3 text-sm text-blue-600 dark:text-blue-400 hover:underline"
                    >
                      View on Google Scholar <ExternalLink className="w-3 h-3" />
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
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};
