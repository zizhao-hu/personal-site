import { Header } from "@/components/custom/header";
import { Rocket, Bot, Brain, Globe, Cpu, Shield, Sparkles, ExternalLink, Github } from "lucide-react";

interface Project {
  title: string;
  description: string;
  tags: string[];
  status: "active" | "research" | "concept";
  icon: React.ElementType;
  color: string;
  link?: string;
  github?: string;
}

const projects: Project[] = [
  {
    title: "DREAM-C2L",
    description: "Reproducible development environments for curriculum learning research. Building robust dataloaders and training pipelines for progressive AI model training.",
    tags: ["Python", "PyTorch", "Curriculum Learning", "Open Source"],
    status: "active",
    icon: Brain,
    color: "blue",
    github: "https://github.com/zizhao-hu",
  },
  {
    title: "Project Orion",
    description: "Industry research project at Handshake AI focusing on multi-agent orchestration for enterprise workflows and autonomous task completion.",
    tags: ["Multi-Agent", "LLM", "Enterprise AI", "Handshake AI"],
    status: "active",
    icon: Bot,
    color: "purple",
  },
  {
    title: "Project Canary",
    description: "Safety-focused research on synthetic data generation and model collapse prevention. Ensuring AI systems remain stable through iterative training.",
    tags: ["Synthetic Data", "Model Safety", "Research"],
    status: "research",
    icon: Shield,
    color: "green",
  },
];

const futureVision = [
  {
    title: "Agentic AI Systems",
    description: "Moving from 'answer my question' to 'do this for me.' Multi-agent ecosystems where specialized models collaborate—one plans, another executes, a third verifies.",
    icon: Bot,
    insight: "Users need autonomy. AI that can browse, book, coordinate, and handle tasks without constant supervision.",
  },
  {
    title: "Test-Time Compute & Reasoning",
    description: "Scaling inference over training. Models that 'think longer' before responding, allocating compute dynamically to solve complex logic and reasoning problems.",
    icon: Brain,
    insight: "Users need reliability. No more confidently stated hallucinations—systems that check their own work.",
  },
  {
    title: "World Models & Physical AI",
    description: "Training AI on video and sensor data to understand cause-and-effect. Critical for the robotics surge and real-world AI applications.",
    icon: Globe,
    insight: "Users need contextual awareness. AI that can 'see' and understand physical environments, not just parse text.",
  },
  {
    title: "Small, Specialized & Sovereign",
    description: "Edge AI and Mixture of Experts (MoE) models that run locally on phones and laptops. Cheaper, faster, and more accurate for specialized domains.",
    icon: Cpu,
    insight: "Users need privacy and speed. Local AI without sending sensitive data to distant cloud servers.",
  },
];

const StatusBadge = ({ status }: { status: Project["status"] }) => {
  const styles = {
    active: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
    research: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
    concept: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400",
  };
  
  return (
    <span className={`px-2 py-1 text-xs font-medium rounded ${styles[status]}`}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
};

export const Projects = () => {
  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      
      <main className="flex-1 overflow-y-auto pb-24">
        <div className="max-w-4xl mx-auto px-4 py-8">
          {/* Hero */}
          <div className="mb-12">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
              Projects
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed">
              Building the next generation of AI systems—from multi-agent orchestration to 
              safe synthetic data pipelines. Here's what I'm working on and where AI is headed.
            </p>
          </div>

          {/* Current Projects */}
          <section className="mb-16">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
              <Rocket className="w-5 h-5 text-orange-500" />
              Current Projects
            </h2>
            <div className="space-y-4">
              {projects.map((project) => {
                const Icon = project.icon;
                return (
                  <div
                    key={project.title}
                    className="p-6 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50 hover:shadow-lg transition-all group"
                  >
                    <div className="flex items-start gap-4">
                      <div className={`w-12 h-12 rounded-xl bg-${project.color}-100 dark:bg-${project.color}-900/30 flex items-center justify-center flex-shrink-0`}>
                        <Icon className={`w-6 h-6 text-${project.color}-600 dark:text-${project.color}-400`} />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h3 className="font-semibold text-gray-900 dark:text-white text-lg">
                            {project.title}
                          </h3>
                          <StatusBadge status={project.status} />
                        </div>
                        <p className="text-gray-600 dark:text-gray-400 mb-3">
                          {project.description}
                        </p>
                        <div className="flex flex-wrap gap-2 mb-3">
                          {project.tags.map((tag) => (
                            <span
                              key={tag}
                              className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                        <div className="flex gap-3">
                          {project.github && (
                            <a
                              href={project.github}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
                            >
                              <Github className="w-4 h-4" /> GitHub
                            </a>
                          )}
                          {project.link && (
                            <a
                              href={project.link}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-1 text-sm text-blue-600 dark:text-blue-400 hover:underline"
                            >
                              Learn more <ExternalLink className="w-3 h-3" />
                            </a>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          {/* Future of AI Vision */}
          <section className="mb-12">
            <div className="flex items-center gap-3 mb-6">
              <Sparkles className="w-6 h-6 text-yellow-500" />
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                The Future of AI (2026+)
              </h2>
            </div>
            <p className="text-gray-600 dark:text-gray-400 mb-6">
              The era of "just add more data and parameters" is hitting diminishing returns. 
              The industry is pivoting from building <strong>oracles</strong> (models that talk) 
              to building <strong>partners</strong> (systems that act).
            </p>
            
            <div className="grid md:grid-cols-2 gap-4">
              {futureVision.map((item) => {
                const Icon = item.icon;
                return (
                  <div
                    key={item.title}
                    className="p-5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gradient-to-br from-gray-50 to-white dark:from-gray-800/50 dark:to-gray-800"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <Icon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                      <h3 className="font-semibold text-gray-900 dark:text-white">
                        {item.title}
                      </h3>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                      {item.description}
                    </p>
                    <div className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800">
                      <p className="text-xs text-blue-700 dark:text-blue-300">
                        <strong>Key Insight:</strong> {item.insight}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          {/* CTA */}
          <section className="p-6 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white text-center">
            <h3 className="text-xl font-semibold mb-2">Interested in Collaboration?</h3>
            <p className="text-blue-100 mb-4">
              I'm open to research partnerships, consulting, and investment opportunities in AI/ML.
            </p>
            <a
              href="mailto:zizhaoh@usc.edu"
              className="inline-flex items-center gap-2 px-6 py-2 bg-white text-blue-600 rounded-lg font-medium hover:bg-blue-50 transition-colors"
            >
              Get in Touch
            </a>
          </section>
        </div>
      </main>
    </div>
  );
};
