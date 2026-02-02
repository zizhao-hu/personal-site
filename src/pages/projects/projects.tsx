import { Header } from "@/components/custom/header";
import { Rocket, Bot, Brain, Globe, Cpu, Shield, Sparkles, ExternalLink, Github, Eye, Zap, Network } from "lucide-react";

interface Project {
  title: string;
  description: string;
  details: string[];
  tags: string[];
  status: "active" | "research" | "prototype" | "concept" | "completed";
  icon: React.ElementType;
  color: string;
  link?: string;
  github?: string;
  image?: string;
}

const projects: Project[] = [
  {
    title: "AgentForge: Multi-Agent Orchestration Framework",
    description: "A framework for building autonomous multi-agent systems where specialized AI agents collaborate on complex tasks. One agent plans, another executes, and a third verifies—creating self-correcting workflows.",
    details: [
      "Agent specialization with role-based task delegation",
      "Inter-agent communication protocol for knowledge sharing",
      "Consensus mechanisms for conflict resolution",
      "Built-in safety guardrails and human-in-the-loop checkpoints"
    ],
    tags: ["Multi-Agent", "LLM Orchestration", "Python", "Autonomous Systems"],
    status: "active",
    icon: Network,
    color: "blue",
    github: "https://github.com/zizhao-hu/agentforge",
    image: "/images/projects/agentforge.png",
  },
  {
    title: "Project Orion",
    description: "Current MOVE Fellowship project at Handshake AI focused on high-quality reasoning refinement, safety injections, and model corrections. Specialized work leveraging PhD-level expertise in AI safety and alignment.",
    details: [
      "High-quality reasoning annotations and corrections",
      "Safety injections and 'rogue behavior' testing",
      "Jailbreak detection and mitigation strategies",
      "PhD-level expertise in targeted model refinement phases"
    ],
    tags: ["AI Safety", "Model Alignment", "Handshake AI", "MOVE Fellowship"],
    status: "active",
    icon: Bot,
    color: "purple",
  },
  {
    title: "DREAM-C2L: Continual Learning Framework",
    description: "Open-source framework for continual learning research. Enabling AI systems to learn continuously without catastrophic forgetting, adapting to new data while preserving prior knowledge.",
    details: [
      "Difficulty-aware sample ordering algorithms",
      "Replay-based and regularization methods for knowledge retention",
      "Reproducible experiment pipelines for HPC clusters",
      "Integration with PyTorch Lightning and Weights & Biases"
    ],
    tags: ["Continual Learning", "PyTorch", "Open Source", "Research"],
    status: "active",
    icon: Brain,
    color: "green",
    github: "https://github.com/zizhao-hu/dream-c2l",
    image: "/images/projects/dream.png",
  },
  {
    title: "ReasonChain: Test-Time Compute Scaling",
    description: "Research prototype exploring how to make LLMs 'think longer' before responding. Implementing chain-of-thought verification where models check their own reasoning before committing to an answer.",
    details: [
      "Multi-step reasoning with self-verification loops",
      "Confidence calibration and uncertainty quantification",
      "Dynamic compute allocation based on problem complexity",
      "Hallucination detection through reasoning trace analysis"
    ],
    tags: ["Test-Time Compute", "Reasoning", "LLM Safety", "Research"],
    status: "research",
    icon: Zap,
    color: "yellow",
  },
  {
    title: "VisionGround: World Models for Physical AI",
    description: "Building AI that understands cause-and-effect in the physical world. Training models on video data to predict outcomes—if a glass falls, it breaks. Critical foundation for robotics applications.",
    details: [
      "Video prediction models for physical dynamics",
      "Cause-effect reasoning from visual observations",
      "Sim-to-real transfer for robotic manipulation",
      "Multimodal fusion of vision, language, and proprioception"
    ],
    tags: ["World Models", "Robotics", "Video Understanding", "Embodied AI"],
    status: "research",
    icon: Globe,
    color: "cyan",
    image: "/images/projects/visionground.png",
  },
  {
    title: "Project Canary",
    description: "Foundational MOVE Fellowship project (Sept-Oct 2025) — a community-driven effort to train and refine frontier AI models. Completed 15,000+ tasks across 15 domains, improving Review 1 approval rates from 10% to 40%.",
    details: [
      "High-volume task generation across CS, Math, Medicine, Physics domains",
      "Core contributor in Computer Science domain",
      "Quality improvement: raised approval rates from 10% to 40%",
      "Precursor to Project Orion's specialized refinement phase"
    ],
    tags: ["AI Training", "Data Generation", "Handshake AI", "MOVE Fellowship"],
    status: "completed",
    icon: Shield,
    color: "green",
  },
  {
    title: "EdgeLLM: Sovereign AI on Device",
    description: "Exploring efficient small language models that run entirely on-device. Privacy-preserving AI that never sends data to the cloud—your AI assistant that respects your data sovereignty.",
    details: [
      "Model quantization and pruning for edge deployment",
      "On-device fine-tuning with federated learning",
      "Specialized domain adapters for legal, medical, code",
      "Offline-first architecture with optional cloud sync"
    ],
    tags: ["Edge AI", "Privacy", "Small Models", "Mobile"],
    status: "prototype",
    icon: Cpu,
    color: "indigo",
  },
  {
    title: "SynthVision: Multimodal Data Generation",
    description: "Pipeline for generating high-quality synthetic vision-language training data. Creating diverse, balanced datasets without the privacy concerns of web-scraped data.",
    details: [
      "Controllable image-text pair generation",
      "Automatic quality assessment and filtering",
      "Bias detection and mitigation in generated data",
      "Scalable generation with GPU-efficient diffusion models"
    ],
    tags: ["Synthetic Data", "Vision-Language", "Data Generation", "Diffusion"],
    status: "prototype",
    icon: Eye,
    color: "pink",
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
    prototype: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
    concept: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400",
    completed: "bg-gray-100 text-gray-700 dark:bg-gray-700/50 dark:text-gray-300",
  };

  const labels = {
    active: "Active",
    research: "Research",
    prototype: "Prototype",
    concept: "Concept",
    completed: "Completed",
  };

  return (
    <span className={`px-1.5 py-0.5 text-xs font-medium rounded ${styles[status]}`}>
      {labels[status]}
    </span>
  );
};

export const Projects = () => {
  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />

      <main className="flex-1 overflow-y-auto pb-24">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6">
          {/* Hero */}
          <div className="mb-6">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Projects
            </h1>
            <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
              Building the infrastructure for autonomous, self-improving AI—from multi-agent
              orchestration to safe synthetic data pipelines.
            </p>
          </div>

          {/* Current Projects */}
          <section className="mb-8">
            <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-1.5">
              <Rocket className="w-4 h-4 text-orange-500" />
              Projects
            </h2>
            <div className="space-y-3">
              {projects.map((project) => {
                const Icon = project.icon;
                return (
                  <div
                    key={project.title}
                    className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50 hover:shadow-md transition-shadow group flex flex-col md:flex-row gap-4"
                  >
                    {project.image && (
                      <div className="w-full md:w-32 h-32 md:h-auto rounded-lg overflow-hidden flex-shrink-0 border border-gray-100 dark:border-gray-700">
                        <img
                          src={project.image}
                          alt={project.title}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                        />
                      </div>
                    )}
                    <div className="flex items-start gap-3 flex-1 min-w-0">
                      <div className={`w-8 h-8 rounded-lg bg-${project.color}-100 dark:bg-${project.color}-900/30 flex items-center justify-center flex-shrink-0 mt-1`}>
                        <Icon className={`w-4 h-4 text-${project.color}-600 dark:text-${project.color}-400`} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <h3 className="font-semibold text-gray-900 dark:text-white text-sm">
                            {project.title}
                          </h3>
                          <StatusBadge status={project.status} />
                        </div>
                        <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                          {project.description}
                        </p>

                        {/* Details */}
                        <ul className="mb-2 space-y-0.5">
                          {project.details.map((detail, idx) => (
                            <li key={idx} className="text-xs text-gray-500 dark:text-gray-500 flex items-start gap-1.5">
                              <span className="text-blue-500 mt-0.5">•</span>
                              {detail}
                            </li>
                          ))}
                        </ul>

                        <div className="flex flex-wrap gap-1 mb-2">
                          {project.tags.map((tag) => (
                            <span
                              key={tag}
                              className="px-1.5 py-0.5 text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded"
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
                              className="inline-flex items-center gap-1 text-xs text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
                            >
                              <Github className="w-3 h-3" /> GitHub
                            </a>
                          )}
                          {project.link && (
                            <a
                              href={project.link}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-1 text-xs text-blue-600 dark:text-blue-400 hover:underline"
                            >
                              Learn more <ExternalLink className="w-2.5 h-2.5" />
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
          <section className="mb-6">
            <div className="flex items-center gap-2 mb-4">
              <Sparkles className="w-4 h-4 text-yellow-500" />
              <h2 className="text-base font-semibold text-gray-900 dark:text-white">
                The Future of AI (2026+)
              </h2>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400 mb-4">
              The industry is pivoting from building <strong>oracles</strong> (models that talk)
              to building <strong>partners</strong> (systems that act).
            </p>

            <div className="grid md:grid-cols-2 gap-3">
              {futureVision.map((item) => {
                const Icon = item.icon;
                return (
                  <div
                    key={item.title}
                    className="p-3 rounded-lg border border-gray-200 dark:border-gray-700 bg-gradient-to-br from-gray-50 to-white dark:from-gray-800/50 dark:to-gray-800"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <Icon className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
                      <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                        {item.title}
                      </h3>
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                      {item.description}
                    </p>
                    <div className="p-2 rounded-md bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800">
                      <p className="text-xs text-blue-700 dark:text-blue-300">
                        <strong>Insight:</strong> {item.insight}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          {/* CTA */}
          <section className="p-4 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600 text-white text-center">
            <h3 className="text-base font-semibold mb-1">Interested in Collaboration?</h3>
            <p className="text-xs text-blue-100 mb-3">
              Open to research partnerships, consulting, and investment opportunities.
            </p>
            <a
              href="mailto:zizhaoh@usc.edu"
              className="inline-flex items-center gap-1.5 px-4 py-1.5 bg-white text-blue-600 rounded-md text-sm font-medium hover:bg-blue-50 transition-colors"
            >
              Get in Touch
            </a>
          </section>
        </div>
      </main>
    </div>
  );
};
