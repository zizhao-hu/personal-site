import { Header } from "@/components/custom/header";
import { Rocket, Bot, Brain, Globe, Cpu, Shield, Sparkles, ExternalLink, Github, Eye, Zap, Network, Database } from "lucide-react";

interface Project {
  title: string;
  description: string;
  details: string[];
  tags: string[];
  status: "active" | "research" | "prototype" | "concept";
  icon: React.ElementType;
  color: string;
  link?: string;
  github?: string;
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
    github: "https://github.com/zizhao-hu",
  },
  {
    title: "Project Orion",
    description: "Industry research project at Handshake AI developing enterprise-grade autonomous agents. Focus on agents that can browse, plan, and execute complex business workflows without constant supervision.",
    details: [
      "Autonomous web navigation and data extraction",
      "Calendar coordination and scheduling automation",
      "Multi-step task completion with error recovery",
      "Enterprise security and compliance integration"
    ],
    tags: ["Agentic AI", "Enterprise", "Handshake AI", "Production"],
    status: "active",
    icon: Bot,
    color: "purple",
  },
  {
    title: "DREAM-C2L: Curriculum Learning Framework",
    description: "Open-source framework for curriculum learning research. Intelligent training curricula that optimize how models learn by dynamically ordering and weighting training samples.",
    details: [
      "Difficulty-aware sample ordering algorithms",
      "Self-paced learning with automatic curriculum generation",
      "Reproducible experiment pipelines for HPC clusters",
      "Integration with PyTorch Lightning and Weights & Biases"
    ],
    tags: ["Curriculum Learning", "PyTorch", "Open Source", "Research"],
    status: "active",
    icon: Brain,
    color: "green",
    github: "https://github.com/zizhao-hu",
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
  },
  {
    title: "Project Canary: Synthetic Data Safety",
    description: "Research on preventing model collapse when training on AI-generated data. Developing methods to detect and filter low-quality synthetic samples that could degrade model performance.",
    details: [
      "Model collapse detection and early warning systems",
      "Synthetic data quality scoring and filtering",
      "Diversity preservation in iterative training",
      "Safe data mixing strategies for foundation models"
    ],
    tags: ["Synthetic Data", "Model Safety", "Data Quality", "Research"],
    status: "research",
    icon: Shield,
    color: "red",
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
  };
  
  const labels = {
    active: "Active",
    research: "Research",
    prototype: "Prototype",
    concept: "Concept",
  };
  
  return (
    <span className={`px-2 py-1 text-xs font-medium rounded ${styles[status]}`}>
      {labels[status]}
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
              Building the infrastructure for autonomous, self-improving AI—from multi-agent 
              orchestration to safe synthetic data pipelines. Here's what I'm working on 
              and where the field is heading.
            </p>
          </div>

          {/* Current Projects */}
          <section className="mb-16">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
              <Rocket className="w-5 h-5 text-orange-500" />
              Projects
            </h2>
            <div className="space-y-6">
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
                        <div className="flex items-center gap-3 mb-2 flex-wrap">
                          <h3 className="font-semibold text-gray-900 dark:text-white text-lg">
                            {project.title}
                          </h3>
                          <StatusBadge status={project.status} />
                        </div>
                        <p className="text-gray-600 dark:text-gray-400 mb-3">
                          {project.description}
                        </p>
                        
                        {/* Details */}
                        <ul className="mb-4 space-y-1">
                          {project.details.map((detail, idx) => (
                            <li key={idx} className="text-sm text-gray-500 dark:text-gray-500 flex items-start gap-2">
                              <span className="text-blue-500 mt-1">•</span>
                              {detail}
                            </li>
                          ))}
                        </ul>
                        
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
              to building <strong>partners</strong> (systems that act). My research directly addresses these trends.
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
