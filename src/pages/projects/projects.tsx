import { Header } from "@/components/custom/header";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Bot, Brain, Globe, Cpu, Sparkles, ExternalLink, Github, LayoutGrid, List } from "lucide-react";
import { tagPillClass, tagBadgeClass } from "@/lib/tag-colors";
import { projects } from "@/data/projects";

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

const statusConfig: Record<string, { label: string; class: string }> = {
  active: { label: "Active", class: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400" },
  research: { label: "Research", class: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400" },
  prototype: { label: "Prototype", class: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400" },
  concept: { label: "Concept", class: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400" },
  completed: { label: "Completed", class: "bg-gray-100 text-gray-700 dark:bg-gray-700/50 dark:text-gray-300" },
};

const statusFilters = [
  { id: "all", label: "All Projects" },
  { id: "active", label: "Active" },
  { id: "research", label: "Research" },
  { id: "prototype", label: "Prototype" },
  { id: "completed", label: "Completed" },
];

const allTags = [...new Set(projects.flatMap(p => p.tags))];
const tagCounts = allTags.reduce((acc, tag) => {
  acc[tag] = projects.filter(p => p.tags.includes(tag)).length;
  return acc;
}, {} as Record<string, number>);

export const Projects = () => {
  const [activeStatus, setActiveStatus] = useState<string>("all");
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<"list" | "grid">("list");
  const navigate = useNavigate();

  const filteredProjects = projects.filter((project) => {
    const statusMatch = activeStatus === "all" || project.status === activeStatus;
    const tagMatch = selectedTags.length === 0 || selectedTags.some(t => project.tags.includes(t));
    return statusMatch && tagMatch;
  });

  const toggleTag = (tag: string) => {
    setSelectedTags(prev => prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]);
  };

  const handleProjectClick = (slug: string) => {
    navigate(`/projects/${slug}`);
  };

  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      <main className="flex-1 overflow-y-auto pb-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
          {/* Hero */}
          <div className="mb-6">
            <h1 className="text-2xl font-bold font-heading text-foreground mb-1">
              Projects
            </h1>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Building the infrastructure for autonomous, self-improving AI—from multi-agent
              orchestration to safe synthetic data pipelines.
            </p>
          </div>

          {/* Main Layout: Sidebar + Content */}
          <div className="flex gap-6">

            {/* ── Left Sidebar ── */}
            <aside className="hidden lg:block w-56 flex-shrink-0">
              <div className="sticky top-6 space-y-6">

                {/* Status Filter */}
                <div>
                  <h3 className="text-[10px] uppercase tracking-wider font-heading text-muted-foreground mb-2 px-2">
                    Status
                  </h3>
                  <nav className="space-y-0.5">
                    {statusFilters.map((filter) => {
                      const count = filter.id === "all"
                        ? projects.length
                        : projects.filter(p => p.status === filter.id).length;
                      return (
                        <button
                          key={filter.id}
                          onClick={() => setActiveStatus(filter.id)}
                          className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs transition-all duration-150 ${activeStatus === filter.id
                            ? "bg-brand-orange/10 text-brand-orange font-medium"
                            : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                            }`}
                        >
                          <span className="font-heading">{filter.label}</span>
                          <span className={`text-[10px] tabular-nums ${activeStatus === filter.id ? "text-brand-orange" : "text-muted-foreground/60"
                            }`}>
                            {count}
                          </span>
                        </button>
                      );
                    })}
                  </nav>
                </div>

                <div className="h-px bg-border" />

                {/* Tags */}
                <div>
                  <div className="flex items-center justify-between mb-2 px-2">
                    <h3 className="text-[10px] uppercase tracking-wider font-heading text-muted-foreground">
                      Tags
                    </h3>
                    {selectedTags.length > 0 && (
                      <button onClick={() => setSelectedTags([])} className="text-[10px] text-brand-orange hover:underline font-heading">
                        Clear
                      </button>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-1.5 px-1">
                    {allTags.map((tag) => (
                      <button
                        key={tag}
                        onClick={() => toggleTag(tag)}
                        className={`px-2 py-0.5 text-[10px] rounded-full transition-all duration-150 font-heading ${tagPillClass(tag, selectedTags.includes(tag))}`}
                      >
                        {tag}
                        <span className="ml-1 opacity-60">{tagCounts[tag]}</span>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="h-px bg-border" />

                {/* Stats */}
                <div className="px-2">
                  <h3 className="text-[10px] uppercase tracking-wider font-heading text-muted-foreground mb-2">
                    Stats
                  </h3>
                  <div className="space-y-1.5 text-xs text-muted-foreground">
                    <div className="flex justify-between">
                      <span>Total projects</span>
                      <span className="font-medium text-foreground">{projects.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Active</span>
                      <span className="font-medium text-foreground">{projects.filter(p => p.status === "active").length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Tags</span>
                      <span className="font-medium text-foreground">{allTags.length}</span>
                    </div>
                  </div>
                </div>
              </div>
            </aside>

            {/* ── Main Content ── */}
            <div className="flex-1 min-w-0">

              {/* Mobile tabs + view toggle */}
              <div className="flex items-center justify-between mb-4">
                <div className="lg:hidden flex gap-0 border-b border-border overflow-x-auto">
                  {statusFilters.map((filter) => (
                    <button
                      key={filter.id}
                      onClick={() => setActiveStatus(filter.id)}
                      className={`relative px-3 py-2 text-xs font-heading whitespace-nowrap transition-colors ${activeStatus === filter.id
                        ? "text-brand-orange"
                        : "text-muted-foreground hover:text-foreground"
                        }`}
                    >
                      {filter.label}
                      {activeStatus === filter.id && (
                        <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-brand-orange" />
                      )}
                    </button>
                  ))}
                </div>

                <div className="flex items-center gap-0.5 bg-muted rounded-lg p-0.5">
                  <button
                    onClick={() => setViewMode("list")}
                    className={`p-1.5 rounded-md transition-colors ${viewMode === "list" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"
                      }`}
                  >
                    <List className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => setViewMode("grid")}
                    className={`p-1.5 rounded-md transition-colors ${viewMode === "grid" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"
                      }`}
                  >
                    <LayoutGrid className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Active filters */}
              {(activeStatus !== "all" || selectedTags.length > 0) && (
                <div className="mb-3 flex items-center gap-2 text-xs text-muted-foreground flex-wrap">
                  <span>Showing {filteredProjects.length} of {projects.length}</span>
                  {selectedTags.map(tag => (
                    <span key={tag} className="px-1.5 py-0.5 bg-brand-orange/10 text-brand-orange rounded text-[10px] font-heading">
                      {tag}
                    </span>
                  ))}
                </div>
              )}

              {/* Project Cards */}
              <div className={viewMode === "grid" ? "grid grid-cols-1 sm:grid-cols-2 gap-4" : "flex flex-col gap-3"}>
                {filteredProjects.map((project) => {
                  const Icon = project.icon;
                  const status = statusConfig[project.status];
                  return viewMode === "list" ? (
                    <div
                      key={project.title}
                      onClick={() => handleProjectClick(project.slug)}
                      className="group bg-card border border-border rounded-xl p-4 hover:shadow-elevation-2 dark:hover:shadow-elevation-2-dark hover:border-brand-orange/20 transition-all flex flex-col md:flex-row gap-4 cursor-pointer"
                    >
                      {project.image && (
                        <div className="w-full md:w-36 h-32 md:h-auto rounded-lg overflow-hidden flex-shrink-0 border border-border">
                          <img src={project.image} alt={project.title} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
                        </div>
                      )}
                      <div className="flex items-start gap-3 flex-1 min-w-0">
                        <div className="w-8 h-8 rounded-lg bg-muted flex items-center justify-center flex-shrink-0 mt-1">
                          <Icon className="w-4 h-4 text-brand-orange" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1 flex-wrap">
                            <h3 className="font-semibold font-heading text-foreground text-sm group-hover:text-brand-orange transition-colors">
                              {project.title}
                            </h3>
                            <span className={`px-1.5 py-0.5 text-[10px] font-medium rounded ${status.class}`}>
                              {status.label}
                            </span>
                          </div>
                          <p className="text-xs text-muted-foreground mb-2">{project.description}</p>
                          <ul className="mb-2 space-y-0.5">
                            {project.details.map((detail, idx) => (
                              <li key={idx} className="text-xs text-muted-foreground/80 flex items-start gap-1.5">
                                <span className="text-brand-orange mt-0.5">•</span>
                                {detail}
                              </li>
                            ))}
                          </ul>
                          <div className="flex flex-wrap gap-1 mb-2">
                            {project.tags.map((tag) => (
                              <span key={tag} className={`px-1.5 py-0.5 text-[10px] rounded font-heading ${tagBadgeClass(tag)}`}>
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
                                className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                                onClick={(e) => e.stopPropagation()}
                              >
                                <Github className="w-3 h-3" /> GitHub
                              </a>
                            )}
                            {project.link && (
                              <a
                                href={project.link}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 text-xs text-brand-orange hover:underline"
                                onClick={(e) => e.stopPropagation()}
                              >
                                Learn more <ExternalLink className="w-2.5 h-2.5" />
                              </a>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div
                      key={project.title}
                      onClick={() => handleProjectClick(project.slug)}
                      className="group bg-card border border-border rounded-xl p-4 hover:shadow-elevation-2 dark:hover:shadow-elevation-2-dark hover:border-brand-orange/20 transition-all flex flex-col cursor-pointer"
                    >
                      {project.image && (
                        <div className="w-full h-32 rounded-lg overflow-hidden border border-border mb-3">
                          <img src={project.image} alt={project.title} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
                        </div>
                      )}
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-6 h-6 rounded bg-muted flex items-center justify-center">
                          <Icon className="w-3 h-3 text-brand-orange" />
                        </div>
                        <span className={`px-1.5 py-0.5 text-[10px] font-medium rounded ${status.class}`}>
                          {status.label}
                        </span>
                      </div>
                      <h3 className="font-semibold font-heading text-foreground text-sm mb-1 group-hover:text-brand-orange transition-colors">{project.title}</h3>
                      <p className="text-xs text-muted-foreground mb-3 line-clamp-2 flex-1">{project.description}</p>
                      <div className="flex flex-wrap gap-1">
                        {project.tags.slice(0, 3).map((tag) => (
                          <span key={tag} className={`px-1.5 py-0.5 text-[10px] rounded font-heading ${tagBadgeClass(tag)}`}>
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>

              {filteredProjects.length === 0 && (
                <div className="text-center py-16">
                  <p className="text-muted-foreground text-sm mb-2">No projects found matching your filters.</p>
                  <button
                    onClick={() => { setActiveStatus("all"); setSelectedTags([]); }}
                    className="text-brand-orange hover:underline text-xs font-heading"
                  >
                    Clear all filters
                  </button>
                </div>
              )}

              {/* Future of AI Vision */}
              <section className="mt-8">
                <div className="flex items-center gap-2 mb-4">
                  <Sparkles className="w-4 h-4 text-brand-orange" />
                  <h2 className="text-base font-semibold font-heading text-foreground">
                    The Future of AI (2026+)
                  </h2>
                </div>
                <p className="text-xs text-muted-foreground mb-4">
                  The industry is pivoting from building <strong>oracles</strong> (models that talk)
                  to building <strong>partners</strong> (systems that act).
                </p>

                <div className="grid md:grid-cols-2 gap-3">
                  {futureVision.map((item) => {
                    const VIcon = item.icon;
                    return (
                      <div
                        key={item.title}
                        className="p-3 rounded-xl border border-border bg-card hover:shadow-elevation-1 dark:hover:shadow-elevation-1-dark transition-all"
                      >
                        <div className="flex items-center gap-2 mb-2">
                          <VIcon className="w-3.5 h-3.5 text-brand-orange" />
                          <h3 className="text-sm font-semibold font-heading text-foreground">
                            {item.title}
                          </h3>
                        </div>
                        <p className="text-xs text-muted-foreground mb-2">
                          {item.description}
                        </p>
                        <div className="p-2 rounded-lg bg-brand-orange/5 border border-brand-orange/10">
                          <p className="text-xs text-brand-orange">
                            <strong>Insight:</strong> {item.insight}
                          </p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>

              {/* CTA */}
              <section className="mt-6 p-4 rounded-xl bg-gradient-to-r from-brand-orange to-brand-clay text-white text-center">
                <h3 className="text-base font-semibold font-heading mb-1">Interested in Collaboration?</h3>
                <p className="text-xs text-white/80 mb-3">
                  Open to research partnerships, consulting, and investment opportunities.
                </p>
                <a
                  href="mailto:zizhaoh@usc.edu"
                  className="inline-flex items-center gap-1.5 px-4 py-1.5 bg-white text-brand-orange rounded-lg text-sm font-medium font-heading hover:bg-white/90 transition-colors"
                >
                  Get in Touch
                </a>
              </section>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};
