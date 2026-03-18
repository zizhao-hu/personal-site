'use client';

import { Linkedin, Github, Mail, BookOpen, Sparkles, Brain, Rocket, Database } from 'lucide-react';

export const Overview = () => {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 bg-background">
      {/* Hero Section */}
      <div className="flex flex-col md:flex-row items-center md:items-start gap-6 mb-8 mt-2">
        {/* Profile Picture */}
        <div className="relative group">
          <div className="absolute -inset-1 bg-gradient-to-r from-brand-orange to-brand-blue rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200"></div>
          <div className="relative w-32 h-32 rounded-2xl overflow-hidden border border-white/10 dark:border-black/50 shadow-2xl flex-shrink-0 bg-background">
            <img
              src="/images/profile.jpg"
              alt="Zizhao Hu"
              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                const parent = target.parentElement;
                if (parent) {
                  parent.innerHTML = `
                    <div class="w-full h-full bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white text-3xl font-bold">
                      ZH
                    </div>
                  `;
                }
              }}
            />
          </div>
        </div>

        {/* Main Info */}
        <div className="flex-1 text-center md:text-left">
          <div className="flex flex-col md:flex-row md:items-center gap-2 md:gap-4 mb-2">
            <h1 className="text-3xl font-bold tracking-tight text-foreground font-heading">
              Zizhao Hu
            </h1>
            <div className="flex items-center justify-center md:justify-start gap-1.5">
              <span className="flex h-2 w-2 rounded-full bg-green-500"></span>
              <span className="text-[10px] font-medium uppercase tracking-wider text-green-600 dark:text-green-400">Available for Collaboration</span>
            </div>
          </div>

          <p className="text-lg text-brand-orange font-medium mb-3 font-heading">
            AI Researcher · PhD Student at USC · GLAMOUR Lab · MINDS · MOVE Fellow
          </p>

          <blockquote className="relative pl-4 border-l-2 border-brand-orange/40 mb-4 max-w-2xl">
            <p className="text-sm italic text-gray-600 dark:text-gray-400 leading-relaxed">
              &ldquo;Understanding how LLMs remember and forget&mdash;then using that knowledge to build faster, leaner inference and better training data.&rdquo;
            </p>
            <cite className="block mt-1 text-[11px] not-italic text-gray-500 dark:text-gray-500">
              &mdash; Zizhao Hu, PhD Student at USC &middot; GLAMOUR Lab &amp; MINDS Group
            </cite>
          </blockquote>

        </div>
      </div>

      {/* What I Do */}
      <div className="mb-6">
        <div className="grid md:grid-cols-3 gap-4">
          {/* Pillar 1 */}
          <div className="group relative p-4 rounded-xl border border-border bg-card hover:border-brand-orange/30 transition-all duration-300">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-7 h-7 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                <Brain className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="text-sm font-semibold text-foreground font-heading">LLM Memorization</h3>
            </div>
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              How LLMs remember &amp; forget — unlearning, KV-cache management, continual learning, and reasoning under memory constraints
            </p>
          </div>

          {/* Pillar 2 */}
          <div className="group relative p-4 rounded-xl border border-border bg-card hover:border-brand-orange/30 transition-all duration-300">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-7 h-7 rounded-lg bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center">
                <Sparkles className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
              </div>
              <h3 className="text-sm font-semibold text-foreground font-heading">Inference Optimization</h3>
            </div>
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              Efficient attention, KV-cache compression, sparse &amp; low-rank methods for faster, leaner LLM serving at scale
            </p>
          </div>

          {/* Pillar 3 */}
          <div className="group relative p-4 rounded-xl border border-border bg-card hover:border-brand-orange/30 transition-all duration-300">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-7 h-7 rounded-lg bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center">
                <Database className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
              </div>
              <h3 className="text-sm font-semibold text-foreground font-heading">Synthetic Data Curation</h3>
            </div>
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              Generate-validate pipelines, quality filtering, model-collapse prevention, and safety-oriented data curation for LLM training
            </p>
          </div>
        </div>
      </div>

      {/* My Vision */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-brand-orange/20 to-brand-blue/20 flex items-center justify-center flex-shrink-0">
            <Rocket className="w-3.5 h-3.5 text-brand-orange" />
          </div>
          <h3 className="text-sm font-semibold text-foreground font-heading">My Vision</h3>
        </div>

        {/* Three-column comparison: Human | Stage | AI */}
        <div className="rounded-xl border border-border overflow-hidden mb-3">
          {/* Header row */}
          <div className="grid grid-cols-[1fr_auto_1fr]">
            <div className="px-2 py-1.5 bg-amber-50 dark:bg-amber-900/15 text-center border-b border-r border-border">
              <span className="text-[10px] font-semibold text-amber-700 dark:text-amber-300">🧠 Human</span>
            </div>
            <div className="px-2 py-1.5 bg-gray-50 dark:bg-gray-800/50 text-center border-b border-r border-border">
              <span className="text-[10px] font-semibold text-muted-foreground">Stage</span>
            </div>
            <div className="px-2 py-1.5 bg-blue-50 dark:bg-blue-900/15 text-center border-b border-border">
              <span className="text-[10px] font-semibold text-blue-700 dark:text-blue-300">🤖 AI</span>
            </div>
          </div>

          {/* Row 1: Origin */}
          <div className="grid grid-cols-[1fr_auto_1fr] border-b border-border">
            <div className="px-2 py-2 bg-amber-50/50 dark:bg-amber-900/5 border-r border-border min-w-0">
              <p className="text-[10px] text-muted-foreground leading-snug">Natural selection shapes the newborn brain&apos;s topology</p>
            </div>
            <div className="px-2 py-2 bg-gray-50/50 dark:bg-gray-800/30 border-r border-border flex items-center">
              <span className="text-[10px] font-bold text-foreground whitespace-nowrap">Origin</span>
            </div>
            <div className="px-2 py-2 bg-blue-50/50 dark:bg-blue-900/5 min-w-0">
              <p className="text-[10px] text-muted-foreground leading-snug">Architecture &amp; pretraining shape the base model&apos;s wiring</p>
            </div>
          </div>

          {/* Row 2: Early Learning */}
          <div className="grid grid-cols-[1fr_auto_1fr] border-b border-border">
            <div className="px-2 py-2 bg-amber-50/50 dark:bg-amber-900/5 border-r border-border min-w-0">
              <p className="text-[10px] text-muted-foreground leading-snug">Guided experience forms task-based memory</p>
            </div>
            <div className="px-2 py-2 bg-gray-50/50 dark:bg-gray-800/30 border-r border-border flex items-center">
              <span className="text-[10px] font-bold text-foreground whitespace-nowrap">Learning</span>
            </div>
            <div className="px-2 py-2 bg-blue-50/50 dark:bg-blue-900/5 min-w-0">
              <p className="text-[10px] text-muted-foreground leading-snug">SFT builds task-specific skills through curated instruction</p>
            </div>
          </div>

          {/* Row 3: Recall */}
          <div className="grid grid-cols-[1fr_auto_1fr] border-b border-border">
            <div className="px-2 py-2 bg-amber-50/50 dark:bg-amber-900/5 border-r border-border min-w-0">
              <p className="text-[10px] text-muted-foreground leading-snug">Brain rewires recall pathways with practice</p>
            </div>
            <div className="px-2 py-2 bg-gray-50/50 dark:bg-gray-800/30 border-r border-border flex items-center">
              <span className="text-[10px] font-bold text-foreground whitespace-nowrap">Recall</span>
            </div>
            <div className="px-2 py-2 bg-blue-50/50 dark:bg-blue-900/5 min-w-0">
              <p className="text-[10px] text-muted-foreground leading-snug">KV management controls which memories stay active</p>
            </div>
          </div>

          {/* Row 4: Tools */}
          <div className="grid grid-cols-[1fr_auto_1fr] border-b border-border">
            <div className="px-2 py-2 bg-amber-50/50 dark:bg-amber-900/5 border-r border-border min-w-0">
              <p className="text-[10px] text-muted-foreground leading-snug">Build tools — books, calculators — to extend cognition</p>
            </div>
            <div className="px-2 py-2 bg-gray-50/50 dark:bg-gray-800/30 border-r border-border flex items-center">
              <span className="text-[10px] font-bold text-foreground whitespace-nowrap">Tools</span>
            </div>
            <div className="px-2 py-2 bg-blue-50/50 dark:bg-blue-900/5 min-w-0">
              <p className="text-[10px] text-muted-foreground leading-snug">RAG &amp; tool use augment models with external knowledge</p>
            </div>
          </div>

          {/* Row 5: Evolution */}
          <div className="grid grid-cols-[1fr_auto_1fr]">
            <div className="px-2 py-2 bg-amber-50/50 dark:bg-amber-900/5 border-r border-border min-w-0">
              <p className="text-[10px] text-muted-foreground leading-snug">Trial-and-error drives individual &amp; societal evolution</p>
            </div>
            <div className="px-2 py-2 bg-gray-50/50 dark:bg-gray-800/30 border-r border-border flex items-center">
              <span className="text-[10px] font-bold text-foreground whitespace-nowrap">Evolve</span>
            </div>
            <div className="px-2 py-2 bg-blue-50/50 dark:bg-blue-900/5 min-w-0">
              <p className="text-[10px] text-muted-foreground leading-snug">Diversity + verification: generate, verify, and improve</p>
            </div>
          </div>
        </div>

        {/* Closing statement */}
        <p className="text-[11px] text-muted-foreground leading-relaxed text-center italic">
          Specialized AI will develop distinct memory profiles — just as human experts develop domain intuition. <span className="text-foreground font-medium not-italic">Diversity with verification</span> is how both human societies and AI systems evolve.
        </p>
      </div>
      <div className="flex items-center justify-center gap-2 flex-wrap">
        <a
          href="https://scholar.google.com/citations?user=A8J42tQAAAAJ"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-brand-orange hover:opacity-90 text-white transition-all text-xs font-medium font-heading"
        >
          <BookOpen className="w-3.5 h-3.5" />
          Google Scholar
        </a>
        <a
          href="https://linkedin.com/in/zizhao-hu"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 transition-colors text-xs font-medium"
        >
          <Linkedin className="w-3.5 h-3.5" />
          LinkedIn
        </a>
        <a
          href="https://github.com/zizhao-hu"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 transition-colors text-xs font-medium"
        >
          <Github className="w-3.5 h-3.5" />
          GitHub
        </a>
        <a
          href="mailto:zizhaoh@usc.edu"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 transition-colors text-xs font-medium"
        >
          <Mail className="w-3.5 h-3.5" />
          Contact
        </a>
      </div>
    </div>
  );
};
