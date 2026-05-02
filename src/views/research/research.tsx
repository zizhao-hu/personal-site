'use client';

import { Header } from "@/components/custom/header";
import {
  ArrowUpRight, Brain, Database, Zap,
} from "lucide-react";

const DIRECTIONS = [
  {
    icon: Brain,
    tag: 'memory · context · harness',
    title: 'Agentic Memory · Context · Harness',
    desc: 'Continual learning across all three agentic layers — model memory (KV cache, weights, unlearning), context (skills, instructions), and harness (driver code, tools).',
    accent: 'text-blue-500 dark:text-blue-400',
  },
  {
    icon: Zap,
    tag: 'latency',
    title: 'Low-Latency AI',
    desc: 'Efficient attention architectures, KV-cache compression, and recurrent transformers.',
    accent: 'text-amber-500 dark:text-amber-400',
  },
  {
    icon: Database,
    tag: 'data',
    title: 'Synthetic Data · Behavior Steering',
    desc: 'Generate-validate pipelines, model-collapse prevention, and steering model behavior toward safety and personalization.',
    accent: 'text-emerald-500 dark:text-emerald-400',
  },
];

type Paper = {
  year: number;
  title: string;
  venue: string;
  type: 'conference' | 'preprint' | 'in-progress';
  summary: string;
  link?: string;
  highlight?: boolean;
};

// Newest first.
const PAPERS: Paper[] = [
  {
    year: 2026,
    title: 'SHRED: Document Unlearning via Self-Distillation and Entropy Demotion',
    venue: 'in submission · NeurIPS 2026',
    type: 'in-progress',
    summary:
      'A document-level unlearning method that combines self-distillation on retain data with entropy demotion on the forget set. Removes targeted knowledge from LLMs without catastrophic damage to unrelated capabilities.',
  },
  {
    year: 2026,
    title: 'PRISM: Probe-guided Iterative Smoothness Minimization for Persona Routing',
    venue: 'in progress',
    type: 'in-progress',
    summary:
      'A framework for efficient persona routing in LLMs that enforces dual-space smoothness — across instruction inputs and behavior outputs — via lightweight probes. Targets controllable behavior without full retraining.',
  },
  {
    year: 2026,
    title: 'AttendTwice: Long-Context Inference via Dynamic Token-Level KV-Cache Selection',
    venue: 'in progress',
    type: 'in-progress',
    summary:
      'A two-pass attention scheme that dynamically selects which KV-cache tokens to attend to per query, enabling long-context inference at a fraction of the standard memory footprint.',
  },
  {
    year: 2025,
    title: 'Multimodal Synthetic Data Finetuning and Model Collapse',
    venue: 'ACM ICMI 2025',
    type: 'conference',
    summary:
      'Studies how vision-language models degrade when fine-tuned on AI-generated multimodal data. Characterizes the collapse dynamics specific to the multimodal regime and proposes mitigation strategies that preserve diversity across modalities.',
    link: 'https://scholar.google.com/citations?user=A8J42tQAAAAJ',
    highlight: true,
  },
  {
    year: 2024,
    title: 'Lateralization MLP: A Simple Brain-inspired Architecture for Diffusion',
    venue: 'preprint',
    type: 'preprint',
    summary:
      'A brain-inspired MLP architecture with hemispheric lateralization applied to diffusion models. Shows competitive sample quality at reduced parameter count, suggesting structured asymmetry as an inductive bias for generative modeling.',
    link: 'https://scholar.google.com/citations?user=A8J42tQAAAAJ',
  },
  {
    year: 2024,
    title: 'Static Key Attention in Vision',
    venue: 'preprint',
    type: 'preprint',
    summary:
      'A more efficient attention variant for vision transformers that pre-computes a static key projection, reducing per-token compute while maintaining downstream task performance.',
    link: 'https://scholar.google.com/citations?user=A8J42tQAAAAJ',
  },
];

const SERVICE = [
  'Reviewer · NeurIPS 2024–2025',
  'Reviewer · ICLR 2024–2025',
  'Reviewer · ICML 2024–2025',
  'TA · DSCI 552 (USC)',
];

export const Research = () => {
  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />

      <main className="flex-1 overflow-y-auto pb-24">
        <div className="relative font-mono text-[13px] leading-relaxed text-foreground">
          {/* Subtle dot grid backdrop, same as overview */}
          <div
            aria-hidden
            className="pointer-events-none absolute inset-0 opacity-[0.18] dark:opacity-[0.12]"
            style={{
              backgroundImage: 'radial-gradient(currentColor 1px, transparent 1px)',
              backgroundSize: '22px 22px',
              color: 'hsl(var(--muted-foreground))',
              maskImage: 'linear-gradient(to bottom, black 0%, black 60%, transparent 100%)',
              WebkitMaskImage: 'linear-gradient(to bottom, black 0%, black 60%, transparent 100%)',
            }}
          />

          <div className="relative max-w-3xl mx-auto px-4 sm:px-6 py-10">
            {/* Directions — same pillars as the front page */}
            <Section label="directions">
              <div className="grid sm:grid-cols-3 gap-3">
                {DIRECTIONS.map((d) => (
                  <article
                    key={d.tag}
                    className="border border-border bg-background hover:border-foreground/30 transition-colors p-3"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10.5px] uppercase tracking-wider text-muted-foreground">
                        {d.tag}
                      </span>
                      <d.icon className={`w-3.5 h-3.5 ${d.accent}`} />
                    </div>
                    <h3 className="text-[13px] font-semibold text-foreground mb-1.5">
                      {d.title}
                    </h3>
                    <p className="text-[11.5px] leading-relaxed text-muted-foreground">
                      {d.desc}
                    </p>
                  </article>
                ))}
              </div>
            </Section>

            {/* Recent Papers */}
            <Section label="recent papers">
              <ol className="space-y-3">
                {PAPERS.map((p, i) => (
                  <li
                    key={p.title}
                    className={`relative border border-border bg-background hover:border-foreground/30 transition-colors p-3 ${
                      p.highlight ? 'border-brand-orange/40' : ''
                    }`}
                  >
                    <div className="flex items-baseline gap-2 mb-1.5 flex-wrap">
                      <span className="text-[10.5px] tabular-nums text-muted-foreground/70">
                        {String(i + 1).padStart(2, '0')}
                      </span>
                      <span className="text-[10.5px] uppercase tracking-wider text-muted-foreground">
                        {p.year}
                      </span>
                      <span
                        className={`text-[9.5px] uppercase tracking-wider px-1.5 py-px ${
                          p.type === 'conference'
                            ? 'text-emerald-700 dark:text-emerald-400 border border-emerald-500/30 bg-emerald-500/[0.08]'
                            : p.type === 'in-progress'
                              ? 'text-amber-700 dark:text-amber-400 border border-amber-500/30 bg-amber-500/[0.08]'
                              : 'text-muted-foreground border border-border'
                        }`}
                      >
                        {p.type}
                      </span>
                      {p.highlight && (
                        <span className="text-[9.5px] uppercase tracking-wider px-1.5 py-px text-brand-orange border border-brand-orange/40 bg-brand-orange/[0.06]">
                          featured
                        </span>
                      )}
                      <span className="ml-auto text-[10.5px] text-muted-foreground/80 italic">
                        {p.venue}
                      </span>
                    </div>
                    <h3 className="text-[13.5px] font-semibold text-foreground leading-snug mb-1">
                      {p.title}
                    </h3>
                    <p className="text-[11.5px] leading-relaxed text-muted-foreground">
                      {p.summary}
                    </p>
                    {p.link && (
                      <a
                        href={p.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 mt-2 text-[11px] text-muted-foreground hover:text-foreground underline decoration-dotted underline-offset-2 transition-colors"
                      >
                        view <ArrowUpRight className="w-3 h-3" />
                      </a>
                    )}
                  </li>
                ))}
              </ol>

              <a
                href="https://scholar.google.com/citations?user=A8J42tQAAAAJ"
                target="_blank"
                rel="noopener noreferrer"
                className="group inline-flex items-center gap-1.5 mt-4 px-2.5 py-1 text-[11px] text-muted-foreground border border-border bg-background hover:border-foreground/30 hover:text-foreground transition-colors"
              >
                full list on Google Scholar
                <ArrowUpRight className="w-3 h-3 opacity-70 group-hover:opacity-100" />
              </a>
            </Section>

            {/* Service */}
            <Section label="academic service" lastSection>
              <ul className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                {SERVICE.map((s) => (
                  <li
                    key={s}
                    className="text-[11.5px] text-muted-foreground border border-border bg-background px-2.5 py-1.5"
                  >
                    {s}
                  </li>
                ))}
              </ul>
            </Section>
          </div>
        </div>
      </main>
    </div>
  );
};

const Section = ({
  label,
  children,
  lastSection = false,
}: {
  label: string;
  children: React.ReactNode;
  lastSection?: boolean;
}) => (
  <section className={lastSection ? '' : 'mb-12'}>
    <div className="flex items-center gap-3 mb-4">
      <span className="text-[10.5px] uppercase tracking-[0.18em] text-muted-foreground">
        {label}
      </span>
      <span className="flex-1 h-px bg-border" />
    </div>
    {children}
  </section>
);
