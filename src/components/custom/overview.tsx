'use client';

import {
  Linkedin, Github, Mail, BookOpen,
  Brain, Zap, Database,
  ArrowUpRight,
  Atom, Bird, Bot, Layers, Wand2, Network, RefreshCw, Telescope,
} from 'lucide-react';
import {
  GeorgiaTechLogo, USCLogo, GoogleCloudLogo, ScaleAILogo, HandshakeAILogo,
  MetaLogo, OpenAILogo,
} from './brand-logos';

const PILLARS = [
  {
    icon: Brain,
    tag: 'memory',
    title: 'Memorization · Continual Learning',
    desc: 'How LLMs remember & forget — unlearning, KV-cache management, and continual adaptation under memory constraints.',
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

type PathStage = {
  tag: string;
  icon: typeof Atom;
  title: string;
  context: string;
  accent: string;
  current?: boolean;
};

const PATH: PathStage[] = [
  {
    tag: 'physics',
    icon: Atom,
    title: 'Physics',
    context: 'photonics & metasurface design · dynamic systems',
    accent: 'text-blue-500 dark:text-blue-400',
  },
  {
    tag: 'biophysics',
    icon: Bird,
    title: 'Agile Systems · Animal Flight',
    context: 'bio-inspired flight, sensing, and locomotion',
    accent: 'text-emerald-500 dark:text-emerald-400',
  },
  {
    tag: 'robotics',
    icon: Bot,
    title: 'Robotics · Reinforcement Learning',
    context: 'policy learning for physical control and agent behavior',
    accent: 'text-yellow-500 dark:text-yellow-400',
  },
  {
    tag: 'vae',
    icon: Layers,
    title: 'Continual Learning · VAE',
    context: 'regularization design for variational autoencoders',
    accent: 'text-amber-500 dark:text-amber-400',
  },
  {
    tag: 'multimodal',
    icon: Wand2,
    title: 'Multimodal Generation',
    context: 'diffusion models · vision-language model architecture',
    accent: 'text-violet-500 dark:text-violet-400',
  },
  {
    tag: 'agents',
    icon: Network,
    title: 'Multi-Agent Systems',
    context: 'coordination, division of labor, mutual verification',
    accent: 'text-rose-500 dark:text-rose-400',
    current: true,
  },
  {
    tag: 'continual_v2',
    icon: RefreshCw,
    title: 'Continual Learning · v2',
    context: 'multi-agent · in-context learning · unlearning',
    accent: 'text-cyan-500 dark:text-cyan-400',
    current: true,
  },
  {
    tag: 'horizon',
    icon: Telescope,
    title: 'World Models · Low-Latency AI',
    context: 'predictive world models and the architectures to serve them in real time',
    accent: 'text-brand-orange',
  },
];

const LINKS = [
  { href: 'https://scholar.google.com/citations?user=A8J42tQAAAAJ', label: 'scholar', icon: BookOpen },
  { href: 'https://linkedin.com/in/zizhao-hu', label: 'linkedin', icon: Linkedin },
  { href: 'https://github.com/zizhao-hu', label: 'github', icon: Github },
  { href: 'mailto:zizhaohu3@gmail.com', label: 'email', icon: Mail },
];

type Collaborator = {
  name: string;
  role: string;
  href?: string;
  Logo: (props: { className?: string }) => JSX.Element;
};

const COLLABORATORS: Collaborator[] = [
  {
    name: 'Georgia Tech',
    role: 'BS · alumni',
    href: 'https://www.gatech.edu/',
    Logo: GeorgiaTechLogo,
  },
  {
    name: 'USC',
    role: 'MS + PhD · current',
    href: 'https://www.usc.edu/',
    Logo: USCLogo,
  },
  {
    name: 'Handshake AI',
    role: 'MOVE Fellow',
    href: 'https://joinhandshake.com/',
    Logo: HandshakeAILogo,
  },
  {
    name: 'OpenAI',
    role: 'client',
    href: 'https://openai.com/',
    Logo: OpenAILogo,
  },
  {
    name: 'Meta',
    role: 'client',
    href: 'https://about.meta.com/',
    Logo: MetaLogo,
  },
  {
    name: 'Google Cloud',
    role: 'client',
    href: 'https://cloud.google.com/',
    Logo: GoogleCloudLogo,
  },
  {
    name: 'Scale AI',
    role: 'client',
    href: 'https://scale.com/',
    Logo: ScaleAILogo,
  },
];

export const Overview = () => {
  return (
    <div className="relative font-mono text-[13px] leading-relaxed text-foreground">
      {/* Subtle dot grid backdrop */}
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
        {/* Hero */}
        <header className="mb-8 flex items-start gap-4">
          <div className="relative flex-shrink-0">
            <div className="w-16 h-16 sm:w-20 sm:h-20 overflow-hidden border border-border bg-muted/40">
              <img
                src="/images/profile.jpg"
                alt="Zizhao Hu"
                className="w-full h-full object-cover"
                onError={(e) => {
                  const t = e.target as HTMLImageElement;
                  t.style.display = 'none';
                  const p = t.parentElement;
                  if (p) p.innerHTML = `<div class="w-full h-full flex items-center justify-center text-foreground font-mono font-bold text-sm">ZH</div>`;
                }}
              />
            </div>
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 flex-wrap mb-1">
              <h1 className="text-2xl sm:text-[28px] font-semibold tracking-tight text-foreground leading-tight">
                Zizhao Hu
              </h1>
              <span className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-emerald-700 dark:text-emerald-400">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                available
              </span>
            </div>
            <p className="text-[12.5px] text-muted-foreground">
              CS PhD <span className="text-foreground">@</span> USC ·{' '}
              <a
                href="https://glamor-usc.github.io/"
                target="_blank"
                rel="noopener noreferrer"
                className="underline decoration-dotted underline-offset-2 text-foreground hover:text-brand-orange transition-colors"
              >
                GLAMOR Lab
              </a>{' '}
              · advised by{' '}
              <a
                href="https://jessethomason.com/"
                target="_blank"
                rel="noopener noreferrer"
                className="underline decoration-dotted underline-offset-2 text-foreground hover:text-brand-orange transition-colors"
              >
                Jesse Thomason
              </a>
            </p>

            {/* Inline personal links */}
            <div className="flex flex-wrap gap-1.5 mt-2.5">
              {LINKS.map((l) => (
                <a
                  key={l.label}
                  href={l.href}
                  target={l.href.startsWith('mailto:') ? undefined : '_blank'}
                  rel={l.href.startsWith('mailto:') ? undefined : 'noopener noreferrer'}
                  className="group inline-flex items-center gap-1.5 px-2 py-0.5 text-[11px] text-muted-foreground border border-border bg-background hover:border-foreground/30 hover:text-foreground transition-colors"
                >
                  <l.icon className="w-3 h-3 opacity-70 group-hover:opacity-100" />
                  <span>{l.label}</span>
                </a>
              ))}
            </div>
          </div>
        </header>

        {/* Tagline */}
        <p className="mb-4 text-[14px] text-foreground/85 leading-relaxed max-w-2xl">
          <span className="text-foreground font-semibold">Context is the new weight.</span>{' '}
          Low-latency control of what to{' '}
          <span className="text-brand-orange font-semibold">1.</span> remember,{' '}
          <span className="text-brand-orange font-semibold">2.</span> forget, and{' '}
          <span className="text-brand-orange font-semibold">3.</span> explore from the real world
          decides the next generation of artificial intelligence.
        </p>

        {/* Keyword tags */}
        <div className="flex flex-wrap gap-1.5 mb-12">
          {[
            'continual learning',
            'multiagent',
            'multimodal agent',
            'model unlearning',
            'safety alignment',
            'post-training',
            'KV cache',
            'synthetic data',
          ].map((kw) => (
            <span
              key={kw}
              className="inline-flex items-center px-2 py-0.5 text-[10.5px] text-muted-foreground border border-border bg-background hover:border-foreground/30 hover:text-foreground transition-colors"
            >
              {kw}
            </span>
          ))}
        </div>

        {/* What I work on */}
        <Section label="what i work on">
          <div className="grid sm:grid-cols-3 gap-3">
            {PILLARS.map((p) => (
              <article
                key={p.tag}
                className="border border-border bg-background hover:border-foreground/30 transition-colors p-3"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10.5px] uppercase tracking-wider text-muted-foreground">
                    {p.tag}
                  </span>
                  <p.icon className={`w-3.5 h-3.5 ${p.accent}`} />
                </div>
                <h3 className="text-[13px] font-semibold text-foreground mb-1.5">
                  {p.title}
                </h3>
                <p className="text-[11.5px] leading-relaxed text-muted-foreground">
                  {p.desc}
                </p>
              </article>
            ))}
          </div>
        </Section>

        {/* My path */}
        <Section label="my path">
          <ol className="relative">
            {PATH.map((p, i) => {
              const Icon = p.icon;
              const isLast = i === PATH.length - 1;
              return (
                <li key={p.tag} className="relative pl-12 pb-4 last:pb-0">
                  {!isLast && (
                    <span className="absolute left-[15px] top-8 bottom-0 w-px bg-border" />
                  )}
                  <div
                    className={`absolute left-0 top-0 w-8 h-8 flex items-center justify-center border ${
                      p.current
                        ? 'border-brand-orange bg-brand-orange/10'
                        : 'border-border bg-background'
                    }`}
                  >
                    <Icon className={`w-4 h-4 ${p.accent}`} />
                  </div>

                  <div className="flex items-baseline gap-2 flex-wrap min-h-[20px]">
                    <span className="text-[10.5px] uppercase tracking-wider text-muted-foreground">
                      {p.tag}
                    </span>
                    {p.current && (
                      <span className="text-[9.5px] font-semibold tracking-wider uppercase inline-flex items-center gap-1 px-1.5 py-px text-emerald-700 dark:text-emerald-400 border border-emerald-500/30 bg-emerald-500/[0.08]">
                        <span className="w-1 h-1 rounded-full bg-emerald-500 animate-pulse" />
                        current
                      </span>
                    )}
                    <span className="ml-auto text-[10px] text-muted-foreground/60 tabular-nums">
                      {String(i + 1).padStart(2, '0')}
                    </span>
                  </div>

                  <h3 className="text-[13.5px] font-semibold text-foreground mt-1 leading-snug">
                    {p.title}
                  </h3>
                  <p className="text-[11.5px] text-muted-foreground mt-0.5 leading-relaxed">
                    {p.context}
                  </p>
                </li>
              );
            })}
          </ol>
        </Section>

        {/* Collaborators */}
        <Section label="collaborators" lastSection>
          <ul className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {COLLABORATORS.map((c) => {
              const Logo = c.Logo;
              const inner = (
                <div className="flex items-center gap-2.5 border border-border bg-background hover:border-foreground/30 transition-colors p-2.5 h-full">
                  <Logo className="w-9 h-9 flex-shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="text-[12.5px] font-semibold text-foreground leading-tight truncate flex items-center gap-1">
                      {c.name}
                      {c.href && (
                        <ArrowUpRight className="w-3 h-3 opacity-0 group-hover:opacity-60 transition-opacity flex-shrink-0" />
                      )}
                    </div>
                    <div className="text-[11px] text-muted-foreground mt-0.5 truncate">
                      {c.role}
                    </div>
                  </div>
                </div>
              );
              return (
                <li key={c.name} className="group">
                  {c.href ? (
                    <a
                      href={c.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block"
                    >
                      {inner}
                    </a>
                  ) : (
                    inner
                  )}
                </li>
              );
            })}
          </ul>
        </Section>

      </div>
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

