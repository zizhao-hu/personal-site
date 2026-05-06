'use client';

import { Header } from '@/components/custom/header';
import { useRouter } from 'next/navigation';
import { ArrowUpRight, PenTool, Presentation, Rocket, Gamepad2 } from 'lucide-react';

const TOOLS = [
  {
    id: 'pipeline-designer',
    tag: 'canvas',
    title: 'AI Pipeline Designer',
    desc: 'Interactive canvas for designing AI/ML pipeline diagrams. Drag, connect, export publication-ready figures.',
    path: '/tools/pipeline-designer',
    icon: PenTool,
    accent: 'text-blue-500 dark:text-blue-400',
  },
  {
    id: 'slide-maker',
    tag: 'slides',
    title: 'Research Slide Maker',
    desc: 'Build research presentation slides from a template sequence. Drag, resize, edit, export to Google Slides.',
    path: '/tools/slide-maker',
    icon: Presentation,
    accent: 'text-emerald-500 dark:text-emerald-400',
  },
  {
    id: 'starship-sim',
    tag: 'sim',
    title: 'Starship Lunar Mission',
    desc: 'Ultra-realistic 3D Starship launch and Moon landing. Babylon.js with SpaceX broadcast-style telemetry.',
    path: '/tools/starship-sim',
    icon: Rocket,
    accent: 'text-amber-500 dark:text-amber-400',
  },
  {
    id: 'maplestory',
    tag: 'game',
    title: 'Ludibrium · MapleStory',
    desc: 'Playable HTML5 MapleStory clone — spawns directly in Ludibrium\'s Eos Tower with Stoneperson mobs, jumping, and skill combos. Vendored & modified from liwenone/maplestory.',
    path: '/tools/maplestory',
    icon: Gamepad2,
    accent: 'text-rose-500 dark:text-rose-400',
  },
];

export const Tools = () => {
  const router = useRouter();

  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      <main className="flex-1 overflow-y-auto pb-24">
        <div className="relative font-mono text-[13px] leading-relaxed text-foreground">
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
            <Section label="tools" lastSection>
              <div className="grid sm:grid-cols-3 gap-3">
                {TOOLS.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => router.push(t.path)}
                    className="group text-left border border-border bg-background hover:border-foreground/30 transition-colors p-3"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10.5px] uppercase tracking-wider text-muted-foreground">
                        {t.tag}
                      </span>
                      <t.icon className={`w-3.5 h-3.5 ${t.accent}`} />
                    </div>
                    <h3 className="text-[13px] font-semibold text-foreground mb-1.5 flex items-center gap-1">
                      {t.title}
                      <ArrowUpRight className="w-3 h-3 opacity-0 group-hover:opacity-60 transition-opacity" />
                    </h3>
                    <p className="text-[11.5px] leading-relaxed text-muted-foreground">
                      {t.desc}
                    </p>
                  </button>
                ))}
              </div>
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
