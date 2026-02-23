import type { Metadata } from 'next';
import { Projects } from '@/views/projects/projects';

export const metadata: Metadata = {
  title: 'Projects',
  description: 'Open-source projects and tools by Zizhao Hu — AI systems, multi-agent frameworks, and research infrastructure.',
};

export default function ProjectsPage() {
  return <Projects />;
}
