import type { Metadata } from 'next';
import { Research } from '@/views/research/research';

export const metadata: Metadata = {
  title: 'Research',
  description: "Zizhao Hu's research in multi-agent systems, synthetic data, brain-inspired architectures, and continual learning.",
};

export default function ResearchPage() {
  return <Research />;
}
