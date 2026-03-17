import type { Metadata } from 'next';
import { ContinualLearningResearch } from '@/views/research/continual-learning';

export const metadata: Metadata = {
  title: 'Multi-Agent Systems Research',
  description: 'Research on multi-agent LLM collaboration, orchestration, debate-based verification, and scalable agent workflows.',
};

export default function ContinualLearningResearchPage() {
  return <ContinualLearningResearch />;
}
