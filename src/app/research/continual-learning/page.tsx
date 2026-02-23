import type { Metadata } from 'next';
import { ContinualLearningResearch } from '@/views/research/continual-learning';

export const metadata: Metadata = {
  title: 'Continual Learning Research',
  description: 'Research on continual learning without catastrophic forgetting — lifelong model training and knowledge retention.',
};

export default function ContinualLearningResearchPage() {
  return <ContinualLearningResearch />;
}
