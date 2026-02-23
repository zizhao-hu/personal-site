import type { Metadata } from 'next';
import { SyntheticDataResearch } from '@/views/research/synthetic-data';

export const metadata: Metadata = {
  title: 'Synthetic Data Research',
  description: 'Generate-validate loops, self-improving data pipelines, and preventing model collapse in synthetic data.',
};

export default function SyntheticDataResearchPage() {
  return <SyntheticDataResearch />;
}
