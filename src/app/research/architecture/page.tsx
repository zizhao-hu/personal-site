import type { Metadata } from 'next';
import { ArchitectureResearch } from '@/views/research/architecture';

export const metadata: Metadata = {
  title: 'Neural Architecture Research',
  description: 'Brain-inspired neural architectures, novel attention mechanisms, and efficient model design.',
};

export default function ArchitectureResearchPage() {
  return <ArchitectureResearch />;
}
