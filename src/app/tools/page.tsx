import type { Metadata } from 'next';
import { Tools } from '@/views/tools/tools';

export const metadata: Metadata = {
  title: 'Tools',
  description: 'Interactive AI tools and demonstrations built by Zizhao Hu.',
};

export default function ToolsPage() {
  return <Tools />;
}
