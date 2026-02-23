import type { Metadata } from 'next';
import { PipelineDesigner } from '@/views/tools/pipeline-designer';

export const metadata: Metadata = {
  title: 'Pipeline Designer',
  description: 'Visual ML pipeline designer tool for creating training workflows.',
};

export default function PipelineDesignerPage() {
  return <PipelineDesigner />;
}
