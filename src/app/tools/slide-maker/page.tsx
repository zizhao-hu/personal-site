import type { Metadata } from 'next';
import { SlideMaker } from '@/views/tools/slide-maker';

export const metadata: Metadata = {
  title: 'Slide Maker',
  description: 'AI-powered presentation slide generator for research and technical talks.',
};

export default function SlideMakerPage() {
  return <SlideMaker />;
}
