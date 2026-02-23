import type { Metadata } from 'next';
import { Tutorials } from '@/views/tutorials/tutorials';

export const metadata: Metadata = {
  title: 'Tutorials',
  description: 'Hands-on tutorials on AI, machine learning, and software engineering by Zizhao Hu.',
};

export default function TutorialsPage() {
  return <Tutorials />;
}
