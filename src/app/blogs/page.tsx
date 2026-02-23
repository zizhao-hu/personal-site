import type { Metadata } from 'next';
import { Blogs } from '@/views/blogs/blogs';

export const metadata: Metadata = {
  title: 'Blog',
  description: 'Thoughts on AI research, machine learning, safety, and technology by Zizhao Hu.',
};

export default function BlogsPage() {
  return <Blogs />;
}
