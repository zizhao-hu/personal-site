import type { Metadata } from 'next';
import { tutorials } from '@/data/tutorials';
import { TutorialDetailClient } from './tutorial-detail-client';

export async function generateStaticParams() {
  return tutorials.map((t) => ({
    slug: t.slug,
  }));
}

export async function generateMetadata({ params }: { params: { slug: string } }): Promise<Metadata> {
  const tutorial = tutorials.find(t => t.slug === params.slug);
  if (!tutorial) {
    return { title: 'Tutorial Not Found' };
  }
  return {
    title: tutorial.title,
    description: tutorial.description,
  };
}

export default function TutorialDetailPage({ params }: { params: { slug: string } }) {
  return <TutorialDetailClient slug={params.slug} />;
}
