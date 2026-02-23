'use client';

import { TutorialDetail } from '@/views/tutorials/tutorial-detail';

export function TutorialDetailClient({ slug }: { slug: string }) {
  return <TutorialDetail slugOverride={slug} />;
}
