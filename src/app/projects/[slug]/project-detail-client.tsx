'use client';

import { ProjectDetail } from '@/views/projects/project-detail';

export function ProjectDetailClient({ slug }: { slug: string }) {
  return <ProjectDetail slugOverride={slug} />;
}
