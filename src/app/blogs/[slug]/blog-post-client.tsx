'use client';

import { BlogPost } from '@/views/blogs/blog-post';

export function BlogPostClient({ slug }: { slug: string }) {
  return <BlogPost slugOverride={slug} />;
}
