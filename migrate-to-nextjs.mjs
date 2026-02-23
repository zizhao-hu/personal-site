/**
 * Vite SPA → Next.js App Router Migration Script
 * 
 * This script:
 * 1. Creates the app/ directory structure
 * 2. Copies page components as 'use client' wrappers importing existing components
 * 3. Adds 'use client' directives to all components using React hooks or browser APIs
 * 4. Replaces react-router-dom imports with next/navigation equivalents
 * 5. Generates sitemap.ts and robots.ts
 */

import fs from 'fs';
import path from 'path';

const SRC = 'src';
const APP = 'src/app';

// ── STEP 1: Fix react-router-dom imports in ALL source files ──

function fixReactRouterImports(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            if (entry.name === 'node_modules' || entry.name === '.next' || entry.name === 'app') continue;
            fixReactRouterImports(fullPath);
        } else if (entry.name.endsWith('.tsx') || entry.name.endsWith('.ts')) {
            let content = fs.readFileSync(fullPath, 'utf-8');
            if (!content.includes('react-router-dom')) continue;

            console.log(`  Fixing imports in: ${fullPath}`);

            // Add 'use client' if not already present (files using router need it)
            if (!content.startsWith("'use client'") && !content.startsWith('"use client"')) {
                content = "'use client';\n\n" + content;
            }

            // Replace useNavigate with useRouter
            if (content.includes('useNavigate')) {
                content = content.replace(
                    /import\s*\{([^}]*)\}\s*from\s*['"]react-router-dom['"];?/g,
                    (match, imports) => {
                        const importList = imports.split(',').map(s => s.trim()).filter(Boolean);
                        const nextImports = [];
                        const linkImports = [];

                        for (const imp of importList) {
                            if (imp === 'useNavigate') nextImports.push('useRouter');
                            else if (imp === 'useParams') nextImports.push('useParams');
                            else if (imp === 'useLocation') nextImports.push('usePathname');
                            else if (imp === 'Link') linkImports.push('Link');
                            // Drop BrowserRouter, Routes, Route, etc.
                        }

                        let result = '';
                        if (nextImports.length > 0) {
                            result += `import { ${nextImports.join(', ')} } from 'next/navigation';\n`;
                        }
                        if (linkImports.length > 0) {
                            result += `import Link from 'next/link';\n`;
                        }
                        return result;
                    }
                );

                // Replace useNavigate() -> useRouter(), navigate() -> router.push()
                content = content.replace(/const\s+navigate\s*=\s*useNavigate\(\);?/g, 'const router = useRouter();');
                content = content.replace(/navigate\(([^)]+)\)/g, 'router.push($1)');
            }

            // Replace useParams import only
            if (content.includes('useParams') && !content.includes('next/navigation')) {
                content = content.replace(
                    /import\s*\{([^}]*)\}\s*from\s*['"]react-router-dom['"];?/g,
                    (match, imports) => {
                        const importList = imports.split(',').map(s => s.trim()).filter(Boolean);
                        const nextImports = [];
                        for (const imp of importList) {
                            if (imp === 'useParams') nextImports.push('useParams');
                            else if (imp === 'useNavigate') nextImports.push('useRouter');
                            else if (imp === 'useLocation') nextImports.push('usePathname');
                        }
                        return nextImports.length > 0
                            ? `import { ${nextImports.join(', ')} } from 'next/navigation';`
                            : '';
                    }
                );
            }

            // Handle Link from react-router-dom → Link from next/link
            // next/link uses href instead of to
            content = content.replace(/import\s*\{[^}]*Link[^}]*\}\s*from\s*['"]react-router-dom['"];?/g,
                "import Link from 'next/link';");
            content = content.replace(/<Link\s+to=/g, '<Link href=');

            // Clean up any remaining react-router-dom imports
            content = content.replace(/import\s*\{[^}]*\}\s*from\s*['"]react-router-dom['"];?\n?/g, '');

            fs.writeFileSync(fullPath, content, 'utf-8');
        }
    }
}

// ── STEP 2: Add 'use client' to components that use React hooks ──

function addUseClientDirectives(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            if (entry.name === 'node_modules' || entry.name === '.next' || entry.name === 'app') continue;
            addUseClientDirectives(fullPath);
        } else if (entry.name.endsWith('.tsx') || entry.name.endsWith('.ts')) {
            let content = fs.readFileSync(fullPath, 'utf-8');

            // Skip if already has 'use client'
            if (content.startsWith("'use client'") || content.startsWith('"use client"')) continue;

            // Check if file uses client-side features
            const needsUseClient =
                content.includes('useState') ||
                content.includes('useEffect') ||
                content.includes('useRef') ||
                content.includes('useContext') ||
                content.includes('useCallback') ||
                content.includes('useMemo') ||
                content.includes('useReducer') ||
                content.includes('onClick') ||
                content.includes('onChange') ||
                content.includes('onSubmit') ||
                content.includes('onError') ||
                content.includes('onMouseEnter') ||
                content.includes('localStorage') ||
                content.includes('window.') ||
                content.includes('document.') ||
                content.includes('useTheme') ||
                content.includes('framer-motion') ||
                content.includes('useRouter') ||
                content.includes('usePathname');

            if (needsUseClient) {
                console.log(`  Adding 'use client' to: ${fullPath}`);
                content = "'use client';\n\n" + content;
                fs.writeFileSync(fullPath, content, 'utf-8');
            }
        }
    }
}

// ── STEP 3: Create App Router pages ──

function createAppPages() {
    // Map of route -> page config
    const pages = [
        {
            route: 'research',
            import: '@/pages/research/research',
            component: 'Research',
            title: 'Research',
            description: 'Zizhao Hu\'s research in multi-agent systems, synthetic data, brain-inspired architectures, and continual learning.',
        },
        {
            route: 'research/llm-vlm',
            import: '@/pages/research/llm-vlm',
            component: 'LlmVlmResearch',
            title: 'LLM & VLM Research',
            description: 'Research on Large Language Models and Vision-Language Models — multi-agent interaction, training paradigms, and model architecture.',
        },
        {
            route: 'research/architecture',
            import: '@/pages/research/architecture',
            component: 'ArchitectureResearch',
            title: 'Neural Architecture Research',
            description: 'Brain-inspired neural architectures, novel attention mechanisms, and efficient model design.',
        },
        {
            route: 'research/continual-learning',
            import: '@/pages/research/continual-learning',
            component: 'ContinualLearningResearch',
            title: 'Continual Learning Research',
            description: 'Research on continual learning without catastrophic forgetting — lifelong model training and knowledge retention.',
        },
        {
            route: 'research/synthetic-data',
            import: '@/pages/research/synthetic-data',
            component: 'SyntheticDataResearch',
            title: 'Synthetic Data Research',
            description: 'Generate-validate loops, self-improving data pipelines, and preventing model collapse in synthetic data.',
        },
        {
            route: 'projects',
            import: '@/pages/projects/projects',
            component: 'Projects',
            title: 'Projects',
            description: 'Open-source projects and tools by Zizhao Hu — AI systems, multi-agent frameworks, and research infrastructure.',
        },
        {
            route: 'blogs',
            import: '@/pages/blogs/blogs',
            component: 'Blogs',
            title: 'Blog',
            description: 'Thoughts on AI research, machine learning, safety, and technology by Zizhao Hu.',
        },
        {
            route: 'tutorials',
            import: '@/pages/tutorials/tutorials',
            component: 'Tutorials',
            title: 'Tutorials',
            description: 'Hands-on tutorials on AI, machine learning, and software engineering by Zizhao Hu.',
        },
        {
            route: 'chat',
            import: '@/pages/chat/chat',
            component: 'Chat',
            title: 'Chat with Zizhao',
            description: 'Chat with an AI powered by Zizhao Hu\'s research and background knowledge.',
        },
        {
            route: 'tools',
            import: '@/pages/tools/tools',
            component: 'Tools',
            title: 'Tools',
            description: 'Interactive AI tools and demonstrations built by Zizhao Hu.',
        },
        {
            route: 'tools/pipeline-designer',
            import: '@/pages/tools/pipeline-designer',
            component: 'PipelineDesigner',
            title: 'Pipeline Designer',
            description: 'Visual ML pipeline designer tool for creating training workflows.',
        },
        {
            route: 'tools/slide-maker',
            import: '@/pages/tools/slide-maker',
            component: 'SlideMaker',
            title: 'Slide Maker',
            description: 'AI-powered presentation slide generator for research and technical talks.',
        },
        {
            route: 'tools/starship-sim',
            import: '@/pages/tools/starship-sim',
            component: 'StarshipSim',
            title: 'Starship Simulator',
            description: 'Interactive SpaceX Starship flight simulator with realistic physics.',
        },
    ];

    for (const page of pages) {
        const dir = path.join(APP, page.route);
        fs.mkdirSync(dir, { recursive: true });

        const content = `import type { Metadata } from 'next';
import { ${page.component} } from '${page.import}';

export const metadata: Metadata = {
  title: '${page.title}',
  description: '${page.description}',
};

export default function ${page.component}Page() {
  return <${page.component} />;
}
`;
        fs.writeFileSync(path.join(dir, 'page.tsx'), content, 'utf-8');
        console.log(`  Created: ${dir}/page.tsx`);
    }
}

// ── STEP 4: Create dynamic route pages with generateStaticParams ──

function createDynamicPages() {
    // Blog detail
    const blogDir = path.join(APP, 'blogs/[slug]');
    fs.mkdirSync(blogDir, { recursive: true });

    fs.writeFileSync(path.join(blogDir, 'page.tsx'), `import type { Metadata } from 'next';
import { blogPosts, getBlogBySlug } from '@/data/blog-posts';
import { BlogPostClient } from './blog-post-client';

export async function generateStaticParams() {
  return blogPosts.map((post) => ({
    slug: post.slug,
  }));
}

export async function generateMetadata({ params }: { params: { slug: string } }): Promise<Metadata> {
  const post = getBlogBySlug(params.slug);
  if (!post) {
    return { title: 'Post Not Found' };
  }
  return {
    title: post.title,
    description: post.excerpt,
    openGraph: {
      title: post.title,
      description: post.excerpt,
      type: 'article',
      publishedTime: post.date,
      authors: [post.author?.name || 'Zizhao Hu'],
      tags: post.tags,
      ...(post.coverImage ? { images: [{ url: post.coverImage }] } : {}),
    },
    twitter: {
      card: 'summary_large_image',
      title: post.title,
      description: post.excerpt,
    },
  };
}

export default function BlogPostPage({ params }: { params: { slug: string } }) {
  return <BlogPostClient slug={params.slug} />;
}
`, 'utf-8');

    // Create client wrapper for blog post
    fs.writeFileSync(path.join(blogDir, 'blog-post-client.tsx'), `'use client';

import { BlogPost } from '@/pages/blogs/blog-post';

export function BlogPostClient({ slug }: { slug: string }) {
  return <BlogPost slugOverride={slug} />;
}
`, 'utf-8');
    console.log(`  Created: ${blogDir}/page.tsx + blog-post-client.tsx`);

    // Project detail
    const projectDir = path.join(APP, 'projects/[slug]');
    fs.mkdirSync(projectDir, { recursive: true });

    fs.writeFileSync(path.join(projectDir, 'page.tsx'), `import type { Metadata } from 'next';
import { projects } from '@/data/projects';
import { ProjectDetailClient } from './project-detail-client';

export async function generateStaticParams() {
  return projects.map((p) => ({
    slug: p.slug,
  }));
}

export async function generateMetadata({ params }: { params: { slug: string } }): Promise<Metadata> {
  const project = projects.find(p => p.slug === params.slug);
  if (!project) {
    return { title: 'Project Not Found' };
  }
  return {
    title: project.title,
    description: project.tagline,
  };
}

export default function ProjectDetailPage({ params }: { params: { slug: string } }) {
  return <ProjectDetailClient slug={params.slug} />;
}
`, 'utf-8');

    fs.writeFileSync(path.join(projectDir, 'project-detail-client.tsx'), `'use client';

import { ProjectDetail } from '@/pages/projects/project-detail';

export function ProjectDetailClient({ slug }: { slug: string }) {
  return <ProjectDetail slugOverride={slug} />;
}
`, 'utf-8');
    console.log(`  Created: ${projectDir}/page.tsx + project-detail-client.tsx`);

    // Tutorial detail
    const tutorialDir = path.join(APP, 'tutorials/[slug]');
    fs.mkdirSync(tutorialDir, { recursive: true });

    fs.writeFileSync(path.join(tutorialDir, 'page.tsx'), `import type { Metadata } from 'next';
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
`, 'utf-8');

    fs.writeFileSync(path.join(tutorialDir, 'tutorial-detail-client.tsx'), `'use client';

import { TutorialDetail } from '@/pages/tutorials/tutorial-detail';

export function TutorialDetailClient({ slug }: { slug: string }) {
  return <TutorialDetail slugOverride={slug} />;
}
`, 'utf-8');
    console.log(`  Created: ${tutorialDir}/page.tsx + tutorial-detail-client.tsx`);
}

// ── STEP 5: Create sitemap.ts ──

function createSitemap() {
    fs.writeFileSync(path.join(APP, 'sitemap.ts'), `import { MetadataRoute } from 'next';
import { blogPosts } from '@/data/blog-posts';
import { projects } from '@/data/projects';
import { tutorials } from '@/data/tutorials';

const BASE_URL = 'https://zizhaohu.com';

export default function sitemap(): MetadataRoute.Sitemap {
  const staticPages = [
    { url: BASE_URL, lastModified: new Date(), changeFrequency: 'weekly' as const, priority: 1 },
    { url: \`\${BASE_URL}/research\`, lastModified: new Date(), changeFrequency: 'monthly' as const, priority: 0.8 },
    { url: \`\${BASE_URL}/research/llm-vlm\`, lastModified: new Date(), changeFrequency: 'monthly' as const, priority: 0.7 },
    { url: \`\${BASE_URL}/research/architecture\`, lastModified: new Date(), changeFrequency: 'monthly' as const, priority: 0.7 },
    { url: \`\${BASE_URL}/research/continual-learning\`, lastModified: new Date(), changeFrequency: 'monthly' as const, priority: 0.7 },
    { url: \`\${BASE_URL}/research/synthetic-data\`, lastModified: new Date(), changeFrequency: 'monthly' as const, priority: 0.7 },
    { url: \`\${BASE_URL}/projects\`, lastModified: new Date(), changeFrequency: 'weekly' as const, priority: 0.8 },
    { url: \`\${BASE_URL}/blogs\`, lastModified: new Date(), changeFrequency: 'weekly' as const, priority: 0.9 },
    { url: \`\${BASE_URL}/tutorials\`, lastModified: new Date(), changeFrequency: 'monthly' as const, priority: 0.7 },
    { url: \`\${BASE_URL}/tools\`, lastModified: new Date(), changeFrequency: 'monthly' as const, priority: 0.6 },
    { url: \`\${BASE_URL}/chat\`, lastModified: new Date(), changeFrequency: 'monthly' as const, priority: 0.5 },
  ];

  const blogPages = blogPosts.map((post) => ({
    url: \`\${BASE_URL}/blogs/\${post.slug}\`,
    lastModified: new Date(post.date),
    changeFrequency: 'monthly' as const,
    priority: 0.8,
  }));

  const projectPages = projects.map((project) => ({
    url: \`\${BASE_URL}/projects/\${project.slug}\`,
    lastModified: new Date(),
    changeFrequency: 'monthly' as const,
    priority: 0.7,
  }));

  const tutorialPages = tutorials.map((tutorial) => ({
    url: \`\${BASE_URL}/tutorials/\${tutorial.slug}\`,
    lastModified: new Date(),
    changeFrequency: 'monthly' as const,
    priority: 0.7,
  }));

  return [...staticPages, ...blogPages, ...projectPages, ...tutorialPages];
}
`, 'utf-8');
    console.log('  Created: sitemap.ts');
}

// ── STEP 6: Create robots.ts ──

function createRobots() {
    fs.writeFileSync(path.join(APP, 'robots.ts'), `import { MetadataRoute } from 'next';

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: '*',
      allow: '/',
    },
    sitemap: 'https://zizhaohu.com/sitemap.xml',
  };
}
`, 'utf-8');
    console.log('  Created: robots.ts');
}

// ── Run all steps ──

console.log('\\n🚀 Starting Vite → Next.js Migration\\n');

console.log('Step 1: Fixing react-router-dom imports...');
fixReactRouterImports(SRC);

console.log('\\nStep 2: Adding "use client" directives...');
addUseClientDirectives(path.join(SRC, 'components'));
addUseClientDirectives(path.join(SRC, 'pages'));
addUseClientDirectives(path.join(SRC, 'context'));
addUseClientDirectives(path.join(SRC, 'lib'));

console.log('\\nStep 3: Creating App Router pages...');
createAppPages();

console.log('\\nStep 4: Creating dynamic route pages with generateStaticParams...');
createDynamicPages();

console.log('\\nStep 5: Creating sitemap.ts...');
createSitemap();

console.log('\\nStep 6: Creating robots.ts...');
createRobots();

console.log('\\n✅ Migration script complete!');
console.log('\\nNext steps:');
console.log('  1. Fix BlogPost/ProjectDetail/TutorialDetail to accept slugOverride prop');
console.log('  2. Fix Header component Link usage');
console.log('  3. Run next dev to test');
