/**
 * Fix detail pages to accept slugOverride prop for Next.js page wrappers.
 * Also re-apply 'use client' and react-router-dom -> next/navigation fixes.
 */
import fs from 'fs';

function fixFile(filePath, componentName, dataFnName) {
    let content = fs.readFileSync(filePath, 'utf-8');

    // Add 'use client' at top if not present
    if (!content.startsWith("'use client'")) {
        content = "'use client';\n\n" + content;
    }

    // Fix react-router-dom imports
    content = content.replace(
        /import\s*\{[^}]*\}\s*from\s*["']react-router-dom["'];?/g,
        "import { useParams, useRouter } from 'next/navigation';"
    );

    // Fix component declaration to accept slugOverride
    // Pattern: export const ComponentName = () => {
    const compRegex = new RegExp(
        `export const ${componentName} = \\(\\)\\s*=>\\s*\\{`
    );
    content = content.replace(compRegex,
        `export const ${componentName} = ({ slugOverride }: { slugOverride?: string }) => {`
    );

    // Fix useParams destructure: const { slug } = useParams<...>();
    content = content.replace(
        /const\s*\{\s*slug\s*\}\s*=\s*useParams<[^>]*>\(\);?/g,
        'const params = useParams();\n  const slug = slugOverride || (params?.slug as string);'
    );

    // Fix navigate -> router.push
    content = content.replace(
        /const\s+navigate\s*=\s*useNavigate\(\);?/g,
        'const router = useRouter();'
    );
    content = content.replace(/navigate\(/g, 'router.push(');

    fs.writeFileSync(filePath, content, 'utf-8');
    console.log(`Fixed: ${filePath}`);
}

// Fix all three detail pages
fixFile('src/pages/projects/project-detail.tsx', 'ProjectDetail', 'getProjectBySlug');
fixFile('src/pages/tutorials/tutorial-detail.tsx', 'TutorialDetail', 'getTutorialBySlug');

// Also fix the blogs page (list page) which uses useNavigate
let blogsContent = fs.readFileSync('src/pages/blogs/blogs.tsx', 'utf-8');
if (!blogsContent.startsWith("'use client'")) {
    blogsContent = "'use client';\n\n" + blogsContent;
}
blogsContent = blogsContent.replace(
    /import\s*\{[^}]*\}\s*from\s*["']react-router-dom["'];?/g,
    "import { useRouter } from 'next/navigation';"
);
blogsContent = blogsContent.replace(
    /const\s+navigate\s*=\s*useNavigate\(\);?/g,
    'const router = useRouter();'
);
blogsContent = blogsContent.replace(/navigate\(/g, 'router.push(');
fs.writeFileSync('src/pages/blogs/blogs.tsx', blogsContent, 'utf-8');
console.log('Fixed: src/pages/blogs/blogs.tsx');

// Fix projects list page
let projContent = fs.readFileSync('src/pages/projects/projects.tsx', 'utf-8');
if (!projContent.startsWith("'use client'")) {
    projContent = "'use client';\n\n" + projContent;
}
projContent = projContent.replace(
    /import\s*\{[^}]*\}\s*from\s*["']react-router-dom["'];?/g,
    "import { useRouter } from 'next/navigation';"
);
projContent = projContent.replace(
    /const\s+navigate\s*=\s*useNavigate\(\);?/g,
    'const router = useRouter();'
);
projContent = projContent.replace(/navigate\(/g, 'router.push(');
fs.writeFileSync('src/pages/projects/projects.tsx', projContent, 'utf-8');
console.log('Fixed: src/pages/projects/projects.tsx');

// Fix tutorials list page
let tutContent = fs.readFileSync('src/pages/tutorials/tutorials.tsx', 'utf-8');
if (!tutContent.startsWith("'use client'")) {
    tutContent = "'use client';\n\n" + tutContent;
}
tutContent = tutContent.replace(
    /import\s*\{[^}]*\}\s*from\s*["']react-router-dom["'];?/g,
    "import { useRouter } from 'next/navigation';"
);
tutContent = tutContent.replace(
    /const\s+navigate\s*=\s*useNavigate\(\);?/g,
    'const router = useRouter();'
);
tutContent = tutContent.replace(/navigate\(/g, 'router.push(');
fs.writeFileSync('src/pages/tutorials/tutorials.tsx', tutContent, 'utf-8');
console.log('Fixed: src/pages/tutorials/tutorials.tsx');

// Fix Header component
let headerContent = fs.readFileSync('src/components/custom/header.tsx', 'utf-8');
if (!headerContent.startsWith("'use client'")) {
    headerContent = "'use client';\n\n" + headerContent;
}
headerContent = headerContent.replace(
    /import\s*\{[^}]*\}\s*from\s*["']react-router-dom["'];?/g,
    "import Link from 'next/link';\nimport { usePathname } from 'next/navigation';"
);
// useLocation -> usePathname
headerContent = headerContent.replace(/useLocation\(\)/g, 'usePathname()');
headerContent = headerContent.replace(/location\.pathname/g, 'pathname');
// If it used const location = ..., change to const pathname = ...
headerContent = headerContent.replace(/const\s+location\s*=\s*usePathname\(\)/g, 'const pathname = usePathname()');
// Fix Link to prop -> href prop
headerContent = headerContent.replace(/<Link\s+to=/g, '<Link href=');
fs.writeFileSync('src/components/custom/header.tsx', headerContent, 'utf-8');
console.log('Fixed: src/components/custom/header.tsx');

// Fix research pages
const researchPages = [
    'src/pages/research/research.tsx',
    'src/pages/research/llm-vlm.tsx',
    'src/pages/research/architecture.tsx',
    'src/pages/research/continual-learning.tsx',
    'src/pages/research/synthetic-data.tsx',
];
for (const page of researchPages) {
    let c = fs.readFileSync(page, 'utf-8');
    if (!c.startsWith("'use client'") && (c.includes('useState') || c.includes('useEffect') || c.includes('onClick') || c.includes('useRouter') || c.includes('useNavigate'))) {
        c = "'use client';\n\n" + c;
    }
    c = c.replace(
        /import\s*\{[^}]*\}\s*from\s*["']react-router-dom["'];?/g,
        "import Link from 'next/link';\nimport { useRouter } from 'next/navigation';"
    );
    c = c.replace(/const\s+navigate\s*=\s*useNavigate\(\);?/g, 'const router = useRouter();');
    c = c.replace(/navigate\(/g, 'router.push(');
    c = c.replace(/<Link\s+to=/g, '<Link href=');
    fs.writeFileSync(page, c, 'utf-8');
    console.log(`Fixed: ${page}`);
}

// Fix tools pages
const toolPages = [
    'src/pages/tools/tools.tsx',
    'src/pages/tools/pipeline-designer.tsx',
    'src/pages/tools/slide-maker.tsx',
    'src/pages/tools/starship-sim.tsx',
];
for (const page of toolPages) {
    let c = fs.readFileSync(page, 'utf-8');
    if (!c.startsWith("'use client'") && (c.includes('useState') || c.includes('useEffect') || c.includes('onClick') || c.includes('useRouter') || c.includes('useNavigate') || c.includes('window'))) {
        c = "'use client';\n\n" + c;
    }
    c = c.replace(
        /import\s*\{[^}]*\}\s*from\s*["']react-router-dom["'];?/g,
        "import Link from 'next/link';\nimport { useRouter } from 'next/navigation';"
    );
    c = c.replace(/const\s+navigate\s*=\s*useNavigate\(\);?/g, 'const router = useRouter();');
    c = c.replace(/navigate\(/g, 'router.push(');
    c = c.replace(/<Link\s+to=/g, '<Link href=');
    fs.writeFileSync(page, c, 'utf-8');
    console.log(`Fixed: ${page}`);
}

// Fix chat page
let chatContent = fs.readFileSync('src/pages/chat/chat.tsx', 'utf-8');
if (!chatContent.startsWith("'use client'") && (chatContent.includes('useState') || chatContent.includes('useEffect'))) {
    chatContent = "'use client';\n\n" + chatContent;
}
chatContent = chatContent.replace(
    /import\s*\{[^}]*\}\s*from\s*["']react-router-dom["'];?/g,
    ""
);
fs.writeFileSync('src/pages/chat/chat.tsx', chatContent, 'utf-8');
console.log('Fixed: src/pages/chat/chat.tsx');

// Fix floating-chat.tsx
let fcContent = fs.readFileSync('src/components/custom/floating-chat.tsx', 'utf-8');
if (!fcContent.startsWith("'use client'")) {
    fcContent = "'use client';\n\n" + fcContent;
}
fcContent = fcContent.replace(
    /import\s*\{[^}]*\}\s*from\s*["']react-router-dom["'];?/g,
    ""
);
fs.writeFileSync('src/components/custom/floating-chat.tsx', fcContent, 'utf-8');
console.log('Fixed: src/components/custom/floating-chat.tsx');

// Fix overview.tsx
let ovContent = fs.readFileSync('src/components/custom/overview.tsx', 'utf-8');
if (!ovContent.startsWith("'use client'") && (ovContent.includes('useState') || ovContent.includes('onClick'))) {
    ovContent = "'use client';\n\n" + ovContent;
}
ovContent = ovContent.replace(
    /import\s*\{[^}]*\}\s*from\s*["']react-router-dom["'];?/g,
    ""
);
fs.writeFileSync('src/components/custom/overview.tsx', ovContent, 'utf-8');
console.log('Fixed: src/components/custom/overview.tsx');

console.log('\n✅ All detail page fixes applied!');
