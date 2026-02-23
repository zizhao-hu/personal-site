const fs = require('fs');

// Fix ThemeContext - add 'use client' and guard localStorage
let tc = fs.readFileSync('src/context/ThemeContext.tsx', 'utf-8');
if (!tc.startsWith("'use client'")) {
    tc = "'use client';\n\n" + tc;
}
tc = tc.replace(
    'const savedTheme = localStorage.getItem',
    "if (typeof window === 'undefined') return false;\n    const savedTheme = localStorage.getItem"
);
fs.writeFileSync('src/context/ThemeContext.tsx', tc);
console.log('Fixed ThemeContext');

// Fix ProjectDetail
let pd = fs.readFileSync('src/pages/projects/project-detail.tsx', 'utf-8');
if (!pd.startsWith("'use client'")) {
    pd = "'use client';\n\n" + pd;
}
pd = pd.replace(/import\s*\{[^}]*\}\s*from\s*["']react-router-dom["'];?/g, "import { useParams, useRouter } from 'next/navigation';");
pd = pd.replace(/export const ProjectDetail = \(\)\s*=>\s*\{/, 'export const ProjectDetail = ({ slugOverride }: { slugOverride?: string }) => {');
pd = pd.replace(/const\s*\{\s*slug\s*\}\s*=\s*useParams<[^>]*>\(\);?/g, 'const params = useParams();\n    const slug = slugOverride || (params?.slug as string);');
pd = pd.replace(/const\s+navigate\s*=\s*useNavigate\(\);?/g, 'const router = useRouter();');
pd = pd.replace(/navigate\(/g, 'router.push(');
fs.writeFileSync('src/pages/projects/project-detail.tsx', pd);
console.log('Fixed ProjectDetail');

// Fix TutorialDetail
let td = fs.readFileSync('src/pages/tutorials/tutorial-detail.tsx', 'utf-8');
if (!td.startsWith("'use client'")) {
    td = "'use client';\n\n" + td;
}
td = td.replace(/import\s*\{[^}]*\}\s*from\s*["']react-router-dom["'];?/g, "import { useParams, useRouter } from 'next/navigation';");
td = td.replace(/export const TutorialDetail = \(\)\s*=>\s*\{/, 'export const TutorialDetail = ({ slugOverride }: { slugOverride?: string }) => {');
td = td.replace(/const\s*\{\s*slug\s*\}\s*=\s*useParams<[^>]*>\(\);?/g, 'const params = useParams();\n  const slug = slugOverride || (params?.slug as string);');
td = td.replace(/const\s+navigate\s*=\s*useNavigate\(\);?/g, 'const router = useRouter();');
td = td.replace(/navigate\(/g, 'router.push(');
fs.writeFileSync('src/pages/tutorials/tutorial-detail.tsx', td);
console.log('Fixed TutorialDetail');
