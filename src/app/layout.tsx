import type { Metadata } from 'next';
import { ThemeProvider } from '@/context/ThemeContext';
import { FloatingChatWrapper } from '@/components/custom/floating-chat-wrapper';
import '@/index.css';

export const metadata: Metadata = {
    title: {
        default: 'Zizhao.md',
        template: '%s — Zizhao.md',
    },
    description:
        'Zizhao Hu — CS PhD at USC GLAMOR Lab. Research on model unlearning, safety alignment, and post-training for LLMs and VLMs.',
    keywords: [
        'Zizhao Hu', 'AI researcher', 'USC PhD', 'model unlearning',
        'safety alignment', 'post-training', 'LLM', 'VLM',
        'KV cache', 'synthetic data', 'GLAMOR Lab', 'Jesse Thomason',
    ],
    authors: [{ name: 'Zizhao Hu' }],
    openGraph: {
        title: 'Zizhao.md',
        description: 'CS PhD at USC GLAMOR Lab. Model unlearning, safety alignment, post-training for LLMs and VLMs.',
        type: 'website',
        locale: 'en_US',
        siteName: 'Zizhao.md',
    },
    twitter: {
        card: 'summary_large_image',
        title: 'Zizhao.md',
        description: 'CS PhD at USC GLAMOR Lab. Model unlearning, safety alignment, post-training for LLMs and VLMs.',
    },
    robots: {
        index: true,
        follow: true,
    },
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" suppressHydrationWarning>
            <head>
                <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
                <link
                    href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Poppins:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap"
                    rel="stylesheet"
                />
            </head>
            <body>
                <ThemeProvider>
                    <div className="w-full h-screen bg-white dark:bg-gray-900 text-gray-900 dark:text-white">
                        {children}
                    </div>
                    <FloatingChatWrapper />
                </ThemeProvider>
            </body>
        </html>
    );
}
