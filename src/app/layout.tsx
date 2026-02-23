import type { Metadata } from 'next';
import { ThemeProvider } from '@/context/ThemeContext';
import { FloatingChatWrapper } from '@/components/custom/floating-chat-wrapper';
import '@/index.css';

export const metadata: Metadata = {
    title: {
        default: 'Zizhao Hu — AI Researcher | PhD Student at USC',
        template: '%s | Zizhao Hu',
    },
    description:
        'Zizhao Hu is a CS PhD student at USC building AI systems that improve themselves. Research in multi-agent systems, synthetic data, brain-inspired architectures, and continual learning.',
    keywords: [
        'Zizhao Hu', 'AI researcher', 'USC PhD', 'multi-agent systems',
        'synthetic data', 'continual learning', 'LLM', 'VLM',
        'brain-inspired architecture', 'GLAMOUR Lab', 'MINDS Group',
    ],
    authors: [{ name: 'Zizhao Hu' }],
    openGraph: {
        title: 'Zizhao Hu — AI Researcher',
        description: 'CS PhD at USC. Multi-agent systems, synthetic data, continual learning.',
        type: 'website',
        locale: 'en_US',
        siteName: 'Zizhao Hu',
    },
    twitter: {
        card: 'summary_large_image',
        title: 'Zizhao Hu — AI Researcher',
        description: 'CS PhD at USC. Multi-agent systems, synthetic data, continual learning.',
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
                    href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Poppins:wght@400;500;600;700&display=swap"
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
