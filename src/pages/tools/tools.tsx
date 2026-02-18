import { Header } from '@/components/custom/header';
import { useNavigate } from 'react-router-dom';
import { Wrench } from 'lucide-react';

const tools = [
    {
        id: 'pipeline-designer',
        title: 'AI Pipeline Designer',
        description: 'Interactive canvas tool for designing and visualizing AI/ML pipeline architectures. Drag, connect, and export publication-ready diagrams.',
        path: '/tools/pipeline-designer',
        icon: '🔧',
        tags: ['Canvas', 'Diagrams', 'Research'],
    },
    {
        id: 'slide-maker',
        title: 'Research Slide Maker',
        description: 'Build academic presentation slides with a pre-defined research template sequence. Drag, resize, and edit elements, then export for Google Slides.',
        path: '/tools/slide-maker',
        icon: '📊',
        tags: ['Slides', 'Presentations', 'Research'],
    },
];

export const Tools = () => {
    const navigate = useNavigate();

    return (
        <div className="flex flex-col min-h-dvh bg-background">
            <Header />
            <div className="flex-1 overflow-y-auto pb-24">
                <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8">
                    {/* Hero */}
                    <div className="mb-8">
                        <div className="flex items-center gap-2 mb-2">
                            <Wrench className="w-5 h-5 text-brand-orange" />
                            <h1 className="text-2xl font-bold font-heading text-foreground">Tools</h1>
                        </div>
                        <p className="text-sm text-muted-foreground max-w-xl">
                            Interactive utilities and design tools built for research and productivity.
                        </p>
                    </div>

                    {/* Tool Cards */}
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                        {tools.map((tool) => (
                            <button
                                key={tool.id}
                                onClick={() => navigate(tool.path)}
                                className="group text-left p-5 rounded-xl bg-card border border-border hover:border-brand-orange/50 hover:shadow-lg transition-all duration-300"
                            >
                                <div className="text-3xl mb-3">{tool.icon}</div>
                                <h2 className="text-sm font-semibold font-heading text-foreground group-hover:text-brand-orange transition-colors mb-1">
                                    {tool.title}
                                </h2>
                                <p className="text-xs text-muted-foreground leading-relaxed mb-3">
                                    {tool.description}
                                </p>
                                <div className="flex flex-wrap gap-1.5">
                                    {tool.tags.map((tag) => (
                                        <span
                                            key={tag}
                                            className="px-2 py-0.5 rounded-full bg-muted text-[10px] font-medium text-muted-foreground uppercase tracking-wider"
                                        >
                                            {tag}
                                        </span>
                                    ))}
                                </div>
                            </button>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};
