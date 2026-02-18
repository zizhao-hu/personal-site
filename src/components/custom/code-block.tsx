import { useState, useCallback } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';

interface CodeBlockProps {
    code: string;
    language?: string;
    showLineNumbers?: boolean;
}

/* Brand-tuned dark theme derived from oneDark */
const customTheme = {
    ...oneDark,
    'pre[class*="language-"]': {
        ...oneDark['pre[class*="language-"]'],
        background: '#1a1a19',
        margin: 0,
        padding: '1rem 1.25rem',
        borderRadius: 0,
        fontSize: '0.8125rem',
        lineHeight: '1.65',
    },
    'code[class*="language-"]': {
        ...oneDark['code[class*="language-"]'],
        background: 'transparent',
        fontSize: '0.8125rem',
        fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace",
    },
};

const languageLabels: Record<string, string> = {
    python: 'Python',
    py: 'Python',
    javascript: 'JavaScript',
    js: 'JavaScript',
    typescript: 'TypeScript',
    ts: 'TypeScript',
    tsx: 'TSX',
    jsx: 'JSX',
    html: 'HTML',
    css: 'CSS',
    bash: 'Bash',
    shell: 'Shell',
    json: 'JSON',
    yaml: 'YAML',
    sql: 'SQL',
    rust: 'Rust',
    go: 'Go',
    cpp: 'C++',
    c: 'C',
    java: 'Java',
    ruby: 'Ruby',
    text: 'Text',
    plaintext: 'Text',
    '': 'Text',
};

export const CodeBlock = ({ code, language = '', showLineNumbers = true }: CodeBlockProps) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = useCallback(async () => {
        try {
            await navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch {
            // Fallback for older browsers
            const textarea = document.createElement('textarea');
            textarea.value = code;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    }, [code]);

    const lang = language.toLowerCase().replace('language-', '');
    const displayLabel = languageLabels[lang] || lang.toUpperCase() || 'Text';
    const lines = code.split('\n').length;

    return (
        <div className="group relative rounded-xl overflow-hidden border border-white/[0.06] bg-[#1a1a19] my-4 shadow-lg">
            {/* Header bar */}
            <div className="flex items-center justify-between px-4 py-2 bg-[#232322] border-b border-white/[0.06]">
                {/* Language badge */}
                <div className="flex items-center gap-2">
                    <div className="flex gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full bg-[#ff5f57]" />
                        <span className="w-2.5 h-2.5 rounded-full bg-[#febc2e]" />
                        <span className="w-2.5 h-2.5 rounded-full bg-[#28c840]" />
                    </div>
                    <span className="text-[10px] font-medium text-gray-400 uppercase tracking-wider ml-2 font-heading">
                        {displayLabel}
                    </span>
                </div>

                {/* Copy button */}
                <button
                    onClick={handleCopy}
                    className="flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-medium transition-all hover:bg-white/10"
                    style={{ color: copied ? '#28c840' : '#888' }}
                    aria-label="Copy code"
                >
                    {copied ? (
                        <>
                            <Check className="w-3 h-3" />
                            <span>Copied!</span>
                        </>
                    ) : (
                        <>
                            <Copy className="w-3 h-3" />
                            <span>Copy</span>
                        </>
                    )}
                </button>
            </div>

            {/* Code content */}
            <SyntaxHighlighter
                language={lang || 'text'}
                style={customTheme}
                showLineNumbers={showLineNumbers && lines > 3}
                lineNumberStyle={{
                    color: '#555',
                    fontSize: '0.7rem',
                    minWidth: '2.5em',
                    paddingRight: '1em',
                    userSelect: 'none',
                }}
                wrapLines
                customStyle={{
                    margin: 0,
                    background: '#1a1a19',
                }}
            >
                {code}
            </SyntaxHighlighter>
        </div>
    );
};

/**
 * Markdown code renderer for ReactMarkdown `components` prop.
 * Handles both inline code and fenced code blocks.
 */
export const markdownCodeComponents = {
    code({
        className,
        children,
        ...props
    }: React.HTMLAttributes<HTMLElement> & { node?: unknown }) {
        const match = /language-(\w+)/.exec(className || '');
        const isInline = !match && !className;
        const codeString = String(children).replace(/\n$/, '');

        if (isInline) {
            return (
                <code
                    className="text-brand-orange bg-brand-orange/10 px-1.5 py-0.5 rounded text-[0.8em] font-mono"
                    {...props}
                >
                    {children}
                </code>
            );
        }

        return <CodeBlock code={codeString} language={match?.[1] || ''} />;
    },
    pre({ children }: { children?: React.ReactNode }) {
        // The CodeBlock is rendered inside <code>, so <pre> just passes through
        return <>{children}</>;
    },
};
