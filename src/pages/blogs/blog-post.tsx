import { Header } from "@/components/custom/header";
import { markdownCodeComponents } from "@/components/custom/code-block";
import { useParams, useNavigate } from "react-router-dom";
import { getBlogBySlug } from "@/data/blog-posts";
import { ArrowLeft, Calendar, Clock, Tag, ChevronDown, List, ExternalLink } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useState, useEffect, useRef, useCallback } from "react";

// Extract headings from markdown content
function extractHeadings(content: string): { id: string; text: string; level: number }[] {
  const headingRegex = /^(#{1,3})\s+(.+)$/gm;
  const headings: { id: string; text: string; level: number }[] = [];
  let match;
  while ((match = headingRegex.exec(content)) !== null) {
    const level = match[1].length;
    const text = match[2].replace(/\*\*/g, '').replace(/\*/g, '').trim();
    const id = text
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-');
    headings.push({ id, text, level });
  }
  return headings;
}

export const BlogPost = () => {
  const { slug } = useParams<{ slug: string }>();
  const navigate = useNavigate();
  const post = slug ? getBlogBySlug(slug) : undefined;
  const [contentOpen, setContentOpen] = useState(false);
  const [activeHeading, setActiveHeading] = useState<string>("");
  const [tocOpen, setTocOpen] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  const headings = post ? extractHeadings(post.content) : [];

  // Observe heading intersections for active TOC highlight
  const observeHeadings = useCallback(() => {
    if (!contentOpen || !contentRef.current) return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveHeading(entry.target.id);
            break;
          }
        }
      },
      { rootMargin: "-80px 0px -60% 0px", threshold: 0.1 }
    );

    const headingElements = contentRef.current.querySelectorAll("h1, h2, h3");
    headingElements.forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, [contentOpen]);

  useEffect(() => {
    const cleanup = observeHeadings();
    return () => cleanup?.();
  }, [observeHeadings]);

  if (!post) {
    return (
      <div className="flex flex-col min-h-dvh bg-background">
        <Header />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <h1 className="text-2xl font-bold font-heading text-foreground mb-4">
              Post Not Found
            </h1>
            <button
              onClick={() => navigate("/blogs")}
              className="text-brand-orange hover:underline font-heading text-sm"
            >
              Back to Blogs
            </button>
          </div>
        </main>
      </div>
    );
  }

  const formatDate = (dateString: string) =>
    new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });

  const scrollToHeading = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
      setActiveHeading(id);
      setTocOpen(false);
    }
  };

  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      <main className="flex-1 overflow-y-auto pb-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
          <div className="flex gap-8">

            {/* ── Main Article Content ── */}
            <article className="flex-1 min-w-0 max-w-4xl">
              {/* Back Button */}
              <button
                onClick={() => navigate("/blogs")}
                className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground mb-4 transition-colors font-heading"
              >
                <ArrowLeft className="w-3 h-3" />
                Back to Blogs
              </button>

              {/* Post Header */}
              <header className="mb-6">
                <h1 className="text-2xl sm:text-3xl font-bold font-heading text-foreground mb-2 leading-tight">
                  {post.title}
                </h1>

                <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground mb-3">
                  <div className="flex items-center gap-1">
                    <Calendar className="w-3 h-3" />
                    <span>{formatDate(post.date)}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    <span>{post.readingTime} read</span>
                  </div>
                </div>

                <div className="flex flex-wrap gap-1.5">
                  {post.tags.map((tag) => (
                    <span
                      key={tag}
                      className="flex items-center gap-1 px-2 py-0.5 text-xs bg-brand-orange/10 text-brand-orange rounded font-heading"
                    >
                      <Tag className="w-2.5 h-2.5" />
                      {tag}
                    </span>
                  ))}
                </div>
              </header>

              {/* Cover Image */}
              {post.coverImage && (
                <div className="mb-8 rounded-xl overflow-hidden border border-border">
                  <img
                    src={post.coverImage}
                    alt={post.title}
                    className="w-full h-auto max-h-[400px] object-cover"
                  />
                </div>
              )}

              {/* ─── TL;DR — rendered as clean blog prose ─── */}
              <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:font-heading prose-headings:text-foreground prose-p:text-muted-foreground prose-strong:text-foreground mb-10">
                <p className="text-muted-foreground leading-relaxed">
                  <strong>The Problem:</strong> {post.tldr.problem}
                </p>
                <p className="text-muted-foreground leading-relaxed">
                  <strong>The Idea:</strong> {post.tldr.idea}
                </p>
                <p className="text-muted-foreground leading-relaxed">
                  <strong>My Solution:</strong> {post.tldr.solution}
                </p>
                <p className="text-muted-foreground leading-relaxed">
                  <strong>The Vision:</strong> {post.tldr.vision}
                </p>
              </div>

              {/* ─── Full Content Toggle ─── */}
              <div className="border border-border rounded-xl overflow-hidden">
                <button
                  onClick={() => setContentOpen(!contentOpen)}
                  className="w-full flex items-center justify-between px-5 py-3.5 bg-muted/40 hover:bg-muted/70 transition-colors text-left group"
                >
                  <div>
                    <span className="text-sm font-medium text-foreground">
                      {contentOpen ? "Hide full article" : "Read the full article"}
                    </span>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      Honestly this part is more for AI agents and search crawlers than for humans — but if you want the deep dive with code examples and all, knock yourself out.
                    </p>
                  </div>
                  <ChevronDown
                    className={`w-4 h-4 text-muted-foreground shrink-0 ml-3 transition-transform duration-200 ${contentOpen ? "rotate-180" : ""}`}
                  />
                </button>

                {contentOpen && (
                  <div className="px-5 py-6 border-t border-border" ref={contentRef}>
                    <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:font-heading prose-headings:text-foreground prose-p:text-muted-foreground prose-a:text-brand-orange prose-strong:text-foreground prose-blockquote:border-brand-orange prose-blockquote:text-muted-foreground prose-li:text-muted-foreground prose-table:border-border prose-th:bg-muted prose-th:border-border prose-td:border-border">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          ...markdownCodeComponents,
                          // Generate IDs for headings so TOC links work
                          h1({ children }) {
                            const text = String(children).replace(/\*\*/g, '').replace(/\*/g, '');
                            const id = text.toLowerCase().replace(/[^a-z0-9\s-]/g, '').replace(/\s+/g, '-').replace(/-+/g, '-');
                            return <h1 id={id}>{children}</h1>;
                          },
                          h2({ children }) {
                            const text = String(children).replace(/\*\*/g, '').replace(/\*/g, '');
                            const id = text.toLowerCase().replace(/[^a-z0-9\s-]/g, '').replace(/\s+/g, '-').replace(/-+/g, '-');
                            return <h2 id={id}>{children}</h2>;
                          },
                          h3({ children }) {
                            const text = String(children).replace(/\*\*/g, '').replace(/\*/g, '');
                            const id = text.toLowerCase().replace(/[^a-z0-9\s-]/g, '').replace(/\s+/g, '-').replace(/-+/g, '-');
                            return <h3 id={id}>{children}</h3>;
                          },
                          table({ children }) {
                            return (
                              <div className="overflow-x-auto my-6 rounded-lg border border-border shadow-sm">
                                <table className="min-w-full divide-y divide-border">
                                  {children}
                                </table>
                              </div>
                            );
                          },
                          thead({ children }) {
                            return (
                              <thead className="bg-muted/70">
                                {children}
                              </thead>
                            );
                          },
                          tbody({ children }) {
                            return (
                              <tbody className="divide-y divide-border [&>tr:nth-child(even)]:bg-muted/30 [&>tr:hover]:bg-muted/50 transition-colors">
                                {children}
                              </tbody>
                            );
                          },
                          th({ children }) {
                            return (
                              <th className="px-4 py-2.5 text-left text-xs font-semibold font-heading text-foreground uppercase tracking-wider border-b border-border">
                                {children}
                              </th>
                            );
                          },
                          td({ children }) {
                            return (
                              <td className="px-4 py-2.5 text-sm text-muted-foreground whitespace-normal">
                                {children}
                              </td>
                            );
                          },
                          a({ href, children }) {
                            const isExternal = href?.startsWith('http');
                            return (
                              <a
                                href={href}
                                target={isExternal ? "_blank" : undefined}
                                rel={isExternal ? "noopener noreferrer" : undefined}
                                className="text-brand-orange hover:text-brand-orange/80 underline decoration-brand-orange/30 hover:decoration-brand-orange transition-colors inline-flex items-center gap-0.5"
                              >
                                {children}
                                {isExternal && <ExternalLink className="w-3 h-3 inline-block" />}
                              </a>
                            );
                          },
                        }}
                      >
                        {post.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                )}
              </div>

              {/* Author Section */}
              <div className="mt-8 pt-4 border-t border-border">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-brand-orange to-brand-clay flex items-center justify-center">
                    <span className="text-white font-bold text-xs">ZH</span>
                  </div>
                  <div>
                    <p className="text-sm font-semibold font-heading text-foreground">
                      {post.author.name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      PhD Student at USC · AI Researcher
                    </p>
                  </div>
                </div>
              </div>
            </article>

            {/* ── Right Sidebar: Table of Contents ── */}
            {headings.length > 0 && (
              <aside className="hidden xl:block w-52 flex-shrink-0">
                <div className="sticky top-6">
                  <div className="flex items-center gap-1.5 mb-3">
                    <List className="w-3.5 h-3.5 text-muted-foreground" />
                    <h3 className="text-[10px] uppercase tracking-wider font-heading text-muted-foreground">
                      On this page
                    </h3>
                  </div>

                  {!contentOpen ? (
                    <p className="text-[11px] text-muted-foreground/60 italic">
                      Expand the full article to navigate sections
                    </p>
                  ) : (
                    <nav className="space-y-0.5 border-l-2 border-border">
                      {headings.map((heading) => (
                        <button
                          key={heading.id}
                          onClick={() => scrollToHeading(heading.id)}
                          className={`block w-full text-left text-[11px] leading-snug transition-all duration-150 border-l-2 -ml-[2px] ${heading.level === 1 ? "pl-3 py-1 font-medium" : ""
                            }${heading.level === 2 ? "pl-3 py-0.5" : ""}${heading.level === 3 ? "pl-5 py-0.5" : ""
                            } ${activeHeading === heading.id
                              ? "border-brand-orange text-brand-orange"
                              : "border-transparent text-muted-foreground hover:text-foreground hover:border-muted-foreground/30"
                            }`}
                        >
                          {heading.text}
                        </button>
                      ))}
                    </nav>
                  )}
                </div>
              </aside>
            )}
          </div>
        </div>

        {/* ── Mobile TOC FAB ── */}
        {contentOpen && headings.length > 0 && (
          <>
            <button
              onClick={() => setTocOpen(!tocOpen)}
              className="xl:hidden fixed bottom-20 right-4 z-40 w-10 h-10 rounded-full bg-brand-orange text-white shadow-lg flex items-center justify-center hover:bg-brand-orange/90 transition-colors"
              title="Table of Contents"
            >
              <List className="w-4 h-4" />
            </button>

            {tocOpen && (
              <>
                <div className="xl:hidden fixed inset-0 z-40 bg-brand-dark/20 backdrop-blur-[2px]" onClick={() => setTocOpen(false)} />
                <div className="xl:hidden fixed bottom-32 right-4 z-50 w-64 max-h-[50vh] overflow-y-auto bg-card border border-border rounded-xl shadow-elevation-4 dark:shadow-elevation-4-dark p-4">
                  <h3 className="text-[10px] uppercase tracking-wider font-heading text-muted-foreground mb-2">
                    On this page
                  </h3>
                  <nav className="space-y-0.5">
                    {headings.map((heading) => (
                      <button
                        key={heading.id}
                        onClick={() => scrollToHeading(heading.id)}
                        className={`block w-full text-left text-xs py-1 transition-colors ${heading.level === 1 ? "font-medium" : ""
                          }${heading.level === 3 ? "pl-3" : ""} ${activeHeading === heading.id
                            ? "text-brand-orange"
                            : "text-muted-foreground hover:text-foreground"
                          }`}
                      >
                        {heading.text}
                      </button>
                    ))}
                  </nav>
                </div>
              </>
            )}
          </>
        )}
      </main>
    </div>
  );
};
