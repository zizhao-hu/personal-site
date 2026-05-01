'use client';

import { Header } from "@/components/custom/header";
import { markdownCodeComponents } from "@/components/custom/code-block";
import { useParams, useRouter } from 'next/navigation';
import { getBlogBySlug, blogPosts, type BlogPost as BlogPostType } from "@/data/blog-posts";
import { ArrowLeft, Clock, ExternalLink } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const formatLongDate = (dateString: string) =>
  new Date(dateString).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

const categoryLabel = (c: BlogPostType["category"]) =>
  c === "ai" ? "AI & ML" : c === "science" ? "Science" : "Economy";

export const BlogPost = ({ slugOverride }: { slugOverride?: string }) => {
  const params = useParams();
  const slug = slugOverride || (params?.slug as string);
  const router = useRouter();
  const post = slug ? getBlogBySlug(slug) : undefined;

  if (!post) {
    return (
      <div className="flex flex-col min-h-dvh bg-background">
        <Header />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <h1 className="text-2xl font-semibold font-sans text-foreground mb-4">
              Post Not Found
            </h1>
            <button
              onClick={() => router.push("/blogs")}
              className="text-brand-orange hover:underline text-sm"
            >
              Back to Blogs
            </button>
          </div>
        </main>
      </div>
    );
  }

  const related = blogPosts
    .filter((p) => p.slug !== post.slug)
    .filter((p) => p.category === post.category || p.tags.some((t) => post.tags.includes(t)))
    .slice(0, 3);

  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      <main className="flex-1 overflow-y-auto pb-24">
        {/* ── Hero (centered, LangChain-style) ── */}
        <section className="border-b border-border">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 pt-8 pb-12 text-center">
            <button
              onClick={() => router.push("/blogs")}
              className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground mb-8 transition-colors"
            >
              <ArrowLeft className="w-3 h-3" />
              Back to Blogs
            </button>

            <div className="mb-6">
              <span className="inline-block text-[10.5px] uppercase tracking-[0.18em] text-muted-foreground border border-border bg-background px-2.5 py-1">
                {categoryLabel(post.category)}
              </span>
            </div>

            <h1 className="font-sans text-3xl sm:text-4xl md:text-5xl font-semibold tracking-tight text-foreground leading-[1.1] mb-8 text-balance max-w-3xl mx-auto">
              {post.title}
            </h1>

            <div className="flex items-center justify-center gap-5 flex-wrap">
              <div className="flex items-center gap-2.5">
                <img
                  src={post.author.avatar}
                  alt={post.author.name}
                  className="w-9 h-9 rounded-full object-cover border border-border"
                  onError={(e) => {
                    const t = e.currentTarget;
                    t.style.display = "none";
                  }}
                />
                <div className="text-left">
                  <div className="text-[13px] font-medium text-foreground leading-tight">
                    {post.author.name}
                  </div>
                  <div className="text-[11px] text-muted-foreground">
                    {formatLongDate(post.date)}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
                <Clock className="w-3 h-3" />
                {post.readingTime}
              </div>
            </div>
          </div>
        </section>

        {/* ── Article body ── */}
        <article className="max-w-3xl mx-auto px-4 sm:px-6 pt-10 pb-12">
          {post.coverImage && (
            <img
              src={post.coverImage}
              alt={post.title}
              className="w-full h-auto rounded-lg border border-border mb-12"
            />
          )}

          <div
            className="prose prose-lg dark:prose-invert max-w-none font-sans
              prose-headings:font-sans prose-headings:font-semibold prose-headings:tracking-tight prose-headings:text-foreground
              prose-h1:text-3xl prose-h1:mt-12 prose-h1:mb-4
              prose-h2:text-2xl prose-h2:mt-12 prose-h2:mb-4
              prose-h3:text-xl prose-h3:mt-10 prose-h3:mb-3
              prose-p:text-foreground/85 prose-p:leading-[1.8]
              prose-strong:text-foreground prose-strong:font-semibold
              prose-em:text-foreground/85
              prose-a:text-brand-orange prose-a:no-underline hover:prose-a:underline
              prose-li:text-foreground/85 prose-li:leading-[1.8]
              prose-blockquote:border-l-2 prose-blockquote:border-brand-orange prose-blockquote:text-muted-foreground prose-blockquote:not-italic prose-blockquote:font-normal
              prose-img:rounded-lg prose-img:border prose-img:border-border prose-img:my-8
              prose-hr:border-border
              prose-code:text-foreground prose-code:bg-muted prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-[0.9em] prose-code:font-mono prose-code:before:content-[''] prose-code:after:content-['']
            "
          >
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                ...markdownCodeComponents,
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
                    <div className="overflow-x-auto my-8 rounded-lg border border-border">
                      <table className="min-w-full divide-y divide-border">{children}</table>
                    </div>
                  );
                },
                thead({ children }) {
                  return <thead className="bg-muted/60">{children}</thead>;
                },
                tbody({ children }) {
                  return (
                    <tbody className="divide-y divide-border">
                      {children}
                    </tbody>
                  );
                },
                th({ children }) {
                  return (
                    <th className="px-4 py-2.5 text-left text-[11px] font-semibold text-foreground uppercase tracking-wider">
                      {children}
                    </th>
                  );
                },
                td({ children }) {
                  return (
                    <td className="px-4 py-2.5 text-[14px] text-foreground/85">
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
        </article>

        {/* ── Related content ── */}
        {related.length > 0 && (
          <section className="border-t border-border bg-muted/20">
            <div className="max-w-5xl mx-auto px-4 sm:px-6 py-12">
              <h3 className="font-sans text-2xl font-semibold tracking-tight text-foreground mb-6">
                Related content
              </h3>
              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {related.map((r) => (
                  <button
                    key={r.slug}
                    onClick={() => router.push(`/blogs/${r.slug}`)}
                    className="group text-left bg-background border border-border hover:border-foreground/30 transition-colors overflow-hidden flex flex-col"
                  >
                    {r.coverImage && (
                      <img
                        src={r.coverImage}
                        alt={r.title}
                        className="w-full aspect-[16/9] object-cover border-b border-border"
                      />
                    )}
                    <div className="p-4 flex-1 flex flex-col gap-3">
                      <span className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                        {categoryLabel(r.category)}
                      </span>
                      <h4 className="font-sans text-[16px] font-semibold leading-snug text-foreground group-hover:text-brand-orange transition-colors">
                        {r.title}
                      </h4>
                      <div className="mt-auto flex items-center justify-between text-[11px] text-muted-foreground">
                        <span>{r.author.name}</span>
                        <span className="inline-flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {r.readingTime}
                        </span>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
};
