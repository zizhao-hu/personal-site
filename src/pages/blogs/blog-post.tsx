import { Header } from "@/components/custom/header";
import { markdownCodeComponents } from "@/components/custom/code-block";
import { useParams, useNavigate } from "react-router-dom";
import { getBlogBySlug } from "@/data/blog-posts";
import { ArrowLeft, Calendar, Clock, Tag } from "lucide-react";
import ReactMarkdown from "react-markdown";

export const BlogPost = () => {
  const { slug } = useParams<{ slug: string }>();
  const navigate = useNavigate();
  const post = slug ? getBlogBySlug(slug) : undefined;

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

  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      <main className="flex-1 overflow-y-auto pb-24">
        <article className="max-w-4xl mx-auto px-4 sm:px-6 py-6">
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
                  className="flex items-center gap-1 px-2 py-0.5 text-xs bg-brand-orange/10 text-brand-orange rounded"
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

          {/* Post Content */}
          <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:font-heading prose-headings:text-foreground prose-p:text-muted-foreground prose-a:text-brand-orange prose-strong:text-foreground prose-blockquote:border-brand-orange prose-blockquote:text-muted-foreground prose-li:text-muted-foreground prose-table:border-border prose-th:bg-muted prose-th:border-border prose-td:border-border">
            <ReactMarkdown
              components={{
                ...markdownCodeComponents,
                table({ children }) {
                  return (
                    <div className="overflow-x-auto my-6">
                      <table className="min-w-full border border-border">
                        {children}
                      </table>
                    </div>
                  );
                },
                th({ children }) {
                  return (
                    <th className="px-4 py-2 bg-muted border border-border text-left font-semibold font-heading text-sm">
                      {children}
                    </th>
                  );
                },
                td({ children }) {
                  return (
                    <td className="px-4 py-2 border border-border text-sm">
                      {children}
                    </td>
                  );
                },
              }}
            >
              {post.content}
            </ReactMarkdown>
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
      </main>
    </div>
  );
};
