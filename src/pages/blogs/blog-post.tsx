import { Header } from "../../components/custom/header";
import { useParams, useNavigate } from "react-router-dom";
import { getBlogBySlug } from "../../data/blog-posts";
import { ArrowLeft, Calendar, Clock, Tag } from "lucide-react";
import ReactMarkdown from "react-markdown";

export const BlogPost = () => {
  const { slug } = useParams<{ slug: string }>();
  const navigate = useNavigate();
  const post = slug ? getBlogBySlug(slug) : undefined;

  if (!post) {
    return (
      <div className="h-full flex flex-col">
        <Header />
        <main className="flex-1 overflow-auto flex items-center justify-center">
          <div className="text-center">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              Post Not Found
            </h1>
            <button
              onClick={() => navigate("/blogs")}
              className="text-blue-600 dark:text-blue-400 hover:underline"
            >
              Back to Blogs
            </button>
          </div>
        </main>
      </div>
    );
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  return (
    <div className="h-full flex flex-col">
      <Header />
      <main className="flex-1 overflow-auto">
        <article className="max-w-4xl mx-auto px-4 sm:px-6 py-8">
          {/* Back Button */}
          <button
            onClick={() => navigate("/blogs")}
            className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white mb-6 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Blogs
          </button>

          {/* Post Header */}
          <header className="mb-8">
            <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-4 leading-tight">
              {post.title}
            </h1>

            <div className="flex flex-wrap items-center gap-4 text-sm text-gray-500 dark:text-gray-400 mb-4">
              <div className="flex items-center gap-1">
                <Calendar className="w-4 h-4" />
                <span>{formatDate(post.date)}</span>
              </div>
              <div className="flex items-center gap-1">
                <Clock className="w-4 h-4" />
                <span>{post.readingTime} read</span>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              {post.tags.map((tag) => (
                <span
                  key={tag}
                  className="flex items-center gap-1 px-3 py-1 text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full"
                >
                  <Tag className="w-3 h-3" />
                  {tag}
                </span>
              ))}
            </div>
          </header>

          {/* Post Content */}
          <div className="prose prose-lg dark:prose-invert max-w-none prose-headings:text-gray-900 dark:prose-headings:text-white prose-p:text-gray-600 dark:prose-p:text-gray-300 prose-a:text-blue-600 dark:prose-a:text-blue-400 prose-code:text-pink-600 dark:prose-code:text-pink-400 prose-code:bg-gray-100 dark:prose-code:bg-gray-800 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-pre:bg-gray-900 dark:prose-pre:bg-gray-950 prose-pre:text-gray-100 prose-strong:text-gray-900 dark:prose-strong:text-white prose-blockquote:border-blue-500 prose-blockquote:text-gray-600 dark:prose-blockquote:text-gray-400 prose-table:border-gray-200 dark:prose-table:border-gray-700 prose-th:bg-gray-50 dark:prose-th:bg-gray-800 prose-th:border-gray-200 dark:prose-th:border-gray-700 prose-td:border-gray-200 dark:prose-td:border-gray-700">
            <ReactMarkdown
              components={{
                code({ node, className, children, ...props }) {
                  const isInline = !className;
                  if (isInline) {
                    return (
                      <code className="text-pink-600 dark:text-pink-400 bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded text-sm" {...props}>
                        {children}
                      </code>
                    );
                  }
                  return (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                },
                pre({ children }) {
                  return (
                    <pre className="bg-gray-900 dark:bg-gray-950 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
                      {children}
                    </pre>
                  );
                },
                table({ children }) {
                  return (
                    <div className="overflow-x-auto my-6">
                      <table className="min-w-full border border-gray-200 dark:border-gray-700">
                        {children}
                      </table>
                    </div>
                  );
                },
                th({ children }) {
                  return (
                    <th className="px-4 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-left font-semibold">
                      {children}
                    </th>
                  );
                },
                td({ children }) {
                  return (
                    <td className="px-4 py-2 border border-gray-200 dark:border-gray-700">
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
          <div className="mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center">
                <span className="text-white font-bold">ZH</span>
              </div>
              <div>
                <p className="font-semibold text-gray-900 dark:text-white">
                  {post.author.name}
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-400">
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
