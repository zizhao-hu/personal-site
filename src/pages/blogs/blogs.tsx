import { Header } from "../../components/custom/header";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Clock, Calendar, Tag, ArrowRight } from "lucide-react";
import { blogPosts } from "../../data/blog-posts";

const categories = [
  { id: "all", label: "All" },
  { id: "ai", label: "AI" },
  { id: "science", label: "Science" },
  { id: "economy", label: "Economy" },
];

export const Blogs = () => {
  const navigate = useNavigate();
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [selectedTags, setSelectedTags] = useState<string[]>([]);

  const allTags = [...new Set(blogPosts.flatMap((post) => post.tags))];

  const filteredPosts = blogPosts.filter((post) => {
    const categoryMatch = activeCategory === "all" || post.category === activeCategory;
    const tagMatch = selectedTags.length === 0 || selectedTags.some((tag) => post.tags.includes(tag));
    return categoryMatch && tagMatch;
  });

  const toggleTag = (tag: string) => {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  return (
    <div className="h-full flex flex-col">
      <Header />
      <main className="flex-1 overflow-auto">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6">
          {/* Page Header */}
          <div className="mb-5">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
              Blog
            </h1>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Thoughts on AI research, machine learning, and technology
            </p>
          </div>

          {/* Category Tabs */}
          <div className="border-b border-gray-200 dark:border-gray-700 mb-4">
            <nav className="flex gap-0">
              {categories.map((category) => (
                <button
                  key={category.id}
                  onClick={() => setActiveCategory(category.id)}
                  className={`relative px-4 py-2 text-xs font-medium transition-colors duration-200 ${
                    activeCategory === category.id
                      ? "text-blue-600 dark:text-blue-400"
                      : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
                  }`}
                >
                  {category.label}
                  {activeCategory === category.id && (
                    <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600 dark:bg-blue-400" />
                  )}
                </button>
              ))}
            </nav>
          </div>

          {/* Tag Filters */}
          <div className="mb-4">
            <div className="flex items-center gap-1.5 flex-wrap">
              <Tag className="w-3.5 h-3.5 text-gray-500 dark:text-gray-400" />
              <span className="text-xs text-gray-500 dark:text-gray-400 mr-1">Filter:</span>
              {allTags.map((tag) => (
                <button
                  key={tag}
                  onClick={() => toggleTag(tag)}
                  className={`px-2 py-0.5 text-xs rounded-full transition-all duration-200 ${
                    selectedTags.includes(tag)
                      ? "bg-blue-600 text-white"
                      : "bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
                  }`}
                >
                  {tag}
                </button>
              ))}
              {selectedTags.length > 0 && (
                <button
                  onClick={() => setSelectedTags([])}
                  className="px-2 py-0.5 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
                >
                  Clear
                </button>
              )}
            </div>
          </div>

          {/* Blog Cards */}
          <div className="grid gap-3">
            {filteredPosts.map((post) => (
              <article
                key={post.id}
                onClick={() => navigate(`/blogs/${post.slug}`)}
                className="group bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
              >
                <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
                  <div className="flex-1">
                    <h2 className="text-base font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors mb-1">
                      {post.title}
                    </h2>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2 line-clamp-2">
                      {post.excerpt}
                    </p>
                    <div className="flex flex-wrap gap-1 mb-2">
                      {post.tags.map((tag) => (
                        <span
                          key={tag}
                          className="px-1.5 py-0.5 text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                    <div className="flex items-center gap-1 text-blue-600 dark:text-blue-400 text-xs font-medium">
                      Read article
                      <ArrowRight className="w-3 h-3 group-hover:translate-x-1 transition-transform" />
                    </div>
                  </div>
                  <div className="flex sm:flex-col items-center sm:items-end gap-3 sm:gap-1 text-xs text-gray-500 dark:text-gray-400">
                    <div className="flex items-center gap-1">
                      <Calendar className="w-3 h-3" />
                      <span>{formatDate(post.date)}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      <span>{post.readingTime}</span>
                    </div>
                  </div>
                </div>
              </article>
            ))}
          </div>

          {filteredPosts.length === 0 && (
            <div className="text-center py-12">
              <p className="text-gray-500 dark:text-gray-400">
                No blog posts found matching your filters.
              </p>
              <button
                onClick={() => {
                  setActiveCategory("all");
                  setSelectedTags([]);
                }}
                className="mt-4 text-blue-600 dark:text-blue-400 hover:underline"
              >
                Clear all filters
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};
