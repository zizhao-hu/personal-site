import { Header } from "../../components/custom/header";
import { useState } from "react";
import { Clock, Calendar, Tag } from "lucide-react";

interface BlogPost {
  id: string;
  title: string;
  excerpt: string;
  date: string;
  readingTime: string;
  category: "ai" | "science" | "economy";
  tags: string[];
  slug: string;
}

const blogPosts: BlogPost[] = [
  {
    id: "1",
    title: "The Future of Large Language Models in Scientific Research",
    excerpt: "Exploring how LLMs are transforming the way we approach scientific discovery and research methodology.",
    date: "2024-01-15",
    readingTime: "8 min",
    category: "ai",
    tags: ["LLM", "Research", "Machine Learning"],
    slug: "llm-scientific-research",
  },
  {
    id: "2",
    title: "Understanding Continual Learning in Neural Networks",
    excerpt: "A deep dive into catastrophic forgetting and modern approaches to lifelong learning in AI systems.",
    date: "2024-01-10",
    readingTime: "12 min",
    category: "ai",
    tags: ["Continual Learning", "Neural Networks", "Deep Learning"],
    slug: "continual-learning-networks",
  },
  {
    id: "3",
    title: "Quantum Computing: From Theory to Practice",
    excerpt: "The current state of quantum computing and its potential impact on computational science.",
    date: "2024-01-05",
    readingTime: "10 min",
    category: "science",
    tags: ["Quantum Computing", "Physics", "Technology"],
    slug: "quantum-computing-practice",
  },
  {
    id: "4",
    title: "The Economics of AI: Market Disruption and Opportunity",
    excerpt: "Analyzing the economic implications of AI adoption across industries and its effect on labor markets.",
    date: "2023-12-28",
    readingTime: "7 min",
    category: "economy",
    tags: ["Economics", "AI Impact", "Labor Market"],
    slug: "economics-of-ai",
  },
  {
    id: "5",
    title: "Multi-Modal Learning: Bridging Vision and Language",
    excerpt: "How modern AI systems learn to understand and connect visual and textual information.",
    date: "2023-12-20",
    readingTime: "9 min",
    category: "ai",
    tags: ["Vision", "NLP", "Multi-Modal"],
    slug: "multimodal-learning",
  },
  {
    id: "6",
    title: "Climate Science and Machine Learning: A Partnership",
    excerpt: "Using ML techniques to improve climate modeling and environmental predictions.",
    date: "2023-12-15",
    readingTime: "11 min",
    category: "science",
    tags: ["Climate", "Machine Learning", "Environment"],
    slug: "climate-ml-partnership",
  },
];

const categories = [
  { id: "all", label: "All" },
  { id: "ai", label: "AI" },
  { id: "science", label: "Science" },
  { id: "economy", label: "Economy" },
];

export const Blogs = () => {
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
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8">
          {/* Category Tabs */}
          <div className="border-b border-gray-200 dark:border-gray-700 mb-6">
            <nav className="flex gap-0">
              {categories.map((category) => (
                <button
                  key={category.id}
                  onClick={() => setActiveCategory(category.id)}
                  className={`relative px-6 py-3 text-sm font-medium transition-colors duration-200 ${
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
          <div className="mb-6">
            <div className="flex items-center gap-2 flex-wrap">
              <Tag className="w-4 h-4 text-gray-500 dark:text-gray-400" />
              <span className="text-sm text-gray-500 dark:text-gray-400 mr-2">Filter by tag:</span>
              {allTags.map((tag) => (
                <button
                  key={tag}
                  onClick={() => toggleTag(tag)}
                  className={`px-3 py-1 text-xs rounded-full transition-all duration-200 ${
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
                  className="px-3 py-1 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
                >
                  Clear all
                </button>
              )}
            </div>
          </div>

          {/* Blog Cards */}
          <div className="grid gap-6">
            {filteredPosts.map((post) => (
              <article
                key={post.id}
                className="group bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6 hover:shadow-md dark:hover:shadow-gray-900/50 transition-all duration-200 cursor-pointer"
              >
                <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
                  <div className="flex-1">
                    <h2 className="text-xl font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors mb-2">
                      {post.title}
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400 mb-4 line-clamp-2">
                      {post.excerpt}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {post.tags.map((tag) => (
                        <span
                          key={tag}
                          className="px-2 py-0.5 text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="flex sm:flex-col items-center sm:items-end gap-4 sm:gap-2 text-sm text-gray-500 dark:text-gray-400">
                    <div className="flex items-center gap-1">
                      <Calendar className="w-4 h-4" />
                      <span>{formatDate(post.date)}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
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
