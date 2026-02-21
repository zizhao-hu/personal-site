import { Header } from "../../components/custom/header";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Clock, Calendar, Tag, LayoutGrid, List, ChevronRight } from "lucide-react";
import { blogPosts } from "../../data/blog-posts";
import { tagPillClass, tagBadgeClass } from "../../lib/tag-colors";

const categories = [
  { id: "all", label: "All Posts", count: blogPosts.length },
  { id: "ai", label: "AI & Machine Learning", count: blogPosts.filter(p => p.category === "ai").length },
  { id: "science", label: "Science", count: blogPosts.filter(p => p.category === "science").length },
  { id: "economy", label: "Economy", count: blogPosts.filter(p => p.category === "economy").length },
];

const allTags = [...new Set(blogPosts.flatMap((post) => post.tags))];

// Count posts per tag
const tagCounts = allTags.reduce((acc, tag) => {
  acc[tag] = blogPosts.filter(p => p.tags.includes(tag)).length;
  return acc;
}, {} as Record<string, number>);

export const Blogs = () => {
  const navigate = useNavigate();
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<"grid" | "list">("list");

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
    <div className="h-full flex flex-col bg-background">
      <Header />
      <main className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
          {/* Page Header */}
          <div className="mb-6">
            <h1 className="text-2xl font-bold font-heading text-foreground mb-1">
              Blog
            </h1>
            <p className="text-sm text-muted-foreground">
              Thoughts on AI research, machine learning, safety, and technology
            </p>
          </div>

          {/* Main Layout: Sidebar + Content */}
          <div className="flex gap-6">

            {/* ── Left Sidebar ── */}
            <aside className="hidden lg:block w-56 flex-shrink-0">
              <div className="sticky top-6 space-y-6">

                {/* Categories */}
                <div>
                  <h3 className="text-[10px] uppercase tracking-wider font-heading text-muted-foreground mb-2 px-2">
                    Categories
                  </h3>
                  <nav className="space-y-0.5">
                    {categories.map((cat) => (
                      <button
                        key={cat.id}
                        onClick={() => setActiveCategory(cat.id)}
                        className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs transition-all duration-150 ${activeCategory === cat.id
                          ? "bg-brand-orange/10 text-brand-orange font-medium"
                          : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                          }`}
                      >
                        <span className="font-heading">{cat.label}</span>
                        <span className={`text-[10px] tabular-nums ${activeCategory === cat.id ? "text-brand-orange" : "text-muted-foreground/60"
                          }`}>
                          {cat.count}
                        </span>
                      </button>
                    ))}
                  </nav>
                </div>

                {/* Divider */}
                <div className="h-px bg-border" />

                {/* Tags */}
                <div>
                  <div className="flex items-center justify-between mb-2 px-2">
                    <h3 className="text-[10px] uppercase tracking-wider font-heading text-muted-foreground">
                      Tags
                    </h3>
                    {selectedTags.length > 0 && (
                      <button
                        onClick={() => setSelectedTags([])}
                        className="text-[10px] text-brand-orange hover:underline font-heading"
                      >
                        Clear
                      </button>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-1.5 px-1">
                    {allTags.map((tag) => (
                      <button
                        key={tag}
                        onClick={() => toggleTag(tag)}
                        className={`px-2 py-0.5 text-[10px] rounded-full transition-all duration-150 font-heading ${tagPillClass(tag, selectedTags.includes(tag))}`}
                      >
                        {tag}
                        <span className="ml-1 opacity-60">{tagCounts[tag]}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Divider */}
                <div className="h-px bg-border" />

                {/* Quick Stats */}
                <div className="px-2">
                  <h3 className="text-[10px] uppercase tracking-wider font-heading text-muted-foreground mb-2">
                    Stats
                  </h3>
                  <div className="space-y-1.5 text-xs text-muted-foreground">
                    <div className="flex justify-between">
                      <span>Total posts</span>
                      <span className="font-medium text-foreground">{blogPosts.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Categories</span>
                      <span className="font-medium text-foreground">{categories.length - 1}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Tags</span>
                      <span className="font-medium text-foreground">{allTags.length}</span>
                    </div>
                  </div>
                </div>
              </div>
            </aside>

            {/* ── Main Content ── */}
            <div className="flex-1 min-w-0">

              {/* Mobile category tabs + view toggle */}
              <div className="flex items-center justify-between mb-4">
                {/* Mobile tabs */}
                <div className="lg:hidden flex gap-0 border-b border-border overflow-x-auto">
                  {categories.map((category) => (
                    <button
                      key={category.id}
                      onClick={() => setActiveCategory(category.id)}
                      className={`relative px-3 py-2 text-xs font-heading whitespace-nowrap transition-colors ${activeCategory === category.id
                        ? "text-brand-orange"
                        : "text-muted-foreground hover:text-foreground"
                        }`}
                    >
                      {category.label}
                      {activeCategory === category.id && (
                        <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-brand-orange" />
                      )}
                    </button>
                  ))}
                </div>

                {/* View mode toggle */}
                <div className="flex items-center gap-0.5 bg-muted rounded-lg p-0.5">
                  <button
                    onClick={() => setViewMode("list")}
                    className={`p-1.5 rounded-md transition-colors ${viewMode === "list" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"
                      }`}
                    title="List view"
                  >
                    <List className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => setViewMode("grid")}
                    className={`p-1.5 rounded-md transition-colors ${viewMode === "grid" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"
                      }`}
                    title="Grid view"
                  >
                    <LayoutGrid className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Mobile tag filters */}
              <div className="lg:hidden mb-4">
                <div className="flex items-center gap-1.5 flex-wrap">
                  <Tag className="w-3 h-3 text-muted-foreground" />
                  {allTags.slice(0, 8).map((tag) => (
                    <button
                      key={tag}
                      onClick={() => toggleTag(tag)}
                      className={`px-2 py-0.5 text-[10px] rounded-full transition-all font-heading ${tagPillClass(tag, selectedTags.includes(tag))}`}
                    >
                      {tag}
                    </button>
                  ))}
                  {selectedTags.length > 0 && (
                    <button onClick={() => setSelectedTags([])} className="text-[10px] text-brand-orange hover:underline font-heading">
                      Clear
                    </button>
                  )}
                </div>
              </div>

              {/* Active filters display */}
              {(activeCategory !== "all" || selectedTags.length > 0) && (
                <div className="mb-3 flex items-center gap-2 text-xs text-muted-foreground">
                  <span>Showing {filteredPosts.length} of {blogPosts.length} posts</span>
                  {selectedTags.length > 0 && (
                    <div className="flex items-center gap-1">
                      <span>tagged</span>
                      {selectedTags.map(tag => (
                        <span key={tag} className="px-1.5 py-0.5 bg-brand-orange/10 text-brand-orange rounded text-[10px] font-heading">
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Blog Cards */}
              <div className={viewMode === "grid"
                ? "grid grid-cols-1 sm:grid-cols-2 gap-4"
                : "flex flex-col gap-3"
              }>
                {filteredPosts.map((post) => viewMode === "list" ? (
                  /* ── List View ── */
                  <article
                    key={post.id}
                    onClick={() => navigate(`/blogs/${post.slug}`)}
                    className="group bg-card border border-border rounded-xl p-4 hover:shadow-elevation-2 dark:hover:shadow-elevation-2-dark hover:border-brand-orange/20 transition-all cursor-pointer flex flex-col sm:flex-row gap-4"
                  >
                    {post.coverImage && (
                      <div className="w-full sm:w-44 h-36 sm:h-auto rounded-lg overflow-hidden flex-shrink-0 border border-border">
                        <img
                          src={post.coverImage}
                          alt={post.title}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                        />
                      </div>
                    )}
                    <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3 flex-1 min-w-0">
                      <div className="flex-1 min-w-0">
                        <h2 className="text-base font-semibold font-heading text-foreground group-hover:text-brand-orange transition-colors mb-1 leading-snug">
                          {post.title}
                        </h2>
                        <p className="text-sm text-muted-foreground mb-2 line-clamp-2">
                          {post.excerpt}
                        </p>
                        <div className="flex flex-wrap gap-1 mb-2">
                          {post.tags.map((tag) => (
                            <span
                              key={tag}
                              className={`px-1.5 py-0.5 text-[10px] rounded font-heading ${tagBadgeClass(tag)}`}
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                        <div className="flex items-center gap-1 text-brand-orange text-xs font-medium font-heading">
                          Read article
                          <ChevronRight className="w-3 h-3 group-hover:translate-x-1 transition-transform" />
                        </div>
                      </div>
                      <div className="flex flex-row sm:flex-col items-center sm:items-end gap-3 sm:gap-1 text-[10px] text-muted-foreground mt-1 flex-shrink-0">
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
                ) : (
                  /* ── Grid View ── */
                  <article
                    key={post.id}
                    onClick={() => navigate(`/blogs/${post.slug}`)}
                    className="group bg-card border border-border rounded-xl overflow-hidden hover:shadow-elevation-2 dark:hover:shadow-elevation-2-dark hover:border-brand-orange/20 transition-all cursor-pointer flex flex-col"
                  >
                    {post.coverImage && (
                      <div className="w-full h-40 overflow-hidden border-b border-border">
                        <img
                          src={post.coverImage}
                          alt={post.title}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                        />
                      </div>
                    )}
                    <div className="p-4 flex-1 flex flex-col">
                      <div className="flex items-center gap-2 text-[10px] text-muted-foreground mb-2">
                        <span>{formatDate(post.date)}</span>
                        <span>·</span>
                        <span>{post.readingTime}</span>
                      </div>
                      <h2 className="text-sm font-semibold font-heading text-foreground group-hover:text-brand-orange transition-colors mb-1.5 leading-snug">
                        {post.title}
                      </h2>
                      <p className="text-xs text-muted-foreground mb-3 line-clamp-2 flex-1">
                        {post.excerpt}
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {post.tags.slice(0, 3).map((tag) => (
                          <span key={tag} className={`px-1.5 py-0.5 text-[10px] rounded font-heading ${tagBadgeClass(tag)}`}>
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </article>
                ))}
              </div>

              {filteredPosts.length === 0 && (
                <div className="text-center py-16">
                  <p className="text-muted-foreground text-sm mb-2">
                    No blog posts found matching your filters.
                  </p>
                  <button
                    onClick={() => {
                      setActiveCategory("all");
                      setSelectedTags([]);
                    }}
                    className="text-brand-orange hover:underline text-xs font-heading"
                  >
                    Clear all filters
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};
