---
description: Add a research paper as a blog post to the personal site
---

## When to trigger
When the user says things like:
- "add a paper to blog"
- "write a blog post about [paper]"
- "blog this paper"
- "add [paper title] to blog"

## Steps

1. Read the blog post template at `.agent/templates/blog-post-template.md` for the exact format
2. If the user provides a paper URL (arXiv, DOI, etc.), read the paper content using `read_url_content`
3. Check the current highest `id` in `src/data/blog-posts.ts` to determine the next ID
4. Write the new blog post entry at the **top** of the `blogPosts` array in `src/data/blog-posts.ts`
5. Follow all writing guidelines from the template (tone, structure, TL;DR format, code examples)
// turbo
6. Build with `npm run build` to verify no errors
7. Ask the user if they want to push
