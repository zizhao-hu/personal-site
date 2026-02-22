# Blog Post Template

## Format
- Add new entries at the **top** of the `blogPosts` array in `src/data/blog-posts.ts`
- Increment `id` from the current highest
- `title`: **Attractive, future-oriented, use-case driven.** Don't just name the paper — frame it around what it enables or changes. Example: instead of "A Survey on Multi-Agent LLMs", write "Multi-Agent AI Is About to Change How We Build Software"
- `slug`: kebab-case from title
- `date`: today's date (YYYY-MM-DD)
- `category`: `"ai"` | `"science"` | `"economy"`

## TL;DR (visible by default)
Use the `tldr` object with these four fields:
- **problem** → What problem does the paper address?
- **idea** → What's the method/approach?
- **solution** → What are the key results?
- **vision** → What's the use case / why it matters?

Keep each field 2-3 sentences max.

## Full Content (hidden behind expand toggle)
The `content` markdown is collapsed by default. Write the deep dive there with code, tables, and analysis.

## Paper Blog Rules
When the source is a research paper:
1. **Extract figures from the paper PDF** — crop/screenshot the pipeline and results figures directly from the paper, save to `/public/images/blogs/`. Do NOT generate new figures.
2. Include the **key results table** — reproduce the main comparison table in markdown
3. Connect to Zizhao's research interests where natural
4. Link to the original paper (arXiv/DOI) at the bottom
5. Tone: senior researcher explaining to a smart peer, not a textbook
