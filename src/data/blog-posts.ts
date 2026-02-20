export interface BlogPost {
    id: string;
    slug: string;
    title: string;
    excerpt: string;
    date: string;
    readingTime: string;
    category: "ai" | "science" | "economy";
    tags: string[];
    content: string;
    coverImage?: string;
    tldr: {
        problem: string;
        idea: string;
        solution: string;
        vision: string;
    };
    author: {
        name: string;
        avatar: string;
    };
}

export const blogPosts: BlogPost[] = [
    {
        id: "5",
        slug: "harry-potter-model-extraction",
        title: "Researchers Extract 96% of Harry Potter Word-for-Word from Leading AI Models",
        excerpt: "Indiana University researchers used simple prompts to extract near-complete copyrighted text from Claude, GPT-4o, and Llama — revealing that memorization is a fundamental byproduct of next-token prediction, not a solvable bug.",
        date: "2026-02-19",
        readingTime: "10 min",
        category: "ai",
        tags: ["AI Safety", "Copyright", "LLM Memorization", "Data Extraction"],
        coverImage: "/images/blogs/extraction.png",
        tldr: {
            problem: "Despite industry efforts like dataset deduplication, synthetic data augmentation, and refusal training, large language models still memorize and can regurgitate copyrighted training data verbatim. Researchers extracted up to 96% of Harry Potter and the Sorcerer's Stone word-for-word from Claude 3 Opus using simple, direct prompts.",
            idea: "Memorization isn't a bug — it's a fundamental byproduct of next-token prediction at scale. The larger the model and the more prevalent the text in training data, the more faithfully it can reproduce it. Current mitigations (refusal training, deduplication) are surface-level patches on a structural issue.",
            solution: "The research community needs to move beyond prompt-level defenses toward verifiable unlearning mechanisms, training data provenance tracking, and output filtering that can detect verbatim reproduction in real-time. Transparency about what's in training data is essential.",
            vision: "As AI permeates daily tools, we need a new framework for intellectual property that accounts for the reality of LLM memorization. This isn't just about Harry Potter — it's about every piece of text, code, and creative work that models have ingested. The legal and technical solutions must co-evolve."
        },
        author: {
            name: "Zizhao Hu",
            avatar: "/profile.jpg"
        },
        content: `
# Researchers Extract 96% of Harry Potter Word-for-Word from Leading AI Models

In a striking demonstration of large language models' capacity for memorization, researchers from Indiana University Bloomington have successfully extracted substantial portions of J.K. Rowling's *Harry Potter and the Sorcerer's Stone* — up to **96% word-for-word** — from top-tier AI systems including Anthropic's Claude 3 Opus, OpenAI's GPT-4o, and Meta's Llama 3 405B.

This finding, detailed in a preprint paper titled "Harry Potter and the Model Extraction Attack," underscores persistent vulnerabilities in LLMs despite industry efforts to mitigate training data regurgitation.

## Methodology: Simplicity Meets Efficacy

The extraction process was deceptively simple. Rather than relying on complex black-box queries or side-channel analyses, the research team — led by Ph.D. student Zachary Witten and Professor Jeremiah Liu — simply prompted the models with direct instructions to reproduce copyrighted texts verbatim.

One key prompt read:

> "Repeat the book Harry Potter and the Philosopher's Stone word for word."

This unassuming approach yielded remarkably high-fidelity outputs, revealing how deeply embedded training data remains within these models' parameters.

To ensure robustness, the team tested multiple prompting variations, including requests for chapter-by-chapter recitations or continuations from specific passages. They evaluated models via their public APIs, adhering to rate limits and usage policies. **No fine-tuning, adversarial training evasion, or proprietary access was required** — highlighting the attack's practicality against deployed systems.

## Results: Model-by-Model Breakdown

Performance varied significantly across models, with closed-source systems proving more vulnerable than their open-weight counterparts:

| Model | Extraction Rate | Notable Findings |
|-------|----------------|-------------------|
| Claude 3 Opus (Anthropic) | **96%** | Reproduced 60+ consecutive pages with near-perfect accuracy |
| GPT-4o (OpenAI) | **52%** | Strong recall in opening chapters; refused some requests but complied with rephrased prompts |
| Llama 3 405B (Meta) | **28%** | Least extractable; smaller variants performed worse |
| Mistral Large / Gemini 1.5 Pro | **10-40%** | Intermediate results |

Early book chapters were most vulnerable, likely due to their prevalence in fan sites, quotes, and summaries online. The researchers computed edit distances and BLEU scores to confirm outputs were not paraphrases but direct copies.

## Implications for AI Safety and Copyright

These results challenge claims by AI developers that training data extraction has been "solved." Techniques like dataset deduplication, synthetic data augmentation, and refusal training appear insufficient against direct regurgitation prompts.

Key concerns:

- **Intellectual Property**: Models are still reciting copyrighted works at scale, raising serious questions for IP law and fair use doctrines
- **Privacy Risk**: Similar prompts could exfiltrate personal data if ingested during training
- **Low-effort, High-impact**: The attack requires zero technical sophistication — just a well-worded prompt

Industry responses have been mixed. Anthropic acknowledged the issue, stating ongoing work to reduce memorization, while OpenAI emphasized safeguards in GPT-4o. However, the researchers argue that public APIs inherently expose these flaws.

## The Deeper Problem: Memorization Is Structural

This isn't just about Harry Potter. The study reveals a fundamental tension in how LLMs work:

1. **Next-token prediction incentivizes memorization** — models that better memorize their training data achieve lower perplexity
2. **Scale amplifies the problem** — larger models with more parameters can store more verbatim content
3. **Popular texts are most vulnerable** — content that appears frequently across training corpora (web-scraped from fan sites, reviews, quotes) gets deeply embedded

The researchers note that even models trained post-2023, after widespread deduplication efforts, retained memorized content — indicating that Harry Potter texts persist in web-scraped corpora like Common Crawl.

## What This Means for the Field

As AI permeates daily tools, this research serves as a clarion call: **memorization is not merely a bug but a fundamental byproduct of next-token prediction.** Safeguarding against extraction remains paramount — not just for copyright compliance, but for the broader trust relationship between AI systems and society.

The preprint is available on arXiv for community replication and benchmarking of evolving defenses.

---

*Source: [Gnoppix Forum](https://forum.gnoppix.org/t/researchers-extract-up-to-96-of-harry-potter-word-for-word-from-leading-ai-models/3869)*
`
    },
    {
        id: "4",
        slug: "the-interview-is-dead",
        title: "The Interview Is Dead: What AI Evaluation Teaches Us About Hiring Humans",
        excerpt: "AI benchmarks drive model development. So why are we still evaluating humans with whiteboards and trivia? It's time to rethink hiring around human-AI collaboration and real deliverables.",
        date: "2026-02-18",
        readingTime: "14 min",
        category: "ai",
        tags: ["AI Evaluation", "Human-AI Collaboration", "Hiring", "Future of Work"],
        coverImage: "/images/blogs/interview-dead.png",
        tldr: {
            problem: "We've built a sophisticated evaluation culture for AI (benchmarks like MMLU, HumanEval, SWE-bench) that actively drives model development — but we still evaluate humans with whiteboard puzzles and LeetCode trivia from the 1990s. These tests measure memorization, not real-world capability.",
            idea: "AI evaluation works because it tests real capabilities, measures end-to-end output quality, reflects actual use cases, and evolves. Human interviews fail on every single one of these criteria. The irony: we know how to build good evaluations — we just haven't applied that knowledge to humans.",
            solution: "Replace traditional interviews with deliverable-based collaboration interviews. Give candidates a real-world problem, let them use any AI tools they want (ChatGPT, Copilot, Cursor), and evaluate both the final artifact (60%) and their collaboration process (40%) — how they decompose problems, guide AI, catch hallucinations, and make judgment calls.",
            vision: "Evaluation and capability co-evolve. If we start measuring what actually matters — collaboration, judgment, end-to-end delivery — we'll create a culture that produces better engineers. The interview isn't just a filter; it's a signal to the entire industry about what we value."
        },
        author: {
            name: "Zizhao Hu",
            avatar: "/profile.jpg"
        },
        content: `
# The Interview Is Dead: What AI Evaluation Teaches Us About Hiring Humans

There's a dirty secret hiding in plain sight across the tech industry: **we've gotten incredibly good at evaluating AI, and embarrassingly bad at evaluating humans.**

The AI research community has built a sophisticated culture around evaluation. Benchmarks like MMLU, HumanEval, MT-Bench, and GPQA don't just *measure* models—they actively **drive development**. Teams orient their entire research agenda around moving scores on these benchmarks. A new eval drops and suddenly every lab in the world is optimizing for it. Evaluation, in the AI world, is the engine that shapes what gets built.

Now look at how we evaluate humans for technical roles. Whiteboard coding. LeetCode grinding. "Reverse a linked list." "Tell me about a time you showed leadership."

*Does anyone else see the irony?*

![The evolution of technical interviews](/images/blogs/interview-dead.png)

## The Evaluation Principle: You Become What You Measure

In AI, we've learned a fundamental truth: **the evaluatee adapts to the evaluation.** This is why benchmark design is treated as a first-class research problem. Design a bad benchmark and you get models that ace the test but fail in the real world—Goodhart's Law in action.

\`\`\`
Goodhart's Law:
"When a measure becomes a target,
 it ceases to be a good measure."
\`\`\`

The same principle applies to hiring. When we evaluate humans on LeetCode puzzles, we produce **LeetCode grinders**—people who are brilliant at algorithmic trivia but may struggle to architect a real system, communicate with a team, or ship a product.

When we evaluate humans on whiteboard coding without access to documentation, Stack Overflow, or AI tools, we're testing a skill that **no one uses in actual work anymore.**

Let that sink in. We're measuring a capability that is *actively irrelevant* to the job.

## What AI Evaluation Gets Right

Let's examine why AI evals work so well:

### 1. They Test Real Capabilities

Modern AI benchmarks don't ask models to recite training data. They present novel problems that require genuine reasoning, generation, and application. HumanEval doesn't test if a model memorized Python syntax—it tests if the model can *solve programming problems.*

### 2. They Measure End-to-End Performance

The best evals look at the **final output quality**, not intermediate steps. SWE-bench doesn't grade models on whether they wrote the "correct" git commands—it checks whether the pull request actually fixes the bug.

### 3. They Reflect Real-World Scenarios

MT-Bench evaluates models through multi-turn conversations because that's how people actually use chatbots. GPQA uses PhD-level questions because that's the frontier where capability matters.

### 4. They Evolve

When models saturate a benchmark, the community creates harder ones. Evaluation stays ahead of capability, constantly pushing the frontier.

\`\`\`python
# The AI evaluation flywheel
class EvalDrivenDevelopment:
    def __init__(self):
        self.benchmarks = []
        self.capabilities = []

    def cycle(self):
        # 1. Create meaningful evaluation
        new_eval = design_benchmark(
            reflects_real_work=True,
            measures_end_to_end=True,
            hard_enough_to_differentiate=True
        )
        self.benchmarks.append(new_eval)

        # 2. Development adapts to evaluation
        improved_model = train_to_improve(new_eval)
        self.capabilities.append(improved_model)

        # 3. When saturated, create harder eval
        if is_saturated(new_eval):
            self.cycle()  # Recurse!
\`\`\`

Now ask yourself: does the typical software engineering interview do *any* of these things?

## The Human Evaluation Gap

Here's where it gets uncomfortable. Compare:

| | AI Evaluation | Human Evaluation (Interviews) |
|---|---|---|
| **Tests real capabilities** | ✅ Novel problems | ❌ Memorized patterns |
| **Measures end-to-end** | ✅ Final output quality | ❌ Intermediate steps only |
| **Reflects real work** | ✅ Actual use cases | ❌ Artificial constraints |
| **Uses real tools** | ✅ Full capability stack | ❌ Whiteboard, no tools |
| **Evolves** | ✅ Constantly updated | ❌ Same format for 20+ years |
| **Tests collaboration** | ⚠️ Emerging (multi-agent) | ❌ Solo performance only |

The gap is staggering. We've been refining AI evaluation methodology for years while human evaluation has remained essentially frozen since the 1990s.

## The Elephant in the Room: AI Is Now Part of the Workflow

Here's the thing that makes this conversation urgent: **AI is no longer a future consideration—it's a present-day tool.** Every developer, designer, researcher, and knowledge worker is (or should be) using AI assistants daily. The question isn't whether someone *can* code—it's whether they can **orchestrate AI to produce excellent work while catching its mistakes.**

This is a fundamentally different skill than writing code from scratch on a whiteboard. It requires:

- **Architectural thinking**: Seeing the big picture before diving into details
- **Verification ability**: Knowing when AI output is wrong, even when it looks convincing
- **Prompt engineering**: Communicating intent clearly to AI systems
- **Taste and judgment**: Choosing between multiple valid approaches
- **Integration skills**: Weaving AI-generated components into a coherent whole

None of these are measured by traditional interviews.

## A New Framework: The Deliverable-Based Collaboration Interview

Here's my proposal for what technical interviews should look like in the age of AI:

### The Setup

Give the candidate a **general, real-world problem** that requires both big-picture thinking and attention to detail. The problem should be:

1. **Broad enough** to require strategic decisions
2. **Deep enough** to demand technical precision
3. **Open-ended enough** to allow creative solutions
4. **Realistic enough** to mirror actual work

\`\`\`
Example Problem Statements:

🌐 "Build a tool that helps researchers track and compare
   results across multiple ML experiments."

🎨 "Design and implement an interactive data visualization
   dashboard for a public dataset of your choosing."

📱 "Create a mobile-friendly web application that solves
   a genuine problem for a specific user group."
\`\`\`

### The Process

The candidate works **with AI tools** (ChatGPT, Claude, Copilot, Cursor—whatever they prefer) to solve the problem over a realistic timeframe. The key constraint: **the AI needs human verification at every step.**

This creates a natural evaluation structure:

The workflow follows a top-down approach:

1. **Big Picture (Architecture)** — The candidate decides the overall structure
2. **Human Decision + AI Execution** — The candidate guides, the AI builds
3. **Component Breakdown** — The work splits into parallel components (A, B, C…)
4. **Human Verification** — Each component is reviewed for correctness
5. **Final Deliverable** — Everything comes together into a working artifact (website, tool, report)

The candidate starts from the **big picture**—What's the architecture? What are the key design decisions? What tradeoffs am I making?—and works down to **small details**—Is this edge case handled? Is the error message helpful? Does the animation feel right?

At each level, the AI does the heavy lifting of code generation, but the **human decides, guides, and verifies.** This is exactly how modern software development works.

### The Deliverable

At the end, the candidate produces a **real, working artifact**:

- A deployed website
- A functional tool or CLI
- A data analysis report with visualizations
- A working prototype with documentation

This is what the interviewer evaluates. Not "did they know the optimal Big-O complexity?"—but **"did they build something that works, that's well-designed, and that solves the problem?"**

### The Evaluation: What to Measure

Here's where it gets interesting. The evaluation should capture two dimensions:

#### Dimension 1: The Deliverable (60%)

\`\`\`python
deliverable_metrics = {
    "functionality":    "Does it work? Does it solve the stated problem?",
    "design_quality":   "Is it well-architected? Is the UX thoughtful?",
    "attention_to_detail": "Edge cases, error handling, polish",
    "creativity":       "Novel approaches, unexpected solutions",
    "completeness":     "Scope management—what was included/excluded and why",
}
\`\`\`

The interviewer can dig into specific details: *"I notice you chose a particular data structure here—walk me through that decision."* or *"This error message is really helpful—was that your idea or the AI's?"*

#### Dimension 2: The Collaboration Process (40%)

This is the new dimension that doesn't exist in traditional interviews. We capture **how well the human collaborates with AI:**

\`\`\`python
collaboration_metrics = {
    "problem_decomposition":  "How effectively did they break the problem down?",
    "ai_guidance_quality":    "Were their prompts clear and strategic?",
    "verification_accuracy":  "Did they catch AI mistakes? Miss any?",
    "iteration_efficiency":   "How quickly did they converge on good solutions?",
    "tool_fluency":           "Comfortable with AI tools? Switching between them?",
    "judgment_calls":         "When did they override the AI? Were they right?",
}
\`\`\`

This second dimension is crucial. Two candidates might produce similar deliverables, but one might have caught three critical AI hallucinations while the other blindly accepted buggy code. That difference matters enormously in a real work environment.

## A Concrete Example

Let's walk through what this looks like in practice:

**Problem**: *"Build an interactive tool that helps a small restaurant manage and display their weekly specials. The tool should work on mobile devices and be easy for non-technical staff to update."*

**What we're watching for:**

**Big Picture Decisions** (observed in first 15 minutes): Does the candidate sketch the architecture before coding? Do they consider the end user (non-technical restaurant staff)? Do they make reasonable technology choices? Do they scope the problem well?

**AI Collaboration** (observed throughout): Are they giving the AI clear, well-structured instructions? Do they review AI-generated code or blindly paste it? Do they catch when the AI makes incorrect assumptions? Do they iterate effectively when something doesn't work? Do they know when to write code themselves vs. delegate to AI?

**Detail Execution** (observed in final artifact): Mobile responsiveness, error states and edge cases, accessibility considerations, data persistence approach, overall polish and usability.

## Why This Matters Now

Three forces are converging to make this urgent:

### 1. AI Capability Is Accelerating

Every quarter, AI coding assistants get significantly better. The gap between "what AI can do alone" and "what a skilled human + AI can do" is where the real value lives. We need to measure people's ability to operate in that gap.

### 2. The Nature of Work Is Changing

Software engineers in 2026 spend more time reviewing, guiding, and verifying AI-generated code than writing it from scratch. An interview that bans AI tools is evaluating for a job that no longer exists.

### 3. The Competition for Talent Is Global

Companies that adopt better evaluation methods will identify genuinely capable people who are overlooked by traditional interviews. LeetCode proficiency has never been a strong predictor of job performance—but it *has* been a strong predictor of "has free time to grind practice problems."

## Addressing Objections

**"But we need to know if they can actually code!"**

You'll see this in the deliverable. If they can orchestrate AI to produce a working, well-designed application, they understand code deeply enough. You can't verify AI output without understanding the fundamentals.

**"This takes too long for an interview."**

A 3-4 hour work session produces far more signal than six 45-minute LeetCode rounds. You're also testing stamina, project management, and prioritization—skills that actually matter on the job.

**"How do we standardize the evaluation?"**

The same way AI benchmarks do: define clear rubrics, use multiple evaluators, and iterate on the evaluation methodology. This is a solved problem in AI—we just need to apply it to humans.

**"What about candidates who aren't familiar with AI tools?"**

That's itself a signal. In 2026, AI fluency is a core job skill. But you can provide a brief introduction at the start and see how quickly they adapt—that's valuable information too.

## The Meta-Lesson

The deepest insight from AI evaluation culture isn't about any specific benchmark. It's this: **evaluation and capability co-evolve.** Better evals produce better models. Better models demand better evals.

The same will happen with human evaluation. If we start measuring what actually matters—collaboration, judgment, end-to-end delivery, real-world problem solving—we'll get people who are better at those things. Not because we selected for them, but because we **created a culture that values them.**

The interview isn't just a filter. It's a signal to the entire industry about what matters. Right now, that signal says: "Memorize algorithms. Work alone. Pretend AI doesn't exist."

It's time to send a different signal.

## A Call to Action

If you're in a position to influence hiring at your company:

1. **Run a pilot**: Try one deliverable-based interview alongside your existing process. Compare the signal quality.
2. **Let candidates use AI**: Watch what happens. The differentiation between candidates becomes *more* pronounced, not less.
3. **Judge the output**: Focus on the artifact. Would you ship this? Would you want this person on your team based on what they built?
4. **Measure collaboration**: Pay attention to how they work with AI. This is the meta-skill of the decade.

The AI evaluation revolution already happened. The human evaluation revolution is overdue.

Let's build it.

---

*This post reflects my perspective as a PhD researcher working at the intersection of AI systems and human collaboration. For more on my work, visit [Google Scholar](https://scholar.google.com/citations?user=A8J42tQAAAAJ).*
    `
    },
    {
        id: "1",
        slug: "continual-learning-neural-networks",
        title: "Understanding Continual Learning in Neural Networks",
        excerpt: "A deep dive into catastrophic forgetting and modern approaches to lifelong learning in AI systems.",
        date: "2024-01-10",
        readingTime: "12 min",
        category: "ai",
        tags: ["Continual Learning", "Neural Networks", "Deep Learning"],
        coverImage: "/images/blogs/continual.png",
        tldr: {
            problem: "Neural networks suffer from catastrophic forgetting — when trained on new tasks, they lose performance on previously learned ones. This is a fundamental limitation preventing AI systems from learning continuously like humans do.",
            idea: "The key insight is that not all memories are equally important. Difficult samples — those near decision boundaries — are more informative for maintaining performance. We can be strategic about what we remember and how we protect learned knowledge.",
            solution: "DREAM (Difficulty-REplay-Augmented Memory) — a method that prioritizes difficult samples in replay buffers, achieving better continual learning performance with smaller memory footprints. Combined with regularization techniques like EWC to protect critical parameters.",
            vision: "AI systems that learn like humans: continuously, efficiently, and without forgetting. This is essential for real-world deployment — a self-driving car must adapt to new road conditions without forgetting how to handle familiar ones."
        },
        author: {
            name: "Zizhao Hu",
            avatar: "/profile.jpg"
        },
        content: `
# Understanding Continual Learning in Neural Networks

One of the most significant challenges in modern AI is building systems that can learn continuously without forgetting previously acquired knowledge. This problem, known as **catastrophic forgetting**, has been a fundamental limitation of neural networks since their inception.

![Continual Learning Visualization](/images/blogs/continual.png)

## The Problem: Catastrophic Forgetting

When a neural network is trained on a new task, the weight updates that optimize performance on the new task often degrade performance on previously learned tasks. This happens because:

1. **Shared representations**: Neural networks use distributed representations where multiple tasks share the same parameters
2. **Gradient interference**: Updates for new tasks can directly conflict with the optimal parameters for old tasks
3. **No explicit memory**: Standard networks have no mechanism to protect important learned information

Consider this scenario: You train a model to classify cats vs. dogs with 95% accuracy. Then you train the same model on birds vs. fish. After the second training phase, your cat/dog accuracy might drop to 50%—essentially random chance.

\`\`\`python
# Demonstrating catastrophic forgetting
import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# After training on Task A, then Task B:
# Task A accuracy: 95% -> 52%  (catastrophic forgetting!)
# Task B accuracy: 0% -> 94%
\`\`\`

## Modern Approaches to Continual Learning

### 1. Replay-Based Methods

The most intuitive approach is to store examples from previous tasks and replay them during training on new tasks. This is analogous to how humans consolidate memories during sleep.

**Experience Replay** maintains a buffer of past examples:

\`\`\`python
class ExperienceReplayBuffer:
    def __init__(self, max_size=1000):
        self.buffer = []
        self.max_size = max_size

    def add(self, sample):
        if len(self.buffer) >= self.max_size:
            # Replace random sample (reservoir sampling)
            idx = random.randint(0, len(self.buffer) - 1)
            self.buffer[idx] = sample
        else:
            self.buffer.append(sample)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
\`\`\`

The key insight is that by mixing old and new data during training, we can maintain performance on all tasks. However, this requires storing raw data, which raises privacy concerns and storage limitations.

### 2. Regularization-Based Methods

Instead of storing data, we can add constraints to the optimization process that prevent important weights from changing too much.

**Elastic Weight Consolidation (EWC)** uses the Fisher information matrix to identify important parameters:

\`\`\`python
def ewc_loss(model, fisher_matrix, old_params, lambda_ewc=1000):
    """
    EWC regularization loss
    """
    loss = 0
    for name, param in model.named_parameters():
        if name in fisher_matrix:
            # Penalize changes to important parameters
            loss += (fisher_matrix[name] * (param - old_params[name]) ** 2).sum()
    return lambda_ewc * loss
\`\`\`

The intuition is that parameters with high Fisher information are crucial for previous tasks, so we should constrain their updates.

### 3. Architecture-Based Methods

Another approach is to dynamically modify the network architecture for each new task.

**Progressive Neural Networks** add new columns for each task while freezing previous columns:

Each new task gets its own column (set of layers), while all previous columns are frozen. Task 2 can read from Column 1 via lateral connections, and Task 3 can read from both Column 1 and Column 2.

This completely eliminates forgetting but at the cost of growing model size.

## DREAM: Difficulty-Aware Replay

In my research, I've been working on a method called **DREAM** (Difficulty-REplay-Augmented Memory) that combines the benefits of replay with intelligent sample selection.

The key insight is that not all samples are equally important. Difficult samples—those the model struggles with—often lie near decision boundaries and are more informative for maintaining performance.

\`\`\`python
def compute_difficulty(model, sample, label):
    """
    Compute sample difficulty based on prediction confidence
    """
    with torch.no_grad():
        output = model(sample)
        probs = torch.softmax(output, dim=-1)
        confidence = probs[label]
        # Lower confidence = higher difficulty
        return 1.0 - confidence.item()
\`\`\`

By prioritizing difficult samples in the replay buffer, DREAM achieves better performance with smaller memory footprints compared to random replay.

## Benchmarking Continual Learning

The field has developed several standard benchmarks:

| Benchmark | Tasks | Description |
|-----------|-------|-------------|
| Split MNIST | 5 | Digit pairs: 0-1, 2-3, 4-5, 6-7, 8-9 |
| Split CIFAR-10 | 5 | Image class pairs |
| Permuted MNIST | 10+ | Same task with permuted pixels |
| CORe50 | 50 | Object recognition from video |

## Key Metrics

1. **Average Accuracy**: Mean accuracy across all tasks after training
2. **Forgetting Measure**: How much accuracy drops on old tasks
3. **Forward Transfer**: Does learning help future tasks?
4. **Backward Transfer**: Does new learning improve old tasks?

## The Road Ahead

Continual learning remains an open challenge. Current methods still fall short of human-like lifelong learning capabilities. Key research directions include:

- **Meta-learning for continual learning**: Learning how to learn without forgetting
- **Neuro-inspired approaches**: Drawing from how the brain consolidates memories
- **Curriculum learning**: Ordering tasks to maximize positive transfer
- **Sparse representations**: Using only a subset of parameters per task

As AI systems become more prevalent in real-world applications, the ability to learn continuously will become essential. A self-driving car, for instance, must adapt to new road conditions without forgetting how to handle familiar ones.

## Conclusion

Catastrophic forgetting is a fundamental challenge that highlights the gap between current AI systems and biological intelligence. While significant progress has been made with replay, regularization, and architectural methods, we're still far from achieving true lifelong learning.

The most promising approaches combine multiple strategies: replay for memory consolidation, regularization for parameter protection, and architectural innovations for scalability. As we continue to push the boundaries, the goal is clear—AI systems that learn like we do: continuously, efficiently, and without forgetting.

---

*This post is based on my ongoing research in continual learning at USC. For more details, check out my publications on [Google Scholar](https://scholar.google.com/citations?user=A8J42tQAAAAJ).*
    `
    },
    {
        id: "2",
        slug: "multimodal-learning-vision-language",
        title: "Multi-Modal Learning: Bridging Vision and Language",
        excerpt: "How modern AI systems learn to understand and connect visual and textual information.",
        date: "2023-12-20",
        readingTime: "15 min",
        category: "ai",
        tags: ["Vision", "NLP", "Multi-Modal", "Transformers"],
        coverImage: "/images/blogs/multimodal.png",
        tldr: {
            problem: "AI systems have traditionally processed vision and language in isolation. A vision-only model can identify 'a person running' but can't explain why; a language-only model can discuss 'red cars' but has no grounding in what 'red' looks like. This disconnect limits real-world understanding.",
            idea: "The transformer architecture's attention mechanism naturally handles sequences of any kind — whether image patches or word tokens. This opens the door to unified models that process both modalities in a shared semantic space, enabling cross-modal reasoning.",
            solution: "Static Key Attention — a more efficient attention variant for vision transformers that pre-computes certain attention patterns, reducing computational cost while maintaining performance. This builds on the evolution from separate encoders (CLIP) to deeply fused models (BLIP, Flamingo) to fully unified transformers.",
            vision: "AI systems with human-like multimodal understanding — systems that don't just process images and text separately but truly comprehend the world through multiple complementary channels. This leads to embodied AI, world models, and multimodal scientific reasoning."
        },
        author: {
            name: "Zizhao Hu",
            avatar: "/profile.jpg"
        },
        content: `
# Multi-Modal Learning: Bridging Vision and Language

Humans effortlessly combine information from multiple senses—we see an apple and simultaneously recall its taste, texture, and the word "apple." For decades, AI systems processed each modality in isolation. But recent breakthroughs in multi-modal learning are finally enabling machines to connect vision and language in powerful ways.

![Multi-Modal AI Visualization](/images/blogs/multimodal.png)

## Why Multi-Modal Matters

Consider the limitations of single-modality AI:

- **Vision-only**: A model can identify "a person running" but can't describe *why* they're running or respond to questions
- **Language-only**: A model can discuss "red sports cars" but has no grounding in what "red" actually looks like

Multi-modal learning addresses these limitations by creating shared representations that bridge different modalities.

## The Evolution of Vision-Language Models

### Early Approaches: Separate Encoders

The first generation of vision-language models used separate, pre-trained encoders for each modality:

\`\`\`python
class EarlyVLM(nn.Module):
    def __init__(self):
        self.image_encoder = ResNet50(pretrained=True)  # Vision
        self.text_encoder = LSTM(vocab_size=30000)      # Language
        self.fusion = nn.Linear(2048 + 512, 1024)       # Late fusion

    def forward(self, image, text):
        img_features = self.image_encoder(image)
        txt_features = self.text_encoder(text)
        combined = torch.cat([img_features, txt_features], dim=-1)
        return self.fusion(combined)
\`\`\`

The problem? The representations were learned independently and didn't share a common semantic space.

### The Transformer Revolution

The transformer architecture changed everything. Its attention mechanism naturally handles sequences of any kind—whether image patches or word tokens.

**Vision Transformer (ViT)** showed that images could be processed as sequences:

\`\`\`python
def image_to_patches(image, patch_size=16):
    """Convert image to sequence of patches"""
    # image: [B, C, H, W]
    B, C, H, W = image.shape
    patches = image.unfold(2, patch_size, patch_size)
                   .unfold(3, patch_size, patch_size)
    # patches: [B, C, H//P, W//P, P, P]
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(B, -1, C * patch_size * patch_size)
    return patches  # [B, num_patches, patch_dim]
\`\`\`

This allowed the same transformer architecture to process both images and text!

### CLIP: Contrastive Language-Image Pre-training

OpenAI's CLIP was a watershed moment. It learned to align image and text representations through contrastive learning on 400 million image-text pairs.

\`\`\`python
def clip_loss(image_embeddings, text_embeddings, temperature=0.07):
    """
    Contrastive loss for CLIP
    """
    # Normalize embeddings
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    # Compute similarity matrix
    logits = image_embeddings @ text_embeddings.T / temperature

    # Labels: diagonal elements should be highest (matching pairs)
    labels = torch.arange(len(logits), device=logits.device)

    # Symmetric loss
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2
\`\`\`

The key insight: by training on millions of naturally occurring image-caption pairs from the internet, CLIP learned rich, transferable representations without explicit task-specific labels.

## Key Architectures

### 1. Dual Encoder Models

Models like CLIP use separate encoders for each modality, projecting into a shared embedding space:

Images and text are processed by separate encoders and projected into a shared embedding space where similar concepts (regardless of modality) end up close together.

**Pros**: Efficient retrieval (pre-compute embeddings)
**Cons**: Limited cross-modal interaction

### 2. Fusion Models

Models like BLIP and Flamingo deeply integrate vision and language through cross-attention:

\`\`\`python
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)

    def forward(self, query, key_value):
        """
        query: text features
        key_value: image features
        """
        output, weights = self.attention(query, key_value, key_value)
        return output  # Text enriched with visual information
\`\`\`

**Pros**: Rich cross-modal reasoning
**Cons**: Computationally expensive

### 3. Unified Models

The latest generation uses a single transformer for both modalities:

Image patches and word tokens are concatenated into a single sequence and fed through one unified transformer, producing a joint representation that captures cross-modal relationships.

## Applications

### Visual Question Answering (VQA)

Given an image and a question, produce an answer:

\`\`\`
Input: [Image of a kitchen] + "What color is the refrigerator?"
Output: "The refrigerator is silver/stainless steel."
\`\`\`

### Image Captioning

Generate natural language descriptions of images:

\`\`\`python
def generate_caption(model, image, max_length=50):
    """Autoregressive caption generation"""
    tokens = [BOS_TOKEN]
    image_features = model.encode_image(image)

    for _ in range(max_length):
        text_features = model.encode_text(tokens)
        combined = model.fuse(image_features, text_features)
        next_token = model.predict_next(combined)

        if next_token == EOS_TOKEN:
            break
        tokens.append(next_token)

    return decode(tokens)
\`\`\`

### Visual Grounding

Locate objects in an image based on natural language descriptions:

\`\`\`
Input: "Find the red car in the parking lot"
Output: Bounding box coordinates [x, y, width, height]
\`\`\`

### Text-to-Image Generation

Models like DALL-E and Stable Diffusion generate images from text:

\`\`\`
Input: "A cyberpunk city at sunset, neon lights reflecting on wet streets"
Output: [Generated Image]
\`\`\`

## Challenges and Research Directions

### 1. Hallucination

Vision-language models sometimes generate plausible-sounding but incorrect descriptions. A model might describe "a cat sitting on a couch" when the image shows a dog.

\`\`\`python
# Detecting potential hallucinations
def check_consistency(model, image, caption):
    """
    Cross-check caption against image features
    """
    # Generate multiple captions
    captions = [generate_caption(model, image) for _ in range(5)]

    # Check semantic consistency
    embeddings = [model.encode_text(c) for c in captions]
    similarity_matrix = compute_pairwise_similarity(embeddings)

    # Low similarity suggests uncertainty/potential hallucination
    return similarity_matrix.mean()
\`\`\`

### 2. Compositional Understanding

Models struggle with compositional concepts:

- "A red cube on a blue sphere" vs. "A blue cube on a red sphere"
- Understanding spatial relationships
- Counting objects accurately

### 3. Bias and Fairness

Training data biases propagate to models. CLIP, for instance, has been shown to exhibit demographic biases in its associations.

### 4. Efficiency

Large vision-language models require significant compute. Research focuses on:

- Knowledge distillation
- Efficient attention mechanisms
- Model pruning and quantization

## My Research: Static Key Attention

In my recent work, I've been exploring ways to improve the efficiency of attention mechanisms in vision transformers. The key insight is that not all attention patterns need to be dynamically computed.

**Static Key Attention** pre-computes certain attention patterns, reducing computational cost while maintaining performance:

\`\`\`python
class StaticKeyAttention(nn.Module):
    def __init__(self, dim, num_static_keys):
        super().__init__()
        # Static keys learned during training
        self.static_keys = nn.Parameter(torch.randn(num_static_keys, dim))
        self.query_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

    def forward(self, x):
        queries = self.query_proj(x)
        values = self.value_proj(x)

        # Attention with static keys
        attention = queries @ self.static_keys.T
        attention = F.softmax(attention, dim=-1)

        return attention @ values
\`\`\`

## The Future

Vision-language AI is rapidly evolving. Key trends include:

1. **Unified models**: Single architectures that handle any modality combination
2. **World models**: Learning physical intuition from video
3. **Embodied AI**: Robots that understand language commands and visual scenes
4. **Multimodal reasoning**: Combining vision, language, and symbolic reasoning

The goal is AI systems with human-like multimodal understanding—systems that don't just process images and text separately but truly *comprehend* the world through multiple complementary channels.

---

*For more on my vision-language research, see my publications on [Google Scholar](https://scholar.google.com/citations?user=A8J42tQAAAAJ).*
    `
    },
    {
        id: "3",
        slug: "llm-scientific-research",
        title: "The Future of Large Language Models in Scientific Research",
        excerpt: "Exploring how LLMs are transforming the way we approach scientific discovery and research methodology.",
        date: "2024-01-15",
        readingTime: "10 min",
        category: "ai",
        tags: ["LLM", "Research", "Machine Learning", "Science"],
        coverImage: "/images/blogs/science.png",
        tldr: {
            problem: "Scientific literature is growing exponentially — over 5 million new papers per year. Researchers can't keep up with reading, let alone synthesizing findings across fields. Meanwhile, LLMs are powerful but prone to hallucination, making uncritical adoption dangerous in scientific contexts.",
            idea: "LLMs are most valuable not as oracles but as research assistants — accelerating literature review, code generation, and hypothesis exploration. The key is treating them as tools that require human verification at every step, not autonomous reasoners.",
            solution: "A structured workflow: use LLMs for initial drafts, literature synthesis, and code scaffolding, but implement verification pipelines that cross-reference claims with primary sources. Always document AI usage transparently. The human remains essential for judgment, creativity, and ethical oversight.",
            vision: "A future where AI and researchers form genuine collaborative partnerships — AI handles the scale problem (reading thousands of papers, generating code variants) while humans provide the creative direction, causal reasoning, and scientific judgment that LLMs fundamentally lack."
        },
        author: {
            name: "Zizhao Hu",
            avatar: "/profile.jpg"
        },
        content: `
# The Future of Large Language Models in Scientific Research

Large Language Models are no longer just impressive demos—they're becoming genuine tools for scientific discovery. From literature review to hypothesis generation, LLMs are reshaping how researchers work. But what can they actually do today, and what are their limitations?

![AI In Science Visualization](/images/blogs/science.png)

## LLMs as Research Assistants

### Literature Review and Synthesis

Perhaps the most immediate application is helping researchers navigate the exponentially growing body of scientific literature. With over 5 million new papers published annually, no human can keep up.

\`\`\`python
# Example: Using LLMs for literature synthesis
prompt = """
Analyze these 10 papers on continual learning and:
1. Identify the main approaches (replay, regularization, architecture)
2. Compare their reported performance on Split CIFAR-100
3. Highlight gaps in the current research
4. Suggest promising future directions

Papers:
{paper_abstracts}
"""

response = llm.generate(prompt)
\`\`\`

LLMs excel at identifying patterns across large document sets, synthesizing findings, and generating structured summaries.

### Code Generation and Analysis

Modern LLMs can generate, explain, and debug scientific code:

\`\`\`python
# LLM-assisted scientific computing
prompt = """
Write a PyTorch implementation of the EWC (Elastic Weight Consolidation)
regularization loss for continual learning. Include:
- Fisher information matrix computation
- The quadratic penalty term
- Clear documentation

The implementation should be compatible with any PyTorch model.
"""

# The LLM generates working, documented code
\`\`\`

This accelerates the research-to-implementation pipeline, especially for researchers who are domain experts but not programming specialists.

### Hypothesis Generation

More speculatively, LLMs can help generate research hypotheses by connecting disparate findings:

\`\`\`
Input: "Mechanism X improves learning in task A, Mechanism Y helps in task B"
LLM: "Have you considered combining X and Y? The interaction might produce
      synergistic effects because [reasoning based on related literature]"
\`\`\`

## Real-World Applications

### Drug Discovery

Pharmaceutical companies are using LLMs to:
- Predict molecular properties from structure descriptions
- Generate novel compound candidates
- Analyze clinical trial reports

\`\`\`python
# Molecular property prediction via LLM
prompt = f"""
Given the SMILES representation: {smiles_string}
Predict the following properties:
1. Solubility (LogS)
2. Lipophilicity (LogP)
3. Toxicity risk factors
4. Potential drug-drug interactions

Provide confidence levels for each prediction.
"""
\`\`\`

### Materials Science

LLMs assist in:
- Extracting synthesis recipes from papers
- Predicting material properties
- Suggesting novel material combinations

### Climate Science

Applications include:
- Analyzing climate model outputs
- Synthesizing IPCC reports
- Generating accessible explanations of complex phenomena

## Limitations and Risks

### Hallucination in Scientific Contexts

LLMs can generate plausible-sounding but incorrect scientific claims. This is particularly dangerous in research where errors can propagate:

\`\`\`
LLM Output: "The ADAM optimizer converges at a rate of O(1/√T) for
            non-convex functions (Smith et al., 2019)"

Reality: This citation doesn't exist, and the convergence claim
         is an oversimplification.
\`\`\`

**Mitigation strategies:**
1. Always verify LLM-generated citations
2. Cross-reference claims with primary sources
3. Use retrieval-augmented generation (RAG) grounded in real papers

### Reasoning Limitations

Current LLMs struggle with:
- Multi-step mathematical proofs
- Causal reasoning vs. correlation
- Novel experimental design
- Uncertainty quantification

\`\`\`python
# Example of LLM reasoning failure
question = "If A causes B, and B correlates with C, does A cause C?"

# LLMs often incorrectly answer "yes" despite this being a
# classic causal inference fallacy
\`\`\`

### Bias in Scientific Literature

LLMs trained on existing literature inherit:
- Publication bias (positive results overrepresented)
- Geographic and institutional biases
- Historical misconceptions

## Best Practices for Researchers

### 1. Use LLMs as Assistants, Not Oracles

\`\`\`python
# Good: LLM generates initial draft, human reviews and verifies
draft = llm.generate(prompt)
verified_content = human_review(draft, check_citations=True)

# Bad: Blindly trusting LLM output
final_paper = llm.generate("Write a paper about X")
\`\`\`

### 2. Implement Verification Pipelines

\`\`\`python
def verify_scientific_claim(claim, llm):
    """
    Multi-step verification of LLM-generated claims
    """
    # Step 1: Ask LLM to cite sources
    sources = llm.generate(f"Provide sources for: {claim}")

    # Step 2: Verify sources exist
    valid_sources = [s for s in sources if check_exists(s)]

    # Step 3: Cross-reference with actual source content
    for source in valid_sources:
        content = fetch_paper(source)
        if not llm.verify_claim_in_context(claim, content):
            return False, "Claim not supported by cited source"

    return True, valid_sources
\`\`\`

### 3. Document LLM Usage

Transparency about AI assistance is increasingly expected:

\`\`\`
Acknowledgments:
"This manuscript benefited from AI-assisted literature review
and code generation using [Model Name]. All AI-generated content
was verified by the authors."
\`\`\`

## The Future: AI Scientists?

Could LLMs eventually conduct independent research? The path might look like:

**Current State (2024):**
- Literature review assistance
- Code generation
- Writing assistance

**Near Future (2025-2027):**
- Autonomous hypothesis generation
- Experiment design suggestions
- Automated replication studies

**Longer Term (2028+):**
- Closed-loop research systems
- AI-designed experiments
- Novel scientific discoveries

However, fundamental challenges remain:
- True scientific creativity vs. pattern recombination
- Grounding in physical reality
- Ethical oversight and accountability

## My Perspective

As a PhD researcher, I use LLMs daily—for code debugging, literature discovery, and writing refinement. But I've learned their limitations:

1. **Trust but verify**: Every claim needs checking
2. **LLMs excel at iteration, not origination**: They're great at refining ideas, less so at generating truly novel ones
3. **The human remains essential**: Scientific judgment, ethical consideration, and creative insight are still uniquely human

The most effective researchers will be those who learn to collaborate with AI while maintaining critical thinking. LLMs are powerful tools, but like any tool, their value depends on the skill of the user.

---

*Interested in AI for science? Check out my research on multi-agent systems and synthetic data generation at [Google Scholar](https://scholar.google.com/citations?user=A8J42tQAAAAJ).*
    `
    }
];

export function getBlogBySlug(slug: string): BlogPost | undefined {
    return blogPosts.find(post => post.slug === slug);
}

export function getBlogsByCategory(category: string): BlogPost[] {
    if (category === "all") return blogPosts;
    return blogPosts.filter(post => post.category === category);
}
