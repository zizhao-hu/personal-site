import { Brain, Shield, Sparkles, ScanEye } from "lucide-react";

export interface Project {
    title: string;
    slug: string;
    description: string;
    details: string[];
    tags: string[];
    status: "active" | "research" | "prototype" | "concept" | "completed";
    icon: React.ElementType;
    color: string;
    link?: string;
    github?: string;
    image?: string;
    content?: string;
}

export const projects: Project[] = [
    {
        title: "Physics-Based AI Image Detection",
        slug: "physics-ai-detection",
        description: "Detecting AI-generated images through physics-based reasoning — analyzing depth maps, brightness-depth consistency, and light estimation to expose how AI fails to model real-world physics. A 3-feature classifier achieves 68.3% accuracy with just depth gradients and brightness edge analysis.",
        details: [
            "Key finding: AI images are paradoxically too physically consistent — overly smooth brightness-depth relationships and symmetric depth distributions",
            "Yet AI images show sharper depth gradients (d=0.653), revealing lack of true 3D understanding",
            "3-feature model (grad_mean, brightness_at_depth_edges, n_valid_patches) outperforms full 27-feature model by 13.3pp",
            "27 physics features across depth statistics, brightness-depth coupling, and light estimation pipelines",
            "PCA reveals real/fake signal is not dominant — scene complexity dominates, requiring targeted feature selection"
        ],
        tags: ["Computer Vision", "AI Detection", "Physics-Based", "Depth Estimation", "Research"],
        status: "active",
        icon: ScanEye,
        color: "rose",
        github: "https://github.com/zizhao-hu/vlmdraw",
        image: "/images/projects/vlmdraw-depth.png",
        content: `
# Physics-Based AI Image Detection

Can you detect AI-generated images by checking if they obey the laws of physics? That's the core question behind this project.

## The Insight

AI image generators (DALL-E, Midjourney, Stable Diffusion) produce visually stunning images — but they don't actually understand physics. They approximate what scenes *look like* without modeling how light, depth, and surface interactions *actually work*.

This project exploits that gap.

## Three Physics Pipelines

We extract 27 features from three complementary physics-based analysis pipelines, applied to 30 real images (COCO) and 30 AI-generated images (AIGenBench):

### 1. Depth Map Statistics
Using monocular depth estimation, we analyze the statistical properties of predicted depth maps:
- **gradient mean/std** — how sharply depth transitions occur
- **skewness/kurtosis** — asymmetry and tail behavior of depth distributions
- **entropy** — information content of the depth field

### 2. Brightness-Depth Consistency
In real photographs, brightness and depth are physically coupled — objects farther away tend to have different illumination characteristics. We measure:
- **Pearson/Spearman correlation** between brightness and depth
- **Local patch correlations** — spatial consistency of the brightness-depth relationship
- **Brightness at depth edges** — what happens to brightness where depth changes sharply

### 3. Light Estimation
Real scenes have consistent lighting from a single dominant source. AI-generated images often have subtle inconsistencies:
- **Global residual** — how well a single light model fits the scene
- **Angular deviation** — variance in estimated light direction across patches
- **Fraction anomalous patches** — percentage of regions with inconsistent lighting

## Key Findings

### The Paradox of AI Images

![Depth Map Statistics: Real vs. AI-Generated](/images/projects/vlmdraw-depth.png)

**AI-generated images are simultaneously too consistent AND too sharp:**

| Finding | What It Means |
|---------|--------------|
| **Smooth brightness-depth** | AI produces overly uniform brightness-depth relationships (d=0.445) |
| **Symmetric depth distributions** | Real images have skewed depth (0.556 vs 0.236) — AI makes everything too balanced |
| **More uniform lighting** | Fewer anomalous patches in AI images (41% vs 51%) |
| **But sharper depth gradients** | AI images have abrupt depth transitions (d=0.653) — the strongest signal |

The last finding is the most interesting: **AI generators create sharp, almost cartoon-like depth boundaries** because they haven't learned that real-world depth transitions are gradual (due to actual 3D geometry, not learned textures).

### Feature Ranking by Effect Size

The top discriminative features ranked by Cohen's d:

| Rank | Feature | Cohen's d | Direction |
|------|---------|-----------|-----------|
| 1 | Depth gradient mean | **0.653** | Fake > Real |
| 2 | Brightness at depth edges | **0.547** | Real > Fake |
| 3 | Local correlation (abs mean) | 0.445 | Fake > Real |
| 4 | Fraction strong correlation | 0.428 | Fake > Real |
| 5 | Gradient magnitude correlation | 0.376 | Fake > Real |
| 6 | Depth skewness | 0.356 | Real > Fake |
| 7 | Fraction anomalous patches | 0.307 | Real > Fake |

### Less Is More: The Classifier

![Brightness-Depth Features: Real vs. AI-Generated](/images/projects/vlmdraw-brightness.png)

We trained logistic regression classifiers with leave-one-out cross-validation:

| Model | Accuracy | F1 | Train Acc | Gap |
|-------|----------|-----|-----------|-----|
| All 27 features | 55.0% | 0.542 | 81.7% | 26.7pp |
| **3 features** | **68.3%** | **0.678** | 71.7% | **3.4pp** |

The 3-feature model uses only **grad_mean**, **brightness_at_depth_edges**, and **n_valid_patches** — and generalizes far better because:
- 27 features overfit on 60 samples
- The top features capture complementary physics violations
- Minimal train/test gap (3.4pp) indicates genuine signal, not memorization

### The Decision Rule

Classify as **FAKE** if:

\`\`\`
1.43 × z(grad_mean) − 1.26 × z(brightness_at_depth_edges) + 0.04 > 0
\`\`\`

Translation: an image is likely AI-generated if it has **sharp depth gradients** (doesn't understand 3D geometry) combined with **low brightness variation at depth edges** (doesn't understand light-surface interaction).

### PCA Analysis

![Light Estimation Features: Real vs. AI-Generated](/images/projects/vlmdraw-light.png)

27 features compress into 6 principal components capturing ~77% of variance:

| PC | Variance | Interpretation |
|----|----------|----------------|
| PC1 | 24.1% | Scene complexity (entropy, skewness, kurtosis) |
| PC2 | 15.3% | Lighting quality (angular deviation, anomalous patches) |
| PC3 | 13.8% | Brightness-depth coupling (pearson_r, spearman_r) |
| PC4 | 10.6% | Depth distribution shape (std, iqr, grad_mean) |
| PC5 | 7.5% | Information content (mutual_information) |

**Critical insight**: The real/fake signal is NOT the dominant axis of variation. Scene-level variation (complexity, depth range) dominates, which is why targeted feature selection outperforms using all features.

## Limitations & Future Work

- **Sample size**: 30 images per class is a proof-of-concept — scaling to thousands would strengthen results
- **Generator diversity**: Tested on AIGenBench; extending to DALL-E 3, Midjourney v6, Flux would test generalizability
- **Depth estimator dependency**: Results depend on the quality of the monocular depth model
- **Complementary approaches**: Physics features could be combined with frequency-domain or learned detectors for higher accuracy

## Conclusion

This project demonstrates that **physics-based reasoning can detect AI-generated images** without any training on specific generators. The key insight — AI images are paradoxically too consistent while having unnaturally sharp depth transitions — reveals a fundamental limitation of current image generators: they learn to approximate visual appearance without genuinely understanding the physical world that produces those appearances.
`
    },
    {
        title: "DREAM-C2L: Continual Learning Framework",
        slug: "dream-c2l",
        description: "Open-source framework for continual learning research. Enabling AI systems to learn continuously without catastrophic forgetting, adapting to new data while preserving prior knowledge.",
        details: [
            "Difficulty-aware sample ordering algorithms",
            "Replay-based and regularization methods for knowledge retention",
            "Reproducible experiment pipelines for HPC clusters",
            "Integration with PyTorch Lightning and Weights & Biases"
        ],
        tags: ["Continual Learning", "PyTorch", "Open Source", "Research"],
        status: "active",
        icon: Brain,
        color: "green",
        github: "https://github.com/zizhao-hu/dream-c2l",
        image: "/images/projects/dream.png",
        content: `
# DREAM-C2L: Continual Learning Framework

An open-source framework for continual learning research, designed for reproducibility and scalability on HPC clusters.

## What Is Continual Learning?

Traditional neural networks suffer from **catastrophic forgetting** — when trained on new data, they lose performance on previously learned tasks. Continual learning aims to solve this: how can a model learn new things without forgetting old ones?

## The DREAM Framework

DREAM-C2L (Difficulty-aware REplay And Memory for Curriculum-to-Lifelong learning) introduces a principled approach to ordering and replaying training samples.

### Core Ideas

- **Curriculum-aware sample ordering**: Instead of random shuffling, order training examples by difficulty. Easy samples first builds a strong foundation; hard samples later refine decision boundaries.
- **Replay buffer management**: Maintain a balanced memory of past experiences, selected to maximize coverage of the learned distribution.
- **Regularization**: Constrain how much model weights can change when learning new tasks, preventing catastrophic forgetting.

### Key Features

- **Modular pipeline**: Swap replay strategies, regularization methods, and difficulty metrics independently
- **HPC-ready**: Built-in SLURM job management, multi-GPU training, checkpointing
- **Reproducible**: Full experiment tracking with Weights & Biases integration
- **PyTorch Lightning backbone**: Clean training loops with automatic mixed precision

## Research Applications

The framework supports multiple continual learning scenarios:
- **Class-incremental**: New classes appear over time
- **Task-incremental**: New tasks with explicit boundaries
- **Domain-incremental**: Same task, shifting data distributions

## Results

Our difficulty-aware approach shows consistent improvements over random ordering baselines across CIFAR-100, TinyImageNet, and ImageNet-subset benchmarks.

The key insight: **the order in which a model sees data matters as much as what data it sees**. By presenting examples in a curriculum-informed order, the model builds more robust internal representations that resist forgetting.
`
    },

    {
        title: "Project Canary",
        slug: "project-canary",
        description: "Foundational MOVE Fellowship project (Sept-Oct 2025) — a community-driven effort to train and refine frontier AI models. Completed 15,000+ tasks across 15 domains, improving Review 1 approval rates from 10% to 40%.",
        details: [
            "High-volume task generation across CS, Math, Medicine, Physics domains",
            "Core contributor in Computer Science domain",
            "Quality improvement: raised approval rates from 10% to 40%",
            "Precursor to Project Orion's specialized refinement phase"
        ],
        tags: ["AI Training", "Data Generation", "Handshake AI", "MOVE Fellowship"],
        status: "completed",
        icon: Shield,
        color: "green",
        content: `
# Project Canary — MOVE Fellowship Foundation

The foundational phase of the MOVE Fellowship at Handshake AI (Sept–Oct 2025), focused on large-scale task generation to train and refine frontier AI models.

## Overview

Project Canary was a community-driven initiative where subject-matter experts contributed training tasks across 15 academic domains. The goal: generate high-quality, diverse training data that would push frontier models toward deeper domain expertise.

## My Contribution

As a **core contributor in the Computer Science domain**, I:
- Generated and reviewed tasks covering algorithms, data structures, systems design, machine learning, and software engineering
- Contributed to over **15,000 tasks** across the fellowship
- Focused on tasks requiring PhD-level reasoning — problems that couldn't be solved by simply retrieving information

## Impact

### Quality Metrics
The most significant achievement was improving task quality:

| Metric | Before | After |
|--------|--------|-------|
| Review 1 approval rate | **10%** | **40%** |
| Tasks completed | — | 15,000+ |
| Domains covered | — | 15 |

A 4× improvement in first-review approval rates meant significantly less rework, faster iteration, and higher-quality training data reaching the model.

### Lessons Learned

1. **Domain expertise matters**: Generic annotators produce generic data. PhD-level contributors produce training signal that actually moves the needle on hard problems.
2. **Quality over quantity**: A single well-crafted reasoning task is worth more than a hundred trivial ones.
3. **Review feedback loops**: Tight review cycles with specific feedback accelerate quality improvement dramatically.

## Legacy

Project Canary laid the groundwork for **Project Orion** — the specialized refinement phase that followed, where the broad data generation shifted to targeted reasoning, safety, and red-teaming work.
`
    },
    {
        title: "Project Orion",
        slug: "project-orion",
        description: "Advanced MOVE Fellowship phase (Nov 2025) — specialized refinement of frontier AI models. Focused on high-quality reasoning chains, safety injections, and red-teaming through jailbreak testing. One-month intensive following Project Canary.",
        details: [
            "High-quality reasoning refinement and chain-of-thought improvement",
            "Safety injection tasks: embedding guardrails into model behavior",
            "Red-teaming and jailbreak testing for frontier models",
            "Built on Canary foundations with deeper specialization in CS domain"
        ],
        tags: ["AI Safety", "Reasoning", "Red-Teaming", "Handshake AI", "MOVE Fellowship"],
        status: "completed",
        icon: Sparkles,
        color: "orange",
        content: `
# Project Orion — Advanced Model Refinement

The advanced phase of the MOVE Fellowship at Handshake AI (Nov 2025), shifting from broad data generation to specialized model refinement.

## From Canary to Orion

While Project Canary focused on volume and coverage across 15 domains, **Project Orion** narrowed the focus to three critical areas:

1. **Reasoning refinement** — improving how models think through complex problems
2. **Safety injections** — embedding guardrails directly into model behavior
3. **Red-teaming** — finding and addressing vulnerabilities through jailbreak testing

## Reasoning Refinement

### Chain-of-Thought Improvement
Not just getting the right answer, but ensuring the model's *reasoning process* is sound:
- Identifying cases where models reach correct answers through flawed logic
- Rewiring reasoning chains to be logically sound, step-by-step
- Creating training examples that demonstrate expert-level problem decomposition

### Quality Over Previous Phase
Where Canary tasks tested "can you solve this?", Orion tasks tested "can you solve this *correctly, for the right reasons, showing your work*?"

## Safety Injections

### Embedding Guardrails
Safety isn't a filter bolted on after training — it needs to be woven into the model's core behavior:
- Creating scenarios where the model must recognize and refuse harmful requests
- Building training data for graceful refusals that explain *why* something is problematic
- Edge cases: requests that seem benign but could enable harm

### The Subtlety Problem
The hardest cases aren't obvious "how do I build a bomb" requests. They're:
- Multi-step requests where each step seems innocent
- Context-dependent requests that require judgment
- Dual-use knowledge that has both legitimate and harmful applications

## Red-Teaming & Jailbreak Testing

### Finding Vulnerabilities
Systematically probing the model for failure modes:
- **Prompt injection**: Embedding instructions that override system prompts
- **Role-play attacks**: Getting models to "play a character" that bypasses safety
- **Encoding tricks**: Using base64, rot13, or other encodings to sneak past filters
- **Multi-turn manipulation**: Slowly escalating across a conversation

### Why This Matters
Every vulnerability found in testing is one that won't be exploited in production. Red-teaming is the immune system of AI safety.

## Impact

Building on Canary's foundation, Orion produced targeted, high-impact training data that directly improved model reasoning quality and safety behavior. The one-month intensive demonstrated that focused expert refinement can achieve more than months of broad data generation.
`
    },

];

export function getProjectBySlug(slug: string): Project | undefined {
    return projects.find(p => p.slug === slug);
}
