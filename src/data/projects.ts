import { Brain, Globe, Cpu, Shield, Sparkles, Eye, Zap, ScanEye } from "lucide-react";

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
        title: "ReasonChain: Test-Time Compute Scaling",
        slug: "reasonchain",
        description: "Research prototype exploring how to make LLMs 'think longer' before responding. Implementing chain-of-thought verification where models check their own reasoning before committing to an answer.",
        details: [
            "Multi-step reasoning with self-verification loops",
            "Confidence calibration and uncertainty quantification",
            "Dynamic compute allocation based on problem complexity",
            "Hallucination detection through reasoning trace analysis"
        ],
        tags: ["Test-Time Compute", "Reasoning", "LLM Safety", "Research"],
        status: "research",
        icon: Zap,
        color: "yellow",
        content: `
# ReasonChain: Test-Time Compute Scaling

Can we make language models more reliable by letting them "think longer" on hard problems?

## The Problem

Large language models are fast but error-prone. They generate responses in a single forward pass, with no mechanism to double-check their own work. This leads to:
- **Confident hallucinations**: stating incorrect facts with high confidence
- **Reasoning errors**: making logical mistakes in multi-step problems
- **Inconsistency**: giving different answers to the same question

## The Approach

ReasonChain implements **test-time compute scaling** — allocating more computation at inference for harder problems, instead of spending it all during training.

### Core Components

1. **Chain-of-Thought Verification**: The model generates a reasoning trace, then a separate verification pass checks each step for logical consistency
2. **Confidence Calibration**: Before answering, the model estimates its own uncertainty — flagging low-confidence responses for additional processing
3. **Dynamic Compute Allocation**: Easy questions get fast responses; hard questions trigger multiple reasoning attempts that are then compared and reconciled

### How It Works

\`\`\`
Input Question
    ↓
Initial Reasoning (Chain of Thought)
    ↓
Self-Verification Pass
    ↓ (if inconsistency detected)
Re-reasoning with Constraints
    ↓
Confidence Estimation
    ↓ (if low confidence)
Multiple Independent Attempts → Consensus
    ↓
Final Answer
\`\`\`

## Key Insight

The industry is shifting from "bigger models" to "smarter inference." A smaller model that checks its own work can outperform a larger model that doesn't. This aligns with OpenAI's o1/o3 approach and DeepMind's research on test-time compute.

## Status

This is an active research prototype. Current focus areas:
- Benchmarking against standard LLM reasoning tasks (GSM8K, MATH, ARC)
- Measuring the compute-accuracy tradeoff
- Exploring when verification helps vs. when it hurts (simple factual questions don't benefit)
`
    },
    {
        title: "VisionGround: World Models for Physical AI",
        slug: "visionground",
        description: "Building AI that understands cause-and-effect in the physical world. Training models on video data to predict outcomes—if a glass falls, it breaks. Critical foundation for robotics applications.",
        details: [
            "Video prediction models for physical dynamics",
            "Cause-effect reasoning from visual observations",
            "Sim-to-real transfer for robotic manipulation",
            "Multimodal fusion of vision, language, and proprioception"
        ],
        tags: ["World Models", "Robotics", "Video Understanding", "Embodied AI"],
        status: "research",
        icon: Globe,
        color: "cyan",
        image: "/images/projects/visionground.png",
        content: `
# VisionGround: World Models for Physical AI

Training AI to understand cause-and-effect in the physical world — the foundation for robotics and embodied intelligence.

## The Vision

Current AI can describe a scene in perfect English but has no idea what will happen next. Drop a glass? It shatters. Push a ball off a table? It falls. These are trivial for humans but deeply challenging for AI systems that have only learned from text and static images.

**VisionGround** bridges this gap by learning *world models* — internal simulations of how the physical world works.

## Approach

### Video Prediction
Train models on large-scale video datasets to predict what happens next. Not pixel-perfect prediction, but understanding the *dynamics*:
- Objects fall when unsupported
- Liquids flow and splash
- Rigid objects bounce, deformable objects compress
- Forces propagate through contact chains

### Causal Reasoning
Go beyond correlation to causation:
- **Intervention**: "If I push this block, what happens to the tower?"
- **Counterfactual**: "Would the tower have fallen if the block were heavier?"
- **Transfer**: Apply learned physics to novel objects and scenarios

### Sim-to-Real Pipeline
1. Learn physics in simulation (fast, cheap, diverse)
2. Fine-tune on real video (ground truth dynamics)
3. Deploy on robots (closed-loop control)

## Why This Matters

The robotics revolution is bottlenecked by *understanding*, not *manipulation*. Modern robot arms are mechanically capable. What they lack is a model of the world that lets them predict consequences of actions.

World models are the missing piece between "AI that talks" and "AI that acts."

## Research Directions

- **Multimodal fusion**: Combine vision, language instructions, and proprioceptive feedback
- **Compositional generalization**: Understand novel object combinations from familiar components
- **Safety-critical prediction**: Reliable prediction for high-stakes manipulation tasks
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
    {
        title: "EdgeLLM: Sovereign AI on Device",
        slug: "edgellm",
        description: "Exploring efficient small language models that run entirely on-device. Privacy-preserving AI that never sends data to the cloud—your AI assistant that respects your data sovereignty.",
        details: [
            "Model quantization and pruning for edge deployment",
            "On-device fine-tuning with federated learning",
            "Specialized domain adapters for legal, medical, code",
            "Offline-first architecture with optional cloud sync"
        ],
        tags: ["Edge AI", "Privacy", "Small Models", "Mobile"],
        status: "prototype",
        icon: Cpu,
        color: "indigo",
        content: `
# EdgeLLM: Sovereign AI on Device

What if your AI assistant never sent a single byte to the cloud?

## The Promise

Large language models are powerful but centralized. Every query you send to ChatGPT, Claude, or Gemini travels to a data center, gets processed, and comes back. Your data — your questions, your documents, your thoughts — all pass through someone else's servers.

**EdgeLLM** explores a different model: AI that runs entirely on your device.

## Technical Approach

### Model Compression
Making LLMs small enough to run on phones and laptops:
- **Quantization**: Reducing model precision from 32-bit to 4-bit (8× smaller) with minimal quality loss
- **Pruning**: Removing redundant neurons and attention heads
- **Distillation**: Training small models to mimic large ones

### Specialized Adapters
Instead of one giant general-purpose model, use small specialized modules:
- **Legal adapter**: Contract analysis, regulatory compliance
- **Medical adapter**: Symptom analysis, drug interactions (with appropriate disclaimers)
- **Code adapter**: Completion, refactoring, bug detection
- Each adapter adds <100MB on top of a base model

### Federated Learning
Improve the model from usage without sharing data:
- Train locally on user interactions
- Share only model weight updates (not data) for aggregation
- Each device contributes to collective improvement while keeping data private

## Why "Sovereign"?

Data sovereignty means you control where your data lives. With EdgeLLM:
- **No internet required**: Full functionality offline
- **No data leakage**: Queries never leave your device
- **No vendor lock-in**: Your AI, your rules
- **No latency**: Instant responses without network round-trips

## Status

Currently prototyping with:
- Llama 3.2 1B/3B quantized to 4-bit (runs on M1 MacBooks)
- ONNX runtime for cross-platform deployment
- Custom adapter training pipeline
`
    },
    {
        title: "SynthVision: Multimodal Data Generation",
        slug: "synthvision",
        description: "Pipeline for generating high-quality synthetic vision-language training data. Creating diverse, balanced datasets without the privacy concerns of web-scraped data.",
        details: [
            "Controllable image-text pair generation",
            "Automatic quality assessment and filtering",
            "Bias detection and mitigation in generated data",
            "Scalable generation with GPU-efficient diffusion models"
        ],
        tags: ["Synthetic Data", "Vision-Language", "Data Generation", "Diffusion"],
        status: "prototype",
        icon: Eye,
        color: "pink",
        content: `
# SynthVision: Multimodal Data Generation

Building the data factory for next-generation vision-language models — without scraping the internet.

## The Data Problem

Training multimodal AI requires millions of image-text pairs. Traditional approaches scrape the web, inheriting:
- **Privacy violations**: People's photos used without consent
- **Bias amplification**: Web data reflects societal biases
- **Quality issues**: Noisy, misaligned image-caption pairs
- **Legal risk**: Copyright and licensing concerns

**SynthVision** generates training data from scratch, giving full control over quality, diversity, and fairness.

## Pipeline Architecture

### 1. Controllable Generation
Specify what you need:
- Scene descriptions (indoor, outdoor, aerial, microscopic)
- Object compositions (specific objects in specific arrangements)
- Demographic representation (balanced across age, gender, ethnicity)
- Edge cases (unusual lighting, occlusion, rare objects)

### 2. Quality Assessment
Automatic filtering pipeline:
- **Image quality**: Resolution, artifacts, coherence
- **Text alignment**: Does the caption accurately describe the image?
- **Diversity metrics**: Are we generating enough variety, or mode-collapsing?
- **Bias detection**: Statistical checks for demographic imbalances

### 3. Scalable Generation
Optimized for throughput:
- Batch generation with GPU-efficient diffusion models
- Parallel text generation with LLMs
- Distributed processing across multiple nodes

## Applications

- **Training data augmentation**: Supplement real data with synthetic examples for underrepresented scenarios
- **Privacy-preserving medical imaging**: Generate realistic medical images without patient data
- **Robotics simulation**: Synthetic scenes for sim-to-real transfer
- **Bias mitigation**: Generate balanced datasets to counteract biases in real-world data

## Status

Prototype stage — currently benchmarking generated data quality against LAION-5B and CC-12M.
`
    },
];

export function getProjectBySlug(slug: string): Project | undefined {
    return projects.find(p => p.slug === slug);
}
