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
  author: {
    name: string;
    avatar: string;
  };
}

export const blogPosts: BlogPost[] = [
  {
    id: "1",
    slug: "continual-learning-neural-networks",
    title: "Understanding Continual Learning in Neural Networks",
    excerpt: "A deep dive into catastrophic forgetting and modern approaches to lifelong learning in AI systems.",
    date: "2024-01-10",
    readingTime: "12 min",
    category: "ai",
    tags: ["Continual Learning", "Neural Networks", "Deep Learning"],
    author: {
      name: "Zizhao Hu",
      avatar: "/profile.jpg"
    },
    content: `
# Understanding Continual Learning in Neural Networks

One of the most significant challenges in modern AI is building systems that can learn continuously without forgetting previously acquired knowledge. This problem, known as **catastrophic forgetting**, has been a fundamental limitation of neural networks since their inception.

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

\`\`\`
Task 1:  [Column 1]
Task 2:  [Column 1 (frozen)] -> [Column 2]
Task 3:  [Column 1 (frozen)] -> [Column 2 (frozen)] -> [Column 3]
\`\`\`

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
    author: {
      name: "Zizhao Hu",
      avatar: "/profile.jpg"
    },
    content: `
# Multi-Modal Learning: Bridging Vision and Language

Humans effortlessly combine information from multiple senses—we see an apple and simultaneously recall its taste, texture, and the word "apple." For decades, AI systems processed each modality in isolation. But recent breakthroughs in multi-modal learning are finally enabling machines to connect vision and language in powerful ways.

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

\`\`\`
Image -> Image Encoder -> [Image Embedding]
                                 ↓
                          Shared Space
                                 ↑
 Text ->  Text Encoder -> [Text Embedding]
\`\`\`

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

\`\`\`
[CLS] [Image Patch 1] ... [Image Patch N] [SEP] [Word 1] ... [Word M] [SEP]
                            ↓
                    Unified Transformer
                            ↓
                    [Joint Representation]
\`\`\`

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
    author: {
      name: "Zizhao Hu",
      avatar: "/profile.jpg"
    },
    content: `
# The Future of Large Language Models in Scientific Research

Large Language Models are no longer just impressive demos—they're becoming genuine tools for scientific discovery. From literature review to hypothesis generation, LLMs are reshaping how researchers work. But what can they actually do today, and what are their limitations?

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
