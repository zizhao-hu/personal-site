export interface Tutorial {
    id: string;
    slug: string;
    title: string;
    description: string;
    difficulty: "beginner" | "intermediate" | "advanced";
    estimatedTime: string;
    topics: string[];
    series?: string;
    content: string;
    prerequisites?: string[];
}

export const tutorials: Tutorial[] = [
    {
        id: "1",
        slug: "getting-started-webllm",
        title: "Getting Started with WebLLM",
        description: "Learn how to run large language models directly in the browser using WebGPU. No server required!",
        difficulty: "beginner",
        estimatedTime: "15 min",
        topics: ["WebLLM", "WebGPU", "JavaScript"],
        series: "WebLLM Fundamentals",
        prerequisites: ["Basic JavaScript knowledge", "Familiarity with async/await"],
        content: `
# Getting Started with WebLLM

WebLLM enables running large language models directly in your browser using WebGPU acceleration. This means you can build AI-powered applications without sending data to external servers—everything runs locally on the user's device.

## Prerequisites

Before we begin, make sure you have:
- A browser that supports WebGPU (Chrome 113+ or Edge 113+)
- Basic knowledge of JavaScript and async/await
- Node.js installed for the development setup

## What is WebLLM?

WebLLM is a library developed by the MLC AI team that brings large language models to the browser. It uses:

- **WebGPU**: A new web standard for GPU-accelerated computation
- **MLC-LLM**: Machine Learning Compilation for efficient model execution
- **Quantized Models**: Compressed model weights that fit in browser memory

## Step 1: Setting Up Your Project

First, create a new project and install WebLLM:

\`\`\`bash
# Create a new Vite project
npm create vite@latest webllm-demo -- --template vanilla-ts
cd webllm-demo

# Install WebLLM
npm install @mlc-ai/web-llm
\`\`\`

## Step 2: Basic WebLLM Setup

Create a simple chat interface. In your \`main.ts\`:

\`\`\`typescript
import * as webllm from "@mlc-ai/web-llm";

// Available models - smaller models load faster
const MODEL_ID = "SmolLM2-360M-Instruct-q4f16_1-MLC";

class WebLLMChat {
  private engine: webllm.MLCEngine | null = null;

  async initialize(onProgress: (progress: string) => void) {
    // Create the engine
    this.engine = new webllm.MLCEngine();

    // Set up progress callback
    this.engine.setInitProgressCallback((report) => {
      onProgress(\`Loading: \${report.text} (\${Math.round(report.progress * 100)}%)\`);
    });

    // Load the model
    await this.engine.reload(MODEL_ID);
    onProgress("Ready!");
  }

  async chat(message: string): Promise<string> {
    if (!this.engine) throw new Error("Engine not initialized");

    const response = await this.engine.chat.completions.create({
      messages: [{ role: "user", content: message }],
      temperature: 0.7,
      max_tokens: 256,
    });

    return response.choices[0].message.content || "";
  }
}

// Usage
const chat = new WebLLMChat();

document.querySelector<HTMLDivElement>("#app")!.innerHTML = \`
  <div>
    <h1>WebLLM Chat</h1>
    <div id="status">Initializing...</div>
    <input type="text" id="input" placeholder="Type a message..." disabled />
    <button id="send" disabled>Send</button>
    <div id="response"></div>
  </div>
\`;

const statusEl = document.getElementById("status")!;
const inputEl = document.getElementById("input") as HTMLInputElement;
const sendBtn = document.getElementById("send") as HTMLButtonElement;
const responseEl = document.getElementById("response")!;

// Initialize
chat.initialize((status) => {
  statusEl.textContent = status;
  if (status === "Ready!") {
    inputEl.disabled = false;
    sendBtn.disabled = false;
  }
});

// Handle send
sendBtn.addEventListener("click", async () => {
  const message = inputEl.value;
  if (!message) return;

  sendBtn.disabled = true;
  responseEl.textContent = "Thinking...";

  const response = await chat.chat(message);
  responseEl.textContent = response;

  sendBtn.disabled = false;
  inputEl.value = "";
});
\`\`\`

## Step 3: Adding Streaming Responses

For a better user experience, stream the response token by token:

\`\`\`typescript
async chatStream(
  message: string,
  onToken: (token: string) => void
): Promise<void> {
  if (!this.engine) throw new Error("Engine not initialized");

  const response = await this.engine.chat.completions.create({
    messages: [{ role: "user", content: message }],
    temperature: 0.7,
    max_tokens: 256,
    stream: true,  // Enable streaming
  });

  // Iterate over the stream
  for await (const chunk of response) {
    const token = chunk.choices[0]?.delta?.content || "";
    if (token) {
      onToken(token);
    }
  }
}

// Usage with streaming
let fullResponse = "";
await chat.chatStream(message, (token) => {
  fullResponse += token;
  responseEl.textContent = fullResponse;
});
\`\`\`

## Step 4: Handling Model Loading

Model loading can take time (downloading ~200MB-2GB depending on the model). Here's how to provide good feedback:

\`\`\`typescript
interface LoadingState {
  stage: "downloading" | "caching" | "initializing" | "ready";
  progress: number;
  text: string;
}

function parseProgress(report: webllm.InitProgressReport): LoadingState {
  const text = report.text.toLowerCase();

  if (text.includes("fetch")) {
    return {
      stage: "downloading",
      progress: report.progress,
      text: "Downloading model weights..."
    };
  }

  if (text.includes("cache")) {
    return {
      stage: "caching",
      progress: report.progress,
      text: "Caching model for faster loads..."
    };
  }

  if (text.includes("loading") || text.includes("init")) {
    return {
      stage: "initializing",
      progress: report.progress,
      text: "Initializing model..."
    };
  }

  return {
    stage: "ready",
    progress: 1,
    text: "Ready!"
  };
}
\`\`\`

## Step 5: Conversation Memory

For multi-turn conversations, maintain message history:

\`\`\`typescript
interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}

class ConversationalChat {
  private engine: webllm.MLCEngine | null = null;
  private messages: Message[] = [];

  // Add system prompt
  setSystemPrompt(prompt: string) {
    this.messages = [{ role: "system", content: prompt }];
  }

  async chat(userMessage: string): Promise<string> {
    if (!this.engine) throw new Error("Engine not initialized");

    // Add user message to history
    this.messages.push({ role: "user", content: userMessage });

    // Get response with full history
    const response = await this.engine.chat.completions.create({
      messages: this.messages,
      temperature: 0.7,
      max_tokens: 256,
    });

    const assistantMessage = response.choices[0].message.content || "";

    // Add assistant response to history
    this.messages.push({ role: "assistant", content: assistantMessage });

    return assistantMessage;
  }

  clearHistory() {
    // Keep system prompt if present
    this.messages = this.messages.filter(m => m.role === "system");
  }
}
\`\`\`

## Available Models

WebLLM supports various models. Smaller models are faster but less capable:

| Model | Size | Use Case |
|-------|------|----------|
| SmolLM2-360M-Instruct | ~200MB | Quick responses, simple tasks |
| Llama-3.2-1B-Instruct | ~600MB | Better quality, still fast |
| Llama-3.2-3B-Instruct | ~1.5GB | Good balance of speed/quality |
| Phi-3.5-mini-instruct | ~2GB | Strong reasoning |

## Common Issues

### WebGPU Not Available

\`\`\`typescript
async function checkWebGPU(): Promise<boolean> {
  if (!navigator.gpu) {
    console.error("WebGPU not supported in this browser");
    return false;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    console.error("No GPU adapter found");
    return false;
  }

  return true;
}
\`\`\`

### Memory Issues

If the model fails to load, try a smaller model or check available GPU memory.

## Next Steps

Now that you have basic WebLLM working:

1. **Add a proper UI**: Build a chat interface with message history
2. **Implement error handling**: Handle network issues and GPU errors gracefully
3. **Explore other models**: Try different models for your use case
4. **Add features**: Implement copy-to-clipboard, message editing, etc.

In the next tutorial, we'll build a complete React chat interface with WebLLM.

---

*This tutorial is part of the WebLLM Fundamentals series.*
    `
    },
    {
        id: "2",
        slug: "llm-from-scratch-preparation",
        title: "Train Your LLM from Scratch — Stage 0: Preparation",
        description: "Set up the environment, build a tokenizer, and create the data pipeline for training a language model from scratch.",
        difficulty: "advanced",
        estimatedTime: "35 min",
        topics: ["LLM Training", "PyTorch", "Tokenization", "Data Pipeline"],
        series: "Train Your LLM from Scratch",
        prerequisites: ["Python proficiency", "PyTorch basics", "Understanding of transformer architecture"],
        content: `
# Stage 0: Preparation

Before writing a single training loop, you need three things: an environment that won't crash mid-run, a tokenizer that can represent your data, and a data pipeline that feeds batches efficiently. Skip any of these and you'll waste GPU hours debugging.

## Environment Setup

\`\`\`bash
# Create a clean environment
conda create -n llm-train python=3.11 -y
conda activate llm-train

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tokenizers wandb accelerate
pip install flash-attn --no-build-isolation  # FlashAttention-2

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
\`\`\`

## Building a Tokenizer

We'll train a BPE tokenizer from scratch using HuggingFace \\\`tokenizers\\\`. This is the same approach used by Llama, GPT, and Mistral.

\`\`\`python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def train_tokenizer(
    data_files: list[str],
    vocab_size: int = 32_000,
    save_path: str = "tokenizer.json"
):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|pad|>", "<|eos|>", "<|bos|>", "<|unk|>"],
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train(data_files, trainer)
    tokenizer.save(save_path)
    print(f"Trained tokenizer: {tokenizer.get_vocab_size()} tokens")
    return tokenizer

# Train on your corpus
tokenizer = train_tokenizer(["corpus_part1.txt", "corpus_part2.txt"])

# Test it
encoded = tokenizer.encode("The transformer architecture revolutionized NLP.")
print(f"Tokens: {encoded.tokens}")
print(f"IDs:    {encoded.ids}")
\`\`\`

**Key decisions:**
- **Vocab size**: 32K is a good default. Larger (64K) improves multilingual; smaller (16K) reduces embedding params
- **Special tokens**: At minimum you need pad, eos, bos, unk. Add \\\`<|im_start|>\\\` and \\\`<|im_end|>\\\` if you plan to instruction-tune later

## Data Pipeline

Efficient data loading is critical. We pack multiple documents into fixed-length sequences to maximize GPU utilization:

\`\`\`python
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

class PretrainingDataset(Dataset):
    """Concatenates all documents and chunks into fixed-length sequences."""

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        seq_length: int = 2048,
        stride: int = 2048,  # No overlap by default
    ):
        self.seq_length = seq_length
        self.stride = stride

        # Tokenize and concatenate all files
        all_ids = []
        for file in sorted(Path(data_dir).glob("*.txt")):
            text = file.read_text(encoding="utf-8")
            encoded = tokenizer.encode(text)
            all_ids.extend(encoded.ids)
            all_ids.append(tokenizer.token_to_id("<|eos|>"))

        self.data = np.array(all_ids, dtype=np.uint16)
        self.n_chunks = max(1, (len(self.data) - seq_length) // stride)
        print(f"Dataset: {len(self.data):,} tokens → {self.n_chunks:,} chunks")

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_length + 1  # +1 for target shift

        chunk = torch.tensor(self.data[start:end], dtype=torch.long)
        x = chunk[:-1]   # Input
        y = chunk[1:]    # Target (shifted by 1)
        return x, y

def create_dataloaders(
    train_dir: str,
    val_dir: str,
    tokenizer,
    batch_size: int = 8,
    seq_length: int = 2048,
):
    train_ds = PretrainingDataset(train_dir, tokenizer, seq_length)
    val_ds = PretrainingDataset(val_dir, tokenizer, seq_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader
\`\`\`

## Data Preparation Checklist

| Step | Action | Why |
|------|--------|-----|
| 1 | Deduplicate documents | Prevents memorization of repeated text |
| 2 | Filter low-quality text | Removes boilerplate, ads, HTML artifacts |
| 3 | Shuffle at document level | Prevents domain clustering in batches |
| 4 | Split train/val (99/1) | Val set should be representative but small |
| 5 | Tokenize and save as binary | Avoids re-tokenizing every training run |

## Hyperparameter Cheat Sheet

For a ~125M parameter model (good for learning):

\`\`\`python
config = {
    "vocab_size": 32_000,
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "d_ff": 3072,          # 4 * d_model
    "seq_length": 2048,
    "dropout": 0.1,
    "batch_size": 8,       # Per GPU
    "gradient_accumulation": 4,
    "lr": 3e-4,
    "warmup_steps": 1000,
    "total_steps": 100_000,
    "weight_decay": 0.1,
}
\`\`\`

---

*Next: Stage 1 — Pretraining, where we build the model and write the training loop.*
    `
  },
    {
        id: "4",
        slug: "llm-from-scratch-pretraining",
        title: "Train Your LLM from Scratch — Stage 1: Pretraining",
        description: "Build the GPT model architecture, implement the training loop with mixed precision, and validate with perplexity.",
        difficulty: "advanced",
        estimatedTime: "45 min",
        topics: ["LLM Training", "PyTorch", "Pretraining", "Mixed Precision"],
        series: "Train Your LLM from Scratch",
        prerequisites: ["Completion of Stage 0: Preparation"],
        content: `
# Stage 1: Pretraining

Pretraining is where a language model learns to predict the next token. This is the most compute-intensive phase — you're teaching the model the statistical patterns of language from raw text.

## Model Architecture

We'll build a decoder-only transformer (GPT-style). This is the same architecture used by GPT, Llama, and Mistral:

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """RMSNorm (used by Llama, Mistral instead of LayerNorm)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        # Project Q, K, V in one shot
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # [B, heads, T, d_k]

        # Scaled dot-product attention with causal mask
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """SwiGLU feedforward (used by Llama, Mistral)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params / 1e6:.1f}M")

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss
\`\`\`

## Training Loop

Here's the complete training loop with mixed precision, gradient accumulation, and cosine LR schedule:

\`\`\`python
from torch.cuda.amp import autocast, GradScaler
import wandb

def train(model, train_loader, val_loader, config):
    device = torch.device("cuda")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.95),
        weight_decay=config["weight_decay"],
    )

    # Cosine LR with warmup
    def lr_schedule(step):
        if step < config["warmup_steps"]:
            return step / config["warmup_steps"]
        progress = (step - config["warmup_steps"]) / (config["total_steps"] - config["warmup_steps"])
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    scaler = GradScaler()  # For mixed precision

    wandb.init(project="llm-from-scratch", config=config)

    global_step = 0
    for epoch in range(config.get("epochs", 1)):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with autocast(dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / config["gradient_accumulation"]

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config["gradient_accumulation"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % 100 == 0:
                    wandb.log({
                        "train/loss": loss.item() * config["gradient_accumulation"],
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/step": global_step,
                    })

                if global_step % 1000 == 0:
                    val_loss = validate(model, val_loader, device)
                    wandb.log({"val/loss": val_loss, "val/perplexity": math.exp(val_loss)})
                    print(f"Step {global_step} | Val loss: {val_loss:.4f} | PPL: {math.exp(val_loss):.2f}")
                    save_checkpoint(model, optimizer, global_step)

                if global_step >= config["total_steps"]:
                    return
\`\`\`

## Validation

Perplexity is the standard metric for pretraining — lower is better. A perplexity of 20 means the model is "20-way confused" on average.

\`\`\`python
@torch.no_grad()
def validate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0
    n_batches = 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with autocast(dtype=torch.bfloat16):
            _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1
        if n_batches >= max_batches:
            break

    model.train()
    return total_loss / n_batches

def save_checkpoint(model, optimizer, step, path="checkpoints"):
    Path(path).mkdir(exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }, f"{path}/step_{step}.pt")
\`\`\`

## What to Watch For

| Metric | Healthy | Unhealthy |
|--------|---------|-----------|
| Training loss | Smooth downward curve | Spikes, plateaus early |
| Val loss | Tracks train loss closely | Diverges from train loss |
| Gradient norm | Stable around 0.1–1.0 | Exploding (>10) or vanishing |
| Learning rate | Smooth warmup → cosine decay | — |
| Perplexity | Steadily decreasing | Stuck above 100 after 10K steps |

## Generating Text (Sanity Check)

\`\`\`python
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8):
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], device=device)

    for _ in range(max_new_tokens):
        logits, _ = model(x[:, -2048:])  # Truncate to max seq len
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        x = torch.cat([x, next_id], dim=1)

        if next_id.item() == tokenizer.token_to_id("<|eos|>"):
            break

    return tokenizer.decode(x[0].tolist())
\`\`\`

---

*Next: Stage 2 — Instruction Tuning, where we teach the model to follow instructions.*
    `
  },
    {
        id: "5",
        slug: "llm-from-scratch-instruction-tuning",
        title: "Train Your LLM from Scratch — Stage 2: Instruction Tuning",
        description: "Fine-tune your pretrained model to follow instructions using SFT, chat templates, and evaluation on benchmarks.",
        difficulty: "advanced",
        estimatedTime: "40 min",
        topics: ["LLM Training", "SFT", "Instruction Tuning", "LoRA"],
        series: "Train Your LLM from Scratch",
        prerequisites: ["Completion of Stage 1: Pretraining"],
        content: `
# Stage 2: Instruction Tuning

A pretrained model can complete text, but it can't follow instructions. Instruction tuning (SFT — Supervised Fine-Tuning) teaches the model to respond helpfully to user requests. This is the step that turns a "text completer" into a "chatbot."

## Chat Template

First, define a chat format your model will learn:

\`\`\`python
CHAT_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>"""

def format_conversation(system: str, user: str, assistant: str) -> str:
    return CHAT_TEMPLATE.format(
        system=system, user=user, assistant=assistant
    )

# Example
formatted = format_conversation(
    system="You are a helpful assistant.",
    user="What is gradient descent?",
    assistant="Gradient descent is an optimization algorithm..."
)
\`\`\`

## Instruction Dataset

\`\`\`python
from datasets import load_dataset
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, tokenizer, max_length=2048, split="train"):
        # Use a public instruction dataset
        raw = load_dataset("tatsu-lab/alpaca", split=split)
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for item in raw:
            instruction = item["instruction"]
            if item.get("input"):
                instruction += f"\\n\\nInput: {item['input']}"

            text = format_conversation(
                system="You are a helpful assistant.",
                user=instruction,
                assistant=item["output"],
            )
            ids = tokenizer.encode(text).ids
            if len(ids) <= max_length:
                self.samples.append(ids)

        print(f"Instruction dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]

        # Pad to max_length
        padded = ids + [self.tokenizer.token_to_id("<|pad|>")] * (self.max_length - len(ids))
        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)

        # Mask: only compute loss on the assistant's response
        # Find where assistant response starts
        y[:len(ids) // 2] = -1  # Simplified — mask system + user tokens
        return x, y
\`\`\`

## LoRA: Parameter-Efficient Fine-Tuning

Full fine-tuning updates all parameters. LoRA freezes the base model and adds small trainable matrices, reducing memory by ~10x:

\`\`\`python
class LoRALinear(nn.Module):
    """Low-Rank Adaptation for efficient fine-tuning."""
    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 32):
        super().__init__()
        self.base = base_layer
        self.base.weight.requires_grad = False  # Freeze base

        d_in, d_out = base_layer.in_features, base_layer.out_features
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.scale = alpha / rank

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scale
        return base_out + lora_out

def apply_lora(model, rank=16, alpha=32):
    """Apply LoRA to all attention projection layers."""
    for name, module in model.named_modules():
        if isinstance(module, CausalSelfAttention):
            module.qkv = LoRALinear(module.qkv, rank, alpha)
            module.proj = LoRALinear(module.proj, rank, alpha)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total ({100*trainable/total:.1f}%)")
\`\`\`

## SFT Training Loop

\`\`\`python
def instruction_tune(model, train_ds, val_ds, config):
    device = torch.device("cuda")
    model = model.to(device)

    # Only optimize LoRA parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5,            # Much lower LR than pretraining
        weight_decay=0.01,
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    for epoch in range(3):  # SFT typically needs only 1-3 epochs
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with autocast(dtype=torch.bfloat16):
                _, loss = model(x, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        val_loss = validate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Train: {epoch_loss/len(train_loader):.4f} | Val: {val_loss:.4f}")
\`\`\`

## Evaluation

After instruction tuning, evaluate on standard benchmarks:

\`\`\`python
def evaluate_instruction_following(model, tokenizer, test_prompts):
    """Manual evaluation of instruction following quality."""
    results = []
    for prompt in test_prompts:
        text = format_conversation(
            system="You are a helpful assistant.",
            user=prompt,
            assistant="",
        )
        # Remove the final <|im_end|> so the model generates the response
        text = text.rsplit("<|im_end|>", 1)[0]

        response = generate(model, tokenizer, text, max_new_tokens=256)
        results.append({"prompt": prompt, "response": response})
        print(f"Q: {prompt}")
        print(f"A: {response}\\n")
    return results

test_prompts = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to find the nth Fibonacci number.",
    "What are the pros and cons of remote work?",
    "Summarize the key ideas of the attention mechanism.",
]
\`\`\`

## SFT Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| Catastrophic forgetting | Model loses general knowledge | Lower LR, use LoRA, fewer epochs |
| Overfitting | Val loss increases after epoch 1 | More data, higher dropout, early stopping |
| Template leakage | Model outputs \\\`<|im_start|>\\\` in responses | Ensure proper masking of special tokens |
| Repetition | Model loops the same phrase | Add repetition penalty during generation |

---

*Next: Stage 3 — Reasoning, where we teach the model to think step by step.*
    `
  },
    {
        id: "6",
        slug: "llm-from-scratch-reasoning",
        title: "Train Your LLM from Scratch — Stage 3: Reasoning",
        description: "Add chain-of-thought reasoning, implement RLHF/DPO alignment, and evaluate on reasoning benchmarks.",
        difficulty: "advanced",
        estimatedTime: "45 min",
        topics: ["LLM Training", "RLHF", "DPO", "Chain-of-Thought", "Reasoning"],
        series: "Train Your LLM from Scratch",
        prerequisites: ["Completion of Stage 2: Instruction Tuning"],
        content: `
# Stage 3: Reasoning & Alignment

An instruction-tuned model follows commands, but it doesn't *think*. This stage teaches the model to reason step-by-step and aligns its behavior with human preferences using RLHF or DPO.

## Chain-of-Thought Training Data

The key insight: if you train on data that contains explicit reasoning steps, the model learns to reason. We create "thinking" traces:

\`\`\`python
COT_TEMPLATE = """<|im_start|>system
You are a helpful assistant. Think step by step before answering.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
<think>
{reasoning}
</think>

{answer}<|im_end|>"""

# Example training data
cot_examples = [
    {
        "question": "If a train travels 120 km in 2 hours, what is its speed in m/s?",
        "reasoning": """Step 1: Find speed in km/h.
Speed = distance / time = 120 km / 2 h = 60 km/h.

Step 2: Convert km/h to m/s.
1 km = 1000 m, 1 h = 3600 s.
60 km/h = 60 × 1000 / 3600 = 16.67 m/s.""",
        "answer": "The train's speed is approximately 16.67 m/s."
    },
    {
        "question": "A store has a 25% off sale. If an item costs $80, and there's an additional 10% member discount applied after, what's the final price?",
        "reasoning": """Step 1: Apply the 25% sale discount.
25% of $80 = $20. Price after sale: $80 - $20 = $60.

Step 2: Apply the 10% member discount on the sale price.
10% of $60 = $6. Price after member discount: $60 - $6 = $54.

Step 3: The discounts are applied sequentially, not combined.
Total discount is not 35% — it's 25% then 10% of the reduced price.""",
        "answer": "The final price is $54.00."
    },
]
\`\`\`

## Rejection Sampling for Reasoning Data

Generate multiple attempts and keep only the ones that arrive at the correct answer:

\`\`\`python
def generate_cot_data(model, tokenizer, problems, n_samples=8, temperature=0.7):
    """Generate chain-of-thought data via rejection sampling."""
    good_samples = []

    for problem in problems:
        prompt = f"""<|im_start|>system
Think step by step.<|im_end|>
<|im_start|>user
{problem['question']}<|im_end|>
<|im_start|>assistant
<think>
"""
        candidates = []
        for _ in range(n_samples):
            response = generate(model, tokenizer, prompt,
                              max_new_tokens=512, temperature=temperature)
            # Check if the final answer matches
            if problem["answer"] in response:
                candidates.append(response)

        if candidates:
            # Keep the shortest correct reasoning (Occam's razor)
            best = min(candidates, key=len)
            good_samples.append({
                "question": problem["question"],
                "response": best,
            })

    print(f"Generated {len(good_samples)}/{len(problems)} valid CoT samples")
    return good_samples
\`\`\`

## DPO: Direct Preference Optimization

DPO is a simpler alternative to RLHF. Instead of training a reward model, you directly optimize on preference pairs (chosen vs rejected):

\`\`\`python
class DPOTrainer:
    def __init__(self, model, ref_model, tokenizer, beta=0.1, lr=5e-7):
        self.model = model
        self.ref_model = ref_model  # Frozen copy of the SFT model
        self.tokenizer = tokenizer
        self.beta = beta

        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01
        )

    def compute_log_probs(self, model, input_ids, labels):
        """Compute log probabilities of the target tokens."""
        logits, _ = model(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        # Mask padding
        mask = (labels != -1).float()
        return (token_log_probs * mask).sum(-1) / mask.sum(-1)

    def dpo_loss(self, chosen_ids, chosen_labels, rejected_ids, rejected_labels):
        """DPO loss: maximize margin between chosen and rejected."""
        # Policy log probs
        pi_chosen = self.compute_log_probs(self.model, chosen_ids, chosen_labels)
        pi_rejected = self.compute_log_probs(self.model, rejected_ids, rejected_labels)

        # Reference log probs (no gradient)
        with torch.no_grad():
            ref_chosen = self.compute_log_probs(self.ref_model, chosen_ids, chosen_labels)
            ref_rejected = self.compute_log_probs(self.ref_model, rejected_ids, rejected_labels)

        # DPO objective
        pi_logratios = pi_chosen - pi_rejected
        ref_logratios = ref_chosen - ref_rejected
        logits = self.beta * (pi_logratios - ref_logratios)

        loss = -F.logsigmoid(logits).mean()
        return loss

    def train_step(self, batch):
        loss = self.dpo_loss(
            batch["chosen_ids"], batch["chosen_labels"],
            batch["rejected_ids"], batch["rejected_labels"],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
\`\`\`

## Preference Data Format

\`\`\`python
preference_pairs = [
    {
        "prompt": "Explain recursion.",
        "chosen": "<think>\\nRecursion is when a function calls itself...\\n</think>\\n\\nRecursion is a programming concept where a function calls itself to solve smaller instances of the same problem...",
        "rejected": "Recursion is when something is recursive."
    },
    {
        "prompt": "Is water wet?",
        "chosen": "<think>\\nThis is a nuanced question. 'Wet' typically means...\\n</think>\\n\\nThis depends on how you define 'wet.' If wet means 'covered in water,' then water itself isn't wet — it *makes* things wet...",
        "rejected": "Yes, water is wet because it's a liquid."
    },
]
\`\`\`

## Evaluation: Reasoning Benchmarks

\`\`\`python
def evaluate_reasoning(model, tokenizer, benchmark="gsm8k"):
    """Evaluate on GSM8K (grade school math)."""
    dataset = load_dataset("gsm8k", "main", split="test")
    correct = 0
    total = 0

    for item in dataset:
        prompt = f"""<|im_start|>system
Solve step by step. Put your final numerical answer after "#### ".<|im_end|>
<|im_start|>user
{item['question']}<|im_end|>
<|im_start|>assistant
<think>
"""
        response = generate(model, tokenizer, prompt, max_new_tokens=512, temperature=0.0)

        # Extract the final answer
        predicted = extract_answer(response)
        expected = item["answer"].split("####")[-1].strip()

        if predicted == expected:
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"{benchmark} accuracy: {accuracy:.1%} ({correct}/{total})")
    return accuracy
\`\`\`

## The Full Training Pipeline

| Stage | What | Data | Epochs | LR | Result |
|-------|------|------|--------|----|----|
| 0 | Preparation | Raw text corpus | — | — | Tokenizer + data pipeline |
| 1 | Pretraining | Raw text | 1 | 3e-4 | Next-token predictor |
| 2 | SFT | Instruction pairs | 1–3 | 2e-5 | Instruction follower |
| 3a | CoT | Reasoning traces | 1–2 | 1e-5 | Step-by-step thinker |
| 3b | DPO | Preference pairs | 1 | 5e-7 | Aligned reasoner |

## What You've Built

By completing all four stages, you've built a model that:
1. **Understands language** (pretraining)
2. **Follows instructions** (SFT)
3. **Thinks before answering** (chain-of-thought)
4. **Prefers good answers over bad ones** (DPO alignment)

This is the same pipeline used by frontier models — just at a smaller scale. The architecture, loss functions, and training stages are identical.

---

*This completes the "Train Your LLM from Scratch" series. For production-scale training, explore DeepSpeed, FSDP, and multi-node distributed training.*
    `
  },

    {
        id: "3",
        slug: "transformer-architecture",
        title: "Understanding Transformer Architecture",
        description: "A visual guide to the transformer architecture that powers modern LLMs like GPT and LLaMA.",
        difficulty: "intermediate",
        estimatedTime: "25 min",
        topics: ["Deep Learning", "NLP", "Attention"],
        series: "ML Fundamentals",
        prerequisites: ["Linear algebra basics", "Neural network fundamentals", "Python/PyTorch experience"],
        content: `
# Understanding Transformer Architecture

The Transformer architecture, introduced in the landmark paper "Attention Is All You Need" (2017), revolutionized natural language processing and laid the foundation for modern LLMs like GPT, LLaMA, and Claude. In this tutorial, we'll break down the architecture piece by piece.

## The Big Picture

At its core, a Transformer processes sequences by:
1. Converting tokens to embeddings
2. Adding positional information
3. Processing through attention and feedforward layers
4. Producing output predictions

\`\`\`
Input Tokens → Embeddings → [N × Transformer Blocks] → Output
                    ↓
            Each block contains:
            - Multi-Head Attention
            - Feed-Forward Network
            - Layer Normalization
            - Residual Connections
\`\`\`

## Step 1: Token Embeddings

First, we convert discrete tokens (words, subwords) into continuous vectors:

\`\`\`python
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len] → [batch_size, seq_len, d_model]
        # Scale by sqrt(d_model) as per original paper
        return self.embedding(x) * (self.d_model ** 0.5)

# Example
vocab_size = 50000
d_model = 512
embedding = TokenEmbedding(vocab_size, d_model)

tokens = torch.tensor([[1, 42, 156, 7]])  # [1, 4]
embedded = embedding(tokens)  # [1, 4, 512]
\`\`\`

## Step 2: Positional Encoding

Unlike RNNs, Transformers process all tokens in parallel. To capture sequence order, we add positional information:

\`\`\`python
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create position encodings
        position = torch.arange(max_seq_len).unsqueeze(1)  # [max_seq_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
\`\`\`

**Why sinusoidal?** The sine/cosine functions allow the model to:
- Learn relative positions (PE[pos+k] can be represented as a function of PE[pos])
- Generalize to longer sequences than seen during training

## Step 3: Self-Attention (The Core Innovation)

Self-attention computes relationships between all pairs of tokens:

\`\`\`python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,    # [batch, heads, seq_len, d_k]
        key: torch.Tensor,      # [batch, heads, seq_len, d_k]
        value: torch.Tensor,    # [batch, heads, seq_len, d_v]
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = query.size(-1)

        # Compute attention scores
        # [batch, heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (for causal attention in decoders)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights
\`\`\`

**Intuition**: Each token "queries" for relevant information from all other tokens. The dot product between query and key determines relevance, and values carry the actual information.

## Step 4: Multi-Head Attention

Instead of single attention, we use multiple "heads" to capture different types of relationships:

\`\`\`python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size = query.size(0)

        # Project and reshape for multi-head: [batch, seq, d_model] → [batch, heads, seq, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, _ = self.attention(Q, K, V, mask)

        # Concatenate heads: [batch, heads, seq, d_k] → [batch, seq, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Final projection
        return self.W_o(attn_output)
\`\`\`

**Why multiple heads?** Different heads can focus on:
- Syntactic relationships (subject-verb agreement)
- Semantic relationships (word meanings)
- Positional patterns (nearby words)

## Step 5: Feed-Forward Network

After attention, each position passes through a feedforward network:

\`\`\`python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Modern transformers use GELU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, d_model]
        x = self.linear1(x)      # [batch, seq, d_ff]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)      # [batch, seq, d_model]
        return x
\`\`\`

Typically, \`d_ff = 4 * d_model\`. This expansion allows the model to process information in a higher-dimensional space.

## Step 6: Transformer Block

Combining everything with residual connections and layer normalization:

\`\`\`python
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        pre_norm: bool = True  # Modern transformers use pre-norm
    ):
        super().__init__()
        self.pre_norm = pre_norm

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if self.pre_norm:
            # Pre-norm (GPT-style)
            attn_out = self.attention(
                self.norm1(x), self.norm1(x), self.norm1(x), mask
            )
            x = x + self.dropout(attn_out)

            ff_out = self.ff(self.norm2(x))
            x = x + self.dropout(ff_out)
        else:
            # Post-norm (original transformer)
            attn_out = self.attention(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_out))

            ff_out = self.ff(x)
            x = self.norm2(x + self.dropout(ff_out))

        return x
\`\`\`

## Step 7: Complete Decoder (GPT-style)

Putting it all together for a decoder-only model (like GPT):

\`\`\`python
class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len] token indices
        seq_len = x.size(1)

        # Create causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        mask = ~mask  # Invert: True = attend, False = mask

        # Embedding + positional encoding
        x = self.token_embedding(x)
        x = self.pos_encoding(x)

        # Transformer blocks
        for layer in self.layers:
            x = layer(x, mask)

        # Output projection
        x = self.norm(x)
        logits = self.output(x)  # [batch, seq, vocab_size]

        return logits
\`\`\`

## Key Concepts Summary

| Component | Purpose |
|-----------|---------|
| Token Embedding | Convert discrete tokens to vectors |
| Positional Encoding | Add sequence order information |
| Self-Attention | Model relationships between all tokens |
| Multi-Head | Capture different types of relationships |
| Feed-Forward | Process each position independently |
| Layer Norm | Stabilize training |
| Residual Connections | Enable gradient flow in deep networks |

## Modern Improvements

Since the original paper, several improvements have been made:

1. **Pre-normalization**: Apply LayerNorm before (not after) attention and FFN
2. **Rotary Position Embeddings (RoPE)**: Better handling of relative positions
3. **Grouped Query Attention (GQA)**: More efficient multi-head attention
4. **SwiGLU Activation**: Improved feedforward networks

## Next Steps

Now that you understand the architecture:
- Implement a small GPT from scratch
- Experiment with different hyperparameters
- Explore pre-trained models on Hugging Face
- Study specific improvements like FlashAttention

---

*This tutorial is part of the ML Fundamentals series. Understanding transformers is essential for working with modern LLMs.*
    `
    }
];

export function getTutorialBySlug(slug: string): Tutorial | undefined {
    return tutorials.find(t => t.slug === slug);
}

export function getTutorialsBySeries(series: string): Tutorial[] {
    return tutorials.filter(t => t.series === series);
}

export function getTutorialsByDifficulty(difficulty: string): Tutorial[] {
    if (difficulty === "all") return tutorials;
    return tutorials.filter(t => t.difficulty === difficulty);
}
