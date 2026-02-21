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
    slug: "llm-from-scratch",
    title: "Train Your LLM from Scratch",
    description: "A comprehensive guide to training a language model from scratch � from data preparation and tokenization through pretraining, instruction tuning, and reasoning with RLHF/DPO.",
    difficulty: "advanced",
    estimatedTime: "2 hours",
    topics: ["LLM Training", "PyTorch", "Pretraining", "SFT", "LoRA", "RLHF", "DPO"],
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

---
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

---
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

---
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


---

# Inference & Efficiency Metrics

These metrics measure how well an AI model runs on hardware and its responsiveness in production.

## Throughput (Tokens Per Second)

[Throughput](https://en.wikipedia.org/wiki/Throughput) measures the total volume of output tokens a model generates every second. High TPS is critical for high-traffic applications and [batch processing](https://en.wikipedia.org/wiki/Batch_processing).

For a given model, throughput depends on:
- **Hardware**: GPU type ([A100](https://www.nvidia.com/en-us/data-center/a100/), [H100](https://www.nvidia.com/en-us/data-center/h100/)), number of GPUs, interconnect bandwidth
- **Batch size**: Larger batches improve throughput but increase latency
- **Quantization**: [INT8/INT4 quantization](https://huggingface.co/docs/optimum/concept_guides/quantization) reduces memory and increases speed at the cost of some quality
- **Serving framework**: [vLLM](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), and [SGLang](https://github.com/sgl-project/sglang) provide optimized inference kernels

| Model Size | Typical TPS (A100) | Typical TPS (H100) |
|-----------|-------------------|-------------------|
| 7B | 40-80 | 80-150 |
| 13B | 25-50 | 50-100 |
| 70B | 8-15 | 20-40 |

## Time to First Token (TTFT)

[TTFT](https://en.wikipedia.org/wiki/Latency_(engineering)) is the delay between a user sending a prompt and seeing the very first character of the response. Sub-200ms is the standard for a "snappy" user experience.

TTFT is dominated by the **prefill phase** � where the model processes all input tokens in parallel through [KV-cache](https://arxiv.org/abs/2211.05102) computation. Techniques to reduce TTFT:

- **[Speculative decoding](https://arxiv.org/abs/2302.01318)**: Use a small draft model to propose tokens, verified by the large model
- **Prefix caching**: Cache the KV states of common system prompts
- **[Chunked prefill](https://arxiv.org/abs/2309.17453)**: Break long prompts into chunks to overlap with decode

## Context Window

The [context window](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)#Context_window) is the "short-term memory" of the model, measured in tokens. A larger window allows the AI to process entire books or massive codebases in a single pass.

| Model | Context Length | ~Pages of Text |
|-------|--------------|----------------|
| GPT-4o | 128K | ~300 pages |
| Claude 3.5 | 200K | ~500 pages |
| Gemini 1.5 Pro | 2M | ~5,000 pages |

Key techniques for extending context:
- **[RoPE scaling](https://arxiv.org/abs/2104.09864)**: Rotary Position Embeddings with frequency scaling
- **[Ring Attention](https://arxiv.org/abs/2310.01889)**: Distributes long sequences across GPUs
- **[Sliding Window Attention](https://arxiv.org/abs/2004.05150)**: Used by [Mistral](https://mistral.ai/) to limit attention to local windows

## GPU & Memory Utilization

Tracks how much hardware resources ([VRAM](https://en.wikipedia.org/wiki/Video_random-access_memory)) the model consumes. Lower utilization per query allows for more simultaneous users.

Key optimization techniques:
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691)**: Reduces memory from O(n^2) to O(n) for attention computation
- **[PagedAttention](https://arxiv.org/abs/2309.06180)**: Used by vLLM, manages KV-cache memory like OS virtual memory pages
- **[Continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)**: Dynamically adds/removes requests from running batches
- **Model parallelism**: [Tensor](https://arxiv.org/abs/1909.08053), [pipeline](https://arxiv.org/abs/1811.06965), and [expert parallelism](https://arxiv.org/abs/2101.03961) for large models

---

# Quality & Intelligence Metrics

These quantify how "smart" or accurate a model's outputs are.

## Perplexity

[Perplexity](https://en.wikipedia.org/wiki/Perplexity) is a mathematical measure of how "surprised" a model is by new data. Lower is better, indicating the model has a stronger internal grasp of language patterns.

Perplexity = exp(average cross-entropy loss). A perplexity of 10 means the model is, on average, "10-way uncertain" about each next token.

| Stage | Typical Perplexity |
|-------|--------------------|
| Early pretraining | 100-1000+ |
| Converged pretraining | 5-15 |
| Domain-specific fine-tune | 3-8 |

**Important caveat**: Perplexity only measures [next-token prediction](https://en.wikipedia.org/wiki/Language_model) quality on a held-out set. A model with great perplexity can still produce bad instruction-following results.

## Accuracy & F1 Score

Standard metrics for classification and extraction tasks:

- **[Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision)**: Percentage of correct predictions overall
- **[Precision](https://en.wikipedia.org/wiki/Precision_and_recall)**: Of items flagged as positive, how many actually are? (Reduces false positives)
- **[Recall](https://en.wikipedia.org/wiki/Precision_and_recall)**: Of all actual positives, how many did we find? (Reduces false negatives)
- **[F1 Score](https://en.wikipedia.org/wiki/F-score)**: The harmonic mean of precision and recall � the "gold standard" for balancing both

For LLM benchmarks, the most commonly referenced evaluations include:
- **[MMLU](https://arxiv.org/abs/2009.03300)**: 57 subjects ranging from STEM to humanities
- **[HumanEval](https://arxiv.org/abs/2107.03374)**: Code generation benchmark
- **[GSM8K](https://arxiv.org/abs/2110.14168)**: Grade school math reasoning
- **[HellaSwag](https://arxiv.org/abs/1905.07830)**: Commonsense reasoning

## Elo Rating (Human Preference)

[Elo rating](https://en.wikipedia.org/wiki/Elo_rating_system), borrowed from chess, is used by the [LMSYS Chatbot Arena](https://chat.lmsys.org/) to rank models based on blind side-by-side human testing. Users see two anonymous model outputs and pick the better one.

This is arguably the most reliable quality signal because:
- It captures **holistic quality** (helpfulness, safety, style, accuracy)
- It's **resistant to benchmark gaming** � models can't overfit to specific test sets
- It reflects **real user preferences**, not proxy metrics

## Hallucination Rate

The [hallucination](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence)) rate measures how frequently a model generates factually incorrect or unsupported information. This is one of the biggest challenges in deploying LLMs.

Two types of hallucination:
- **Intrinsic**: Contradicts the provided source material
- **Extrinsic**: Generates information not supported by any source

Mitigation strategies:
- **[Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401)**: Ground responses in retrieved documents
- **[Chain-of-thought prompting](https://arxiv.org/abs/2201.11903)**: Force step-by-step reasoning
- **Citation training**: Train models to cite sources (as done in this tutorial's markdown!)
- **Confidence calibration**: Train models to say "I don't know" when uncertain

## Scaling Laws

Both efficiency and quality metrics improve with scale, but in predictable ways described by [scaling laws](https://arxiv.org/abs/2001.08361):

- **Compute-optimal training** ([Chinchilla scaling](https://arxiv.org/abs/2203.15556)): The optimal model size and data size grow proportionally with compute budget
- **Inference scaling**: Techniques like [test-time compute](https://arxiv.org/abs/2408.03314) allow models to "think longer" on harder problems, trading latency for quality
- **Data scaling**: [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) showed that high-quality data can substitute for raw scale



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
    },
  {
    id: "4",
    slug: "antigravity-fast-prototyping",
    title: "Antigravity: Fast Prototype, Fast Mistake, Fast Fix",
    description: "Use the Antigravity AI coding agent to prototype at SpaceX speed � iterate rapidly, embrace failures, and fix forward. Includes setup with the Auto Accept extension for maximum velocity.",
    difficulty: "beginner",
    estimatedTime: "15 min",
    topics: ["Antigravity", "AI Coding", "Rapid Prototyping", "Developer Tools"],
    series: "Developer Workflow",
    prerequisites: ["VS Code or Cursor installed", "Basic programming knowledge"],
    content: `
# Antigravity: Fast Prototype, Fast Mistake, Fast Fix

> "If things are not failing, you are not innovating enough." � [Elon Musk](https://en.wikipedia.org/wiki/Elon_Musk)

## The SpaceX Philosophy Applied to Software

[SpaceX](https://en.wikipedia.org/wiki/SpaceX) builds rockets differently than NASA. Instead of spending years perfecting blueprints, they **build, launch, explode, learn, rebuild**. [Starship](https://en.wikipedia.org/wiki/SpaceX_Starship) has gone through rapid unscheduled disassemblies (explosions) � and each one taught more than a year of simulation.

This same philosophy applies to software development with AI agents:

| Traditional Development | SpaceX / Antigravity Way |
|------------------------|-------------------------|
| Plan for weeks | Prototype in minutes |
| Avoid all errors | Embrace errors as data |
| Ship when "perfect" | Ship when functional, iterate |
| Debug cautiously | Fix forward aggressively |
| Write every line manually | Let AI generate, you curate |

## What is Antigravity?

Antigravity is an agentic AI coding assistant built into your editor. Unlike autocomplete tools that suggest the next line, Antigravity:

- **Reads your entire codebase** � understands project structure, dependencies, and patterns
- **Executes multi-step plans** � creates files, edits code, runs commands, fixes errors
- **Iterates autonomously** � when a build fails, it reads the error and fixes it
- **Learns from context** � remembers your preferences, style, and project conventions

The key insight: **Antigravity doesn't need to be perfect on the first try.** It needs to be fast enough that the cycle of *generate ? test ? fix ? test* is faster than manually writing code.

## Setup: Maximum Velocity Mode

### Step 1: Install the Auto Accept Extension

The biggest bottleneck in agentic coding is the approval loop � every file edit, every command requires you to click "Accept." For maximum prototyping speed, install the **Antigravity Auto Accept** extension:

**[Antigravity Auto Accept](https://open-vsx.org/extension/pesosz/antigravity-auto-accept)** by pesosz

This extension automatically accepts Antigravity's proposed changes, letting the agent work at full speed. You review the *results*, not every individual step.

> **Credit**: Thanks to VincentHH for the recommendation � "Tried everything. This extension worked for me."

### Step 2: Set Up Your Workflow

\\\`\\\`\\\`
Recommended workflow with Auto Accept enabled:

1. Describe what you want in plain language
2. Let Antigravity generate the full implementation
3. Review the result (not each step)
4. If wrong ? describe the fix ? let it iterate
5. If right ? commit and move on
\\\`\\\`\\\`

### Step 3: Create a Workflow File

Antigravity supports workflow files that define repeatable processes. Create one in your project:

\\\`\\\`\\\`markdown
<!-- .agent/workflows/rapid-prototype.md -->
---
description: Rapid prototyping workflow
---

1. Create the component/feature in a single file first
2. Get it working (ignore styling)
// turbo
3. Run the dev server to verify
4. Add styling and polish
// turbo
5. Run tests
6. Commit with descriptive message
\\\`\\\`\\\`

The \\\`// turbo\\\` annotations tell Antigravity to auto-run those steps without asking for permission.

## The Fast Mistake Philosophy

The key mental shift:

### Old Way: Prevent Mistakes
- Spend 30 minutes thinking about the perfect architecture
- Write code carefully, line by line
- Test only when "done"
- Find bugs after investing hours

### New Way: Embrace Mistakes
- Describe the feature in 2 sentences
- Let Antigravity generate a working prototype in 2 minutes
- Test immediately
- Find issues in minutes, not hours
- Fix forward � describe the problem, let AI fix it

### Why This Works

The cost of a mistake has fundamentally changed:

| Era | Cost of a Mistake | Fix Time |
|-----|-------------------|----------|
| Manual coding | Hours of debugging | Hours |
| With AI autocomplete | Less debugging, still manual | 30+ minutes |
| With Antigravity | AI reads the error and fixes it | 1-2 minutes |
| With Auto Accept | Fully autonomous fix cycle | Seconds |

When mistakes are cheap and fixes are fast, the optimal strategy is to **try more things**, not to think longer.

## Real-World Example

Here's how a rapid prototyping session looks:

**Prompt**: "Build a dashboard page with a chart showing user signups over time, a stats card row, and a recent activity feed"

**What happens**:
1. Antigravity creates the page component (~30 seconds)
2. Adds chart library import, generates mock data (~15 seconds)
3. Builds the stats cards and activity feed (~20 seconds)
4. Starts the dev server to verify (~10 seconds)
5. Finds a build error (missing dependency) ? installs it ? rebuilds (~15 seconds)
6. **Total time: ~90 seconds for a working dashboard**

The equivalent manual workflow takes 30-60 minutes. Even if Antigravity's first attempt needs 3 rounds of fixes, you're still 10x faster.

## When NOT to Use Auto Accept

Auto Accept is for prototyping, not production deployments:

- ? Running \\\`rm -rf\\\` or destructive commands
- ? Pushing to production branches
- ? Modifying authentication or security code
- ? Database migrations
- ? Creating new components
- ? Building UI prototypes
- ? Writing tests
- ? Refactoring existing code
- ? Adding features to development branches

## Coming Soon

This tutorial will be expanded with:
- Video walkthroughs of real prototyping sessions
- Advanced workflow configurations
- Team collaboration patterns with Antigravity
- Benchmarks: manual vs. AI-assisted development speed

---

*This is the first tutorial in the Developer Workflow series. Antigravity is designed to make the "fast prototype, fast mistake, fast fix" cycle as frictionless as possible.*
    `
  },
  {
    id: "5",
    slug: "cs-vocabulary-for-vibe-coders",
    title: "The CS Vocabulary You Need for Vibe Coding",
    description: "Every term, concept, and piece of jargon you need to understand what AI is doing and to express what you want � no code required. Organized by domain: OS, networking, web dev, hardware, Linux, clusters, AI/ML, databases, security, and more.",
    difficulty: "beginner",
    estimatedTime: "40 min",
    topics: ["Computer Science", "Vocabulary", "Vibe Coding", "Developer Fundamentals"],
    series: "Developer Workflow",
    prerequisites: ["Curiosity", "An AI coding assistant"],
    content: `
# The CS Vocabulary You Need for Vibe Coding

> You don't need to write code anymore. But you DO need to speak the language.

Vibe coding means letting AI write the code while you direct the vision. But to direct effectively, you need to understand the **vocabulary** � the terms AI uses in its explanations, the concepts behind error messages, and the jargon that lets you describe what you actually want.

This is not a coding tutorial. There are **zero code examples**. This is a reference dictionary organized by domain, designed so you can look up any term you encounter while working with an AI coding agent.

---

## Operating Systems (OS)

The operating system is the software layer between your hardware and your applications. When AI mentions "processes" or "memory," this is what it's talking about.

| Term | What It Means |
|------|--------------|
| **Process** | A running instance of a program. Your browser is a process. Your code editor is a process. Each has its own memory space. |
| **Thread** | A lightweight unit of execution within a process. A single process can have multiple threads running simultaneously (like a browser rendering a page while downloading a file). |
| **Kernel** | The core of the OS. It manages hardware resources, memory, and process scheduling. You almost never interact with it directly, but errors that mention "kernel panic" mean something went very wrong at this level. |
| **File System** | How your OS organizes files on disk. Common types: NTFS (Windows), ext4 (Linux), APFS (Mac). When AI says "write to the file system," it means saving a file. |
| **PATH** | An environment variable that tells your OS where to find executable programs. When you type "python" in a terminal, the OS searches your PATH to find where Python is installed. |
| **Environment Variable** | A named value stored by the OS that programs can read. API keys, configuration settings, and system paths are commonly stored this way. Example: \`HOME\`, \`PATH\`, \`NODE_ENV\`. |
| **stdin / stdout / stderr** | The three standard streams. stdin = input going INTO a program. stdout = normal output coming OUT. stderr = error output. When AI says "pipe stdout," it means redirect the output somewhere. |
| **Daemon** | A background process that runs continuously without user interaction. Web servers, database servers, and system services are daemons. |
| **Symlink (Symbolic Link)** | A shortcut/alias that points to another file or directory. Unlike a copy, it references the original � if the original moves, the symlink breaks. |
| **Permission** | Who can read, write, or execute a file. On Linux/Mac, you see things like \`rwxr-xr-x\` � this encodes owner/group/others permissions. \`chmod\` changes permissions. |
| **Shell** | The program that interprets your text commands. Bash, Zsh, PowerShell, and Fish are all shells. When AI says "run this in your shell," it means open a terminal and type it. |

---

## Networking

Every web application communicates over a network. These terms come up constantly when building anything that talks to the internet.

| Term | What It Means |
|------|--------------|
| **IP Address** | A unique numerical address for a device on a network. IPv4 looks like \`192.168.1.1\`. IPv6 looks like \`2001:0db8:85a3::8a2e:0370:7334\`. |
| **Port** | A number (0-65535) that identifies a specific service on a machine. Web servers commonly use port 80 (HTTP) or 443 (HTTPS). Your dev server might run on port 3000 or 5173. |
| **DNS** | Domain Name System � translates human-readable domain names (google.com) into IP addresses. When someone says "DNS isn't resolving," it means the name-to-address lookup is failing. |
| **HTTP / HTTPS** | Hypertext Transfer Protocol. The language browsers and servers use to communicate. HTTPS is the encrypted version (the S stands for Secure). |
| **Request / Response** | The fundamental pattern of web communication. A client sends a **request** (e.g., "give me this page"), and the server sends back a **response** (e.g., the HTML content). |
| **GET / POST / PUT / DELETE** | HTTP methods (verbs). GET = retrieve data. POST = send/create data. PUT = update data. DELETE = remove data. These map to CRUD operations (Create, Read, Update, Delete). |
| **API** | Application Programming Interface � a defined way for programs to communicate with each other. When AI says "call the API," it means send a structured request to a service and get data back. |
| **REST** | Representational State Transfer � an architectural style for APIs. REST APIs use HTTP methods and URL paths to organize resources (e.g., GET /users/123 retrieves user 123). |
| **GraphQL** | An alternative to REST where you specify exactly what data you want in your query, and the server returns only that. |
| **WebSocket** | A protocol for real-time, two-way communication between client and server. Unlike HTTP (request-response), WebSockets keep a connection open for continuous data flow. Used for chat, live updates, gaming. |
| **CORS** | Cross-Origin Resource Sharing � a browser security feature that blocks requests from one domain to another unless the server explicitly allows it. The most common source of "why won't my API call work" frustration. |
| **SSL / TLS** | Secure Sockets Layer / Transport Layer Security � encryption protocols that make HTTPS work. When AI mentions "SSL certificate," it's about proving a server's identity and encrypting traffic. |
| **Latency** | The time delay between sending a request and receiving a response. Low latency = fast. High latency = slow. Measured in milliseconds (ms). |
| **Bandwidth** | The maximum rate of data transfer. Think of it as the width of a pipe � latency is how long water takes to travel through, bandwidth is how much water can flow at once. |
| **CDN** | Content Delivery Network � a global network of servers that cache your content close to users. If your website's images load fast worldwide, it's probably using a CDN (e.g., Cloudflare, AWS CloudFront). |
| **Load Balancer** | A system that distributes incoming traffic across multiple servers so no single server gets overwhelmed. |
| **Proxy / Reverse Proxy** | A proxy sits between client and server. A forward proxy acts on behalf of the client (like a VPN). A reverse proxy acts on behalf of the server (like Nginx routing requests to different backend services). |
| **TCP / UDP** | Transport protocols. TCP = reliable, ordered delivery (used for web, email). UDP = fast, no guarantees (used for video streaming, gaming). |

---

## Web Development

The vocabulary of building things people interact with in a browser.

| Term | What It Means |
|------|--------------|
| **Frontend** | The part users see and interact with � HTML, CSS, JavaScript running in the browser. |
| **Backend** | The server-side logic, databases, and APIs that power the frontend. Users don't see this directly. |
| **Full-Stack** | Working on both frontend and backend. |
| **HTML** | HyperText Markup Language � the structure of web pages. Tags like \`<div>\`, \`<h1>\`, \`<p>\` define what's on the page. |
| **CSS** | Cascading Style Sheets � controls how HTML looks. Colors, fonts, layout, animations. |
| **JavaScript (JS)** | The programming language of the web. Runs in browsers and (via Node.js) on servers. |
| **TypeScript (TS)** | JavaScript with type annotations. Catches errors before code runs. When AI generates \`.tsx\` files, that's TypeScript + JSX. |
| **JSX / TSX** | A syntax extension that lets you write HTML-like code inside JavaScript/TypeScript. Used by React. |
| **DOM** | Document Object Model � the browser's internal representation of a web page as a tree of objects. When AI says "manipulate the DOM," it means change what's displayed on the page. |
| **Component** | A reusable, self-contained piece of UI. A button, a card, a navigation bar � each is typically a component. Modern web dev is component-based. |
| **State** | Data that changes over time and affects what's displayed. A counter value, whether a menu is open, the current user � all state. |
| **Props** | Properties passed from a parent component to a child component. How components communicate in React/Vue/etc. |
| **Hook** | A React concept � functions that let components use state and lifecycle features. \`useState\`, \`useEffect\`, \`useRef\` are the common ones. |
| **Routing** | Mapping URLs to different pages/views. When you go to /about, the router shows the About page. |
| **SPA** | Single Page Application � a web app that loads once and dynamically updates content without full page reloads. React, Vue, Angular apps are SPAs. |
| **SSR / SSG** | Server-Side Rendering / Static Site Generation. SSR generates HTML on the server for each request. SSG generates all HTML at build time. Both improve SEO and initial load speed compared to pure SPAs. |
| **Hydration** | The process where a server-rendered HTML page becomes interactive by attaching JavaScript event handlers on the client side. |
| **Build Tool** | Software that transforms your source code into optimized production files. Vite, Webpack, esbuild, Turbopack are build tools. |
| **Bundle** | The final JavaScript/CSS files that get sent to the browser, created by the build tool. "Bundle size" matters because bigger bundles = slower load. |
| **Minification** | Removing whitespace, comments, and shortening variable names to make code files smaller. |
| **Tree Shaking** | Removing unused code from the final bundle. If you import one function from a library, tree shaking ensures the rest of the library isn't included. |
| **Hot Module Replacement (HMR)** | Updating code in the browser instantly during development without a full page reload. When AI says "the dev server supports HMR," your changes appear instantly. |
| **Responsive Design** | Making layouts adapt to different screen sizes (phone, tablet, desktop). |
| **Viewport** | The visible area of a web page in the browser. Mobile viewports are smaller than desktop viewports. |
| **Semantic HTML** | Using HTML tags that convey meaning (e.g., \`<nav>\`, \`<article>\`, \`<header>\`) instead of generic \`<div>\` for everything. Important for accessibility and SEO. |
| **Accessibility (a11y)** | Making web content usable by people with disabilities � screen readers, keyboard navigation, color contrast, ARIA attributes. |

---

## Databases

Where data lives permanently. Every app with user accounts, posts, or any persistent data uses a database.

| Term | What It Means |
|------|--------------|
| **SQL** | Structured Query Language � the language for querying relational databases. SELECT, INSERT, UPDATE, DELETE are the core operations. |
| **NoSQL** | Databases that don't use traditional SQL tables. Document stores (MongoDB), key-value stores (Redis), graph databases (Neo4j). |
| **Schema** | The structure definition of your data � what fields exist, their types, and relationships. |
| **Migration** | A versioned change to your database schema. Like git commits, but for database structure. "Run migrations" = apply pending schema changes. |
| **ORM** | Object-Relational Mapping � a library that lets you interact with databases using your programming language's objects instead of raw SQL. Prisma, SQLAlchemy, TypeORM are ORMs. |
| **Primary Key** | A unique identifier for each record in a table. Usually an auto-incrementing ID or a UUID. |
| **Foreign Key** | A field that references the primary key of another table, creating a relationship between them. |
| **Index** | A data structure that speeds up database lookups, like the index in a book. Without an index, the database scans every row (slow). |
| **Query** | A request for data from a database. "The query is slow" = the database lookup takes too long. |
| **Transaction** | A group of database operations that either ALL succeed or ALL fail. No partial changes. Critical for financial operations. |
| **CRUD** | Create, Read, Update, Delete � the four basic database operations. Most applications are fundamentally CRUD apps with nice UIs. |
| **Redis** | An in-memory data store used for caching, session management, and real-time features. Extremely fast because data lives in RAM, not disk. |
| **Connection Pool** | A cache of database connections that are reused rather than creating new ones for each request. Improves performance. |

---

## Hardware & Performance

Understanding what your code actually runs on helps you understand performance issues and constraints.

| Term | What It Means |
|------|--------------|
| **CPU** | Central Processing Unit � the "brain" that executes instructions. Clock speed (GHz) and core count determine raw computation power. |
| **GPU** | Graphics Processing Unit � originally for rendering graphics, now essential for AI/ML because it can do thousands of parallel computations simultaneously. NVIDIA GPUs (CUDA) dominate AI training. |
| **RAM** | Random Access Memory � fast, temporary storage that programs use while running. When your computer "runs out of memory," it's running out of RAM. |
| **VRAM** | Video RAM � memory on the GPU. AI models need to fit in VRAM to run efficiently. "OOM" (Out Of Memory) errors during model inference usually mean insufficient VRAM. |
| **Storage (SSD/HDD)** | Permanent data storage. SSDs (Solid State Drives) are fast. HDDs (Hard Disk Drives) are slow but cheap. NVMe SSDs are the fastest. |
| **Cache** | Fast, small storage that keeps frequently accessed data close to the processor. L1/L2/L3 caches are on the CPU. Caching is a universal pattern � keep copies of expensive-to-compute things for fast retrieval. |
| **Bottleneck** | The slowest component that limits overall system performance. Could be CPU, GPU, memory, disk, or network. |
| **Throughput** | How much work a system can do per unit of time. Requests per second, tokens per second, frames per second. |
| **Concurrency** | Doing multiple things at once (or appearing to). A web server handling 1000 simultaneous connections uses concurrency. |
| **Parallelism** | Actually executing multiple computations simultaneously on different CPU/GPU cores. Related to but different from concurrency. |

---

## Linux & Command Line

Most servers run Linux. Most AI development happens on Linux. These terms come up constantly.

| Term | What It Means |
|------|--------------|
| **Terminal / Console** | A text-based interface for running commands. On Mac: Terminal. On Windows: PowerShell or WSL. On Linux: any terminal emulator. |
| **Bash** | The most common Linux shell. When AI writes shell scripts, they're usually Bash scripts. |
| **sudo** | "Super User Do" � run a command with administrator/root privileges. Required for installing system software, changing system files, etc. |
| **apt / yum / brew** | Package managers for installing software. apt (Debian/Ubuntu), yum (CentOS/RHEL), brew (macOS). Like app stores for command-line tools. |
| **SSH** | Secure Shell � a protocol for remotely connecting to and controlling another computer via an encrypted connection. "SSH into the server" = connect to a remote machine's terminal. |
| **SCP / rsync** | Tools for copying files between machines. SCP is simpler, rsync is smarter (only copies changes). |
| **grep** | Search for text patterns in files. "Grep for that error message" = search your codebase for a specific string. |
| **pipe ( \\| )** | Sends the output of one command as input to another. \`cat file.txt \\| grep "error"\` reads a file and filters for lines containing "error." |
| **curl / wget** | Command-line tools for making HTTP requests and downloading files. "Curl the endpoint" = send a request to a URL from the terminal. |
| **cron** | A scheduler that runs commands at specified intervals. "Set up a cron job" = schedule a task to run automatically (daily backup, hourly data sync, etc.). |
| **systemd** | The init system that manages services on modern Linux. \`systemctl start nginx\` starts the Nginx web server; \`systemctl enable\` makes it start on boot. |
| **container** | A lightweight, isolated environment for running applications. Docker containers package your app with all its dependencies so it runs the same everywhere. |
| **Docker** | The most popular containerization platform. A Dockerfile defines how to build a container image. Docker Compose manages multi-container applications. |
| **Volume** | Persistent storage for containers. Without a volume, data inside a container disappears when the container stops. |
| **WSL** | Windows Subsystem for Linux � lets you run a real Linux environment inside Windows. Essential for Windows developers working with Linux-native tools. |

---

## Clusters & Infrastructure

When one machine isn't enough, you need clusters. This is how AI training, big websites, and cloud services work.

| Term | What It Means |
|------|--------------|
| **Cluster** | A group of computers working together as a single system. Used for distributed computing, AI training, and high-availability services. |
| **Node** | A single machine in a cluster. A 4-node cluster = 4 connected computers. |
| **Cloud** | Renting someone else's computers. AWS, Google Cloud, Azure provide servers, storage, and services you pay for by usage. |
| **Instance** | A single virtual machine in the cloud. "Spin up an instance" = create a new virtual server. |
| **VM (Virtual Machine)** | A software emulation of a computer that runs on physical hardware. Multiple VMs can run on one physical server. |
| **Kubernetes (K8s)** | An orchestration system for managing containers at scale. Automatically handles deployment, scaling, load balancing, and failure recovery. |
| **SLURM** | A job scheduler for HPC (High Performance Computing) clusters. Common in academic research and AI training. You submit jobs and SLURM allocates GPU/CPU resources. |
| **HPC** | High Performance Computing � powerful computing clusters used for scientific research, simulations, and large-scale model training. |
| **Distributed Training** | Training an AI model across multiple GPUs or machines simultaneously. Techniques: data parallelism (split data), model parallelism (split model), pipeline parallelism (split layers). |
| **CI/CD** | Continuous Integration / Continuous Deployment � automated pipelines that build, test, and deploy your code when you push changes. GitHub Actions, GitLab CI, Jenkins are CI/CD tools. |
| **Serverless** | Cloud functions that run on demand without managing servers. AWS Lambda, Vercel Functions, Cloudflare Workers. You pay per execution, not per server-hour. |
| **Microservice** | An architectural pattern where an application is split into small, independent services that communicate via APIs. Opposite of a "monolith" (one big application). |
| **Scaling (Horizontal vs. Vertical)** | Vertical = bigger machine (more RAM, faster CPU). Horizontal = more machines. Horizontal scaling is usually preferred because it has no upper limit. |
| **Auto-scaling** | Automatically adding or removing servers based on demand. Handle traffic spikes without manual intervention. |

---

## Version Control (Git)

How developers track changes, collaborate, and avoid disasters.

| Term | What It Means |
|------|--------------|
| **Repository (Repo)** | A project folder tracked by Git. Contains all files and their complete history of changes. |
| **Commit** | A saved snapshot of your changes with a description. Like a checkpoint in a video game. |
| **Branch** | A parallel version of your code. Create a branch to work on a feature without affecting the main code. |
| **Merge** | Combining changes from one branch into another. "Merge your feature branch into main." |
| **Pull Request (PR)** | A request to merge your branch into another. Other developers review your changes before merging. Also called Merge Request (MR) on GitLab. |
| **Conflict** | When two branches modify the same lines differently. Git can't auto-merge, so you have to manually choose which version to keep. |
| **Clone** | Copy a remote repository to your local machine. |
| **Fork** | Create your own copy of someone else's repository on GitHub, so you can modify it independently. |
| **Stash** | Temporarily save uncommitted changes without committing them. Useful when you need to switch branches quickly. |
| **.gitignore** | A file that tells Git which files to NOT track. Node_modules, .env files, build outputs, and large datasets should be in .gitignore. |
| **HEAD** | Points to the current commit you're working on. "Detached HEAD" means you're looking at a specific past commit, not a branch tip. |
| **Rebase** | Rewriting commit history by moving your branch's starting point. Makes history cleaner but can be dangerous if misused. |

---

## AI & Machine Learning

The vocabulary of the tools that are writing your code.

| Term | What It Means |
|------|--------------|
| **Model** | A trained neural network � the "brain" that generates text, images, or predictions. GPT-4, Claude, Llama are language models. |
| **LLM** | Large Language Model � a model trained on massive text datasets to understand and generate language. The AI writing your code is an LLM. |
| **VLM** | Vision-Language Model � an LLM that can also process images. Can "see" screenshots, diagrams, or photos and reason about them. |
| **Token** | The basic unit of text that models process. Roughly 3/4 of a word. "Hello world" � 2 tokens. Context limits are measured in tokens. |
| **Context Window** | How much text a model can "see" at once. GPT-4 has a 128K context window. Larger = can read more of your codebase simultaneously. |
| **Inference** | Running a trained model to get outputs. When AI generates code for you, that's inference. |
| **Training** | The process of teaching a model by showing it data. Pretraining = learning from massive datasets. Fine-tuning = specializing on specific tasks. |
| **Fine-tuning** | Training an existing model further on domain-specific data to improve its performance on a particular task. |
| **Prompt** | The text input you give to an AI model. Prompt engineering = crafting inputs that produce better outputs. |
| **Temperature** | A parameter that controls randomness in AI outputs. Low temperature (0.0) = deterministic, focused. High temperature (1.0+) = creative, random. |
| **Hallucination** | When an AI generates plausible-sounding but factually incorrect information. It "hallucinates" functions that don't exist, APIs with wrong signatures, or made-up facts. |
| **Embedding** | A numerical representation of text/images in a high-dimensional space. Similar concepts have similar embeddings. Used for search, recommendations, and clustering. |
| **Vector Database** | A database optimized for storing and searching embeddings. Pinecone, Weaviate, Chroma are vector databases. Used for semantic search and RAG. |
| **RAG** | Retrieval-Augmented Generation � giving an AI access to external data (documents, codebase, web) so it can ground its answers in facts rather than relying solely on training data. |
| **Agent** | An AI system that can take actions � run code, browse the web, call APIs, edit files � not just generate text. Antigravity is an agent. |
| **MoE** | Mixture of Experts � an architecture where only a subset of the model's parameters are active for each input. Makes models efficient despite having many total parameters. |
| **RLHF** | Reinforcement Learning from Human Feedback � training an AI to align with human preferences by having humans rank outputs in order of quality. |
| **Overfitting** | When a model memorizes training data instead of learning general patterns. It performs well on training data but poorly on new data. |
| **Epoch** | One complete pass through the entire training dataset. Training takes multiple epochs. |
| **Loss** | A number that measures how wrong the model's predictions are. Training aims to minimize loss. Lower = better. |
| **Batch Size** | How many examples the model processes at once during training. Larger batch sizes use more memory but can train faster. |
| **Learning Rate** | How big of steps the model takes during training. Too high = overshoots and doesn't learn. Too low = learns too slowly. |
| **Gradient** | The direction and magnitude of change needed to reduce loss. "Gradient descent" = iteratively adjusting model parameters in the direction that reduces error. |
| **Transformer** | The neural network architecture behind modern LLMs. Key innovation: the attention mechanism, which lets the model weigh which parts of the input are relevant to each output. |
| **Attention** | A mechanism that lets models focus on relevant parts of the input when producing output. "Self-attention" = each word attends to every other word in the sequence. |
| **Diffusion Model** | An AI architecture for generating images by learning to gradually remove noise. Stable Diffusion, DALL-E, and Midjourney use this approach. |
| **LoRA** | Low-Rank Adaptation � a technique for fine-tuning large models efficiently by only training a small number of additional parameters instead of the full model. |
| **Quantization** | Reducing model precision (e.g., from 32-bit to 4-bit numbers) to use less memory and run faster, with minimal quality loss. "4-bit quantized" models run on consumer GPUs. |

---

## Statistics & Math Concepts

You don't need to do the math, but understanding these terms helps you interpret AI outputs and model performance.

| Term | What It Means |
|------|--------------|
| **Mean / Average** | Sum of values divided by count. The most basic summary statistic. |
| **Variance / Standard Deviation** | How spread out values are from the mean. Low variance = consistent. High variance = unpredictable. |
| **Distribution** | The pattern of how values are spread. Normal distribution (bell curve) is the most common. |
| **Correlation** | How strongly two variables move together. Correlation does NOT imply causation � ice cream sales and drowning deaths are correlated (both increase in summer). |
| **Regression** | Predicting a continuous number (price, temperature) from input features. Linear regression draws a best-fit line through data. |
| **Classification** | Predicting a category (spam/not spam, cat/dog) from input features. |
| **Precision / Recall** | Precision = of all items you identified as positive, what fraction actually were? Recall = of all actual positives, what fraction did you catch? There's usually a trade-off. |
| **Accuracy** | The percentage of correct predictions. Can be misleading � if 99% of emails aren't spam, a model that predicts "not spam" for everything has 99% accuracy but is useless. |
| **F1 Score** | The harmonic mean of precision and recall. A balanced metric when both matter. |
| **Perplexity** | How "surprised" a language model is by text. Lower perplexity = the model predicts the text well. Used to evaluate language models. |
| **Dimensionality** | The number of features/variables in a dataset. A 768-dimensional embedding has 768 numbers describing each item. |
| **p-value** | The probability of seeing your result by random chance. p < 0.05 is traditionally considered "statistically significant." |

---

## Physics & Systems Thinking

Concepts borrowed from physics that appear in computing and AI.

| Term | What It Means |
|------|--------------|
| **Entropy** | A measure of disorder or information content. In information theory: high entropy = unpredictable, lots of information. In machine learning: used in loss functions and decision trees. |
| **Signal vs. Noise** | Signal = the useful information you want. Noise = random, irrelevant variation that obscures the signal. Better models have higher signal-to-noise ratios. |
| **Equilibrium** | A stable state where opposing forces balance out. In training, reaching equilibrium means the model has converged and isn't improving further. |
| **Convergence** | When a training process stabilizes and stops improving significantly. "The model has converged" = further training won't help much. |
| **Feedback Loop** | Output of a system feeds back as input. Positive feedback loops amplify (exponential growth/collapse). Negative feedback loops stabilize (thermostats, PID controllers). AI self-improvement is a positive feedback loop. |
| **Latent Space** | A compressed, learned representation where similar items are close together. Diffusion models generate images by navigating latent space. |
| **Decay** | Gradual reduction of a value over time. Learning rate decay = lowering the learning rate as training progresses for finer adjustments. Weight decay = regularization that prevents overfitting. |
| **Annealing** | Inspired by metallurgy � gradually reducing temperature (randomness) during optimization to first explore broadly, then fine-tune locally. Simulated annealing is a search algorithm based on this principle. |

---

## Security

Terms you'll encounter when building anything that handles user data or connects to the internet.

| Term | What It Means |
|------|--------------|
| **Authentication (AuthN)** | Verifying WHO you are. Username/password, OAuth, biometrics. "Are you who you claim to be?" |
| **Authorization (AuthZ)** | Verifying WHAT you can do. After authentication, does this user have permission to access this resource? |
| **OAuth** | An authorization framework that lets third-party apps access your data without your password. "Sign in with Google/GitHub" uses OAuth. |
| **JWT** | JSON Web Token � a compact, self-contained token for securely transmitting information. Commonly used for authentication. Contains encoded (not encrypted) user info. |
| **Hashing** | Converting data into a fixed-length string that can't be reversed. Passwords are stored as hashes, not plaintext. SHA-256, bcrypt are hashing algorithms. |
| **Encryption** | Making data unreadable without a key. Unlike hashing, encryption is reversible with the right key. AES, RSA are encryption algorithms. |
| **CSRF** | Cross-Site Request Forgery � tricking a user's browser into making unwanted requests to a site they're logged into. CSRF tokens prevent this. |
| **XSS** | Cross-Site Scripting � injecting malicious scripts into web pages viewed by other users. Input sanitization prevents this. |
| **SQL Injection** | Inserting malicious SQL into input fields to manipulate a database. Parameterized queries prevent this. |
| **API Key** | A secret string that authenticates your app to a service. NEVER commit API keys to git. Use environment variables. |
| **.env File** | A file storing environment variables (API keys, database URLs) that should NEVER be committed to version control. |
| **Rate Limiting** | Restricting how many requests a user/IP can make in a time period. Prevents abuse and ensures fair usage. |

---

## DevOps & Deployment

Getting your code from your laptop to users.

| Term | What It Means |
|------|--------------|
| **Deploy** | Making your application available to users. "Deploy to production" = push your latest code to the live server. |
| **Production (Prod)** | The live environment that real users interact with. "Don't test in production" = don't experiment where real users will be affected. |
| **Staging** | A pre-production environment that mirrors production. Test here before deploying to prod. |
| **Rollback** | Reverting to a previous version when a deployment goes wrong. "Roll back to the last known good version." |
| **Downtime** | When your service is unavailable. "99.9% uptime" = less than 8.76 hours of downtime per year. |
| **Monitoring** | Tracking your application's health, performance, and errors in real time. Datadog, Grafana, New Relic are monitoring tools. |
| **Logging** | Recording events and errors for debugging. Structured logging (JSON format) makes logs searchable. |
| **Nginx** | A popular web server and reverse proxy. Often sits in front of your application, handling SSL, routing, and static files. |
| **PM2** | A process manager for Node.js applications. Keeps your app running, restarts on crash, handles clustering. |
| **Blue-Green Deployment** | Running two identical production environments. Deploy to the inactive one (green), test it, then switch traffic from the active one (blue) to green. Instant rollback by switching back. |

---

## Package Management & Dependencies

How modern software is built from thousands of smaller pieces.

| Term | What It Means |
|------|--------------|
| **Package / Library / Module** | Reusable code someone else wrote that you can use in your project. React, Express, NumPy are packages. |
| **Dependency** | A package your project requires to function. Your project "depends on" React means React is a dependency. |
| **npm / yarn / pnpm** | Package managers for JavaScript/Node.js projects. \`npm install\` downloads all dependencies listed in package.json. |
| **pip** | Package manager for Python. \`pip install torch\` installs PyTorch. |
| **package.json** | The manifest file for Node.js projects. Lists dependencies, scripts, and project metadata. |
| **node_modules** | The folder where npm installs all your dependencies. Can contain thousands of packages. NEVER commit to git. |
| **Lock File** | Pins exact dependency versions (package-lock.json, yarn.lock). Ensures everyone gets the same versions. |
| **Semantic Versioning** | Version numbers in the format MAJOR.MINOR.PATCH (e.g., 2.1.3). MAJOR = breaking changes, MINOR = new features, PATCH = bug fixes. |
| **Breaking Change** | A change that makes existing code stop working. Major version bumps signal breaking changes. |
| **Monorepo** | A single repository containing multiple related projects. Managed with tools like Turborepo, Nx, or Lerna. |

---

*This reference is part of the Developer Workflow series. Bookmark it � you'll come back to this more often than you think. And remember: you don't need to memorize everything here. You just need to recognize the terms when AI uses them, and know enough to ask the right follow-up questions.*
    `
  },
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