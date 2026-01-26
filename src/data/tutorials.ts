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

*This tutorial is part of the WebLLM Fundamentals series. Continue to "Building a Chat Interface with React" for the next steps.*
    `
  },
  {
    id: "2",
    slug: "react-chat-interface",
    title: "Building a Chat Interface with React",
    description: "Create a modern chat UI component with message streaming, typing indicators, and markdown support.",
    difficulty: "intermediate",
    estimatedTime: "30 min",
    topics: ["React", "TypeScript", "Tailwind CSS"],
    prerequisites: ["React basics", "TypeScript fundamentals", "Completion of WebLLM basics tutorial"],
    content: `
# Building a Chat Interface with React

In this tutorial, we'll create a production-ready chat interface using React, TypeScript, and Tailwind CSS. This component will support message streaming, markdown rendering, and a polished user experience.

## What We're Building

A complete chat interface with:
- Real-time message streaming
- Markdown support with syntax highlighting
- Auto-scrolling message list
- Typing indicators
- Copy-to-clipboard functionality
- Responsive design

## Project Setup

Start with a Vite + React + TypeScript project:

\`\`\`bash
npm create vite@latest chat-ui -- --template react-ts
cd chat-ui
npm install
npm install react-markdown @tailwindcss/typography lucide-react
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
\`\`\`

Configure Tailwind in \`tailwind.config.js\`:

\`\`\`javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {},
  },
  plugins: [require("@tailwindcss/typography")],
};
\`\`\`

## Step 1: Define Types

Create \`src/types.ts\`:

\`\`\`typescript
export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}
\`\`\`

## Step 2: Create the Message Component

Create \`src/components/ChatMessage.tsx\`:

\`\`\`tsx
import { Copy, Check, User, Bot } from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Message } from "../types";

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={\`flex gap-4 p-4 \${isUser ? "bg-gray-50" : "bg-white"}\`}>
      {/* Avatar */}
      <div className={\`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 \${
        isUser
          ? "bg-blue-600 text-white"
          : "bg-gray-200 text-gray-700"
      }\`}>
        {isUser ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-sm">
            {isUser ? "You" : "Assistant"}
          </span>
          <span className="text-xs text-gray-500">
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>

        {/* Message content with markdown */}
        <div className="prose prose-sm max-w-none">
          <ReactMarkdown
            components={{
              code({ node, className, children, ...props }) {
                const isInline = !className;
                if (isInline) {
                  return (
                    <code className="bg-gray-100 px-1 rounded text-sm" {...props}>
                      {children}
                    </code>
                  );
                }
                return (
                  <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                    <code className={className} {...props}>
                      {children}
                    </code>
                  </pre>
                );
              },
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        {/* Streaming indicator */}
        {message.isStreaming && (
          <span className="inline-block w-2 h-4 bg-gray-400 animate-pulse ml-1" />
        )}
      </div>

      {/* Copy button */}
      {!isUser && (
        <button
          onClick={copyToClipboard}
          className="p-1.5 rounded hover:bg-gray-100 text-gray-500 self-start"
          title="Copy message"
        >
          {copied ? (
            <Check className="w-4 h-4 text-green-600" />
          ) : (
            <Copy className="w-4 h-4" />
          )}
        </button>
      )}
    </div>
  );
}
\`\`\`

## Step 3: Create the Chat Input

Create \`src/components/ChatInput.tsx\`:

\`\`\`tsx
import { Send } from "lucide-react";
import { useState, useRef, useEffect, KeyboardEvent } from "react";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({ onSend, disabled, placeholder }: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
    }
  }, [input]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (trimmed && !disabled) {
      onSend(trimmed);
      setInput("");
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t bg-white p-4">
      <div className="max-w-3xl mx-auto flex gap-2 items-end">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder || "Type a message..."}
          disabled={disabled}
          rows={1}
          className="flex-1 resize-none rounded-lg border border-gray-300 p-3 focus:outline-none focus:border-blue-500 disabled:bg-gray-100"
        />
        <button
          onClick={handleSend}
          disabled={disabled || !input.trim()}
          className="p-3 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          <Send className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
}
\`\`\`

## Step 4: Create the Message List

Create \`src/components/MessageList.tsx\`:

\`\`\`tsx
import { useEffect, useRef } from "react";
import { Message } from "../types";
import { ChatMessage } from "./ChatMessage";

interface MessageListProps {
  messages: Message[];
}

export function MessageList({ messages }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    const container = containerRef.current;
    const bottom = bottomRef.current;

    if (container && bottom) {
      // Only auto-scroll if user is near the bottom
      const isNearBottom =
        container.scrollHeight - container.scrollTop - container.clientHeight < 100;

      if (isNearBottom) {
        bottom.scrollIntoView({ behavior: "smooth" });
      }
    }
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500">
        <div className="text-center">
          <p className="text-lg font-medium">No messages yet</p>
          <p className="text-sm">Start a conversation below</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="flex-1 overflow-y-auto">
      {messages.map((message) => (
        <ChatMessage key={message.id} message={message} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
\`\`\`

## Step 5: Put It All Together

Create the main Chat component in \`src/components/Chat.tsx\`:

\`\`\`tsx
import { useState, useCallback } from "react";
import { Message } from "../types";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";

interface ChatProps {
  onSendMessage: (
    message: string,
    onToken: (token: string) => void
  ) => Promise<void>;
}

export function Chat({ onSendMessage }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = useCallback(async (content: string) => {
    // Add user message
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);

    // Create placeholder for assistant message
    const assistantId = crypto.randomUUID();
    const assistantMessage: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isStreaming: true,
    };
    setMessages(prev => [...prev, assistantMessage]);
    setIsLoading(true);

    try {
      // Stream the response
      await onSendMessage(content, (token) => {
        setMessages(prev =>
          prev.map(msg =>
            msg.id === assistantId
              ? { ...msg, content: msg.content + token }
              : msg
          )
        );
      });

      // Mark streaming complete
      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantId
            ? { ...msg, isStreaming: false }
            : msg
        )
      );
    } catch (error) {
      // Handle error
      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantId
            ? { ...msg, content: "Sorry, an error occurred.", isStreaming: false }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  }, [onSendMessage]);

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <header className="border-b bg-white p-4">
        <h1 className="text-xl font-semibold text-center">AI Chat</h1>
      </header>

      <MessageList messages={messages} />

      <ChatInput
        onSend={handleSend}
        disabled={isLoading}
        placeholder={isLoading ? "Waiting for response..." : "Type a message..."}
      />
    </div>
  );
}
\`\`\`

## Step 6: Integrate with WebLLM

Update \`src/App.tsx\`:

\`\`\`tsx
import { useState, useEffect, useCallback } from "react";
import * as webllm from "@mlc-ai/web-llm";
import { Chat } from "./components/Chat";

const MODEL_ID = "SmolLM2-360M-Instruct-q4f16_1-MLC";

function App() {
  const [engine, setEngine] = useState<webllm.MLCEngine | null>(null);
  const [loadingProgress, setLoadingProgress] = useState("");
  const [isReady, setIsReady] = useState(false);

  // Initialize WebLLM
  useEffect(() => {
    const initEngine = async () => {
      const newEngine = new webllm.MLCEngine();

      newEngine.setInitProgressCallback((report) => {
        setLoadingProgress(\`\${report.text} (\${Math.round(report.progress * 100)}%)\`);
      });

      await newEngine.reload(MODEL_ID);
      setEngine(newEngine);
      setIsReady(true);
    };

    initEngine();
  }, []);

  // Send message with streaming
  const handleSendMessage = useCallback(async (
    message: string,
    onToken: (token: string) => void
  ) => {
    if (!engine) return;

    const response = await engine.chat.completions.create({
      messages: [{ role: "user", content: message }],
      temperature: 0.7,
      stream: true,
    });

    for await (const chunk of response) {
      const token = chunk.choices[0]?.delta?.content || "";
      if (token) onToken(token);
    }
  }, [engine]);

  if (!isReady) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-gray-600">{loadingProgress || "Initializing..."}</p>
        </div>
      </div>
    );
  }

  return <Chat onSendMessage={handleSendMessage} />;
}

export default App;
\`\`\`

## Enhancements

### Adding Typing Indicator

\`\`\`tsx
function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 p-4">
      <div className="flex gap-1">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
            style={{ animationDelay: \`\${i * 0.15}s\` }}
          />
        ))}
      </div>
      <span className="text-sm text-gray-500 ml-2">Assistant is typing...</span>
    </div>
  );
}
\`\`\`

### Keyboard Shortcuts

\`\`\`typescript
// In ChatInput
const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
  // Cmd/Ctrl + Enter to send
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    handleSend();
    return;
  }

  // Enter without shift to send
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
  // Shift + Enter for new line (default behavior)
};
\`\`\`

## Conclusion

You now have a production-ready chat interface that:
- Streams responses in real-time
- Renders markdown with code highlighting
- Auto-scrolls smoothly
- Handles loading and error states
- Supports keyboard shortcuts

Next steps:
- Add message persistence with localStorage
- Implement conversation history
- Add theme support (light/dark mode)
- Build model selection UI

---

*This tutorial is part of the WebLLM Fundamentals series.*
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
