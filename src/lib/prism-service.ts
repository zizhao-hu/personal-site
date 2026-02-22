/**
 * PRISM — Personal Response & Intelligent Semantic Matching
 * 
 * Provides instant responses by matching user queries against a knowledge base
 * using TF-IDF cosine similarity. No model loading required — works immediately.
 * 
 * The knowledge base is built from:
 *   1. Predefined Q&A pairs (curated answers to common questions)
 *   2. Topic-based knowledge chunks (from zizhao-context.ts + mock-llm-service keyword responses)
 */

import { predefinedQuestions } from '@/data/predefined-qa';

export interface ChatMessage {
    role: "user" | "assistant";
    content: string;
}

interface KnowledgeEntry {
    id: string;
    keywords: string[];       // Search terms for matching
    question?: string;        // Original question if from Q&A
    answer: string;           // The response
    category: string;         // For display
    tfidf?: number[];         // Precomputed vector (computed on init)
}

// ── KNOWLEDGE BASE ──────────────────────────────────────────────
// Combines predefined Q&A + topic-based knowledge for broad coverage

const TOPIC_KNOWLEDGE: Omit<KnowledgeEntry, 'tfidf'>[] = [
    {
        id: 'topic-publications',
        keywords: ['publication', 'paper', 'published', 'research achievement', 'cite', 'citation', 'scholar', 'conference', 'journal', 'icmi', 'neurips', 'iclr', 'icml'],
        answer: "My recent work spans three areas: **'Multimodal Synthetic Data Finetuning and Model Collapse'** at ACM ICMI 2025 — addressing how models degrade when trained on AI-generated data. **'Static Key Attention in Vision'** — novel attention mechanisms for vision models. **'Lateralization MLP'** — brain-inspired architectures for diffusion models. I also serve as a reviewer for NeurIPS, ICLR, and ICML. Find all publications on [Google Scholar](https://scholar.google.com/citations?user=A8J42tQAAAAJ).",
        category: 'research'
    },
    {
        id: 'topic-research',
        keywords: ['research', 'synthetic data', 'thesis', 'dissertation', 'phd topic', 'research focus', 'what do you study'],
        answer: "My research has three pillars: (1) **LLM/VLM multi-agent interaction** — how multiple agents collaborate, debate, and verify each other's outputs through generate-validate loops. (2) **Transformer memory mechanisms** — designing efficient architectures that maximize capability per FLOP. (3) **Synthetic data with generate-validate loops** — self-improving data pipelines that prevent model collapse. My thesis: *AI systems should improve themselves while remaining under human control.*",
        category: 'research'
    },
    {
        id: 'topic-multi-agent',
        keywords: ['multi-agent', 'agent', 'multi agent', 'autonomous', 'agentforge', 'orchestration', 'coordination', 'collaboration', 'self-improving'],
        answer: "Multi-agent systems are my core focus. I'm building **AgentForge** — a multi-agent orchestration framework where specialized agents collaborate on complex tasks with self-correction. My research explores coordination mechanisms, knowledge sharing, generate-validate loops, and how agents can generate their own training signal. This ties into my broader thesis that AI systems should be able to improve themselves while remaining under human control.",
        category: 'research'
    },
    {
        id: 'topic-multimodal',
        keywords: ['multi-modal', 'multimodal', 'fusion', 'vision language', 'vlm', 'vla', 'image', 'visual'],
        answer: "Multi-modal fusion is central to my work on vision-language models (VLMs) and vision-language-action models (VLAs). I'm developing unified multimodal backbones and studying how transformers store and reason over information across modalities. My work on **Multimodal Synthetic Data Finetuning** specifically addresses model collapse when training on AI-generated multimodal data — published at ACM ICMI 2025.",
        category: 'research'
    },
    {
        id: 'topic-current-projects',
        keywords: ['current', 'project', 'building', 'working on', 'now', 'today', 'recent'],
        answer: "I'm working on several interconnected projects: (1) **AgentForge** — Multi-agent orchestration framework with self-correction. (2) **DREAM-C2L** — Open-source continual learning framework. (3) **ReasonChain** — Test-time compute scaling research. (4) **Static Key Attention** — Novel attention mechanisms for vision transformers. All tied to my core thesis: building AI systems that improve themselves while remaining under control.",
        category: 'research'
    },
    {
        id: 'topic-usc',
        keywords: ['usc', 'university', 'school', 'southern california', 'viterbi', 'glamour', 'lab', 'advisor', 'professor', 'thomason', 'rostami'],
        answer: "I'm a CS Ph.D. student at USC's Viterbi School of Engineering, part of the **GLAMOUR Lab** (Prof. Jesse Thomason) and **MINDS Group** (Prof. Mohammad Rostami). USC provides an excellent research environment with access to world-class computing resources, including the CARC cluster for distributed training. My research focuses on multi-agent systems, continual learning, and efficient model architectures.",
        category: 'background'
    },
    {
        id: 'topic-background',
        keywords: ['background', 'experience', 'history', 'where', 'from', 'education', 'georgia tech', 'ilab', 'handshake', 'move fellow'],
        answer: "My journey spans multiple institutions: **Georgia Tech** for undergrad & masters, then **USC** for my PhD. I've done research at USC iLab, Georgia Tech's Agile Systems Lab, and the Photonics Research Group. I was also a **MOVE Fellow at Handshake AI** where I worked on reasoning refinement, safety injections, and jailbreak testing — contributing 15,000+ tasks across 15 domains and raising approval rates from 10% to 40%.",
        category: 'background'
    },
    {
        id: 'topic-skills',
        keywords: ['skill', 'expertise', 'capability', 'tech stack', 'programming', 'language', 'python', 'pytorch', 'framework', 'tool'],
        answer: "**Expert**: Python, PyTorch, distributed training (SLURM/HPC clusters), LLM fine-tuning, multi-agent frameworks. **Proficient**: React/Next.js, MLOps (Weights & Biases, Docker), data pipelines. **Research specialties**: Multi-agent systems, synthetic data generation, brain-inspired architectures, continual learning, model safety/alignment. My code is on [GitHub](https://github.com/zizhao-hu).",
        category: 'background'
    },
    {
        id: 'topic-goals',
        keywords: ['goal', 'future', 'plan', 'ambition', 'dream', 'aspiration', 'next', 'vision'],
        answer: "My long-term goal is to advance AI by developing systems that can learn and improve autonomously. I believe the gap between what AI can do alone and what a skilled human + AI can do together is where the real value lives. I'm interested in academic positions, industry research roles, and entrepreneurial opportunities — anything that lets me build AI that matters at scale.",
        category: 'career'
    },
    {
        id: 'topic-collaboration',
        keywords: ['collaboration', 'consulting', 'work together', 'hire', 'partner', 'advisory', 'invest', 'opportunity', 'contact', 'email', 'reach'],
        answer: "I'm open to: (1) **Research partnerships** on multi-agent systems, synthetic data, or LLM safety. (2) **Technical consulting** on AI architecture, training pipelines, or multi-agent deployment. (3) **Advisory roles** for AI startups. (4) **Investment discussions** — my tech has commercial potential. Best way to reach me: **zizhaoh@usc.edu** | [GitHub](https://github.com/zizhao-hu)",
        category: 'collaboration'
    },
    {
        id: 'topic-roles',
        keywords: ['role', 'job', 'position', 'career', 'looking for', 'hiring', 'team', 'company', 'startup', 'industry'],
        answer: "I thrive at the intersection of research and deployment. Ideal fits: (1) **Research Scientist** at labs like DeepMind, Anthropic, OpenAI. (2) **Applied AI** at companies deploying autonomous agents. (3) **Founding/early engineer** at AI startups where research shapes product. I'm also open to hybrid arrangements continuing my PhD while contributing to industry.",
        category: 'career'
    },
    {
        id: 'topic-personal',
        keywords: ['personal', 'hobby', 'fun', 'outside', 'dance', 'kpop', 'soccer', 'basketball', 'food', 'pet', 'dog', 'gym', 'fitness', 'messi', 'curry', 'spade', 'kcon', 'hot pot', 'kbbq'],
        answer: "I'm a K-pop cover dancer with **Spade A** at USC (performed at KCON!). Huge soccer fan — Messi is the GOAT, no debate. Love watching Steph Curry too. I once had a border collie, one of the smartest dogs ever. 6 AM gym routine, and I'm a hot pot and KBBQ enthusiast. Learning choreography is like debugging code: break it down, iterate, drill until it's perfect. 😄",
        category: 'personal'
    },
    {
        id: 'topic-ai-future',
        keywords: ['ai future', 'ai trend', 'where is ai going', 'next wave', 'agentic', 'predictions', 'industry'],
        answer: "The era of 'just add more data and compute' is hitting diminishing returns. The next wave is **agentic AI** — systems that act, not just respond. Three key shifts: (1) Multi-agent orchestration with specialized models collaborating. (2) Test-time compute scaling — models 'think longer' before responding. (3) Small, sovereign models running locally for privacy. My research directly addresses these trends.",
        category: 'vision'
    },
    {
        id: 'topic-hello',
        keywords: ['hello', 'hi', 'hey', 'greetings', 'whats up', 'how are you', 'good morning', 'good afternoon', 'good evening', 'sup'],
        answer: "Hey! I'm Zizhao Hu — CS PhD student at USC, working on multi-agent AI systems and synthetic data. I'm here to chat about research, career opportunities, collaborations, or just connect. What's on your mind?",
        category: 'general'
    },
    {
        id: 'topic-who',
        keywords: ['who are you', 'about you', 'tell me about yourself', 'introduce yourself', 'what do you do'],
        answer: "I'm **Zizhao Hu** (胡子昭), a CS Ph.D. student at USC building AI systems that improve themselves while remaining under control. My research spans multi-agent interaction, continual learning, and synthetic data generation. I was a MOVE Fellow at Handshake AI and I review for NeurIPS, ICLR, and ICML. Outside the lab, I do K-pop cover dance with Spade A and I'm obsessed with soccer and hot pot. How can I help?",
        category: 'background'
    }
];

// ── TF-IDF VECTOR MATCHING ─────────────────────────────────────

class PrismService {
    private entries: KnowledgeEntry[] = [];
    private vocabulary: Map<string, number> = new Map();
    private idf: number[] = [];
    private ready = false;

    constructor() {
        this.buildKnowledgeBase();
    }

    private tokenize(text: string): string[] {
        return text.toLowerCase()
            .replace(/[^\w\s'-]/g, ' ')
            .split(/\s+/)
            .filter(w => w.length > 1);
    }

    private buildKnowledgeBase() {
        // Merge predefined Q&A + topic knowledge
        const allEntries: KnowledgeEntry[] = [];

        // Add predefined Q&A
        for (const qa of predefinedQuestions) {
            allEntries.push({
                id: qa.id,
                keywords: this.tokenize(qa.question),
                question: qa.question,
                answer: qa.answer,
                category: qa.category,
            });
        }

        // Add topic knowledge
        for (const topic of TOPIC_KNOWLEDGE) {
            allEntries.push({ ...topic });
        }

        // Build vocabulary from all keywords + answers
        const vocabSet = new Set<string>();
        const allDocs: string[][] = [];

        for (const entry of allEntries) {
            const tokens = [...entry.keywords, ...this.tokenize(entry.answer)];
            if (entry.question) tokens.push(...this.tokenize(entry.question));
            allDocs.push(tokens);
            tokens.forEach(t => vocabSet.add(t));
        }

        // Assign indices
        let idx = 0;
        for (const word of vocabSet) {
            this.vocabulary.set(word, idx++);
        }

        // Compute IDF
        const N = allDocs.length;
        this.idf = new Array(this.vocabulary.size).fill(0);
        for (const doc of allDocs) {
            const seen = new Set<string>();
            for (const token of doc) {
                if (!seen.has(token)) {
                    seen.add(token);
                    const i = this.vocabulary.get(token);
                    if (i !== undefined) this.idf[i]++;
                }
            }
        }
        this.idf = this.idf.map(df => Math.log((N + 1) / (df + 1)) + 1);

        // Compute TF-IDF vectors for each entry (using keywords + question as the searchable doc)
        for (const entry of allEntries) {
            const tokens = [...entry.keywords];
            if (entry.question) tokens.push(...this.tokenize(entry.question));
            // Weight keywords more heavily by duplicating them
            const weighted = [...tokens, ...entry.keywords, ...entry.keywords];
            entry.tfidf = this.computeTFIDF(weighted);
        }

        this.entries = allEntries;
        this.ready = true;
    }

    private computeTFIDF(tokens: string[]): number[] {
        const vec = new Array(this.vocabulary.size).fill(0);
        const tf: Record<number, number> = {};

        for (const token of tokens) {
            const i = this.vocabulary.get(token);
            if (i !== undefined) {
                tf[i] = (tf[i] || 0) + 1;
            }
        }

        for (const [i, count] of Object.entries(tf)) {
            const idx = parseInt(i);
            vec[idx] = (1 + Math.log(count)) * this.idf[idx]; // log-normalized TF * IDF
        }

        return vec;
    }

    private cosineSimilarity(a: number[], b: number[]): number {
        let dot = 0, magA = 0, magB = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            magA += a[i] * a[i];
            magB += b[i] * b[i];
        }
        const denom = Math.sqrt(magA) * Math.sqrt(magB);
        return denom === 0 ? 0 : dot / denom;
    }

    /**
     * Find the best matching response for a user query.
     * Returns the answer and a confidence score (0-1).
     */
    match(query: string): { answer: string; confidence: number; category: string; matchedQuestion?: string } {
        if (!this.ready || this.entries.length === 0) {
            return {
                answer: "Hey! I'm Zizhao Hu. I'm still warming up — ask me anything about my research, background, or collaborations!",
                confidence: 0,
                category: 'general'
            };
        }

        const queryTokens = this.tokenize(query);
        const queryVec = this.computeTFIDF(queryTokens);

        let bestScore = -1;
        let bestEntry: KnowledgeEntry | null = null;

        for (const entry of this.entries) {
            if (!entry.tfidf) continue;
            const score = this.cosineSimilarity(queryVec, entry.tfidf);
            if (score > bestScore) {
                bestScore = score;
                bestEntry = entry;
            }
        }

        // Threshold: if similarity is too low, give a generic response
        if (bestScore < 0.05 || !bestEntry) {
            return {
                answer: "That's a great question! I'm not sure I have the perfect answer for that one. I'm best at talking about my research (multi-agent AI, synthetic data), my background (USC, Georgia Tech, Handshake AI), or potential collaborations. Try asking about one of those! 😊",
                confidence: bestScore,
                category: 'general'
            };
        }

        return {
            answer: bestEntry.answer,
            confidence: bestScore,
            category: bestEntry.category,
            matchedQuestion: bestEntry.question,
        };
    }

    /**
     * Generate a response for the chat, considering conversation history.
     */
    generateResponse(messages: ChatMessage[]): string {
        const lastUserMessage = messages.filter(m => m.role === 'user').pop();
        if (!lastUserMessage) {
            return "Hey! I'm Zizhao Hu — CS PhD student at USC. What would you like to know about my research, projects, or background?";
        }

        // Check for exact predefined match first
        const exactMatch = predefinedQuestions.find(
            qa => qa.question.toLowerCase() === lastUserMessage.content.toLowerCase()
        );
        if (exactMatch) return exactMatch.answer;

        // Vector match
        const result = this.match(lastUserMessage.content);
        return result.answer;
    }

    isReady(): boolean {
        return this.ready;
    }
}

// Export singleton
export const prismService = new PrismService();
