export interface PredefinedQA {
  id: string;
  question: string;
  answer: string;
  category: 'research' | 'background' | 'collaboration' | 'career' | 'vision';
  icon: string;
}

export const predefinedQuestions: PredefinedQA[] = [
  {
    id: 'elevator-pitch',
    question: "Give me your 30-second pitch — what makes you unique?",
    answer: "I'm a PhD researcher at USC building the infrastructure for autonomous AI. My work sits at the intersection of three critical areas: multi-agent systems (how AI agents collaborate), synthetic data generation (how we train AI without compromising privacy), and model safety (preventing AI systems from degrading over time). What sets me apart is my dual focus on academic rigor and industry impact—I'm currently a Fellow at Handshake AI working on real-world deployment while publishing at top venues like ACM ICMI. I bridge the gap between 'this works in a paper' and 'this works in production.'",
    category: 'background',
    icon: 'Rocket'
  },
  {
    id: 'future-of-ai',
    question: "Where do you see AI heading in the next 2-3 years?",
    answer: "The era of 'just add more data and compute' is hitting diminishing returns. The next wave is about **agentic AI**—systems that don't just respond but actually act. Instead of chatbots, we're building partners. Three key shifts: (1) Multi-agent orchestration where specialized models collaborate on complex tasks, (2) Test-time compute scaling where models 'think longer' before responding, and (3) Small, sovereign models that run locally on devices for privacy. My research directly addresses these trends—I'm building the coordination frameworks and data pipelines these systems will need.",
    category: 'vision',
    icon: 'Sparkles'
  },
  {
    id: 'investment-opportunity',
    question: "Why should investors be interested in your work?",
    answer: "Three reasons: (1) **Timing** — We're at an inflection point where AI is shifting from demos to deployment, and the infrastructure for autonomous agents is the critical bottleneck. (2) **Dual validation** — My work is published at top academic venues AND being deployed at Handshake AI, proving both scientific rigor and commercial viability. (3) **Defensible expertise** — Multi-agent coordination and synthetic data safety are specialized skills with high barriers to entry. The market for enterprise AI agents is projected to be $100B+ by 2028, and I'm building the foundational tech. I'm open to discussing advisory roles, research partnerships, or venture opportunities.",
    category: 'collaboration',
    icon: 'TrendingUp'
  },
  {
    id: 'publications',
    question: "What's your most impactful research contribution?",
    answer: "My paper on 'Multimodal Synthetic Data Finetuning and Model Collapse' at ACM ICMI 2025 addresses one of the most critical problems in AI: when models are trained on synthetic data from other models, they can degrade over time—what we call 'model collapse.' I developed methods to detect and prevent this, which is essential as the industry increasingly relies on synthetic data for training. This work has implications for any company using AI-generated data, which is basically everyone. You can find all my publications on Google Scholar: https://scholar.google.com/citations?user=A8J42tQAAAAJ",
    category: 'research',
    icon: 'BookOpen'
  },
  {
    id: 'current-project',
    question: "What are you building right now?",
    answer: "I'm working on three active projects: (1) **DREAM-C2L** — An open-source framework for curriculum learning that's already being used by other researchers for reproducible ML experiments. (2) **Project Orion at Handshake AI** — Multi-agent orchestration for enterprise workflows, where specialized AI agents collaborate to complete complex tasks autonomously. (3) **Project Canary** — Safety research on synthetic data generation, ensuring AI training pipelines don't inadvertently degrade model quality. All of these tie back to my core thesis: autonomous AI needs robust coordination and safe data pipelines.",
    category: 'research',
    icon: 'Code'
  },
  {
    id: 'hiring-fit',
    question: "What kind of roles or teams are you looking for?",
    answer: "I thrive at the intersection of research and deployment. Ideal fits include: (1) **Research Scientist** roles at labs like DeepMind, Anthropic, or OpenAI where I can push the frontier on multi-agent systems. (2) **Applied AI** teams at companies deploying autonomous agents (think Adept, Cognition, or enterprise AI startups). (3) **Founding/early engineer** at AI startups where my research background can shape product direction. I'm also open to hybrid arrangements that let me continue my PhD while contributing to industry. What matters most is working on problems that matter at scale.",
    category: 'career',
    icon: 'Target'
  },
  {
    id: 'technical-skills',
    question: "What's your technical stack and expertise level?",
    answer: "**Expert level**: Python, PyTorch, distributed training (SLURM/HPC clusters), LLM fine-tuning, multi-agent frameworks. **Proficient**: Next.js/React for research demos, MLOps (Weights & Biases, Docker), data engineering pipelines. **Research specialties**: Synthetic data generation, curriculum learning, multi-agent coordination, model safety/alignment. I've built systems that run on USC's CARC cluster (thousands of GPU hours) and deployed production AI at Handshake. I use 'uv' for modern Python dependency management and prioritize reproducible research. My code is available on GitHub: https://github.com/zizhao-hu",
    category: 'background',
    icon: 'Terminal'
  },
  {
    id: 'collaboration',
    question: "Are you available for consulting or collaboration?",
    answer: "Yes! I'm open to: (1) **Research partnerships** — If you're working on multi-agent systems, synthetic data, or LLM safety and want academic collaboration. (2) **Technical consulting** — Advising companies on AI architecture, training pipelines, or multi-agent deployment. (3) **Advisory roles** — For AI startups where my research expertise can help shape technical direction. (4) **Investment discussions** — I'm building technology with commercial potential and am open to conversations with VCs and angels focused on AI infrastructure. Best way to reach me: zizhaoh@usc.edu",
    category: 'collaboration',
    icon: 'Handshake'
  },
  {
    id: 'personal-side',
    question: "Tell me something interesting about you outside of AI",
    answer: "Here's something most researchers don't mention: I'm a K-pop cover dancer! I perform with Spade A, USC's K-pop dance group, and I've danced at KCON. I also maintain a strict 6 AM fitness routine focused on muscle building—discipline in the gym translates to discipline in research. I'm a huge foodie (hot pot and KBBQ are my weaknesses) and I've traveled extensively between the US and China. I think the best researchers have lives outside the lab—it keeps you creative and grounded. Plus, learning choreography is surprisingly similar to debugging code: break it down, iterate, and practice until it's perfect.",
    category: 'background',
    icon: 'Music'
  }
];

// Helper function to get conversation starters (just the questions and icons)
export const getConversationStarters = () => {
  return predefinedQuestions.map(qa => ({
    question: qa.question,
    icon: qa.icon
  }));
};

// Helper function to get answer by question
export const getAnswerByQuestion = (question: string): string | null => {
  const qa = predefinedQuestions.find(qa => qa.question === question);
  return qa ? qa.answer : null;
};

// Helper function to get answer by ID
export const getAnswerById = (id: string): string | null => {
  const qa = predefinedQuestions.find(qa => qa.id === id);
  return qa ? qa.answer : null;
};
