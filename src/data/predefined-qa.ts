export interface PredefinedQA {
  id: string;
  question: string;
  answer: string;
  category: 'research' | 'background' | 'collaboration' | 'career';
  icon: string; // Icon name for the conversation starter
}

export const predefinedQuestions: PredefinedQA[] = [
  {
    id: 'publications',
    question: "Where can I find your publications and research profile?",
    answer: "For my complete and up-to-date publication list, I'd recommend checking my Google Scholar profile. You can find all my research papers, citations, and academic contributions there. Here's the link: https://scholar.google.com/citations?user=A8J42tQAAAAJ. My research spans synthetic data generation, multi-agent systems, and multi-modal fusion, with work published in top-tier AI conferences and journals. I'm particularly proud of my contributions to self-improving AI agents.",
    category: 'research',
    icon: 'Brain'
  },
  {
    id: 'current-project',
    question: "What's your current project?",
    answer: "I'm currently working on synthetic data generation frameworks for self-improving AI agents. This project focuses on developing systems that can autonomously generate high-quality training data to improve their own performance over time. The framework combines multi-agent collaboration with advanced data synthesis techniques to create more robust and adaptive AI systems. This work builds on my expertise in synthetic data generation and multi-agent systems, and I'm excited about its potential applications in autonomous AI development.",
    category: 'research',
    icon: 'Code'
  },
  {
    id: 'most-interesting-project',
    question: "What's your most interesting project?",
    answer: "My most interesting project has been developing a multi-agent system for autonomous synthetic data generation. This project was fascinating because it involved creating AI agents that could not only generate synthetic data but also evaluate and improve the quality of that data through collaborative learning. The agents would compete and cooperate to create increasingly better datasets, essentially creating a self-improving data generation system. What made it particularly interesting was watching how the agents developed different strategies for data synthesis and how they learned to coordinate their efforts to achieve better results than any single agent could produce alone.",
    category: 'research',
    icon: 'Rocket'
  },
  {
    id: 'ai-project-help',
    question: "How can your synthetic data expertise help with our AI project?",
    answer: "My expertise in synthetic data generation can help your AI project in several ways. I can design frameworks for generating high-quality training data that's difficult or expensive to collect naturally. This includes creating synthetic datasets that maintain the statistical properties of real data while being more diverse and comprehensive. I can also help implement multi-agent systems that can collaboratively generate and validate synthetic data, ensuring quality and relevance. Additionally, my experience with multi-agent systems can help scale your AI training across multiple agents or environments.",
    category: 'collaboration',
    icon: 'Code'
  },
  {
    id: 'multi-agent-experience',
    question: "Tell me about your multi-agent systems experience and applications",
    answer: "I have extensive experience in multi-agent systems, particularly in collaborative learning and problem-solving. My work includes developing agents that can coordinate to generate synthetic data, share knowledge across different domains, and improve their collective performance through collaboration. I've worked on applications ranging from autonomous data generation to collaborative decision-making systems. My research focuses on creating agents that can communicate effectively, share resources, and learn from each other's experiences while maintaining individual autonomy.",
    category: 'research',
    icon: 'Rocket'
  },
  {
    id: 'collaboration-opportunities',
    question: "What collaboration opportunities are you open to?",
    answer: "I'm open to various collaboration opportunities including research partnerships, consulting on AI and synthetic data projects, and industry collaborations. I'm particularly interested in projects involving synthetic data generation and multi-agent systems. I can help with designing AI training pipelines, implementing multi-agent architectures, or developing synthetic data generation frameworks. I'm also available for academic collaborations, industry consulting, and research partnerships. Feel free to reach out to discuss potential opportunities that align with your needs and my expertise.",
    category: 'collaboration',
    icon: 'GraduationCap'
  },
  {
    id: 'research-industry',
    question: "How does your research apply to industry problems?",
    answer: "My research has direct applications to several industry challenges. Synthetic data generation can help companies overcome data scarcity issues, reduce data collection costs, and improve AI model robustness. Multi-agent systems can be applied to collaborative decision-making and autonomous systems. My work on self-improving AI agents can help create systems that continuously learn and adapt to changing environments. These technologies are particularly relevant for industries dealing with limited data or autonomous systems that need to operate in dynamic environments.",
    category: 'research',
    icon: 'MessageCircle'
  },
  {
    id: 'career-goals',
    question: "What are your career goals and preferred work arrangements?",
    answer: "My career goals focus on advancing AI technology through research and practical applications. I'm interested in roles that combine research with real-world impact, whether in academia, industry research labs, or technology companies. I'm particularly drawn to opportunities that involve synthetic data generation and multi-agent systems. I'm flexible with work arrangements and open to remote work, hybrid models, or on-site positions. I value environments that encourage innovation, collaboration, and continuous learning. I'm also interested in opportunities that allow me to contribute to both theoretical advances and practical implementations.",
    category: 'career',
    icon: 'BotIcon'
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
