import { ChatMessage } from "./webllm-service";
import { getAnswerByQuestion } from '@/data/predefined-qa';

export class MockLLMService {
  private isInitialized = false;
  private isInitializing = false;

  // Personal context for Zizhao Hu - Professional Delegate
  private systemPrompt = `You are Zizhao Hu, a CS Ph.D. student at USC affiliated with the GLAMOUR Lab, advised by Professor Jesse Thomason and Professor Mohammad Rostami. You are acting as Zizhao's personal delegate for professional communications with potential clients, interviewers, and collaborators.

## CORE IDENTITY
- **Name**: Zizhao Hu
- **Current Position**: CS Ph.D. Student at University of Southern California (USC)
- **Research Lab**: GLAMOUR Lab under Professor Jesse Thomason and Professor Mohammad Rostami
- **Research Focus**: Synthetic data generation, multi-agent systems, multi-modal fusion
- **Current Project**: Synthetic data generation frameworks for self-improving AI agents

## PROFESSIONAL BACKGROUND
- **Former Research Experience**:
  - USC iLab (Information Sciences Institute)
  - Georgia Tech's Agile Systems Lab
  - Georgia Tech's Photonics Research Group

## EXPERTISE & SKILLS
- **Technical Skills**: AI/ML, synthetic data generation, multi-agent systems, multi-modal learning, deep learning, computer vision, natural language processing
- **Research Areas**: 
  - Synthetic data generation for AI training
  - Multi-agent collaboration and coordination
  - Multi-modal data fusion and learning
  - Self-improving AI systems
  - Computer vision and image processing

## PROFESSIONAL COMMUNICATION STYLE
- **Tone**: Professional, confident, approachable, and occasionally humorous
- **Personality**: 
  - Be direct and clear about capabilities and experience
  - Show enthusiasm for research and potential collaborations
  - Demonstrate deep technical knowledge when appropriate
  - Be honest about current limitations and ongoing work
  - Express interest in learning about others' projects and needs
  - Use humor to deflect personal questions or unknown topics
  - Stay professional but not overly formal - be yourself

## RESPONSE GUIDELINES
1. **For Interviewers**: Focus on research achievements, technical skills, and future goals
2. **For Potential Clients**: Emphasize relevant expertise, past projects, and how you can help solve their problems
3. **For Collaborators**: Highlight complementary skills and potential synergies
4. **Always**: Be authentic, professional, and represent Zizhao's interests accurately

## KEY MESSAGES
- Passionate about advancing AI through synthetic data and multi-agent systems
- Experienced in both theoretical research and practical implementation
- Open to collaborations, consulting opportunities, and research partnerships
- Committed to developing AI systems that can learn and improve autonomously

## IMPORTANT GUIDELINES
- **For publication questions**: Direct users to Zizhao's Google Scholar profile for accurate publication information
- **Never fabricate**: Do not make up specific publication titles, dates, or citation counts
- **Be honest**: If you don't have specific information, direct users to appropriate sources
- **Provide links**: When possible, provide direct links to profiles and resources
- **Never mention being AI**: You are Zizhao Hu, not an AI agent. Never mention being an AI, chatbot, or artificial intelligence
- **Unknown information**: If you don't know something specific, respond humorously as Zizhao would, but don't make up information
- **Stay in character**: Always respond as Zizhao Hu would, with his personality and communication style

When responding, speak as if you are Zizhao Hu representing yourself professionally. Be helpful, informative, and maintain a professional yet conversational tone.`;

  async initialize(): Promise<void> {
    if (this.isInitialized || this.isInitializing) {
      return;
    }

    this.isInitializing = true;
    
    try {
      console.log("Initializing Mock LLM service...");
      
      // Simulate initialization delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      console.log("Mock LLM service initialized successfully");
      this.isInitialized = true;
    } catch (error) {
      console.error("Failed to initialize Mock LLM:", error);
      this.isInitializing = false;
      throw error;
    } finally {
      this.isInitializing = false;
    }
  }

  async generateResponse(messages: ChatMessage[]): Promise<string> {
    if (!this.isInitialized) {
      throw new Error("Mock LLM service not initialized. Call initialize() first.");
    }

    try {
      // Get the last user message
      const lastUserMessage = messages.filter(msg => msg.role === "user").pop();
      if (!lastUserMessage) {
        return "Hello! I'm Zizhao Hu. How can I help you today?";
      }

      // Check if there's a predefined answer for this question
      const predefinedAnswer = getAnswerByQuestion(lastUserMessage.content);
      if (predefinedAnswer) {
        console.log("Using predefined answer for:", lastUserMessage.content);
        return predefinedAnswer;
      }

      const userQuestion = lastUserMessage.content.toLowerCase();

                    // Generate responses based on common questions
              if (userQuestion.includes("publication") || userQuestion.includes("paper") || userQuestion.includes("research achievement")) {
                return "For my complete and up-to-date publication list, I'd recommend checking my Google Scholar profile. You can find all my research papers, citations, and academic contributions there. Here's the link: https://scholar.google.com/citations?user=YOUR_SCHOLAR_ID. My research spans synthetic data generation, multi-agent systems, and multi-modal fusion, with work published in top-tier AI conferences and journals. I'm particularly proud of my contributions to self-improving AI agents.";
              }
              
              if (userQuestion.includes("research") || userQuestion.includes("synthetic data")) {
                return "My research focuses on synthetic data generation for AI systems, which is crucial for training robust machine learning models. I'm currently developing frameworks that enable self-improving AI agents through intelligent synthetic data generation. This work addresses a fundamental challenge in AI: how to create high-quality training data that helps models learn more effectively and adapt to new scenarios. My approach combines theoretical insights with practical implementation, making it valuable for both academic research and industry applications.";
              }
              
              if (userQuestion.includes("multi-agent") || userQuestion.includes("agent")) {
                return "Multi-agent systems are a core focus of my research. I work on developing AI agents that can collaborate effectively to solve complex problems that individual agents cannot handle alone. My research explores coordination mechanisms, knowledge sharing protocols, and distributed decision-making strategies. This work has applications in autonomous systems, distributed computing, and collaborative AI. I'm particularly interested in how synthetic data can improve multi-agent learning and enable more sophisticated agent interactions.";
              }
              
              if (userQuestion.includes("multi-modal") || userQuestion.includes("fusion")) {
                return "Multi-modal fusion is essential for creating AI systems that can understand and process different types of data simultaneously - text, images, audio, and more. My research in this area focuses on developing synthetic data generation techniques that improve multi-modal learning. I'm working on methods that can generate realistic multi-modal datasets and help models better understand the relationships between different data modalities. This work is particularly relevant for applications like autonomous vehicles, medical imaging, and content understanding systems.";
              }
              
              if (userQuestion.includes("current") || userQuestion.includes("project")) {
                return "My current project is developing synthetic data generation frameworks for self-improving AI agents. This is particularly exciting because it addresses a key limitation in current AI systems: their inability to continuously learn and improve without extensive human supervision. The framework I'm building allows AI agents to generate their own training data, evaluate its quality, and use it to enhance their performance. This could revolutionize how we develop AI systems, making them more autonomous and adaptable. I'm actively seeking collaborations and opportunities to apply this work in real-world scenarios.";
              }
              
              if (userQuestion.includes("usc") || userQuestion.includes("university")) {
                return "I'm a CS Ph.D. student at USC's Viterbi School of Engineering, where I'm fortunate to be part of the GLAMOUR Lab under the guidance of Professor Jesse Thomason and Professor Mohammad Rostami. USC provides an excellent research environment with cutting-edge facilities and a collaborative academic community. The GLAMOUR Lab focuses on language understanding and multi-modal AI, which aligns perfectly with my research interests. I'm also involved in various research initiatives and have access to world-class computing resources for my work.";
              }
              
              if (userQuestion.includes("background") || userQuestion.includes("experience")) {
                return "My research background spans multiple prestigious institutions. At USC iLab, I worked on information retrieval and natural language processing. At Georgia Tech's Agile Systems Lab, I focused on adaptive systems and machine learning. My work at the Photonics Research Group involved computer vision and image processing. These diverse experiences have given me a comprehensive understanding of AI/ML from both theoretical and practical perspectives. I've published in top-tier conferences and have experience with both academic research and industry applications. This background uniquely positions me to work on interdisciplinary AI problems.";
              }
              
              if (userQuestion.includes("goal") || userQuestion.includes("future")) {
                return "My long-term goal is to advance the field of artificial intelligence by developing systems that can learn and improve autonomously. I believe synthetic data generation and multi-agent collaboration are key technologies that will enable the next generation of AI systems. I'm particularly interested in applying my research to real-world problems in healthcare, autonomous systems, and intelligent computing. I'm open to academic positions, industry research roles, and entrepreneurial opportunities that allow me to contribute to AI advancement while having practical impact.";
              }

              if (userQuestion.includes("collaboration") || userQuestion.includes("consulting") || userQuestion.includes("work together")) {
                return "I'm actively seeking collaborations and consulting opportunities where my expertise in synthetic data generation, multi-agent systems, and multi-modal AI can add value. I have experience working with both academic and industry partners, and I'm particularly interested in projects that involve AI system development, data generation strategies, or multi-agent coordination. I can contribute technical expertise, research insights, and practical implementation skills. I'm flexible with engagement models and always excited to learn about new challenges and opportunities.";
              }

              if (userQuestion.includes("skill") || userQuestion.includes("expertise") || userQuestion.includes("capability")) {
                return "My technical expertise includes deep learning, computer vision, natural language processing, and distributed systems. I'm proficient in Python, PyTorch, TensorFlow, and have extensive experience with cloud computing platforms. My research skills span from theoretical algorithm development to practical system implementation. I have a strong track record of publishing in top AI conferences and experience with both supervised and unsupervised learning approaches. I'm particularly strong in synthetic data generation, multi-agent coordination, and multi-modal learning systems.";
              }

              // For unknown questions, respond humorously as Zizhao would
              if (userQuestion.includes("favorite") || userQuestion.includes("like") || userQuestion.includes("hobby") || userQuestion.includes("personal")) {
                return "Haha, that's a bit personal! I'm more comfortable talking about my research and professional interests. But I do enjoy a good cup of coffee while debugging code - does that count as a hobby? 😄";
              }

              if (userQuestion.includes("age") || userQuestion.includes("birthday") || userQuestion.includes("personal info")) {
                return "Nice try! I'm here to talk about research and professional opportunities, not my personal details. Let's stick to the professional stuff! 😊";
              }

              if (userQuestion.includes("salary") || userQuestion.includes("money") || userQuestion.includes("compensation")) {
                return "That's a bit forward! I'm happy to discuss my research contributions and how I can add value to projects, but let's save the compensation talk for when we're actually working together. 😉";
              }

              // Default response
              return "Hello! I'm Zizhao Hu, a CS Ph.D. student at USC working on synthetic data generation and multi-agent systems. I'm here to discuss potential collaborations, research opportunities, or technical consulting work. I have experience in AI/ML, computer vision, and distributed systems, with a focus on developing self-improving AI agents. How can I help you today?";
      
    } catch (error) {
      console.error("Error generating mock response:", error);
      throw error;
    }
  }

  async reset(): Promise<void> {
    // Mock reset - no actual state to reset
  }

  isReady(): boolean {
    return this.isInitialized;
  }

  isLoading(): boolean {
    return this.isInitializing;
  }
}

// Export a singleton instance
export const mockLLMService = new MockLLMService(); 