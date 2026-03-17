import type { Metadata } from 'next';
import { LlmVlmResearch } from '@/views/research/llm-vlm';

export const metadata: Metadata = {
  title: 'AI Memorization Research',
  description: 'Research on how LLMs memorize — in-parameter, in-context, and external retrieval. Forgetting, unlearning, KV management, inference optimization, and continual learning.',
};

export default function LlmVlmResearchPage() {
  return <LlmVlmResearch />;
}
