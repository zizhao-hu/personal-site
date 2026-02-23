import type { Metadata } from 'next';
import { LlmVlmResearch } from '@/views/research/llm-vlm';

export const metadata: Metadata = {
  title: 'LLM & VLM Research',
  description: 'Research on Large Language Models and Vision-Language Models — multi-agent interaction, training paradigms, and model architecture.',
};

export default function LlmVlmResearchPage() {
  return <LlmVlmResearch />;
}
