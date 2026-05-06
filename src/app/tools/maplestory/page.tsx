import type { Metadata } from 'next';
import { Maplestory } from '@/views/tools/maplestory';

export const metadata: Metadata = {
  title: 'MapleStory UI',
  description: 'Embedded subsite — a MapleStory-style web UI built by maiconlara.',
};

export default function MaplestoryPage() {
  return <Maplestory />;
}
