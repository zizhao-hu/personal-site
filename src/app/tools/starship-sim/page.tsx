import type { Metadata } from 'next';
import { StarshipSim } from '@/views/tools/starship-sim';

export const metadata: Metadata = {
  title: 'Starship Simulator',
  description: 'Interactive SpaceX Starship flight simulator with realistic physics.',
};

export default function StarshipSimPage() {
  return <StarshipSim />;
}
