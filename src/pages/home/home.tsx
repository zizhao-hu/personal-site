import { Header } from '@/components/custom/header';
import { Overview } from '@/components/custom/overview';

export const Home = () => {
  return (
    <div className="flex flex-col min-h-dvh bg-background">
      {/* Header */}
      <Header />

      {/* Overview */}
      <div className="flex-1 overflow-y-auto">
        <Overview />
      </div>
    </div>
  );
};
