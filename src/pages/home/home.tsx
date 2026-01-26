import { Header } from '@/components/custom/header';
import { Overview } from '@/components/custom/overview';
import { ConversationStarters } from '@/components/custom/conversation-starters';

export const Home = () => {
  // Handle starter click - now handled by floating chat
  const handleStarterClick = (question: string) => {
    // Dispatch a custom event that the floating chat can listen to
    const event = new CustomEvent('chat-question', { detail: question });
    window.dispatchEvent(event);
  };

  return (
    <div className="flex flex-col min-h-dvh bg-background">
      {/* Header */}
      <Header />
      
      {/* Overview */}
      <div className="flex-shrink-0">
        <Overview />
      </div>
      
      {/* Content */}
      <div className="flex-1 overflow-y-auto pb-24">
        <div className="px-4 md:px-6 py-8">
          <ConversationStarters onStarterClick={handleStarterClick} />
        </div>
      </div>
    </div>
  );
};
