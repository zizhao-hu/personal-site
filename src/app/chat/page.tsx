import type { Metadata } from 'next';
import { Suspense } from 'react';
import { Chat } from '@/views/chat/chat';

export const metadata: Metadata = {
  title: 'Chat with Zizhao',
  description: "Chat with an AI powered by Zizhao Hu's research and background knowledge.",
};

export default function ChatPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-dvh bg-background"><span className="text-muted-foreground text-sm">Loading chat...</span></div>}>
      <Chat />
    </Suspense>
  );
}
