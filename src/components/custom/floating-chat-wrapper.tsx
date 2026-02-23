'use client';

import { usePathname } from 'next/navigation';
import { FloatingChat } from './floating-chat';

export function FloatingChatWrapper() {
    const pathname = usePathname();
    const showFloatingChat = pathname !== '/chat';

    if (!showFloatingChat) return null;
    return <FloatingChat />;
}
