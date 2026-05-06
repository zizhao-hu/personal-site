'use client';

import { usePathname } from 'next/navigation';
import { FloatingChat } from './floating-chat';

const HIDDEN_PATHS = ['/chat', '/tools/maplestory'];

export function FloatingChatWrapper() {
    const pathname = usePathname();
    if (pathname && HIDDEN_PATHS.includes(pathname)) return null;
    return <FloatingChat />;
}
