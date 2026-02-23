import { Header } from '@/components/custom/header';
import { Overview } from '@/components/custom/overview';

export default function HomePage() {
    return (
        <div className="flex flex-col min-h-dvh bg-background">
            <Header />
            <div className="flex-1 overflow-y-auto">
                <Overview />
            </div>
        </div>
    );
}
