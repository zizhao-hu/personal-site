import { useEffect, useRef, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { initStarshipScene, SimState, destroyScene } from './starship-engine';

type Phase = 'prelaunch' | 'ignition' | 'liftoff' | 'maxq' | 'meco' | 'separation' | 'ses' | 'coast' | 'lunar-approach' | 'landing-burn' | 'touchdown' | 'landed' | 'eva' | 'exploration' | 'complete';

const PHASE_LABELS: Record<Phase, string> = {
    prelaunch: 'PRE-LAUNCH',
    ignition: 'IGNITION SEQUENCE',
    liftoff: 'LIFTOFF',
    maxq: 'MAX-Q',
    meco: 'MECO',
    separation: 'STAGE SEPARATION',
    ses: 'SES-1 CONFIRMED',
    coast: 'COAST PHASE',
    'lunar-approach': 'LUNAR APPROACH',
    'landing-burn': 'LANDING BURN',
    touchdown: 'TOUCHDOWN',
    landed: 'SYSTEMS NOMINAL',
    eva: 'EVA IN PROGRESS',
    exploration: 'LUNAR EXPLORATION',
    complete: 'MISSION COMPLETE',
};

const DONE_PHASES = ['landed', 'eva', 'exploration', 'complete'];
const AFTER = (phase: Phase, milestone: Phase) => {
    const order: Phase[] = ['prelaunch', 'ignition', 'liftoff', 'maxq', 'meco', 'separation', 'ses', 'coast', 'lunar-approach', 'landing-burn', 'touchdown', 'landed', 'eva', 'exploration', 'complete'];
    return order.indexOf(phase) > order.indexOf(milestone);
};

export const StarshipSim = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const navigate = useNavigate();
    const [sim, setSim] = useState<SimState>({
        phase: 'prelaunch', altitude: 0, velocity: 0, downrange: 0,
        fuel: 100, thrust: 0, missionTime: -10, throttle: 0,
    });
    const [started, setStarted] = useState(false);

    useEffect(() => {
        if (!canvasRef.current) return;
        const cleanup = initStarshipScene(canvasRef.current, (s: SimState) => setSim({ ...s }));
        return () => { cleanup(); destroyScene(); };
    }, []);

    const handleLaunch = useCallback(() => {
        setStarted(true);
        window.dispatchEvent(new CustomEvent('starship-launch'));
    }, []);

    const fmt = (n: number, d = 1) => n.toFixed(d);
    const fmtTime = (t: number) => {
        const neg = t < 0; const abs = Math.abs(t);
        const m = Math.floor(abs / 60), s = Math.floor(abs % 60);
        return `${neg ? 'T-' : 'T+'}${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    };

    const phase = sim.phase as Phase;
    const isExploring = phase === 'exploration';

    return (
        <div className="relative w-full h-screen bg-black overflow-hidden select-none">
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />

            {/* Top Bar */}
            <div className="absolute top-0 left-0 right-0 z-10 pointer-events-none">
                <div className="flex items-center justify-between px-4 py-2" style={{ background: 'linear-gradient(180deg, rgba(0,0,0,0.8) 0%, transparent 100%)' }}>
                    <button onClick={() => navigate('/tools')}
                        className="pointer-events-auto flex items-center gap-1.5 text-white/60 hover:text-white text-xs transition-colors">
                        <ArrowLeft className="w-3 h-3" /> Back
                    </button>
                    <div className="flex items-center gap-3">
                        <span className="text-[10px] uppercase tracking-[0.3em] text-white/40 font-mono">SPACEX</span>
                        <div className="w-px h-3 bg-white/20" />
                        <span className="text-[10px] uppercase tracking-[0.2em] text-white/60 font-mono">STARSHIP LUNAR MISSION</span>
                    </div>
                    <div className="text-xs font-mono text-white/60">{fmtTime(sim.missionTime)}</div>
                </div>
            </div>

            {/* Phase Banner */}
            <div className="absolute top-12 left-1/2 -translate-x-1/2 z-10 pointer-events-none">
                <div className={`px-4 py-1 rounded text-xs font-mono uppercase tracking-[0.2em] transition-all duration-500 ${phase === 'prelaunch' ? 'bg-white/10 text-white/50' :
                        DONE_PHASES.includes(phase) || phase === 'touchdown' ? 'bg-green-500/20 text-green-400 border border-green-500/30' :
                            phase === 'eva' ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30' :
                                'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
                    }`}>
                    {PHASE_LABELS[phase]}
                </div>
            </div>

            {/* Left Telemetry Panel */}
            <div className="absolute left-4 bottom-4 z-10 pointer-events-none">
                <div className="bg-black/70 backdrop-blur-sm border border-white/10 rounded-lg p-3 w-52" style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace" }}>
                    <div className="text-[9px] uppercase tracking-[0.2em] text-white/30 mb-2">TELEMETRY</div>
                    <TelemetryRow label="ALTITUDE" value={`${fmt(sim.altitude)} km`} />
                    <TelemetryRow label="VELOCITY" value={`${fmt(sim.velocity)} km/s`} />
                    <TelemetryRow label="DOWNRANGE" value={`${fmt(sim.downrange)} km`} />
                    <div className="h-px bg-white/10 my-1.5" />
                    <TelemetryRow label="THRUST" value={`${fmt(sim.thrust, 0)}%`} accent />
                    <TelemetryRow label="FUEL" value={`${fmt(sim.fuel, 0)}%`} warn={sim.fuel < 20} />
                    <div className="mt-2">
                        <div className="text-[8px] text-white/30 mb-0.5">THROTTLE</div>
                        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                            <div className="h-full rounded-full transition-all duration-200" style={{
                                width: `${sim.throttle}%`,
                                background: sim.throttle > 80 ? '#f97316' : sim.throttle > 0 ? '#06b6d4' : '#333',
                            }} />
                        </div>
                    </div>
                </div>
            </div>

            {/* Right Mission Panel */}
            <div className="absolute right-4 bottom-4 z-10 pointer-events-none">
                <div className="bg-black/70 backdrop-blur-sm border border-white/10 rounded-lg p-3 w-48" style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace" }}>
                    <div className="text-[9px] uppercase tracking-[0.2em] text-white/30 mb-2">MISSION</div>
                    <div className="space-y-1">
                        <MissionStep label="LAUNCH" done={AFTER(phase, 'liftoff')} active={phase === 'ignition' || phase === 'liftoff'} />
                        <MissionStep label="MAX-Q" done={AFTER(phase, 'maxq')} active={phase === 'maxq'} />
                        <MissionStep label="MECO" done={AFTER(phase, 'meco')} active={phase === 'meco'} />
                        <MissionStep label="SEPARATION" done={AFTER(phase, 'separation')} active={phase === 'separation'} />
                        <MissionStep label="TRANS-LUNAR" done={AFTER(phase, 'coast')} active={phase === 'coast' || phase === 'ses'} />
                        <MissionStep label="LUNAR LANDING" done={AFTER(phase, 'touchdown')} active={phase === 'lunar-approach' || phase === 'landing-burn'} />
                        <MissionStep label="EVA" done={AFTER(phase, 'eva')} active={phase === 'eva'} />
                        <MissionStep label="EXPLORATION" done={phase === 'complete'} active={phase === 'exploration'} />
                    </div>
                </div>
            </div>

            {/* Rover Controls HUD */}
            {isExploring && (
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10">
                    <div className="bg-black/70 backdrop-blur-sm border border-white/10 rounded-lg px-4 py-2 text-center">
                        <div className="text-[9px] uppercase tracking-[0.2em] text-cyan-400 mb-1 font-mono">ROVER CONTROLS</div>
                        <div className="flex items-center gap-3 text-[10px] text-white/50 font-mono">
                            <span>W/↑ Forward</span>
                            <span className="text-white/20">|</span>
                            <span>S/↓ Back</span>
                            <span className="text-white/20">|</span>
                            <span>A/↓ Left</span>
                            <span className="text-white/20">|</span>
                            <span>D/→ Right</span>
                        </div>
                    </div>
                </div>
            )}

            {/* Launch Button */}
            {!started && (
                <div className="absolute bottom-24 left-1/2 -translate-x-1/2 z-20">
                    <button onClick={handleLaunch}
                        className="px-8 py-3 bg-gradient-to-b from-orange-500 to-red-600 text-white font-mono text-sm uppercase tracking-[0.3em] rounded-lg border border-orange-400/50 hover:from-orange-400 hover:to-red-500 transition-all duration-300 shadow-[0_0_30px_rgba(249,115,22,0.3)] hover:shadow-[0_0_50px_rgba(249,115,22,0.5)]">
                        INITIATE LAUNCH
                    </button>
                </div>
            )}
        </div>
    );
};

const TelemetryRow = ({ label, value, accent, warn }: { label: string; value: string; accent?: boolean; warn?: boolean }) => (
    <div className="flex justify-between items-baseline py-0.5">
        <span className="text-[8px] text-white/30">{label}</span>
        <span className={`text-[11px] tabular-nums ${warn ? 'text-red-400' : accent ? 'text-orange-400' : 'text-white/80'}`}>{value}</span>
    </div>
);

const MissionStep = ({ label, done, active }: { label: string; done: boolean; active: boolean }) => (
    <div className="flex items-center gap-2">
        <div className={`w-1.5 h-1.5 rounded-full ${done ? 'bg-green-400' : active ? 'bg-cyan-400 animate-pulse' : 'bg-white/20'}`} />
        <span className={`text-[9px] ${done ? 'text-green-400/80' : active ? 'text-cyan-400' : 'text-white/20'}`}>{label}</span>
    </div>
);
