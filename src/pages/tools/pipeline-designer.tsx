import { Header } from '@/components/custom/header';
import { useEffect, useRef, useCallback, useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { jsPDF } from 'jspdf';

/* ─── Brand Palette ─── */
const brand = {
    dark: '#141413',
    light: '#faf9f5',
    midGray: '#b0aea5',
    lightGray: '#e8e6dc',
    orange: '#d97757',
    blue: '#6a9bcc',
    green: '#788c5d',
    clay: '#c2968a',
};

const colorSchemes: Record<string, { head: string; body: string; border: string; textTitle: string; textBody: string }> = {
    neutral: { head: brand.midGray, body: brand.light, border: brand.midGray, textTitle: brand.light, textBody: brand.dark },
    orange: { head: brand.orange, body: brand.light, border: brand.orange, textTitle: brand.light, textBody: brand.dark },
    green: { head: brand.green, body: brand.light, border: brand.green, textTitle: brand.light, textBody: brand.dark },
    blue: { head: brand.blue, body: brand.light, border: brand.blue, textTitle: brand.light, textBody: brand.dark },
    dark: { head: brand.dark, body: brand.light, border: brand.dark, textTitle: brand.light, textBody: brand.dark },
    clay: { head: brand.clay, body: brand.light, border: brand.clay, textTitle: brand.light, textBody: brand.dark },
};

/* ─── Types ─── */
interface PNode {
    id: string;
    x: number; y: number; w: number; h: number;
    type: 'title' | 'colLabel' | 'headerNode' | 'minimal' | 'plain' | 'container';
    variant?: string;
    title?: string;
    content?: string;
    stacked?: boolean;
    size?: number;
}

interface PConnection {
    from: string | string[];
    to: string | string[];
    type: 'direct' | 'split' | 'merge';
    dash?: number[];
    orientation?: string;
}

const GRID = 10;
const snap = (v: number) => Math.round(v / GRID) * GRID;

/* ─── Default Layout ─── */
function createDefaultNodes(): PNode[] {
    return [
        { id: 'label_a', x: 40, y: 20, w: 180, h: 20, type: 'title', title: 'Approach A: Inference (Explicit)' },
        { id: 'a_q1', x: 20, y: 120, w: 100, h: 45, type: 'headerNode', variant: 'neutral', title: 'User Prompt', content: '"Integral of x^2"' },
        { id: 'a_r1', x: 140, y: 110, w: 40, h: 65, type: 'minimal', variant: 'orange', title: 'Router' },
        { id: 'col_sys_a', x: 220, y: 30, w: 80, h: 15, type: 'colLabel', title: 'SYSTEM PROMPT' },
        { id: 'a_p1', x: 220, y: 50, w: 80, h: 40, type: 'headerNode', variant: 'neutral', title: 'Scientist', content: 'System Prompt' },
        { id: 'a_p2', x: 220, y: 100, w: 80, h: 40, type: 'headerNode', variant: 'neutral', title: 'Counselor', content: 'System Prompt' },
        { id: 'a_p3', x: 220, y: 150, w: 80, h: 40, type: 'headerNode', variant: 'neutral', title: 'Default', content: 'System Prompt' },
        { id: 'a_q2', x: 320, y: 100, w: 100, h: 45, type: 'headerNode', variant: 'neutral', title: 'User Prompt', content: '"Integral of x^2"' },
        { id: 'a_llm', x: 440, y: 90, w: 40, h: 65, type: 'minimal', variant: 'neutral', title: 'Base LLM' },
        { id: 'col_ans_a', x: 500, y: 30, w: 80, h: 15, type: 'colLabel', title: 'ANSWER' },
        { id: 'a_out1', x: 500, y: 50, w: 80, h: 35, type: 'plain', variant: 'green', content: '"x^3/3 + C"' },
        { id: 'a_out2', x: 500, y: 100, w: 80, h: 35, type: 'plain', variant: 'neutral', content: '"Helpful msg"' },
        { id: 'a_out3', x: 500, y: 150, w: 80, h: 35, type: 'plain', variant: 'neutral', content: '"Assistant reply"' },
        { id: 'label_b', x: 650, y: 20, w: 220, h: 20, type: 'title', title: 'Approach B: Inference (Distilled)' },
        { id: 'bi_q', x: 650, y: 100, w: 100, h: 45, type: 'headerNode', variant: 'neutral', title: 'User Prompt', content: '"Integral of x^2"' },
        { id: 'bi_llm', x: 790, y: 95, w: 50, h: 65, type: 'minimal', variant: 'orange', title: 'PESD LLM' },
        { id: 'bi_ans', x: 880, y: 105, w: 100, h: 45, type: 'plain', variant: 'green', content: '"x^3/3 + C"' },
        { id: 'bt_container', x: 20, y: 280, w: 1060, h: 420, type: 'container', title: 'Approach B: Training (PESD Framework)' },
        { id: 'label_bt1', x: 40, y: 320, w: 220, h: 15, type: 'title', title: '1. Synthetic Query Generation', size: 9 },
        { id: 'label_bt2', x: 380, y: 320, w: 220, h: 15, type: 'title', title: '2. Synthetic Answer Generation', size: 9 },
        { id: 'label_bt3', x: 740, y: 320, w: 220, h: 15, type: 'title', title: '3. Data Pairs Distillation', size: 9 },
        { id: 'col_sys_b1', x: 160, y: 345, w: 80, h: 15, type: 'colLabel', title: 'SYSTEM PROMPT' },
        { id: 'col_sq_b1', x: 320, y: 345, w: 80, h: 15, type: 'colLabel', title: 'SYNTHETIC QUERY' },
        { id: 'bt1_q', x: 40, y: 400, w: 100, h: 45, type: 'headerNode', variant: 'neutral', title: 'User Prompt', content: '"User Query"' },
        { id: 'bt1_sys1', x: 160, y: 370, w: 80, h: 35, type: 'headerNode', variant: 'neutral', title: 'Scientist', content: 'System Prompt' },
        { id: 'bt1_sys2', x: 160, y: 410, w: 80, h: 35, type: 'headerNode', variant: 'neutral', title: 'Counselor', content: 'System Prompt' },
        { id: 'bt1_sys3', x: 160, y: 450, w: 80, h: 35, type: 'headerNode', variant: 'neutral', title: 'Default', content: 'System Prompt' },
        { id: 'bt1_llm', x: 260, y: 395, w: 40, h: 60, type: 'minimal', variant: 'neutral', title: 'Base LLM' },
        { id: 'bt1_sq1', x: 320, y: 370, w: 80, h: 25, type: 'plain', variant: 'blue', content: '"Synth Q1"', stacked: true },
        { id: 'bt1_sq2', x: 320, y: 405, w: 80, h: 25, type: 'plain', variant: 'blue', content: '"Synth Q2"', stacked: true },
        { id: 'bt1_sq3', x: 320, y: 440, w: 80, h: 25, type: 'plain', variant: 'blue', content: '"Synth Q3"', stacked: true },
        { id: 'col_sys_b2', x: 420, y: 345, w: 80, h: 15, type: 'colLabel', title: 'SYSTEM PROMPT' },
        { id: 'col_sq_b2', x: 520, y: 345, w: 80, h: 15, type: 'colLabel', title: 'SYNTHETIC QUERY' },
        { id: 'col_sa_b2', x: 680, y: 345, w: 85, h: 15, type: 'colLabel', title: 'SYNTHETIC ANSWER' },
        { id: 'bt2_sys1', x: 420, y: 370, w: 80, h: 35, type: 'headerNode', variant: 'neutral', title: 'Scientist', content: 'System Prompt' },
        { id: 'bt2_sys2', x: 420, y: 410, w: 80, h: 35, type: 'headerNode', variant: 'neutral', title: 'Counselor', content: 'System Prompt' },
        { id: 'bt2_sys3', x: 420, y: 450, w: 80, h: 35, type: 'headerNode', variant: 'neutral', title: 'Default', content: 'System Prompt' },
        { id: 'bt2_sq1', x: 520, y: 370, w: 80, h: 25, type: 'plain', variant: 'blue', content: '"Synth Q1"', stacked: true },
        { id: 'bt2_sq2', x: 520, y: 405, w: 80, h: 25, type: 'plain', variant: 'blue', content: '"Synth Q2"', stacked: true },
        { id: 'bt2_sq3', x: 520, y: 440, w: 80, h: 25, type: 'plain', variant: 'blue', content: '"Synth Q3"', stacked: true },
        { id: 'bt2_llm', x: 620, y: 395, w: 40, h: 60, type: 'minimal', variant: 'neutral', title: 'Base LLM' },
        { id: 'bt2_sa1', x: 680, y: 370, w: 85, h: 30, type: 'plain', variant: 'blue', content: '"Synth Ans 1"', stacked: true },
        { id: 'bt2_sa2', x: 680, y: 405, w: 85, h: 30, type: 'plain', variant: 'blue', content: '"Synth Ans 2"', stacked: true },
        { id: 'bt2_sa3', x: 680, y: 440, w: 85, h: 30, type: 'plain', variant: 'blue', content: '"Synth Ans 3"', stacked: true },
        { id: 'col_sq_b3', x: 790, y: 345, w: 80, h: 15, type: 'colLabel', title: 'SYNTHETIC QUERY' },
        { id: 'col_sa_b3', x: 950, y: 345, w: 80, h: 15, type: 'colLabel', title: 'SYNTHETIC ANSWER' },
        { id: 'bt3_sq1', x: 790, y: 370, w: 80, h: 28, type: 'plain', variant: 'blue', content: '"Synth Q1"', stacked: true },
        { id: 'bt3_sq2', x: 790, y: 405, w: 80, h: 28, type: 'plain', variant: 'blue', content: '"Synth Q2"', stacked: true },
        { id: 'bt3_sq3', x: 790, y: 440, w: 80, h: 28, type: 'plain', variant: 'blue', content: '"Synth Q3"', stacked: true },
        { id: 'bt3_llm', x: 890, y: 395, w: 40, h: 60, type: 'minimal', variant: 'orange', title: 'Base LLM' },
        { id: 'bt3_sa1', x: 950, y: 370, w: 80, h: 28, type: 'plain', variant: 'blue', content: '"Synth Ans 1"', stacked: true },
        { id: 'bt3_sa2', x: 950, y: 405, w: 80, h: 28, type: 'plain', variant: 'blue', content: '"Synth Ans 2"', stacked: true },
        { id: 'bt3_sa3', x: 950, y: 440, w: 80, h: 28, type: 'plain', variant: 'blue', content: '"Synth Ans 3"', stacked: true },
    ];
}

function createDefaultConnections(): PConnection[] {
    return [
        { from: 'a_r1', to: ['a_p1', 'a_p2', 'a_p3'], type: 'split' },
        { from: ['a_p1', 'a_p2', 'a_p3'], to: 'a_q2', type: 'merge' },
        { from: 'a_q2', to: 'a_llm', type: 'direct' },
        { from: 'a_llm', to: ['a_out1', 'a_out2', 'a_out3'], type: 'split' },
        { from: 'bi_q', to: 'bi_llm', type: 'direct' },
        { from: 'bi_llm', to: 'bi_ans', type: 'direct' },
        { from: 'bi_llm', to: 'bt_container', type: 'direct', dash: [5, 5], orientation: 'vertical' },
        { from: 'bt1_q', to: ['bt1_sys1', 'bt1_sys2', 'bt1_sys3'], type: 'split' },
        { from: ['bt1_sys1', 'bt1_sys2', 'bt1_sys3'], to: 'bt1_llm', type: 'merge' },
        { from: 'bt1_llm', to: ['bt1_sq1', 'bt1_sq2', 'bt1_sq3'], type: 'split' },
        { from: 'bt2_sys1', to: 'bt2_sq1', type: 'direct' },
        { from: 'bt2_sys2', to: 'bt2_sq2', type: 'direct' },
        { from: 'bt2_sys3', to: 'bt2_sq3', type: 'direct' },
        { from: ['bt2_sq1', 'bt2_sq2', 'bt2_sq3'], to: 'bt2_llm', type: 'merge' },
        { from: 'bt2_llm', to: ['bt2_sa1', 'bt2_sa2', 'bt2_sa3'], type: 'split' },
        { from: ['bt3_sq1', 'bt3_sq2', 'bt3_sq3'], to: 'bt3_llm', type: 'merge' },
        { from: 'bt3_llm', to: ['bt3_sa1', 'bt3_sa2', 'bt3_sa3'], type: 'split' },
    ];
}

/* ─── Drawing helpers ─── */
function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number, fill: boolean, fs?: string | null, ss?: string | null) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
    if (fill && fs) { ctx.fillStyle = fs; ctx.fill(); }
    if (ss && ss !== 'transparent') { ctx.strokeStyle = ss; ctx.lineWidth = 1; ctx.stroke(); }
}

function wrapText(ctx: CanvasRenderingContext2D, t: string, x: number, y: number, mw: number, lh: number, fs: number, align: CanvasTextAlign = 'left') {
    if (!t) return;
    ctx.font = `400 ${fs}px 'Lora'`;
    ctx.textAlign = align;
    const words = t.split(' ');
    const lines: string[] = [];
    let currentLine = words[0];
    for (let i = 1; i < words.length; i++) {
        const word = words[i];
        if (ctx.measureText(currentLine + ' ' + word).width < mw) {
            currentLine += ' ' + word;
        } else {
            lines.push(currentLine);
            currentLine = word;
        }
    }
    lines.push(currentLine);
    const totalH = lines.length * lh;
    let startY = y - totalH / 2 + lh / 2 + 2;
    lines.forEach((line) => { ctx.fillText(line, x, startY); startY += lh; });
}

function drawArrow(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number) {
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    const a = Math.atan2(y2 - y1, x2 - x1);
    ctx.save(); ctx.translate(x2, y2); ctx.rotate(a);
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-5, -3); ctx.lineTo(-5, 3);
    ctx.fillStyle = brand.midGray; ctx.fill(); ctx.restore();
}

function drawLine(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number) {
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
}

/* ─── Component ─── */
export const PipelineDesigner = () => {
    const navigate = useNavigate();
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const editorRef = useRef<HTMLTextAreaElement>(null);
    const menuRef = useRef<HTMLDivElement>(null);

    const nodesRef = useRef<PNode[]>(createDefaultNodes());
    const connsRef = useRef<PConnection[]>(createDefaultConnections());

    const draggedRef = useRef<PNode | null>(null);
    const resizedRef = useRef<PNode | null>(null);
    const editingRef = useRef<{ node: PNode; field: string } | null>(null);
    const selBoxRef = useRef<{ x: number; y: number; startX: number; startY: number; w: number; h: number } | null>(null);
    const selectedRef = useRef<Set<string>>(new Set());
    const connectRef = useRef<{ active: boolean; sourceId: string | null }>({ active: false, sourceId: null });
    const mouseRef = useRef({ x: 0, y: 0 });
    const offsetRef = useRef({ x: 0, y: 0 });
    const clipboardRef = useRef<{ nodes: PNode[]; connections: PConnection[] }>({ nodes: [], connections: [] });

    const [connectMode, setConnectMode] = useState(false);
    const [toast, setToast] = useState<string | null>(null);

    const W = 1150;
    const H = 750;

    const showToast = useCallback((msg: string) => {
        setToast(msg);
        setTimeout(() => setToast(null), 2000);
    }, []);

    const getNode = useCallback((id: string) => nodesRef.current.find((n) => n.id === id), []);

    /* ── Rendering ── */
    const render = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const dpr = window.devicePixelRatio || 2;
        canvas.width = W * dpr;
        canvas.height = H * dpr;
        canvas.style.width = W + 'px';
        canvas.style.height = H + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, W, H);

        const nodes = nodesRef.current;
        const connections = connsRef.current;

        // Render nodes
        nodes.forEach((node) => {
            ctx.save();
            const isSelected = selectedRef.current.has(node.id);
            const isSource = connectRef.current.sourceId === node.id;
            const color = colorSchemes[node.variant || 'neutral'] || colorSchemes.neutral;
            ctx.shadowBlur = 0;

            if (node.type === 'container') {
                ctx.setLineDash([10, 5]); ctx.strokeStyle = brand.midGray; ctx.lineWidth = 2;
                roundRect(ctx, node.x, node.y, node.w, node.h, 12, false, null, brand.midGray);
                ctx.setLineDash([]); ctx.fillStyle = brand.midGray; ctx.font = "600 12px 'Poppins'";
                ctx.fillText(node.title || '', node.x + 20, node.y + 25);
            } else if (node.type === 'title') {
                const sz = node.size || 11;
                ctx.fillStyle = brand.dark; ctx.font = `600 ${sz}px 'Poppins'`; ctx.textAlign = 'center';
                ctx.fillText(node.title || '', node.x + node.w / 2, node.y + node.h / 2 + 5);
            } else if (node.type === 'colLabel') {
                ctx.fillStyle = brand.midGray; ctx.font = "600 10px 'Poppins'"; ctx.textAlign = 'center';
                ctx.fillText(node.title || '', node.x + node.w / 2, node.y + node.h / 2 + 5);
            } else if (node.type === 'headerNode') {
                roundRect(ctx, node.x, node.y, node.w, node.h, 6, true, color.body, color.border);
                const hH = 16;
                ctx.fillStyle = color.head;
                roundRect(ctx, node.x, node.y, node.w, hH, 6, true, color.head, 'transparent');
                ctx.fillRect(node.x, node.y + hH - 3, node.w, 3);
                ctx.fillStyle = color.textTitle; ctx.font = "600 8.5px 'Poppins'"; ctx.textAlign = 'left';
                ctx.fillText(node.title || '', node.x + 6, node.y + 10);
                ctx.fillStyle = brand.dark; ctx.font = "400 8.5px 'Lora'";
                wrapText(ctx, node.content || '', node.x + 6, node.y + hH + 11, node.w - 12, 10, 8.5, 'left');
            } else if (node.type === 'minimal') {
                roundRect(ctx, node.x, node.y, node.w, node.h, 6, true, color.head, color.border);
                ctx.fillStyle = color.textTitle; ctx.font = "600 8.5px 'Poppins'"; ctx.textAlign = 'center';
                const txt = (node.title || '').split(' ');
                if (txt.length > 1) {
                    ctx.fillText(txt[0], node.x + node.w / 2, node.y + node.h / 2 - 2);
                    ctx.fillText(txt[1], node.x + node.w / 2, node.y + node.h / 2 + 8);
                } else {
                    ctx.fillText(node.title || '', node.x + node.w / 2, node.y + node.h / 2 + 3);
                }
            } else if (node.type === 'plain') {
                if (node.stacked) roundRect(ctx, node.x - 3, node.y + 3, node.w, node.h, 6, true, brand.light, color.border);
                roundRect(ctx, node.x, node.y, node.w, node.h, 6, true, brand.light, color.border);
                ctx.fillStyle = brand.dark; ctx.font = "400 8.5px 'Lora'";
                wrapText(ctx, node.content || '', node.x + node.w / 2, node.y + node.h / 2, node.w - 10, 10, 8.5, 'center');
            }

            if (isSelected || draggedRef.current === node || isSource) {
                ctx.strokeStyle = isSource ? brand.blue : brand.orange;
                ctx.lineWidth = 2; ctx.setLineDash([4, 2]);
                ctx.strokeRect(node.x - 2, node.y - 2, node.w + 4, node.h + 4);
            }
            ctx.restore();
        });

        // Render connections
        ctx.setLineDash([]); ctx.strokeStyle = brand.midGray; ctx.lineWidth = 1.2;
        connections.forEach((l) => {
            ctx.setLineDash(l.dash || []);
            const sources = Array.isArray(l.from) ? l.from.map((id) => getNode(id)) : [getNode(l.from)];
            const targets = Array.isArray(l.to) ? l.to.map((id) => getNode(id)) : [getNode(l.to)];
            if (sources.some((s) => !s) || targets.some((t) => !t)) return;
            const s = sources[0]!;
            const t = targets[0]!;
            if (l.orientation === 'vertical') {
                drawArrow(ctx, s.x + s.w / 2, s.y + s.h, s.x + s.w / 2, t.y);
            } else if (l.type === 'split') {
                const midX = s.x + s.w + (targets[0]!.x - (s.x + s.w)) / 2;
                drawLine(ctx, s.x + s.w, s.y + s.h / 2, midX, s.y + s.h / 2);
                drawLine(ctx, midX, Math.min(...targets.map((n) => n!.y + n!.h / 2)), midX, Math.max(...targets.map((n) => n!.y + n!.h / 2)));
                targets.forEach((node) => drawArrow(ctx, midX, node!.y + node!.h / 2, node!.x, node!.y + node!.h / 2));
            } else if (l.type === 'merge') {
                const maxX = Math.max(...sources.map((n) => n!.x + n!.w));
                const midX = t.x - (t.x - maxX) / 2;
                drawLine(ctx, midX, Math.min(...sources.map((n) => n!.y + n!.h / 2)), midX, Math.max(...sources.map((n) => n!.y + n!.h / 2)));
                sources.forEach((node) => drawLine(ctx, node!.x + node!.w, node!.y + node!.h / 2, midX, node!.y + node!.h / 2));
                drawArrow(ctx, midX, t.y + t.h / 2, t.x, t.y + t.h / 2);
            } else {
                drawArrow(ctx, s.x + s.w, s.y + s.h / 2, t.x, t.y + t.h / 2);
            }
        });

        // Manual arrow from a_q1 -> a_r1
        const aq1 = getNode('a_q1');
        const ar1 = getNode('a_r1');
        if (aq1 && ar1) drawArrow(ctx, aq1.x + aq1.w, aq1.y + aq1.h / 2, ar1.x, ar1.y + ar1.h / 2);

        // Connection preview
        if (connectRef.current.active && connectRef.current.sourceId) {
            const src = getNode(connectRef.current.sourceId);
            if (src) {
                ctx.save(); ctx.strokeStyle = brand.blue; ctx.lineWidth = 1.5; ctx.setLineDash([5, 5]);
                ctx.beginPath(); ctx.moveTo(src.x + src.w, src.y + src.h / 2);
                ctx.lineTo(mouseRef.current.x, mouseRef.current.y); ctx.stroke(); ctx.restore();
            }
        }

        // Selection box
        const sb = selBoxRef.current;
        if (sb) {
            ctx.save(); ctx.strokeStyle = brand.blue; ctx.setLineDash([5, 5]); ctx.lineWidth = 1;
            ctx.fillStyle = 'rgba(106, 155, 204, 0.1)';
            ctx.fillRect(sb.x, sb.y, sb.w, sb.h);
            ctx.strokeRect(sb.x, sb.y, sb.w, sb.h);
            ctx.restore();
        }
    }, [getNode]);

    /* ── Animation loop ── */
    useEffect(() => {
        let raf: number;
        const loop = () => { render(); raf = requestAnimationFrame(loop); };
        raf = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(raf);
    }, [render]);

    /* ── Mouse position helper ── */
    const getMouse = useCallback((e: MouseEvent | React.MouseEvent) => {
        const canvas = canvasRef.current;
        if (!canvas) return { x: 0, y: 0 };
        const rect = canvas.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }, []);

    /* ── Event handlers ── */
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const onMouseDown = (e: MouseEvent) => {
            const mouse = getMouse(e);
            closeEditor();
            if (menuRef.current) menuRef.current.style.display = 'none';
            if (e.button === 2) return;

            let clicked: PNode | null = null;
            for (let i = nodesRef.current.length - 1; i >= 0; i--) {
                const n = nodesRef.current[i];
                if (mouse.x > n.x && mouse.x < n.x + n.w && mouse.y > n.y && mouse.y < n.y + n.h) {
                    clicked = n; break;
                }
            }

            if (connectRef.current.active && clicked) {
                if (!connectRef.current.sourceId) {
                    connectRef.current.sourceId = clicked.id;
                } else if (connectRef.current.sourceId !== clicked.id) {
                    connsRef.current.push({ from: connectRef.current.sourceId, to: clicked.id, type: 'direct' });
                    connectRef.current = { active: false, sourceId: null };
                    setConnectMode(false);
                }
                return;
            }

            if (clicked) {
                if (['headerNode', 'minimal', 'plain', 'container'].includes(clicked.type) &&
                    mouse.x > clicked.x + clicked.w - 15 && mouse.y > clicked.y + clicked.h - 15) {
                    resizedRef.current = clicked;
                    return;
                }
                draggedRef.current = clicked;
                offsetRef.current = { x: mouse.x, y: mouse.y };
                if (!selectedRef.current.has(clicked.id)) {
                    if (!e.shiftKey) selectedRef.current.clear();
                    selectedRef.current.add(clicked.id);
                }
            } else {
                selectedRef.current.clear();
                selBoxRef.current = { x: mouse.x, y: mouse.y, startX: mouse.x, startY: mouse.y, w: 0, h: 0 };
            }
        };

        const onMouseMove = (e: MouseEvent) => {
            const m = getMouse(e);
            mouseRef.current = m;
            if (resizedRef.current) {
                resizedRef.current.w = snap(Math.max(40, m.x - resizedRef.current.x));
                resizedRef.current.h = snap(Math.max(20, m.y - resizedRef.current.y));
            } else if (draggedRef.current) {
                const dx = snap(m.x - offsetRef.current.x);
                const dy = snap(m.y - offsetRef.current.y);
                if (dx !== 0 || dy !== 0) {
                    nodesRef.current.forEach((n) => {
                        if (selectedRef.current.has(n.id)) { n.x += dx; n.y += dy; }
                    });
                    offsetRef.current.x += dx;
                    offsetRef.current.y += dy;
                }
            } else if (selBoxRef.current) {
                const sb = selBoxRef.current;
                sb.w = m.x - sb.startX;
                sb.h = m.y - sb.startY;
                sb.x = sb.w < 0 ? m.x : sb.startX;
                sb.y = sb.h < 0 ? m.y : sb.startY;
                selectedRef.current.clear();
                nodesRef.current.forEach((n) => {
                    if (n.x > sb.x && n.x + n.w < sb.x + Math.abs(sb.w) &&
                        n.y > sb.y && n.y + n.h < sb.y + Math.abs(sb.h)) {
                        selectedRef.current.add(n.id);
                    }
                });
            }
        };

        const onMouseUp = () => {
            draggedRef.current = null;
            resizedRef.current = null;
            selBoxRef.current = null;
        };

        const onDblClick = (e: MouseEvent) => {
            const mouse = getMouse(e);
            for (let i = nodesRef.current.length - 1; i >= 0; i--) {
                const n = nodesRef.current[i];
                if (mouse.x > n.x && mouse.x < n.x + n.w && mouse.y > n.y && mouse.y < n.y + n.h) {
                    const field = (n.type === 'headerNode' && mouse.y < n.y + 16) ? 'title' : 'content';
                    openEditor(n, field);
                    return;
                }
            }
        };

        const onContextMenu = (e: MouseEvent) => {
            e.preventDefault();
            const mouse = getMouse(e);
            let target: PNode | null = null;
            for (let i = nodesRef.current.length - 1; i >= 0; i--) {
                const n = nodesRef.current[i];
                if (mouse.x > n.x && mouse.x < n.x + n.w && mouse.y > n.y && mouse.y < n.y + n.h) {
                    target = n; break;
                }
            }
            if (target && menuRef.current && containerRef.current) {
                if (!selectedRef.current.has(target.id)) {
                    selectedRef.current.clear();
                    selectedRef.current.add(target.id);
                }
                const cr = containerRef.current.getBoundingClientRect();
                menuRef.current.style.display = 'block';
                menuRef.current.style.left = (e.clientX - cr.left) + 'px';
                menuRef.current.style.top = (e.clientY - cr.top) + 'px';
            }
        };

        const onKeyDown = (e: KeyboardEvent) => {
            if (editingRef.current) return;
            const cmd = e.metaKey || e.ctrlKey;
            if (cmd && e.key.toLowerCase() === 'c') {
                const selNodes = nodesRef.current.filter((n) => selectedRef.current.has(n.id));
                const selConns = connsRef.current.filter((conn) => {
                    const f = Array.isArray(conn.from) ? conn.from : [conn.from];
                    const t = Array.isArray(conn.to) ? conn.to : [conn.to];
                    return f.every((id) => selectedRef.current.has(id)) && t.every((id) => selectedRef.current.has(id));
                });
                clipboardRef.current = { nodes: selNodes.map((n) => ({ ...n })), connections: selConns.map((c) => ({ ...c })) };
                showToast('Copied');
            }
            if (cmd && e.key.toLowerCase() === 'v') {
                if (clipboardRef.current.nodes.length === 0) return;
                const idMap: Record<string, string> = {};
                const newNodes: PNode[] = [];
                clipboardRef.current.nodes.forEach((n) => {
                    const newId = 'pasted_' + Math.random().toString(36).substr(2, 9);
                    idMap[n.id] = newId;
                    const newNode = { ...n, id: newId, x: snap(n.x + 30), y: snap(n.y + 30) };
                    nodesRef.current.push(newNode);
                    newNodes.push(newNode);
                });
                clipboardRef.current.connections.forEach((c) => {
                    const newConn = { ...c };
                    newConn.from = Array.isArray(c.from) ? c.from.map((id) => idMap[id]) : idMap[c.from];
                    newConn.to = Array.isArray(c.to) ? c.to.map((id) => idMap[id]) : idMap[c.to];
                    connsRef.current.push(newConn);
                });
                selectedRef.current.clear();
                newNodes.forEach((n) => selectedRef.current.add(n.id));
                showToast('Pasted');
            }
            if (cmd && (e.key === 'Delete' || e.key === 'Backspace')) {
                nodesRef.current = nodesRef.current.filter((n) => !selectedRef.current.has(n.id));
                connsRef.current = connsRef.current.filter((conn) => {
                    const f = Array.isArray(conn.from) ? conn.from : [conn.from];
                    const t = Array.isArray(conn.to) ? conn.to : [conn.to];
                    return [...f, ...t].every((id) => nodesRef.current.some((n) => n.id === id));
                });
                selectedRef.current.clear();
                showToast('Deleted');
            }
        };

        canvas.addEventListener('mousedown', onMouseDown);
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
        canvas.addEventListener('dblclick', onDblClick);
        canvas.addEventListener('contextmenu', onContextMenu);
        window.addEventListener('keydown', onKeyDown);

        return () => {
            canvas.removeEventListener('mousedown', onMouseDown);
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);
            canvas.removeEventListener('dblclick', onDblClick);
            canvas.removeEventListener('contextmenu', onContextMenu);
            window.removeEventListener('keydown', onKeyDown);
        };
    }, [getMouse, showToast, getNode]);

    /* ── Editor ── */
    const openEditor = (node: PNode, field: string) => {
        editingRef.current = { node, field };
        const ed = editorRef.current;
        if (!ed) return;
        ed.style.display = 'block';
        ed.style.left = node.x + 'px';
        ed.style.top = node.y + 'px';
        ed.style.width = node.w + 'px';
        ed.style.height = node.h + 'px';
        ed.value = (field === 'title' || ['title', 'colLabel', 'minimal', 'container'].includes(node.type))
            ? (node.title || '')
            : (node.content || '');
        ed.focus();
    };

    const closeEditor = () => {
        if (!editingRef.current) return;
        const { node, field } = editingRef.current;
        const ed = editorRef.current;
        if (!ed) return;
        if (field === 'title' || ['title', 'colLabel', 'minimal', 'container'].includes(node.type)) {
            node.title = ed.value;
        } else {
            node.content = ed.value;
        }
        ed.style.display = 'none';
        editingRef.current = null;
    };

    const setVariant = (v: string) => {
        nodesRef.current.forEach((n) => { if (selectedRef.current.has(n.id)) n.variant = v; });
        if (menuRef.current) menuRef.current.style.display = 'none';
    };

    const resetLayout = () => {
        nodesRef.current = createDefaultNodes();
        connsRef.current = createDefaultConnections();
        selectedRef.current.clear();
        showToast('Layout Reset');
    };

    const exportPNG = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const link = document.createElement('a');
        link.download = 'ai-pipeline.png';
        link.href = canvas.toDataURL();
        link.click();
    };

    const exportPDF = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const imgData = canvas.toDataURL('image/png');
        const pdf = new jsPDF({ orientation: 'landscape', unit: 'px', format: [W, H] });
        pdf.addImage(imgData, 'PNG', 0, 0, W, H);
        pdf.save('ai-pipeline.pdf');
        showToast('Exported PDF');
    };

    const toggleConnect = () => {
        const next = !connectRef.current.active;
        connectRef.current = { active: next, sourceId: null };
        setConnectMode(next);
    };

    const colorSwatches = [
        { label: 'Neutral (Grey)', variant: 'neutral', color: brand.midGray },
        { label: 'Blue', variant: 'blue', color: brand.blue },
        { label: 'Orange', variant: 'orange', color: brand.orange },
        { label: 'Green', variant: 'green', color: brand.green },
        { label: 'Clay', variant: 'clay', color: brand.clay },
        { label: 'Dark', variant: 'dark', color: brand.dark },
    ];

    return (
        <div className="flex flex-col min-h-dvh bg-background">
            <Header />
            <div className="flex-1 overflow-y-auto pb-24">
                <div className="max-w-[1200px] mx-auto px-4 py-6">
                    {/* Back */}
                    <button
                        onClick={() => navigate('/tools')}
                        className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground mb-4 transition-colors font-heading"
                    >
                        <ArrowLeft className="w-3.5 h-3.5" />
                        Back to Tools
                    </button>

                    {/* Title */}
                    <div className="mb-4">
                        <h1 className="text-xl font-bold font-heading text-foreground">AI Pipeline Designer</h1>
                        <p className="text-xs text-muted-foreground mt-1">
                            Drag nodes to reposition • Double-click to edit text • Right-click to change color • Ctrl+C/V to copy/paste
                        </p>
                    </div>

                    {/* Canvas */}
                    <div ref={containerRef} className="relative bg-white rounded-xl border border-border overflow-hidden shadow-sm">
                        <canvas ref={canvasRef} style={{ display: 'block', touchAction: 'none' }} />
                        <textarea
                            ref={editorRef}
                            className="absolute hidden bg-white border border-brand-orange px-1 py-0.5 text-[10px] z-[1000] outline-none shadow-md resize-none font-heading"
                            onBlur={closeEditor}
                            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); closeEditor(); } }}
                        />
                        <div
                            ref={menuRef}
                            className="absolute hidden bg-card border border-border rounded-lg shadow-xl py-1.5 z-[2000] min-w-[140px]"
                        >
                            {colorSwatches.map((s) => (
                                <button
                                    key={s.variant}
                                    onClick={() => setVariant(s.variant)}
                                    className="w-full flex items-center gap-2.5 px-3 py-1.5 text-xs font-heading hover:bg-muted transition-colors"
                                >
                                    <span className="w-3 h-3 rounded-sm border border-black/10" style={{ background: s.color }} />
                                    {s.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Controls */}
                    <div className="flex items-center gap-3 mt-4">
                        <button
                            onClick={toggleConnect}
                            className={`px-4 py-2 rounded-lg text-xs font-semibold font-heading transition-all ${connectMode
                                ? 'bg-brand-blue text-white'
                                : 'bg-foreground text-background hover:opacity-90'
                                }`}
                        >
                            {connectMode ? 'Connecting...' : 'Add Connection'}
                        </button>
                        <button
                            onClick={resetLayout}
                            className="px-4 py-2 rounded-lg bg-foreground text-background text-xs font-semibold font-heading hover:opacity-90 transition-all"
                        >
                            Reset Layout
                        </button>
                        <button
                            onClick={exportPNG}
                            className="px-4 py-2 rounded-lg bg-brand-orange text-white text-xs font-semibold font-heading hover:opacity-90 transition-all"
                        >
                            Export PNG
                        </button>
                        <button
                            onClick={exportPDF}
                            className="px-4 py-2 rounded-lg bg-brand-orange text-white text-xs font-semibold font-heading hover:opacity-90 transition-all"
                        >
                            Export PDF
                        </button>
                    </div>
                </div>
            </div>

            {/* Toast */}
            {toast && (
                <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-foreground text-background px-5 py-2.5 rounded-md text-xs font-heading z-[3000] pointer-events-none shadow-lg animate-in fade-in slide-in-from-bottom-2 duration-200">
                    {toast}
                </div>
            )}
        </div>
    );
};
