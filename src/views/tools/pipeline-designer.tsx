'use client';

import { Header } from '@/components/custom/header';
import { useEffect, useRef, useCallback, useState } from 'react';
import { ArrowLeft, Save, FolderOpen, Trash2, Pencil, X, Check, Square, Circle, Diamond, Type, RectangleHorizontal, Hexagon, Plus } from 'lucide-react';
import { useRouter } from 'next/navigation';

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
type ShapeKind = 'rect' | 'rounded-rect' | 'circle' | 'diamond' | 'hexagon';

interface PNode {
    id: string;
    x: number; y: number; w: number; h: number;
    type: 'title' | 'colLabel' | 'headerNode' | 'minimal' | 'plain' | 'container' | 'shape' | 'textbox';
    variant?: string;
    title?: string;
    content?: string;
    stacked?: boolean;
    size?: number;
    shapeKind?: ShapeKind;
    fontSize?: number;
    fontWeight?: string;
    textAlign?: CanvasTextAlign;
    filled?: boolean;
}

interface PConnection {
    from: string | string[];
    to: string | string[];
    type: 'direct' | 'split' | 'merge';
    dash?: number[];
    orientation?: string;
}

interface SavedTemplate {
    id: string;
    name: string;
    nodes: PNode[];
    connections: PConnection[];
    createdAt: number;
    updatedAt: number;
}

const STORAGE_KEY = 'pipeline-designer-templates';

function loadTemplatesFromStorage(): SavedTemplate[] {
    if (typeof window === 'undefined') return [];
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch { return []; }
}

function saveTemplatesToStorage(templates: SavedTemplate[]) {
    if (typeof window === 'undefined') return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(templates));
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

function drawDiamond(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, fill: boolean, fs?: string | null, ss?: string | null) {
    const cx = x + w / 2, cy = y + h / 2;
    ctx.beginPath();
    ctx.moveTo(cx, y);
    ctx.lineTo(x + w, cy);
    ctx.lineTo(cx, y + h);
    ctx.lineTo(x, cy);
    ctx.closePath();
    if (fill && fs) { ctx.fillStyle = fs; ctx.fill(); }
    if (ss && ss !== 'transparent') { ctx.strokeStyle = ss; ctx.lineWidth = 1.5; ctx.stroke(); }
}

function drawHexagon(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, fill: boolean, fs?: string | null, ss?: string | null) {
    const cx = x + w / 2, cy = y + h / 2;
    const inset = w * 0.25;
    ctx.beginPath();
    ctx.moveTo(x + inset, y);
    ctx.lineTo(x + w - inset, y);
    ctx.lineTo(x + w, cy);
    ctx.lineTo(x + w - inset, y + h);
    ctx.lineTo(x + inset, y + h);
    ctx.lineTo(x, cy);
    ctx.closePath();
    if (fill && fs) { ctx.fillStyle = fs; ctx.fill(); }
    if (ss && ss !== 'transparent') { ctx.strokeStyle = ss; ctx.lineWidth = 1.5; ctx.stroke(); }
}

function drawEllipse(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, fill: boolean, fs?: string | null, ss?: string | null) {
    const cx = x + w / 2, cy = y + h / 2;
    ctx.beginPath();
    ctx.ellipse(cx, cy, w / 2, h / 2, 0, 0, Math.PI * 2);
    ctx.closePath();
    if (fill && fs) { ctx.fillStyle = fs; ctx.fill(); }
    if (ss && ss !== 'transparent') { ctx.strokeStyle = ss; ctx.lineWidth = 1.5; ctx.stroke(); }
}

function wrapText(ctx: CanvasRenderingContext2D, t: string, x: number, y: number, mw: number, lh: number, fs: number, align: CanvasTextAlign = 'left', weight: string = '400', font: string = 'Lora') {
    if (!t) return;
    ctx.font = `${weight} ${fs}px '${font}'`;
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

/* ─── Add-element helpers ─── */
let _addCounter = 0;
function genId(prefix: string) { return `${prefix}_${Date.now()}_${++_addCounter}`; }

/* ─── Component ─── */
export const PipelineDesigner = () => {
    const router = useRouter();
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
    const [templates, setTemplates] = useState<SavedTemplate[]>([]);
    const [showTemplates, setShowTemplates] = useState(false);
    const [saveName, setSaveName] = useState('');
    const [showSaveDialog, setShowSaveDialog] = useState(false);
    const [renamingId, setRenamingId] = useState<string | null>(null);
    const [showAddMenu, setShowAddMenu] = useState(false);
    const [renameValue, setRenameValue] = useState('');

    const W = 1150;
    const H = 750;

    const showToast = useCallback((msg: string) => {
        setToast(msg);
        setTimeout(() => setToast(null), 2000);
    }, []);

    const getNode = useCallback((id: string) => nodesRef.current.find((n) => n.id === id), []);

    /* ── Template Storage ── */
    useEffect(() => {
        const loaded = loadTemplatesFromStorage();
        if (loaded.length === 0) {
            // Seed with the default template
            const defaultTemplate: SavedTemplate = {
                id: 'default_' + Date.now(),
                name: 'Default Pipeline',
                nodes: createDefaultNodes(),
                connections: createDefaultConnections(),
                createdAt: Date.now(),
                updatedAt: Date.now(),
            };
            saveTemplatesToStorage([defaultTemplate]);
            setTemplates([defaultTemplate]);
        } else {
            setTemplates(loaded);
        }
    }, []);

    const saveTemplate = useCallback((name: string) => {
        const newTemplate: SavedTemplate = {
            id: 'tmpl_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6),
            name: name.trim() || 'Untitled',
            nodes: nodesRef.current.map(n => ({ ...n })),
            connections: connsRef.current.map(c => ({ ...c })),
            createdAt: Date.now(),
            updatedAt: Date.now(),
        };
        const updated = [newTemplate, ...templates];
        saveTemplatesToStorage(updated);
        setTemplates(updated);
        showToast(`Saved "${newTemplate.name}"`);
        setShowSaveDialog(false);
        setSaveName('');
    }, [templates, showToast]);

    const loadTemplate = useCallback((tmpl: SavedTemplate) => {
        nodesRef.current = tmpl.nodes.map(n => ({ ...n }));
        connsRef.current = tmpl.connections.map(c => ({ ...c }));
        selectedRef.current.clear();
        setShowTemplates(false);
        showToast(`Loaded "${tmpl.name}"`);
    }, [showToast]);

    const deleteTemplate = useCallback((id: string) => {
        const updated = templates.filter(t => t.id !== id);
        saveTemplatesToStorage(updated);
        setTemplates(updated);
        showToast('Template deleted');
    }, [templates, showToast]);

    const renameTemplate = useCallback((id: string, newName: string) => {
        const updated = templates.map(t => t.id === id ? { ...t, name: newName.trim() || t.name, updatedAt: Date.now() } : t);
        saveTemplatesToStorage(updated);
        setTemplates(updated);
        setRenamingId(null);
        setRenameValue('');
    }, [templates]);

    const overwriteTemplate = useCallback((tmpl: SavedTemplate) => {
        const updated = templates.map(t => t.id === tmpl.id ? {
            ...t,
            nodes: nodesRef.current.map(n => ({ ...n })),
            connections: connsRef.current.map(c => ({ ...c })),
            updatedAt: Date.now(),
        } : t);
        saveTemplatesToStorage(updated);
        setTemplates(updated);
        showToast(`Updated "${tmpl.name}"`);
    }, [templates, showToast]);

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
            } else if (node.type === 'shape') {
                const fillColor = node.filled !== false ? color.head : 'transparent';
                const strokeColor = color.border;
                const kind = node.shapeKind || 'rect';
                if (kind === 'rect') {
                    roundRect(ctx, node.x, node.y, node.w, node.h, 0, true, fillColor, strokeColor);
                } else if (kind === 'rounded-rect') {
                    roundRect(ctx, node.x, node.y, node.w, node.h, 10, true, fillColor, strokeColor);
                } else if (kind === 'circle') {
                    drawEllipse(ctx, node.x, node.y, node.w, node.h, true, fillColor, strokeColor);
                } else if (kind === 'diamond') {
                    drawDiamond(ctx, node.x, node.y, node.w, node.h, true, fillColor, strokeColor);
                } else if (kind === 'hexagon') {
                    drawHexagon(ctx, node.x, node.y, node.w, node.h, true, fillColor, strokeColor);
                }
                // Label inside shape
                if (node.title) {
                    const textColor = node.filled !== false ? color.textTitle : brand.dark;
                    ctx.fillStyle = textColor;
                    const fs = node.fontSize || 9;
                    wrapText(ctx, node.title, node.x + node.w / 2, node.y + node.h / 2, node.w - 12, fs + 3, fs, 'center', '600', 'Poppins');
                }
            } else if (node.type === 'textbox') {
                // optional light background
                if (node.filled) {
                    roundRect(ctx, node.x, node.y, node.w, node.h, 4, true, brand.light, brand.lightGray);
                }
                const fs = node.fontSize || 11;
                const weight = node.fontWeight || '400';
                const align = node.textAlign || 'left';
                ctx.fillStyle = brand.dark;
                const tx = align === 'center' ? node.x + node.w / 2 : align === 'right' ? node.x + node.w - 6 : node.x + 6;
                wrapText(ctx, node.content || node.title || '', tx, node.y + node.h / 2, node.w - 12, fs + 4, fs, align, weight, 'Poppins');
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
                if (['headerNode', 'minimal', 'plain', 'container', 'shape', 'textbox'].includes(clicked.type) &&
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
                    if (n.type === 'shape') {
                        openEditor(n, 'title');
                    } else if (n.type === 'textbox') {
                        openEditor(n, 'content');
                    } else {
                        const field = (n.type === 'headerNode' && mouse.y < n.y + 16) ? 'title' : 'content';
                        openEditor(n, field);
                    }
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
        if (node.type === 'textbox') {
            ed.value = node.content || '';
        } else if (node.type === 'shape') {
            ed.value = node.title || '';
        } else {
            ed.value = (field === 'title' || ['title', 'colLabel', 'minimal', 'container'].includes(node.type))
                ? (node.title || '')
                : (node.content || '');
        }
        ed.focus();
    };

    const closeEditor = () => {
        if (!editingRef.current) return;
        const { node, field } = editingRef.current;
        const ed = editorRef.current;
        if (!ed) return;
        if (node.type === 'textbox') {
            node.content = ed.value;
        } else if (node.type === 'shape') {
            node.title = ed.value;
        } else if (field === 'title' || ['title', 'colLabel', 'minimal', 'container'].includes(node.type)) {
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

    /* ── Add elements ── */
    const addShape = (kind: ShapeKind) => {
        const size = kind === 'circle' ? 60 : kind === 'diamond' ? 70 : 80;
        const h = kind === 'diamond' ? 60 : kind === 'circle' ? 60 : 50;
        const node: PNode = {
            id: genId('shape'),
            x: snap(W / 2 - size / 2),
            y: snap(H / 2 - h / 2),
            w: size,
            h,
            type: 'shape',
            shapeKind: kind,
            variant: 'neutral',
            title: kind.charAt(0).toUpperCase() + kind.slice(1),
            filled: true,
        };
        nodesRef.current.push(node);
        selectedRef.current.clear();
        selectedRef.current.add(node.id);
        setShowAddMenu(false);
        showToast(`Added ${kind}`);
    };

    const addTextBox = () => {
        const node: PNode = {
            id: genId('text'),
            x: snap(W / 2 - 60),
            y: snap(H / 2 - 15),
            w: 120,
            h: 30,
            type: 'textbox',
            content: 'Text',
            fontSize: 11,
            fontWeight: '400',
            textAlign: 'center',
            filled: false,
        };
        nodesRef.current.push(node);
        selectedRef.current.clear();
        selectedRef.current.add(node.id);
        setShowAddMenu(false);
        showToast('Added text box');
    };

    const addNodeOfType = (type: PNode['type'], variant: string = 'neutral') => {
        const defaults: Record<string, Partial<PNode>> = {
            headerNode: { w: 100, h: 50, title: 'Header', content: 'Content' },
            minimal: { w: 50, h: 60, title: 'Label' },
            plain: { w: 80, h: 35, content: '"value"' },
            container: { w: 300, h: 200, title: 'Group' },
            colLabel: { w: 80, h: 15, title: 'LABEL' },
            title: { w: 180, h: 20, title: 'Section Title' },
        };
        const d = defaults[type] || { w: 80, h: 40 };
        const node: PNode = {
            id: genId(type),
            x: snap(W / 2 - (d.w || 80) / 2),
            y: snap(H / 2 - (d.h || 40) / 2),
            w: d.w || 80,
            h: d.h || 40,
            type,
            variant,
            ...d,
        };
        nodesRef.current.push(node);
        selectedRef.current.clear();
        selectedRef.current.add(node.id);
        setShowAddMenu(false);
        showToast(`Added ${type}`);
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
                        onClick={() => router.push('/tools')}
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
                    <div className="flex items-center gap-3 mt-4 flex-wrap">
                        {/* Add Elements dropdown */}
                        <div className="relative">
                            <button
                                onClick={() => setShowAddMenu(!showAddMenu)}
                                className={`px-4 py-2 rounded-lg text-xs font-semibold font-heading transition-all flex items-center gap-1.5 ${
                                    showAddMenu ? 'bg-brand-blue text-white' : 'bg-brand-green text-white hover:opacity-90'
                                }`}
                            >
                                <Plus className="w-3.5 h-3.5" />
                                Add Element
                            </button>
                            {showAddMenu && (
                                <div className="absolute top-full left-0 mt-1.5 bg-card border border-border rounded-xl shadow-2xl py-2 z-[3000] min-w-[200px] animate-in fade-in slide-in-from-top-1 duration-150">
                                    <div className="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Shapes</div>
                                    <button onClick={() => addShape('rect')} className="w-full flex items-center gap-2.5 px-3 py-2 text-xs font-heading hover:bg-muted transition-colors">
                                        <Square className="w-3.5 h-3.5 text-muted-foreground" />
                                        Rectangle
                                    </button>
                                    <button onClick={() => addShape('rounded-rect')} className="w-full flex items-center gap-2.5 px-3 py-2 text-xs font-heading hover:bg-muted transition-colors">
                                        <RectangleHorizontal className="w-3.5 h-3.5 text-muted-foreground" />
                                        Rounded Rectangle
                                    </button>
                                    <button onClick={() => addShape('circle')} className="w-full flex items-center gap-2.5 px-3 py-2 text-xs font-heading hover:bg-muted transition-colors">
                                        <Circle className="w-3.5 h-3.5 text-muted-foreground" />
                                        Ellipse / Circle
                                    </button>
                                    <button onClick={() => addShape('diamond')} className="w-full flex items-center gap-2.5 px-3 py-2 text-xs font-heading hover:bg-muted transition-colors">
                                        <Diamond className="w-3.5 h-3.5 text-muted-foreground" />
                                        Diamond
                                    </button>
                                    <button onClick={() => addShape('hexagon')} className="w-full flex items-center gap-2.5 px-3 py-2 text-xs font-heading hover:bg-muted transition-colors">
                                        <Hexagon className="w-3.5 h-3.5 text-muted-foreground" />
                                        Hexagon
                                    </button>
                                    <div className="h-px bg-border my-1.5" />
                                    <div className="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Text</div>
                                    <button onClick={() => addTextBox()} className="w-full flex items-center gap-2.5 px-3 py-2 text-xs font-heading hover:bg-muted transition-colors">
                                        <Type className="w-3.5 h-3.5 text-muted-foreground" />
                                        Text Box
                                    </button>
                                    <div className="h-px bg-border my-1.5" />
                                    <div className="px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Nodes</div>
                                    <button onClick={() => addNodeOfType('headerNode')} className="w-full flex items-center gap-2.5 px-3 py-2 text-xs font-heading hover:bg-muted transition-colors">
                                        <span className="w-3.5 h-3.5 rounded-sm border border-muted-foreground bg-muted flex-shrink-0" />
                                        Header Node
                                    </button>
                                    <button onClick={() => addNodeOfType('minimal')} className="w-full flex items-center gap-2.5 px-3 py-2 text-xs font-heading hover:bg-muted transition-colors">
                                        <span className="w-3.5 h-3.5 rounded-sm bg-muted-foreground flex-shrink-0" />
                                        Minimal Node
                                    </button>
                                    <button onClick={() => addNodeOfType('plain')} className="w-full flex items-center gap-2.5 px-3 py-2 text-xs font-heading hover:bg-muted transition-colors">
                                        <span className="w-3.5 h-3.5 rounded-sm border border-muted-foreground flex-shrink-0" />
                                        Plain Node
                                    </button>
                                    <button onClick={() => addNodeOfType('container')} className="w-full flex items-center gap-2.5 px-3 py-2 text-xs font-heading hover:bg-muted transition-colors">
                                        <span className="w-3.5 h-3.5 rounded-sm border border-dashed border-muted-foreground flex-shrink-0" />
                                        Container
                                    </button>
                                </div>
                            )}
                        </div>
                        <button
                            onClick={() => { setShowSaveDialog(true); setSaveName(''); }}
                            className="px-4 py-2 rounded-lg bg-brand-green text-white text-xs font-semibold font-heading hover:opacity-90 transition-all flex items-center gap-1.5"
                        >
                            <Save className="w-3.5 h-3.5" />
                            Save Template
                        </button>
                        <button
                            onClick={() => setShowTemplates(!showTemplates)}
                            className={`px-4 py-2 rounded-lg text-xs font-semibold font-heading transition-all flex items-center gap-1.5 ${showTemplates ? 'bg-brand-blue text-white' : 'bg-foreground text-background hover:opacity-90'
                                }`}
                        >
                            <FolderOpen className="w-3.5 h-3.5" />
                            Templates ({templates.length})
                        </button>
                        <div className="w-px h-6 bg-border" />
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

                    {/* Save Dialog */}
                    {showSaveDialog && (
                        <div className="fixed inset-0 bg-black/40 z-[4000] flex items-center justify-center backdrop-blur-sm" onClick={() => setShowSaveDialog(false)}>
                            <div className="bg-card border border-border rounded-2xl shadow-2xl p-6 w-full max-w-sm mx-4" onClick={e => e.stopPropagation()}>
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="text-sm font-bold font-heading text-foreground">Save as Template</h3>
                                    <button onClick={() => setShowSaveDialog(false)} className="text-muted-foreground hover:text-foreground transition-colors">
                                        <X className="w-4 h-4" />
                                    </button>
                                </div>
                                <input
                                    type="text"
                                    value={saveName}
                                    onChange={e => setSaveName(e.target.value)}
                                    placeholder="Template name…"
                                    className="w-full px-3 py-2 rounded-lg border border-border bg-background text-sm font-heading outline-none focus:border-brand-orange transition-colors"
                                    autoFocus
                                    onKeyDown={e => { if (e.key === 'Enter') saveTemplate(saveName); }}
                                />
                                <div className="flex justify-end gap-2 mt-4">
                                    <button
                                        onClick={() => setShowSaveDialog(false)}
                                        className="px-4 py-2 rounded-lg text-xs font-semibold font-heading text-muted-foreground hover:text-foreground transition-colors"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        onClick={() => saveTemplate(saveName)}
                                        className="px-4 py-2 rounded-lg bg-brand-green text-white text-xs font-semibold font-heading hover:opacity-90 transition-all"
                                    >
                                        Save
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Templates Panel */}
                    {showTemplates && (
                        <div className="mt-4 bg-card border border-border rounded-2xl shadow-lg overflow-hidden">
                            <div className="flex items-center justify-between px-4 py-3 border-b border-border">
                                <h3 className="text-sm font-bold font-heading text-foreground">Saved Templates</h3>
                                <button onClick={() => setShowTemplates(false)} className="text-muted-foreground hover:text-foreground transition-colors">
                                    <X className="w-4 h-4" />
                                </button>
                            </div>
                            {templates.length === 0 ? (
                                <div className="px-4 py-8 text-center text-xs text-muted-foreground font-heading">
                                    No saved templates yet. Click &quot;Save Template&quot; to save the current diagram.
                                </div>
                            ) : (
                                <div className="divide-y divide-border max-h-[320px] overflow-y-auto">
                                    {templates.map(tmpl => (
                                        <div key={tmpl.id} className="flex items-center gap-3 px-4 py-3 hover:bg-muted/50 transition-colors group">
                                            <div className="flex-1 min-w-0">
                                                {renamingId === tmpl.id ? (
                                                    <div className="flex items-center gap-1.5">
                                                        <input
                                                            type="text"
                                                            value={renameValue}
                                                            onChange={e => setRenameValue(e.target.value)}
                                                            className="flex-1 px-2 py-1 rounded border border-border bg-background text-xs font-heading outline-none focus:border-brand-orange"
                                                            autoFocus
                                                            onKeyDown={e => {
                                                                if (e.key === 'Enter') renameTemplate(tmpl.id, renameValue);
                                                                if (e.key === 'Escape') setRenamingId(null);
                                                            }}
                                                        />
                                                        <button onClick={() => renameTemplate(tmpl.id, renameValue)} className="text-brand-green hover:opacity-80"><Check className="w-3.5 h-3.5" /></button>
                                                        <button onClick={() => setRenamingId(null)} className="text-muted-foreground hover:text-foreground"><X className="w-3.5 h-3.5" /></button>
                                                    </div>
                                                ) : (
                                                    <>
                                                        <p className="text-xs font-semibold font-heading text-foreground truncate">{tmpl.name}</p>
                                                        <p className="text-[10px] text-muted-foreground font-heading mt-0.5">
                                                            {tmpl.nodes.length} nodes · {tmpl.connections.length} connections · {new Date(tmpl.updatedAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                                                        </p>
                                                    </>
                                                )}
                                            </div>
                                            {renamingId !== tmpl.id && (
                                                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                                    <button
                                                        onClick={() => loadTemplate(tmpl)}
                                                        className="px-2.5 py-1.5 rounded-md bg-brand-blue text-white text-[10px] font-semibold font-heading hover:opacity-90 transition-all"
                                                    >
                                                        Load
                                                    </button>
                                                    <button
                                                        onClick={() => overwriteTemplate(tmpl)}
                                                        className="px-2.5 py-1.5 rounded-md bg-brand-orange text-white text-[10px] font-semibold font-heading hover:opacity-90 transition-all"
                                                        title="Overwrite with current diagram"
                                                    >
                                                        Update
                                                    </button>
                                                    <button
                                                        onClick={() => { setRenamingId(tmpl.id); setRenameValue(tmpl.name); }}
                                                        className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-all"
                                                        title="Rename"
                                                    >
                                                        <Pencil className="w-3 h-3" />
                                                    </button>
                                                    <button
                                                        onClick={() => deleteTemplate(tmpl.id)}
                                                        className="p-1.5 rounded-md text-muted-foreground hover:text-red-500 hover:bg-red-50 transition-all"
                                                        title="Delete"
                                                    >
                                                        <Trash2 className="w-3 h-3" />
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
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
