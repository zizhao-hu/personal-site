'use client';

import { Header } from '@/components/custom/header';
import { useEffect, useRef, useCallback, useState } from 'react';
import { ArrowLeft, Save, FolderOpen, Trash2, Pencil, X, Check, Square, Circle, Diamond, Type, RectangleHorizontal, Hexagon, Plus, Grid3X3, Link, RotateCcw, Download, FileDown } from 'lucide-react';
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
    textVAlign?: 'top' | 'middle' | 'bottom';
    filled?: boolean;
}

interface PConnection {
    from: string | string[];
    to: string | string[];
    type: 'direct' | 'split' | 'merge';
    dash?: number[];
    orientation?: string;
    midpoints?: { x: number; y: number }[];
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
    // Position constants
    const A1_X = 20;   // Approach 1 left edge
    const A2_X = 450;  // Approach 2 left edge
    const CT_Y = 285;  // PRISM container top
    const PH_Y = CT_Y + 35;  // phase label row y
    const COL_Y = PH_Y + 20; // column-label row y
    const R_Y = COL_Y + 20;  // node row start y

    // Phase X offsets (inside container, relative to canvas)
    const P1X = 30;
    const P2X = 390;
    const P3X = 780;

    return [
        // --- Approach 1: Inference with PRISM ---
        { id: 'lbl_a1', x: A1_X, y: 20, w: 300, h: 18, type: 'title', title: 'Approach 1: Intent-based Prompting' },
        { id: 'a1_col_usr', x: A1_X, y: 42, w: 80, h: 14, type: 'colLabel', title: 'User' },
        { id: 'a1_col_ans', x: A1_X + 200, y: 42, w: 80, h: 14, type: 'colLabel', title: 'Answer' },
        { id: 'a1_u1', x: A1_X, y: 60, w: 110, h: 32, type: 'plain', variant: 'neutral', content: '"What is the integral of x^2"' },
        { id: 'a1_u2', x: A1_X, y: 100, w: 110, h: 32, type: 'plain', variant: 'neutral', content: '"I feel stressed for exams..."' },
        { id: 'a1_u3', x: A1_X, y: 140, w: 110, h: 32, type: 'plain', variant: 'neutral', content: '"How to use a rice cooker..."' },
        { id: 'a1_llm', x: A1_X + 130, y: 85, w: 50, h: 70, type: 'minimal', variant: 'orange', title: 'PRISMEd LLM' },
        { id: 'a1_ans1', x: A1_X + 200, y: 60, w: 110, h: 32, type: 'plain', variant: 'green', content: '"x^3/3 + C"' },
        { id: 'a1_ans2', x: A1_X + 200, y: 100, w: 110, h: 32, type: 'plain', variant: 'neutral', content: '"I totally understand..."' },
        { id: 'a1_ans3', x: A1_X + 200, y: 140, w: 110, h: 32, type: 'plain', variant: 'neutral', content: '"Here is the step by step..."' },

        // --- Approach 2: Inference with Training-free Persona Routing ---
        { id: 'lbl_a2', x: A2_X, y: 20, w: 390, h: 18, type: 'title', title: 'Approach 2: Intent-based Routing' },
        { id: 'a2_col_pp', x: A2_X, y: 42, w: 110, h: 14, type: 'colLabel', title: 'Persona Prompt' },
        { id: 'a2_col_usr', x: A2_X + 230, y: 42, w: 60, h: 14, type: 'colLabel', title: 'User' },
        { id: 'a2_col_ans', x: A2_X + 380, y: 42, w: 80, h: 14, type: 'colLabel', title: 'Answer' },
        { id: 'a2_p1', x: A2_X, y: 60, w: 110, h: 35, type: 'headerNode', variant: 'neutral', title: 'Scientist', content: 'Be concise...' },
        { id: 'a2_p2', x: A2_X, y: 100, w: 110, h: 35, type: 'headerNode', variant: 'neutral', title: 'Counselor', content: 'Show empathy...' },
        { id: 'a2_p3', x: A2_X, y: 140, w: 110, h: 35, type: 'headerNode', variant: 'neutral', title: 'Default', content: 'Helpful assistant' },
        { id: 'a2_usr', x: A2_X + 130, y: 95, w: 100, h: 34, type: 'plain', variant: 'neutral', content: '"What is the integral of x^2"' },
        { id: 'a2_rtr', x: A2_X + 244, y: 90, w: 40, h: 38, type: 'minimal', variant: 'orange', title: 'Router' },
        { id: 'a2_usr2', x: A2_X + 296, y: 95, w: 100, h: 34, type: 'plain', variant: 'neutral', content: '"What is the integral of x^2"' },
        { id: 'a2_llm', x: A2_X + 408, y: 84, w: 40, h: 50, type: 'minimal', variant: 'neutral', title: 'Base LLM' },
        { id: 'a2_ans1', x: A2_X + 460, y: 60, w: 90, h: 32, type: 'plain', variant: 'green', content: '"x^3/3 + C"' },
        { id: 'a2_ans2', x: A2_X + 460, y: 100, w: 90, h: 32, type: 'plain', variant: 'neutral', content: '"Sure! We first apply..."' },
        { id: 'a2_ans3', x: A2_X + 460, y: 140, w: 90, h: 32, type: 'plain', variant: 'neutral', content: '"Here is the integral..."' },

        // --- PRISM Training Container ---
        { id: 'prism_box', x: 20, y: CT_Y, w: 1060, h: 390, type: 'container', title: 'Persona Routing via Intent-based Self-Modeling (PRISM)' },
        { id: 'lbl_ph1', x: P1X + 20, y: PH_Y, w: 280, h: 14, type: 'title', title: '1. Synthetic Query (SQ) Generation', size: 9 },
        { id: 'lbl_ph2', x: P2X + 20, y: PH_Y, w: 280, h: 14, type: 'title', title: '2. Synthetic Answer (SA) Generation', size: 9 },
        { id: 'lbl_ph3', x: P3X + 20, y: PH_Y, w: 220, h: 14, type: 'title', title: '3. Self-Distillation via PEFT', size: 9 },

        // Phase 1 column labels
        { id: 'p1_col_pp', x: P1X + 85, y: COL_Y, w: 110, h: 14, type: 'colLabel', title: 'Persona Prompt' },
        { id: 'p1_col_sq', x: P1X + 250, y: COL_Y, w: 60, h: 14, type: 'colLabel', title: 'SQ' },
        // Phase 1 nodes
        { id: 'p1_trig', x: P1X, y: R_Y + 30, w: 80, h: 50, type: 'plain', variant: 'neutral', content: '"Create questions related to persona..."' },
        { id: 'p1_pp1', x: P1X + 85, y: R_Y, w: 110, h: 32, type: 'headerNode', variant: 'neutral', title: 'Scientist', content: 'Be concise...' },
        { id: 'p1_pp2', x: P1X + 85, y: R_Y + 40, w: 110, h: 32, type: 'headerNode', variant: 'neutral', title: 'Counselor', content: 'Show empathy...' },
        { id: 'p1_pp3', x: P1X + 85, y: R_Y + 80, w: 110, h: 32, type: 'headerNode', variant: 'neutral', title: 'Default', content: 'Helpful assistant' },
        { id: 'p1_llm', x: P1X + 205, y: R_Y + 28, w: 40, h: 60, type: 'minimal', variant: 'neutral', title: 'Base LLM' },
        { id: 'p1_sq1', x: P1X + 254, y: R_Y, w: 85, h: 28, type: 'plain', variant: 'blue', content: '"Calculate x, y, z..."', stacked: true },
        { id: 'p1_sq2', x: P1X + 254, y: R_Y + 36, w: 85, h: 28, type: 'plain', variant: 'blue', content: '"I have stress..."', stacked: true },
        { id: 'p1_sq3', x: P1X + 254, y: R_Y + 72, w: 85, h: 28, type: 'plain', variant: 'blue', content: '"How to swim..."', stacked: true },

        // Phase 2 column labels
        { id: 'p2_col_pp', x: P2X, y: COL_Y, w: 110, h: 14, type: 'colLabel', title: 'Persona Prompt' },
        { id: 'p2_col_sq', x: P2X + 120, y: COL_Y, w: 60, h: 14, type: 'colLabel', title: 'SQ' },
        { id: 'p2_col_sa', x: P2X + 265, y: COL_Y, w: 60, h: 14, type: 'colLabel', title: 'SA' },
        // Phase 2 nodes
        { id: 'p2_pp1', x: P2X, y: R_Y, w: 110, h: 32, type: 'headerNode', variant: 'neutral', title: 'Scientist', content: 'Be concise...' },
        { id: 'p2_pp2', x: P2X, y: R_Y + 40, w: 110, h: 32, type: 'headerNode', variant: 'neutral', title: 'Counselor', content: 'Show empathy...' },
        { id: 'p2_pp3', x: P2X, y: R_Y + 80, w: 110, h: 32, type: 'headerNode', variant: 'neutral', title: 'Default', content: 'Helpful assistant' },
        { id: 'p2_sq1', x: P2X + 120, y: R_Y, w: 85, h: 28, type: 'plain', variant: 'blue', content: '"Calculate x, y, z..."', stacked: true },
        { id: 'p2_sq2', x: P2X + 120, y: R_Y + 36, w: 85, h: 28, type: 'plain', variant: 'blue', content: '"I have stress..."', stacked: true },
        { id: 'p2_sq3', x: P2X + 120, y: R_Y + 72, w: 85, h: 28, type: 'plain', variant: 'blue', content: '"How to swim..."', stacked: true },
        { id: 'p2_llm', x: P2X + 215, y: R_Y + 28, w: 40, h: 60, type: 'minimal', variant: 'neutral', title: 'Base LLM' },
        { id: 'p2_selfverify', x: P2X + 162, y: R_Y + 12, w: 100, h: 14, type: 'title', title: 'Self-verify', size: 8 },
        { id: 'p2_sa1', x: P2X + 265, y: R_Y, w: 85, h: 28, type: 'plain', variant: 'blue', content: '"x^3/3 + C"', stacked: true },
        { id: 'p2_sa2', x: P2X + 265, y: R_Y + 36, w: 85, h: 28, type: 'plain', variant: 'blue', content: '"I feel sorry..."', stacked: true },
        { id: 'p2_sa3', x: P2X + 265, y: R_Y + 72, w: 85, h: 28, type: 'plain', variant: 'blue', content: '"Here are steps..."', stacked: true },

        // Phase 3 column labels
        { id: 'p3_col_sq', x: P3X, y: COL_Y, w: 60, h: 14, type: 'colLabel', title: 'SQ' },
        { id: 'p3_col_sa', x: P3X + 220, y: COL_Y, w: 60, h: 14, type: 'colLabel', title: 'SA' },
        // Phase 3 nodes
        { id: 'p3_sq1', x: P3X, y: R_Y, w: 85, h: 28, type: 'plain', variant: 'blue', content: '"Calculate x, y, z..."', stacked: true },
        { id: 'p3_sq2', x: P3X, y: R_Y + 36, w: 85, h: 28, type: 'plain', variant: 'blue', content: '"I have stress..."', stacked: true },
        { id: 'p3_sq3', x: P3X, y: R_Y + 72, w: 85, h: 28, type: 'plain', variant: 'blue', content: '"How to swim..."', stacked: true },
        { id: 'p3_base', x: P3X + 95, y: R_Y + 14, w: 40, h: 60, type: 'minimal', variant: 'neutral', title: 'Base LLM' },
        { id: 'p3_distill', x: P3X + 143, y: R_Y + 28, w: 70, h: 14, type: 'title', title: 'Distill', size: 8 },
        { id: 'p3_prism', x: P3X + 148, y: R_Y + 44, w: 54, h: 60, type: 'minimal', variant: 'orange', title: 'PRISMEd LLM' },
        { id: 'p3_sa1', x: P3X + 212, y: R_Y, w: 85, h: 28, type: 'plain', variant: 'blue', content: '"x^3/3 + C"', stacked: true },
        { id: 'p3_sa2', x: P3X + 212, y: R_Y + 36, w: 85, h: 28, type: 'plain', variant: 'blue', content: '"I feel sorry..."', stacked: true },
        { id: 'p3_sa3', x: P3X + 212, y: R_Y + 72, w: 85, h: 28, type: 'plain', variant: 'blue', content: '"Here are steps..."', stacked: true },
    ];
}

function createDefaultConnections(): PConnection[] {
    return [
        // Approach 1
        { from: ['a1_u1', 'a1_u2', 'a1_u3'], to: 'a1_llm', type: 'merge' },
        { from: 'a1_llm', to: ['a1_ans1', 'a1_ans2', 'a1_ans3'], type: 'split' },
        // Approach 2
        { from: ['a2_p1', 'a2_p2', 'a2_p3'], to: 'a2_rtr', type: 'merge' },
        { from: 'a2_usr', to: 'a2_rtr', type: 'direct' },
        { from: 'a2_rtr', to: 'a2_usr2', type: 'direct' },
        { from: 'a2_usr2', to: 'a2_llm', type: 'direct' },
        { from: 'a2_llm', to: ['a2_ans1', 'a2_ans2', 'a2_ans3'], type: 'split' },
        // Phase 1: SQ Generation
        { from: 'p1_trig', to: ['p1_pp1', 'p1_pp2', 'p1_pp3'], type: 'split' },
        { from: ['p1_pp1', 'p1_pp2', 'p1_pp3'], to: 'p1_llm', type: 'merge' },
        { from: 'p1_llm', to: ['p1_sq1', 'p1_sq2', 'p1_sq3'], type: 'split' },
        // Phase 2: SA Generation
        { from: 'p2_pp1', to: 'p2_sq1', type: 'direct' },
        { from: 'p2_pp2', to: 'p2_sq2', type: 'direct' },
        { from: 'p2_pp3', to: 'p2_sq3', type: 'direct' },
        { from: ['p2_sq1', 'p2_sq2', 'p2_sq3'], to: 'p2_llm', type: 'merge' },
        { from: 'p2_llm', to: ['p2_sa1', 'p2_sa2', 'p2_sa3'], type: 'split' },
        // Phase 3: Self-Distillation
        { from: ['p3_sq1', 'p3_sq2', 'p3_sq3'], to: 'p3_base', type: 'merge' },
        { from: 'p3_base', to: 'p3_prism', type: 'direct' },
        { from: ['p3_sq1', 'p3_sq2', 'p3_sq3'], to: 'p3_prism', type: 'merge' },
        { from: 'p3_prism', to: ['p3_sa1', 'p3_sa2', 'p3_sa3'], type: 'split' },
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

function wrapText(ctx: CanvasRenderingContext2D, t: string, x: number, y: number, mw: number, lh: number, fs: number, align: CanvasTextAlign = 'left', weight: string = '400', font: string = 'Lora', vAlign: 'top' | 'middle' | 'bottom' = 'middle', boxH?: number) {
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
    let startY: number;
    if (vAlign === 'top') { startY = y - (boxH ? boxH / 2 : 0) + lh / 2 + 4; }
    else if (vAlign === 'bottom') { startY = y + (boxH ? boxH / 2 : 0) - totalH + lh / 2 - 2; }
    else { startY = y - totalH / 2 + lh / 2 + 2; }
    lines.forEach((line) => { ctx.fillText(line, x, startY); startY += lh; });
}

function drawArrowHead(ctx: CanvasRenderingContext2D, x: number, y: number, angle: number) {
    ctx.save(); ctx.translate(x, y); ctx.rotate(angle);
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-5, -3); ctx.lineTo(-5, 3);
    ctx.fillStyle = ctx.strokeStyle; ctx.fill(); ctx.restore();
}

function drawOrthPath(ctx: CanvasRenderingContext2D, pts: { x: number; y: number }[], arrow: boolean = true) {
    if (pts.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.stroke();
    if (arrow && pts.length >= 2) {
        const last = pts[pts.length - 1];
        const prev = pts[pts.length - 2];
        const a = Math.atan2(last.y - prev.y, last.x - prev.x);
        drawArrowHead(ctx, last.x, last.y, a);
    }
}

/** Build an orthogonal point sequence from source edge to target edge via midpoints */
function buildOrthPoints(
    sx: number, sy: number, tx: number, ty: number,
    midpoints?: { x: number; y: number }[]
): { x: number; y: number }[] {
    if (midpoints && midpoints.length > 0) {
        // Route through each midpoint with L-shaped turns
        const pts: { x: number; y: number }[] = [{ x: sx, y: sy }];
        let cx = sx, cy = sy;
        for (const mp of midpoints) {
            // go horizontal to mp.x, then vertical to mp.y
            pts.push({ x: mp.x, y: cy });
            pts.push({ x: mp.x, y: mp.y });
            cx = mp.x; cy = mp.y;
        }
        // from last midpoint to target: horizontal then vertical
        pts.push({ x: tx, y: cy });
        if (Math.abs(cy - ty) > 1) pts.push({ x: tx, y: ty });
        return pts;
    }
    // Default: single midpoint at halfway X
    const mx = sx + (tx - sx) / 2;
    if (Math.abs(sy - ty) < 2) return [{ x: sx, y: sy }, { x: tx, y: ty }];
    return [{ x: sx, y: sy }, { x: mx, y: sy }, { x: mx, y: ty }, { x: tx, y: ty }];
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
    const dragMidRef = useRef<{ conn: PConnection; idx: number } | null>(null);
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
    const showGridRef = useRef(true);
    const [showGrid, setShowGrid] = useState(true);
    const [panelTab, setPanelTab] = useState<'shapes' | 'nodes' | 'text'>('shapes');
    const [selectionVersion, setSelectionVersion] = useState(0);
    const bumpSelection = useCallback(() => setSelectionVersion(v => v + 1), []);
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

        // Grid dots
        if (showGridRef.current) {
            ctx.save();
            ctx.fillStyle = 'rgba(0,0,0,0.13)';
            for (let gx = 0; gx <= W; gx += GRID) {
                for (let gy = 0; gy <= H; gy += GRID) {
                    ctx.fillRect(gx - 0.5, gy - 0.5, 1, 1);
                }
            }
            ctx.restore();
        }

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
                const containerStroke = color.border;
                // Draw dashed rounded-rect border directly (roundRect resets lineWidth)
                ctx.setLineDash([10, 5]); ctx.strokeStyle = containerStroke; ctx.lineWidth = 2;
                ctx.beginPath();
                const r = 12;
                ctx.moveTo(node.x + r, node.y);
                ctx.arcTo(node.x + node.w, node.y, node.x + node.w, node.y + node.h, r);
                ctx.arcTo(node.x + node.w, node.y + node.h, node.x, node.y + node.h, r);
                ctx.arcTo(node.x, node.y + node.h, node.x, node.y, r);
                ctx.arcTo(node.x, node.y, node.x + node.w, node.y, r);
                ctx.closePath();
                ctx.stroke();
            } else if (node.type === 'title') {
                const sz = node.fontSize || node.size || 11;
                const tw = node.fontWeight || '600';
                const titleColor = color.head;
                ctx.fillStyle = titleColor; ctx.font = `${tw} ${sz}px 'Poppins'`; ctx.textAlign = node.textAlign || 'center';
                ctx.fillText(node.title || '', node.x + node.w / 2, node.y + node.h / 2 + 5);
            } else if (node.type === 'colLabel') {
                const clfs = node.fontSize || 10;
                const clfw = node.fontWeight || '600';
                const clColor = color.head || brand.midGray;
                ctx.fillStyle = clColor; ctx.font = `${clfw} ${clfs}px 'Poppins'`; ctx.textAlign = node.textAlign || 'center';
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
                    roundRect(ctx, node.x, node.y, node.w, node.h, 4, true, color.body, color.border);
                }
                const fs = node.fontSize || 11;
                const weight = node.fontWeight || '400';
                const align = node.textAlign || 'left';
                ctx.fillStyle = color.head;
                const tx = align === 'center' ? node.x + node.w / 2 : align === 'right' ? node.x + node.w - 6 : node.x + 6;
                wrapText(ctx, node.content || node.title || '', tx, node.y + node.h / 2, node.w - 12, fs + 4, fs, align, weight, 'Poppins', (node as any).textVAlign || 'middle', node.h);
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
                // Vertical: top-down orthogonal
                const pts = buildOrthPoints(s.x + s.w / 2, s.y + s.h, t.x + t.w / 2, t.y, l.midpoints);
                drawOrthPath(ctx, pts);
            } else if (l.type === 'split') {
                // Split: source right → vertical bus → each target left
                const defaultMidX = s.x + s.w + (targets[0]!.x - (s.x + s.w)) / 2;
                const midX = (l.midpoints && l.midpoints.length > 0) ? l.midpoints[0].x : defaultMidX;
                // Horizontal from source to midX
                drawLine(ctx, s.x + s.w, s.y + s.h / 2, midX, s.y + s.h / 2);
                // Vertical bus
                const ys = targets.map(n => n!.y + n!.h / 2);
                drawLine(ctx, midX, Math.min(...ys), midX, Math.max(...ys));
                // Arrow from bus to each target
                targets.forEach(node => {
                    const ty = node!.y + node!.h / 2;
                    drawOrthPath(ctx, [{ x: midX, y: ty }, { x: node!.x, y: ty }]);
                });

            } else if (l.type === 'merge') {
                // Merge: each source right → vertical bus → target left
                const maxX = Math.max(...sources.map(n => n!.x + n!.w));
                const defaultMidX = t.x - (t.x - maxX) / 2;
                const midX = (l.midpoints && l.midpoints.length > 0) ? l.midpoints[0].x : defaultMidX;
                // Vertical bus
                const ys = sources.map(n => n!.y + n!.h / 2);
                drawLine(ctx, midX, Math.min(...ys), midX, Math.max(...ys));
                // Horizontal from each source to bus
                sources.forEach(node => drawLine(ctx, node!.x + node!.w, node!.y + node!.h / 2, midX, node!.y + node!.h / 2));
                // Arrow from bus to target
                drawOrthPath(ctx, [{ x: midX, y: t.y + t.h / 2 }, { x: t.x, y: t.y + t.h / 2 }]);

            } else {
                // Direct: orthogonal path through midpoints
                const sx = s.x + s.w, sy = s.y + s.h / 2;
                const tx = t.x, ty = t.y + t.h / 2;
                const pts = buildOrthPoints(sx, sy, tx, ty, l.midpoints);
                drawOrthPath(ctx, pts);

            }
        });


        // Connection preview (orthogonal)
        if (connectRef.current.active && connectRef.current.sourceId) {
            const src = getNode(connectRef.current.sourceId);
            if (src) {
                ctx.save(); ctx.strokeStyle = brand.blue; ctx.lineWidth = 1.5; ctx.setLineDash([5, 5]);
                const sx = src.x + src.w, sy = src.y + src.h / 2;
                const mx = mouseRef.current.x, my = mouseRef.current.y;
                const midX = sx + (mx - sx) / 2;
                ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(midX, sy); ctx.lineTo(midX, my); ctx.lineTo(mx, my); ctx.stroke();
                ctx.restore();
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

            // Check midpoint handles first
            const HIT = 8;
            for (const conn of connsRef.current) {
                if (conn.type === 'split' || conn.type === 'merge') {
                    // For split/merge, the midpoint handle is positioned at the bus X
                    const sources = Array.isArray(conn.from) ? conn.from.map(id => getNode(id)) : [getNode(conn.from)];
                    const targets = Array.isArray(conn.to) ? conn.to.map(id => getNode(id)) : [getNode(conn.to)];
                    if (sources.some(s => !s) || targets.some(t => !t)) continue;
                    const s = sources[0]!, t = targets[0]!;
                    let midX: number, midY: number;
                    if (conn.type === 'split') {
                        midX = (conn.midpoints && conn.midpoints.length > 0) ? conn.midpoints[0].x : s.x + s.w + (t.x - (s.x + s.w)) / 2;
                        midY = s.y + s.h / 2;
                    } else {
                        const maxX = Math.max(...sources.map(n => n!.x + n!.w));
                        midX = (conn.midpoints && conn.midpoints.length > 0) ? conn.midpoints[0].x : t.x - (t.x - maxX) / 2;
                        midY = t.y + t.h / 2;
                    }
                    if (Math.abs(mouse.x - midX) < HIT && Math.abs(mouse.y - midY) < HIT) {
                        if (!conn.midpoints) conn.midpoints = [{ x: midX, y: midY }];
                        dragMidRef.current = { conn, idx: 0 };
                        return;
                    }
                } else if (conn.type === 'direct' && !conn.orientation) {
                    const sf = Array.isArray(conn.from) ? getNode(conn.from[0]) : getNode(conn.from);
                    const tf = Array.isArray(conn.to) ? getNode(conn.to[0]) : getNode(conn.to);
                    if (!sf || !tf) continue;
                    const sx = sf.x + sf.w, sy = sf.y + sf.h / 2;
                    const tx = tf.x, ty = tf.y + tf.h / 2;
                    const mps = conn.midpoints || [{ x: sx + (tx - sx) / 2, y: sy }];
                    for (let i = 0; i < mps.length; i++) {
                        if (Math.abs(mouse.x - mps[i].x) < HIT && Math.abs(mouse.y - mps[i].y) < HIT) {
                            if (!conn.midpoints) conn.midpoints = [{ x: mps[i].x, y: mps[i].y }];
                            dragMidRef.current = { conn, idx: i };
                            return;
                        }
                    }
                }
            }

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
                bumpSelection();
            } else {
                selectedRef.current.clear();
                bumpSelection();
                selBoxRef.current = { x: mouse.x, y: mouse.y, startX: mouse.x, startY: mouse.y, w: 0, h: 0 };
            }
        };

        const onMouseMove = (e: MouseEvent) => {
            const m = getMouse(e);
            mouseRef.current = m;
            if (resizedRef.current) {
                resizedRef.current.w = snap(Math.max(40, m.x - resizedRef.current.x));
                resizedRef.current.h = snap(Math.max(20, m.y - resizedRef.current.y));
            } else if (dragMidRef.current) {
                const dm = dragMidRef.current;
                if (dm.conn.type === 'split' || dm.conn.type === 'merge') {
                    // For split/merge, only allow horizontal (X) movement of the bus
                    dm.conn.midpoints![dm.idx].x = snap(m.x);
                } else {
                    dm.conn.midpoints![dm.idx].x = snap(m.x);
                    dm.conn.midpoints![dm.idx].y = snap(m.y);
                }
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
            if (draggedRef.current) { nodesRef.current.forEach((n) => { if (selectedRef.current.has(n.id)) { n.x = snap(n.x); n.y = snap(n.y); } }); }
            if (resizedRef.current) { resizedRef.current.w = snap(resizedRef.current.w); resizedRef.current.h = snap(resizedRef.current.h); }
            draggedRef.current = null;
            resizedRef.current = null;
            dragMidRef.current = null;
            selBoxRef.current = null;
            bumpSelection();
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
        bumpSelection();
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
        const wasGrid = showGridRef.current;
        showGridRef.current = false;
        render();
        const link = document.createElement('a');
        link.download = 'ai-pipeline.png';
        link.href = canvas.toDataURL();
        link.click();
        showGridRef.current = wasGrid;
    };

    const exportPDF = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const wasGrid = showGridRef.current;
        showGridRef.current = false;
        render();
        const imgData = canvas.toDataURL('image/png');
        const pdf = new jsPDF({ orientation: 'landscape', unit: 'px', format: [W, H] });
        pdf.addImage(imgData, 'PNG', 0, 0, W, H);
        pdf.save('ai-pipeline.pdf');
        showGridRef.current = wasGrid;
        showToast('Exported PDF');
    };

    const toggleGrid = () => { const next = !showGridRef.current; showGridRef.current = next; setShowGrid(next); };

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
        bumpSelection();
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
        bumpSelection();
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
        bumpSelection();
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

    // Selected node for properties
    void selectionVersion; // dependency to re-compute on selection change
    const selectedNodes = nodesRef.current.filter(n => selectedRef.current.has(n.id));
    const selNode = selectedNodes.length === 1 ? selectedNodes[0] : null;
    const setNodeProp = (key: keyof PNode, val: unknown) => {
        if (selectedNodes.length === 0) return;
        selectedNodes.forEach(n => { (n as unknown as Record<string, unknown>)[key] = val; });
        bumpSelection();
    };
    const panelBtnClass = "w-full flex items-center gap-2 px-3 py-2 text-[11px] font-heading hover:bg-muted/70 rounded-md transition-colors text-foreground cursor-pointer";
    const tabBtnClass = (active: boolean) => `flex-1 py-1.5 text-[10px] font-bold font-heading uppercase tracking-wider transition-all rounded-md ${active ? 'bg-foreground text-background shadow-sm' : 'text-muted-foreground hover:text-foreground'}`;
    const sectionLabel = "text-[9px] font-bold uppercase tracking-wider text-muted-foreground px-1 mb-1";
    const hasSelection = selectedNodes.length > 0;
    const selVariant = selNode?.variant;

    return (
        <div className="flex flex-col min-h-dvh bg-background">
            <Header />
            <div className="flex-1 overflow-y-auto pb-24">
                <div className="max-w-[1400px] mx-auto px-4 py-6">
                    <button onClick={() => router.push('/tools')} className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground mb-3 transition-colors font-heading"><ArrowLeft className="w-3.5 h-3.5" />Back to Tools</button>
                    <div className="flex items-center justify-between mb-3">
                        <div>
                            <h1 className="text-lg font-bold font-heading text-foreground">AI Pipeline Designer</h1>
                            <p className="text-[10px] text-muted-foreground mt-0.5">Drag · Double-click to edit · Right-click color · Ctrl+C/V</p>
                        </div>
                        <div className="flex items-center gap-1.5">
                            <button onClick={toggleGrid} className={`px-2.5 py-1.5 rounded-md text-[10px] font-semibold font-heading transition-all ${showGrid ? 'bg-brand-blue text-white' : 'bg-muted text-muted-foreground hover:text-foreground'}`} title="Toggle Grid"><Grid3X3 className="w-3.5 h-3.5" /></button>
                            <button onClick={toggleConnect} className={`px-2.5 py-1.5 rounded-md text-[10px] font-semibold font-heading transition-all ${connectMode ? 'bg-brand-blue text-white' : 'bg-muted text-muted-foreground hover:text-foreground'}`} title="Connect"><Link className="w-3.5 h-3.5" /></button>
                            <button onClick={resetLayout} className="px-2.5 py-1.5 rounded-md bg-muted text-muted-foreground hover:text-foreground text-[10px] font-semibold font-heading transition-all" title="Reset"><RotateCcw className="w-3.5 h-3.5" /></button>
                            <div className="w-px h-5 bg-border mx-1" />
                            <button onClick={() => { setShowSaveDialog(true); setSaveName(''); }} className="px-2.5 py-1.5 rounded-md bg-brand-green text-white text-[10px] font-semibold font-heading hover:opacity-90 transition-all flex items-center gap-1"><Save className="w-3 h-3" /> Save</button>
                            <button onClick={() => setShowTemplates(!showTemplates)} className={`px-2.5 py-1.5 rounded-md text-[10px] font-semibold font-heading transition-all flex items-center gap-1 ${showTemplates ? 'bg-brand-blue text-white' : 'bg-foreground text-background hover:opacity-90'}`}><FolderOpen className="w-3 h-3" /> Templates</button>
                            <div className="w-px h-5 bg-border mx-1" />
                            <button onClick={exportPNG} className="px-2.5 py-1.5 rounded-md bg-brand-orange text-white text-[10px] font-semibold font-heading hover:opacity-90 transition-all flex items-center gap-1"><Download className="w-3 h-3" /> PNG</button>
                            <button onClick={exportPDF} className="px-2.5 py-1.5 rounded-md bg-brand-orange text-white text-[10px] font-semibold font-heading hover:opacity-90 transition-all flex items-center gap-1"><FileDown className="w-3 h-3" /> PDF</button>
                        </div>
                    </div>
                    <div className="flex gap-3">
                        <div className="w-[210px] flex-shrink-0 bg-card border border-border rounded-xl overflow-hidden shadow-sm self-start max-h-[calc(100vh-160px)] overflow-y-auto">
                            {/* === Add Element Tabs === */}
                            <div className="flex gap-0.5 p-1.5 border-b border-border bg-muted/30">
                                <button onClick={() => setPanelTab('shapes')} className={tabBtnClass(panelTab === 'shapes')}>Shapes</button>
                                <button onClick={() => setPanelTab('nodes')} className={tabBtnClass(panelTab === 'nodes')}>Nodes</button>
                                <button onClick={() => setPanelTab('text')} className={tabBtnClass(panelTab === 'text')}>Text</button>
                            </div>
                            <div className="p-2 space-y-0.5">
                                {panelTab === 'shapes' && (<>
                                    <button onClick={() => addShape('rect')} className={panelBtnClass}><Square className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" /> Rectangle</button>
                                    <button onClick={() => addShape('rounded-rect')} className={panelBtnClass}><RectangleHorizontal className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" /> Rounded Rect</button>
                                    <button onClick={() => addShape('circle')} className={panelBtnClass}><Circle className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" /> Ellipse</button>
                                    <button onClick={() => addShape('diamond')} className={panelBtnClass}><Diamond className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" /> Diamond</button>
                                    <button onClick={() => addShape('hexagon')} className={panelBtnClass}><Hexagon className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" /> Hexagon</button>
                                </>)}
                                {panelTab === 'nodes' && (<>
                                    <button onClick={() => addNodeOfType('headerNode')} className={panelBtnClass}><span className="w-3.5 h-3.5 rounded-sm border border-muted-foreground bg-muted flex-shrink-0" /> Header Node</button>
                                    <button onClick={() => addNodeOfType('minimal')} className={panelBtnClass}><span className="w-3.5 h-3.5 rounded-sm bg-muted-foreground flex-shrink-0" /> Minimal</button>
                                    <button onClick={() => addNodeOfType('plain')} className={panelBtnClass}><span className="w-3.5 h-3.5 rounded-sm border border-muted-foreground flex-shrink-0" /> Plain</button>
                                    <button onClick={() => addNodeOfType('container')} className={panelBtnClass}><span className="w-3.5 h-3.5 rounded-sm border border-dashed border-muted-foreground flex-shrink-0" /> Container</button>
                                    <button onClick={() => addNodeOfType('colLabel')} className={panelBtnClass}><span className="w-3.5 h-1 bg-muted-foreground rounded flex-shrink-0" /> Label</button>
                                    <button onClick={() => addNodeOfType('title')} className={panelBtnClass}><span className="w-3.5 h-1.5 bg-foreground rounded flex-shrink-0" /> Title</button>
                                </>)}
                                {panelTab === 'text' && (<>
                                    <button onClick={() => addTextBox()} className={panelBtnClass}><Type className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" /> Text Box</button>
                                </>)}
                            </div>

                            {/* === Unified Properties Panel === */}
                            {hasSelection && (
                                <div className="border-t border-border">
                                    <div className="px-2 py-2">
                                        <p className="text-[9px] font-bold uppercase tracking-wider text-muted-foreground px-1 mb-2">
                                            Properties {selectedNodes.length > 1 ? `(${selectedNodes.length} selected)` : selNode ? `— ${selNode.type}` : ''}
                                        </p>

                                        {/* Color */}
                                        <div className="px-1 mb-3">
                                            <p className={sectionLabel}>Color</p>
                                            <div className="flex gap-1 flex-wrap">
                                                {colorSwatches.map(s => (
                                                    <button
                                                        key={s.variant}
                                                        onClick={() => setVariant(s.variant)}
                                                        className={`w-5 h-5 rounded-sm border hover:scale-110 transition-transform ${selVariant === s.variant ? 'border-foreground ring-1 ring-foreground ring-offset-1' : 'border-black/10'}`}
                                                        style={{ background: s.color }}
                                                        title={s.label}
                                                    />
                                                ))}
                                            </div>
                                        </div>

                                        {/* Fill toggle */}
                                        {selNode && (selNode.type === 'shape' || selNode.type === 'textbox') && (
                                            <div className="px-1 mb-3">
                                                <p className={sectionLabel}>Fill</p>
                                                <div className="flex gap-0.5">
                                                    <button onClick={() => setNodeProp('filled', true)} className={`flex-1 py-1 rounded text-[10px] font-heading transition-all ${selNode.filled ? 'bg-brand-blue text-white' : 'bg-muted/60 text-muted-foreground hover:bg-muted'}`}>Filled</button>
                                                    <button onClick={() => setNodeProp('filled', false)} className={`flex-1 py-1 rounded text-[10px] font-heading transition-all ${!selNode.filled ? 'bg-brand-blue text-white' : 'bg-muted/60 text-muted-foreground hover:bg-muted'}`}>Outline</button>
                                                </div>
                                            </div>
                                        )}

                                        {/* Font Size */}
                                        <div className="px-1 mb-3">
                                            <p className={sectionLabel}>Font Size</p>
                                            <div className="flex gap-0.5">
                                                {[8, 9, 10, 11, 12, 14].map(s => (
                                                    <button key={s} onClick={() => setNodeProp('fontSize', s)} className={`flex-1 py-1 rounded text-[10px] font-heading transition-all ${selNode?.fontSize === s ? 'bg-brand-blue text-white' : 'bg-muted/60 text-muted-foreground hover:bg-muted'}`}>{s}</button>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Font Weight */}
                                        <div className="px-1 mb-3">
                                            <p className={sectionLabel}>Weight</p>
                                            <div className="flex gap-0.5">
                                                {[{ l: 'Light', v: '300' }, { l: 'Normal', v: '400' }, { l: 'Semi', v: '600' }, { l: 'Bold', v: '700' }].map(w => (
                                                    <button key={w.v} onClick={() => setNodeProp('fontWeight', w.v)} className={`flex-1 py-1 rounded text-[10px] font-heading transition-all ${selNode?.fontWeight === w.v ? 'bg-brand-blue text-white' : 'bg-muted/60 text-muted-foreground hover:bg-muted'}`}>{w.l}</button>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Text Alignment 3x3 grid */}
                                        <div className="px-1 mb-2">
                                            <p className={sectionLabel}>Alignment</p>
                                            <div className="grid grid-cols-3 gap-0.5 w-fit">
                                                {(['top', 'middle', 'bottom'] as const).flatMap(v =>
                                                    (['left', 'center', 'right'] as const).map(h => {
                                                        const isH = selNode?.textAlign === h || (!selNode?.textAlign && h === 'left');
                                                        const isV = selNode?.textVAlign === v || (!selNode?.textVAlign && v === 'middle');
                                                        const active = isH && isV;
                                                        const sym: Record<string, string> = {
                                                            'top-left': '↖', 'top-center': '↑', 'top-right': '↗',
                                                            'middle-left': '←', 'middle-center': '·', 'middle-right': '→',
                                                            'bottom-left': '↙', 'bottom-center': '↓', 'bottom-right': '↘'
                                                        };
                                                        return (
                                                            <button
                                                                key={`${v}-${h}`}
                                                                onClick={() => { setNodeProp('textAlign', h); setNodeProp('textVAlign', v); }}
                                                                className={`w-7 h-7 rounded text-[11px] font-bold transition-all ${active ? 'bg-brand-blue text-white shadow-sm' : 'bg-muted/60 text-muted-foreground hover:bg-muted hover:text-foreground'}`}
                                                                title={`${v}-${h}`}
                                                            >{sym[`${v}-${h}`]}</button>
                                                        );
                                                    })
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                        <div className="flex-1 min-w-0">
                            <div ref={containerRef} className="relative bg-white rounded-xl border border-border overflow-hidden shadow-sm">
                                <canvas ref={canvasRef} style={{ display: 'block', touchAction: 'none' }} />
                                <textarea ref={editorRef} className="absolute hidden bg-white border border-brand-orange px-1 py-0.5 text-[10px] z-[1000] outline-none shadow-md resize-none font-heading" onBlur={closeEditor} onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); closeEditor(); } }} />
                                <div ref={menuRef} className="absolute hidden bg-card border border-border rounded-lg shadow-xl py-1.5 z-[2000] min-w-[140px]">{colorSwatches.map((s) => (<button key={s.variant} onClick={() => setVariant(s.variant)} className="w-full flex items-center gap-2.5 px-3 py-1.5 text-xs font-heading hover:bg-muted transition-colors"><span className="w-3 h-3 rounded-sm border border-black/10" style={{ background: s.color }} />{s.label}</button>))}</div>
                                {connectMode && <div className="absolute top-2 left-2 bg-brand-blue text-white px-2.5 py-1 rounded-md text-[10px] font-heading font-semibold shadow-lg animate-pulse">Click source → Click target</div>}
                            </div>
                        </div>
                    </div>
                    {showSaveDialog && (<div className="fixed inset-0 bg-black/40 z-[4000] flex items-center justify-center backdrop-blur-sm" onClick={() => setShowSaveDialog(false)}><div className="bg-card border border-border rounded-2xl shadow-2xl p-6 w-full max-w-sm mx-4" onClick={e => e.stopPropagation()}><div className="flex items-center justify-between mb-4"><h3 className="text-sm font-bold font-heading text-foreground">Save as Template</h3><button onClick={() => setShowSaveDialog(false)} className="text-muted-foreground hover:text-foreground transition-colors"><X className="w-4 h-4" /></button></div><input type="text" value={saveName} onChange={e => setSaveName(e.target.value)} placeholder="Template name…" className="w-full px-3 py-2 rounded-lg border border-border bg-background text-sm font-heading outline-none focus:border-brand-orange transition-colors" autoFocus onKeyDown={e => { if (e.key === 'Enter') saveTemplate(saveName); }} /><div className="flex justify-end gap-2 mt-4"><button onClick={() => setShowSaveDialog(false)} className="px-4 py-2 rounded-lg text-xs font-semibold font-heading text-muted-foreground hover:text-foreground transition-colors">Cancel</button><button onClick={() => saveTemplate(saveName)} className="px-4 py-2 rounded-lg bg-brand-green text-white text-xs font-semibold font-heading hover:opacity-90 transition-all">Save</button></div></div></div>)}
                    {showTemplates && (<div className="mt-3 bg-card border border-border rounded-2xl shadow-lg overflow-hidden"><div className="flex items-center justify-between px-4 py-3 border-b border-border"><h3 className="text-sm font-bold font-heading text-foreground">Saved Templates</h3><button onClick={() => setShowTemplates(false)} className="text-muted-foreground hover:text-foreground transition-colors"><X className="w-4 h-4" /></button></div>{templates.length === 0 ? (<div className="px-4 py-8 text-center text-xs text-muted-foreground font-heading">No saved templates yet.</div>) : (<div className="divide-y divide-border max-h-[280px] overflow-y-auto">{templates.map(tmpl => (<div key={tmpl.id} className="flex items-center gap-3 px-4 py-2.5 hover:bg-muted/50 transition-colors group"><div className="flex-1 min-w-0">{renamingId === tmpl.id ? (<div className="flex items-center gap-1.5"><input type="text" value={renameValue} onChange={e => setRenameValue(e.target.value)} className="flex-1 px-2 py-1 rounded border border-border bg-background text-xs font-heading outline-none focus:border-brand-orange" autoFocus onKeyDown={e => { if (e.key === 'Enter') renameTemplate(tmpl.id, renameValue); if (e.key === 'Escape') setRenamingId(null); }} /><button onClick={() => renameTemplate(tmpl.id, renameValue)} className="text-brand-green hover:opacity-80"><Check className="w-3.5 h-3.5" /></button><button onClick={() => setRenamingId(null)} className="text-muted-foreground hover:text-foreground"><X className="w-3.5 h-3.5" /></button></div>) : (<><p className="text-xs font-semibold font-heading text-foreground truncate">{tmpl.name}</p><p className="text-[10px] text-muted-foreground font-heading mt-0.5">{tmpl.nodes.length} nodes · {tmpl.connections.length} conns</p></>)}</div>{renamingId !== tmpl.id && (<div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity"><button onClick={() => loadTemplate(tmpl)} className="px-2 py-1 rounded-md bg-brand-blue text-white text-[10px] font-semibold font-heading hover:opacity-90">Load</button><button onClick={() => overwriteTemplate(tmpl)} className="px-2 py-1 rounded-md bg-brand-orange text-white text-[10px] font-semibold font-heading hover:opacity-90">Update</button><button onClick={() => { setRenamingId(tmpl.id); setRenameValue(tmpl.name); }} className="p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted"><Pencil className="w-3 h-3" /></button><button onClick={() => deleteTemplate(tmpl.id)} className="p-1 rounded-md text-muted-foreground hover:text-red-500 hover:bg-red-50"><Trash2 className="w-3 h-3" /></button></div>)}</div>))}</div>)}</div>)}
                </div>
            </div>
            {toast && (<div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-foreground text-background px-5 py-2.5 rounded-md text-xs font-heading z-[3000] pointer-events-none shadow-lg animate-in fade-in slide-in-from-bottom-2 duration-200">{toast}</div>)}
        </div>
    );

};