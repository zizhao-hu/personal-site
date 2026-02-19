import { Header } from '@/components/custom/header';
import { useState, useRef, useCallback } from 'react';
import { ArrowLeft, Plus, Trash2, ChevronLeft, ChevronRight, Type, AlignLeft, Square, Circle, Image, Download, Copy, RotateCcw, GripVertical, Minus, Table, FileText } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import pptxgen from 'pptxgenjs';
import { jsPDF } from 'jspdf';

/* ─── Brand ─── */
const B = {
    dark: '#141413', light: '#faf9f5', mid: '#b0aea5', lgray: '#e8e6dc',
    orange: '#d97757', blue: '#6a9bcc', green: '#788c5d', clay: '#c2968a',
};

/* ─── Viterbi Theme ─── */
const VITERBI_BG = '__viterbi__';
const VITERBI = {
    cardinal: '#990000',
    gold: '#FFCC00',
    white: '#FFFFFF',
    bg: '#F8F8F8',
    footerRatio: 0.14, // footer takes 14% of slide height
    goldLineH: 4,       // px height of gold accent line
};

const isViterbiBg = (bg: string) => bg === VITERBI_BG;

/* ─── Types ─── */
interface SlideElement {
    id: string;
    type: 'title' | 'subtitle' | 'body' | 'bullet' | 'box' | 'circle' | 'image' | 'table' | 'divider';
    x: number; y: number; w: number; h: number;
    text?: string;
    color?: string;
    bg?: string;
    fontSize?: number;
    fontWeight?: string;
    textAlign?: 'left' | 'center' | 'right';
    borderRadius?: number;
    rows?: string[][];
    src?: string;
}

interface Slide {
    id: string;
    label: string;
    elements: SlideElement[];
    bg: string;
}

let _id = 0;
const uid = () => `el_${++_id}_${Date.now()}`;
const sid = () => `sl_${++_id}_${Date.now()}`;

/* ─── Aspect Ratio: 16:9 ─── */
const SW = 960;
const SH = 540;

/* ─── Template Slides ─── */
function createTemplateSlides(): Slide[] {
    return [
        {
            id: sid(), label: 'Title', bg: B.dark, elements: [
                { id: uid(), type: 'title', x: 80, y: 160, w: 800, h: 60, text: 'Your Research Title Here', color: B.light, fontSize: 36, fontWeight: '700', textAlign: 'center' },
                { id: uid(), type: 'subtitle', x: 80, y: 240, w: 800, h: 30, text: 'Author Name · Lab Name · Conference 2025', color: B.orange, fontSize: 18, fontWeight: '400', textAlign: 'center' },
                { id: uid(), type: 'divider', x: 380, y: 300, w: 200, h: 3, bg: B.orange },
                { id: uid(), type: 'body', x: 80, y: 320, w: 800, h: 24, text: 'University of Southern California', color: B.mid, fontSize: 14, fontWeight: '400', textAlign: 'center' },
            ],
        },
        {
            id: sid(), label: 'Motivation', bg: B.light, elements: [
                { id: uid(), type: 'title', x: 60, y: 30, w: 600, h: 40, text: 'Motivation & Problem', color: B.dark, fontSize: 28, fontWeight: '700', textAlign: 'left' },
                { id: uid(), type: 'divider', x: 60, y: 80, w: 120, h: 3, bg: B.orange },
                { id: uid(), type: 'bullet', x: 60, y: 110, w: 500, h: 160, text: '• Current methods suffer from X limitation\n• This leads to poor performance on Y\n• No existing work addresses Z', color: B.dark, fontSize: 16, fontWeight: '400', textAlign: 'left' },
                { id: uid(), type: 'box', x: 600, y: 100, w: 320, h: 200, bg: B.lgray, borderRadius: 12 },
                { id: uid(), type: 'body', x: 620, y: 160, w: 280, h: 80, text: '[Figure / Illustration]', color: B.mid, fontSize: 14, fontWeight: '400', textAlign: 'center' },
            ],
        },
        {
            id: sid(), label: 'Related Work', bg: B.light, elements: [
                { id: uid(), type: 'title', x: 60, y: 30, w: 600, h: 40, text: 'Related Work', color: B.dark, fontSize: 28, fontWeight: '700', textAlign: 'left' },
                { id: uid(), type: 'divider', x: 60, y: 80, w: 120, h: 3, bg: B.orange },
                { id: uid(), type: 'box', x: 60, y: 110, w: 260, h: 140, bg: '#f0ede4', borderRadius: 10 },
                { id: uid(), type: 'body', x: 75, y: 120, w: 230, h: 24, text: 'Approach A', color: B.dark, fontSize: 16, fontWeight: '600', textAlign: 'left' },
                { id: uid(), type: 'body', x: 75, y: 150, w: 230, h: 80, text: 'Key insight and limitation of this approach.', color: B.mid, fontSize: 13, fontWeight: '400', textAlign: 'left' },
                { id: uid(), type: 'box', x: 350, y: 110, w: 260, h: 140, bg: '#f0ede4', borderRadius: 10 },
                { id: uid(), type: 'body', x: 365, y: 120, w: 230, h: 24, text: 'Approach B', color: B.dark, fontSize: 16, fontWeight: '600', textAlign: 'left' },
                { id: uid(), type: 'body', x: 365, y: 150, w: 230, h: 80, text: 'Key insight and limitation of this approach.', color: B.mid, fontSize: 13, fontWeight: '400', textAlign: 'left' },
                { id: uid(), type: 'box', x: 640, y: 110, w: 260, h: 140, bg: '#f0ede4', borderRadius: 10 },
                { id: uid(), type: 'body', x: 655, y: 120, w: 230, h: 24, text: 'Approach C', color: B.dark, fontSize: 16, fontWeight: '600', textAlign: 'left' },
                { id: uid(), type: 'body', x: 655, y: 150, w: 230, h: 80, text: 'Key insight and limitation of this approach.', color: B.mid, fontSize: 13, fontWeight: '400', textAlign: 'left' },
                { id: uid(), type: 'bullet', x: 60, y: 280, w: 840, h: 80, text: '• Gap: None of these methods handle [specific challenge]\n• Our work bridges this gap by…', color: B.dark, fontSize: 15, fontWeight: '400', textAlign: 'left' },
            ],
        },
        {
            id: sid(), label: 'Method', bg: B.light, elements: [
                { id: uid(), type: 'title', x: 60, y: 30, w: 600, h: 40, text: 'Proposed Method', color: B.dark, fontSize: 28, fontWeight: '700', textAlign: 'left' },
                { id: uid(), type: 'divider', x: 60, y: 80, w: 120, h: 3, bg: B.orange },
                { id: uid(), type: 'box', x: 60, y: 110, w: 840, h: 260, bg: B.lgray, borderRadius: 12 },
                { id: uid(), type: 'body', x: 300, y: 210, w: 360, h: 40, text: '[Architecture Diagram]', color: B.mid, fontSize: 18, fontWeight: '400', textAlign: 'center' },
                { id: uid(), type: 'bullet', x: 60, y: 400, w: 840, h: 100, text: '• Step 1: Description of first component\n• Step 2: Description of second component\n• Step 3: How they combine for the final output', color: B.dark, fontSize: 15, fontWeight: '400', textAlign: 'left' },
            ],
        },
        {
            id: sid(), label: 'Results', bg: B.light, elements: [
                { id: uid(), type: 'title', x: 60, y: 30, w: 600, h: 40, text: 'Experimental Results', color: B.dark, fontSize: 28, fontWeight: '700', textAlign: 'left' },
                { id: uid(), type: 'divider', x: 60, y: 80, w: 120, h: 3, bg: B.orange },
                {
                    id: uid(), type: 'table', x: 60, y: 110, w: 840, h: 180, color: B.dark, fontSize: 13, rows: [
                        ['Method', 'Metric A', 'Metric B', 'Metric C'],
                        ['Baseline 1', '72.3', '68.1', '45.2'],
                        ['Baseline 2', '74.8', '70.5', '48.9'],
                        ['Ours', '78.6 ↑', '75.2 ↑', '53.1 ↑'],
                    ]
                },
                { id: uid(), type: 'bullet', x: 60, y: 320, w: 840, h: 80, text: '• Our method achieves state-of-the-art on all three metrics\n• +3.8 improvement on Metric A over best baseline', color: B.dark, fontSize: 15, fontWeight: '400', textAlign: 'left' },
            ],
        },
        {
            id: sid(), label: 'Ablation', bg: B.light, elements: [
                { id: uid(), type: 'title', x: 60, y: 30, w: 600, h: 40, text: 'Ablation Study', color: B.dark, fontSize: 28, fontWeight: '700', textAlign: 'left' },
                { id: uid(), type: 'divider', x: 60, y: 80, w: 120, h: 3, bg: B.orange },
                { id: uid(), type: 'box', x: 60, y: 110, w: 400, h: 250, bg: B.lgray, borderRadius: 12 },
                { id: uid(), type: 'body', x: 160, y: 210, w: 200, h: 30, text: '[Chart / Plot]', color: B.mid, fontSize: 14, fontWeight: '400', textAlign: 'center' },
                { id: uid(), type: 'bullet', x: 500, y: 110, w: 400, h: 250, text: '• Removing component A → -4.2%\n• Removing component B → -2.8%\n• Both components are essential\n• Scaling behavior is favorable', color: B.dark, fontSize: 15, fontWeight: '400', textAlign: 'left' },
            ],
        },
        {
            id: sid(), label: 'Conclusion', bg: B.dark, elements: [
                { id: uid(), type: 'title', x: 80, y: 100, w: 800, h: 50, text: 'Summary & Future Work', color: B.light, fontSize: 32, fontWeight: '700', textAlign: 'center' },
                { id: uid(), type: 'divider', x: 380, y: 165, w: 200, h: 3, bg: B.orange },
                { id: uid(), type: 'bullet', x: 120, y: 200, w: 720, h: 180, text: '• We proposed [method name] for [problem]\n• Achieved state-of-the-art results on [benchmarks]\n• Key insight: [one-sentence takeaway]\n\n• Future: Scaling to larger models, new domains', color: B.lgray, fontSize: 16, fontWeight: '400', textAlign: 'left' },
                { id: uid(), type: 'body', x: 80, y: 430, w: 800, h: 24, text: 'Thank you! Questions? → email@university.edu', color: B.orange, fontSize: 16, fontWeight: '400', textAlign: 'center' },
            ],
        },
    ];
}

/* ─── Component ─── */
export const SlideMaker = () => {
    const navigate = useNavigate();
    const [slides, setSlides] = useState<Slide[]>(createTemplateSlides);
    const [currentIdx, setCurrentIdx] = useState(0);
    const [selected, setSelected] = useState<string | null>(null);
    const [dragging, setDragging] = useState<{ id: string; ox: number; oy: number } | null>(null);
    const [resizing, setResizing] = useState<{ id: string; ow: number; oh: number; sx: number; sy: number } | null>(null);
    const [editing, setEditing] = useState<string | null>(null);
    const [toast, setToast] = useState<string | null>(null);
    const slideRef = useRef<HTMLDivElement>(null);

    const slide = slides[currentIdx];
    const showToast = useCallback((m: string) => { setToast(m); setTimeout(() => setToast(null), 1800); }, []);

    const updateSlide = (fn: (s: Slide) => Slide) => {
        setSlides(prev => prev.map((s, i) => i === currentIdx ? fn({ ...s, elements: s.elements.map(e => ({ ...e })) }) : s));
    };

    const updateElement = (id: string, patch: Partial<SlideElement>) => {
        updateSlide(s => ({ ...s, elements: s.elements.map(e => e.id === id ? { ...e, ...patch } : e) }));
    };

    /* ── Drag ── */
    const onPointerDown = (e: React.PointerEvent, el: SlideElement) => {
        e.stopPropagation();
        if (editing === el.id) return;
        setSelected(el.id);
        const rect = slideRef.current?.getBoundingClientRect();
        if (!rect) return;
        const scale = rect.width / SW;
        setDragging({ id: el.id, ox: e.clientX / scale - el.x, oy: e.clientY / scale - el.y });
    };

    const onPointerMove = (e: React.PointerEvent) => {
        const rect = slideRef.current?.getBoundingClientRect();
        if (!rect) return;
        const scale = rect.width / SW;
        if (dragging) {
            const nx = Math.round(e.clientX / scale - dragging.ox);
            const ny = Math.round(e.clientY / scale - dragging.oy);
            updateElement(dragging.id, { x: nx, y: ny });
        }
        if (resizing) {
            const dw = Math.round((e.clientX - resizing.sx) / scale);
            const dh = Math.round((e.clientY - resizing.sy) / scale);
            updateElement(resizing.id, { w: Math.max(30, resizing.ow + dw), h: Math.max(10, resizing.oh + dh) });
        }
    };

    const onPointerUp = () => { setDragging(null); setResizing(null); };

    /* ── Add Elements ── */
    const addElement = (type: SlideElement['type']) => {
        const base: Partial<SlideElement> = { x: 100, y: 200, w: 300, h: 40 };
        const defs: Record<string, Partial<SlideElement>> = {
            title: { ...base, text: 'Title Text', fontSize: 28, fontWeight: '700', color: slide.bg === B.dark ? B.light : B.dark, textAlign: 'left' },
            subtitle: { ...base, text: 'Subtitle', fontSize: 18, fontWeight: '400', color: B.orange, textAlign: 'left' },
            body: { ...base, h: 60, text: 'Body text here', fontSize: 15, fontWeight: '400', color: slide.bg === B.dark ? B.lgray : B.dark, textAlign: 'left' },
            bullet: { ...base, h: 100, w: 400, text: '• Point one\n• Point two\n• Point three', fontSize: 15, fontWeight: '400', color: slide.bg === B.dark ? B.lgray : B.dark, textAlign: 'left' },
            box: { x: 200, y: 150, w: 300, h: 200, bg: B.lgray, borderRadius: 12 },
            circle: { x: 300, y: 200, w: 120, h: 120, bg: B.orange, borderRadius: 999 },
            divider: { x: 60, y: 200, w: 200, h: 3, bg: B.orange },
            image: { x: 200, y: 150, w: 300, h: 200, bg: '#ddd', text: '[Drop image here]', color: B.mid, fontSize: 14, textAlign: 'center' },
            table: { x: 60, y: 150, w: 500, h: 140, color: B.dark, fontSize: 13, rows: [['Col A', 'Col B', 'Col C'], ['val', 'val', 'val']] },
        };
        const el: SlideElement = { id: uid(), type, ...defs[type] } as SlideElement;
        updateSlide(s => ({ ...s, elements: [...s.elements, el] }));
        setSelected(el.id);
    };

    const deleteSelected = () => {
        if (!selected) return;
        updateSlide(s => ({ ...s, elements: s.elements.filter(e => e.id !== selected) }));
        setSelected(null);
        showToast('Deleted');
    };

    const duplicateSelected = () => {
        if (!selected) return;
        const el = slide.elements.find(e => e.id === selected);
        if (!el) return;
        const nEl = { ...el, id: uid(), x: el.x + 20, y: el.y + 20 };
        updateSlide(s => ({ ...s, elements: [...s.elements, nEl] }));
        setSelected(nEl.id);
        showToast('Duplicated');
    };

    /* ── Slides nav ── */
    const addSlide = () => {
        const ns: Slide = {
            id: sid(), label: `Slide ${slides.length + 1}`, bg: B.light, elements: [
                { id: uid(), type: 'title', x: 60, y: 30, w: 600, h: 40, text: 'New Slide', color: B.dark, fontSize: 28, fontWeight: '700', textAlign: 'left' },
                { id: uid(), type: 'divider', x: 60, y: 80, w: 120, h: 3, bg: B.orange },
            ]
        };
        setSlides(p => [...p, ns]);
        setCurrentIdx(slides.length);
        setSelected(null);
    };

    const deleteSlide = () => {
        if (slides.length <= 1) return;
        setSlides(p => p.filter((_, i) => i !== currentIdx));
        setCurrentIdx(Math.max(0, currentIdx - 1));
        setSelected(null);
    };

    const resetAll = () => { setSlides(createTemplateSlides()); setCurrentIdx(0); setSelected(null); showToast('Reset to template'); };

    /* ── Export to PPTX ── */
    const exportPPTX = async () => {
        const pres = new pptxgen();
        pres.layout = 'LAYOUT_WIDE'; // 13.33" x 7.5"
        const PW = 13.33; // presentation width inches
        const PH = 7.5;   // presentation height inches

        const pxToInX = (px: number) => (px / SW) * PW;
        const pxToInY = (px: number) => (px / SH) * PH;
        const hexClean = (hex: string) => hex.replace('#', '');

        slides.forEach(s => {
            const pSlide = pres.addSlide();

            // Background
            if (isViterbiBg(s.bg)) {
                pSlide.background = { color: hexClean(VITERBI.bg) };
                const footerH = PH * VITERBI.footerRatio;
                const footerY = PH - footerH;
                const goldH = (VITERBI.goldLineH / SH) * PH;

                // Gold accent line
                pSlide.addShape(pres.ShapeType.rect, {
                    x: 0, y: footerY, w: PW, h: goldH,
                    fill: { color: hexClean(VITERBI.gold) },
                    line: { width: 0 },
                });
                // Cardinal footer band
                pSlide.addShape(pres.ShapeType.rect, {
                    x: 0, y: footerY + goldH, w: PW, h: footerH - goldH,
                    fill: { color: hexClean(VITERBI.cardinal) },
                    line: { width: 0 },
                });
                // Footer text — left
                pSlide.addText([
                    { text: 'USC ', options: { bold: true, fontSize: 11, color: 'FFFFFF' } },
                    { text: 'Viterbi', options: { bold: false, fontSize: 11, color: 'FFFFFF' } },
                ], {
                    x: 0.3, y: footerY + goldH + 0.05, w: 3, h: 0.3,
                    valign: 'top',
                });
                pSlide.addText('School of Engineering', {
                    x: 0.3, y: footerY + goldH + 0.35, w: 3, h: 0.2,
                    fontSize: 8, color: 'FFFFFF', valign: 'top',
                });
                // Footer text — right
                pSlide.addText('University of Southern California', {
                    x: PW - 4, y: footerY + goldH + 0.15, w: 3.7, h: 0.3,
                    fontSize: 9, color: 'FFFFFF', italic: true, align: 'right',
                });
                // USC shield watermark — top right
                pSlide.addShape(pres.ShapeType.ellipse, {
                    x: PW - 0.9, y: 0.2, w: 0.7, h: 0.7,
                    line: { color: hexClean(VITERBI.gold), width: 1.5 },
                    fill: { type: 'solid', color: hexClean(VITERBI.bg) },
                });
                pSlide.addText('USC', {
                    x: PW - 0.9, y: 0.2, w: 0.7, h: 0.7,
                    fontSize: 8, color: hexClean(VITERBI.gold), bold: true,
                    align: 'center', valign: 'middle', fontFace: 'Georgia',
                });
            } else {
                pSlide.background = { color: hexClean(s.bg) };
            }

            // Elements
            s.elements.forEach(el => {
                const x = pxToInX(el.x);
                const y = pxToInY(el.y);
                const w = pxToInX(el.w);
                const h = pxToInY(el.h);

                if (el.type === 'divider') {
                    pSlide.addShape(pres.ShapeType.rect, {
                        x, y, w, h,
                        fill: { color: hexClean(el.bg || B.orange) },
                        line: { width: 0 },
                    });
                } else if (el.type === 'box') {
                    pSlide.addShape(pres.ShapeType.rect, {
                        x, y, w, h,
                        fill: { color: hexClean(el.bg || B.lgray) },
                        rectRadius: el.borderRadius ? el.borderRadius / 100 : 0,
                        line: { width: 0 },
                    });
                } else if (el.type === 'circle') {
                    pSlide.addShape(pres.ShapeType.ellipse, {
                        x, y, w, h,
                        fill: { color: hexClean(el.bg || B.lgray) },
                        line: { width: 0 },
                    });
                } else if (el.type === 'table' && el.rows) {
                    const tableRows = el.rows.map((row, ri) =>
                        row.map(cell => ({
                            text: cell,
                            options: {
                                fontSize: (el.fontSize || 13) * 0.75,
                                color: hexClean(ri === 0 ? B.light : (el.color || B.dark)),
                                bold: ri === 0,
                                fill: { color: hexClean(ri === 0 ? B.dark : (ri % 2 === 0 ? '#f5f4ef' : B.light)) },
                                fontFace: ri === 0 ? 'Arial' : 'Georgia',
                                valign: 'middle' as const,
                            },
                        }))
                    );
                    pSlide.addTable(tableRows, {
                        x, y, w, h,
                        border: { type: 'solid', pt: 0.5, color: hexClean(B.lgray) },
                        colW: Array(el.rows[0]?.length || 1).fill(w / (el.rows[0]?.length || 1)),
                    });
                } else if (el.text) {
                    // Text elements (title, subtitle, body, bullet, image label)
                    const isBullet = el.type === 'bullet';
                    const fontSizePt = ((el.fontSize || 16) * 0.75);
                    const fontFace = (el.fontSize && el.fontSize >= 20) ? 'Arial' : 'Georgia';
                    pSlide.addText(el.text, {
                        x, y, w, h,
                        fontSize: fontSizePt,
                        fontFace,
                        color: hexClean(el.color || B.dark),
                        bold: el.fontWeight === '700' || el.fontWeight === '600',
                        align: (el.textAlign || 'left') as 'left' | 'center' | 'right',
                        valign: 'top',
                        wrap: true,
                        bullet: isBullet ? { type: 'bullet' } : undefined,
                        lineSpacingMultiple: 1.5,
                    });
                }
            });
        });

        await pres.writeFile({ fileName: 'research-slides.pptx' });
        showToast('Exported PPTX');
    };

    /* ── Export to PDF ── */
    const hexToRgb = (hex: string) => {
        const h = hex.replace('#', '');
        return { r: parseInt(h.substring(0, 2), 16), g: parseInt(h.substring(2, 4), 16), b: parseInt(h.substring(4, 6), 16) };
    };

    const exportPDF = () => {
        const pdf = new jsPDF({ orientation: 'landscape', unit: 'px', format: [SW, SH] });
        slides.forEach((s, si) => {
            if (si > 0) pdf.addPage([SW, SH], 'landscape');
            // Background
            if (isViterbiBg(s.bg)) {
                // Viterbi slide background
                const bgC = hexToRgb(VITERBI.bg);
                pdf.setFillColor(bgC.r, bgC.g, bgC.b);
                pdf.rect(0, 0, SW, SH, 'F');
                // Gold accent line
                const footerY = SH * (1 - VITERBI.footerRatio);
                const goldC = hexToRgb(VITERBI.gold);
                pdf.setFillColor(goldC.r, goldC.g, goldC.b);
                pdf.rect(0, footerY, SW, VITERBI.goldLineH, 'F');
                // Cardinal footer band
                const cardC = hexToRgb(VITERBI.cardinal);
                pdf.setFillColor(cardC.r, cardC.g, cardC.b);
                pdf.rect(0, footerY + VITERBI.goldLineH, SW, SH - footerY - VITERBI.goldLineH, 'F');
                // Footer text
                pdf.setTextColor(255, 255, 255);
                pdf.setFontSize(11);
                pdf.setFont('helvetica', 'bold');
                pdf.text('USC', 30, footerY + VITERBI.goldLineH + 22);
                pdf.setFont('helvetica', 'normal');
                pdf.text(' Viterbi', 30 + pdf.getTextWidth('USC'), footerY + VITERBI.goldLineH + 22);
                pdf.setFontSize(8);
                pdf.text('School of Engineering', 30, footerY + VITERBI.goldLineH + 34);
                pdf.setFontSize(9);
                pdf.text('University of Southern California', SW - 30, footerY + VITERBI.goldLineH + 28, { align: 'right' });
                // Gold shield watermark (simplified circle in top right)
                pdf.setDrawColor(goldC.r, goldC.g, goldC.b);
                pdf.setFillColor(goldC.r, goldC.g, goldC.b);
                pdf.circle(SW - 40, 40, 18, 'S');
                pdf.setFontSize(12);
                pdf.setTextColor(goldC.r, goldC.g, goldC.b);
                pdf.text('USC', SW - 52, 44);
            } else {
                const bg = hexToRgb(s.bg);
                pdf.setFillColor(bg.r, bg.g, bg.b);
                pdf.rect(0, 0, SW, SH, 'F');
            }
            // Elements
            s.elements.forEach(el => {
                if (el.type === 'divider') {
                    const c = hexToRgb(el.bg || B.orange);
                    pdf.setFillColor(c.r, c.g, c.b);
                    pdf.rect(el.x, el.y, el.w, el.h, 'F');
                } else if (el.type === 'box' || el.type === 'circle') {
                    const c = hexToRgb(el.bg || B.lgray);
                    pdf.setFillColor(c.r, c.g, c.b);
                    if (el.type === 'circle') {
                        pdf.ellipse(el.x + el.w / 2, el.y + el.h / 2, el.w / 2, el.h / 2, 'F');
                    } else {
                        pdf.roundedRect(el.x, el.y, el.w, el.h, el.borderRadius || 0, el.borderRadius || 0, 'F');
                    }
                } else if (el.type === 'table' && el.rows) {
                    const cols = el.rows[0]?.length || 1;
                    const colW = el.w / cols;
                    const rowH = el.h / el.rows.length;
                    el.rows.forEach((row, ri) => {
                        row.forEach((cell, ci) => {
                            const isHeader = ri === 0;
                            const cellBg = hexToRgb(isHeader ? B.dark : (ri % 2 === 0 ? '#f5f4ef' : B.light));
                            pdf.setFillColor(cellBg.r, cellBg.g, cellBg.b);
                            pdf.rect(el.x + ci * colW, el.y + ri * rowH, colW, rowH, 'FD');
                            const tc = hexToRgb(isHeader ? B.light : (el.color || B.dark));
                            pdf.setTextColor(tc.r, tc.g, tc.b);
                            pdf.setFontSize(el.fontSize || 13);
                            pdf.setFont('helvetica', isHeader ? 'bold' : 'normal');
                            pdf.text(cell, el.x + ci * colW + 8, el.y + ri * rowH + rowH / 2 + 4);
                        });
                    });
                } else if (el.text) {
                    const tc = hexToRgb(el.color || B.dark);
                    pdf.setTextColor(tc.r, tc.g, tc.b);
                    pdf.setFontSize(el.fontSize || 16);
                    pdf.setFont('helvetica', el.fontWeight === '700' ? 'bold' : (el.fontWeight === '600' ? 'bold' : 'normal'));
                    const lines = pdf.splitTextToSize(el.text, el.w);
                    let textX = el.x;
                    if (el.textAlign === 'center') textX = el.x + el.w / 2;
                    else if (el.textAlign === 'right') textX = el.x + el.w;
                    pdf.text(lines, textX, el.y + (el.fontSize || 16), { align: el.textAlign || 'left', lineHeightFactor: 1.5 });
                }
            });
        });
        pdf.save('research-slides.pdf');
        showToast('Exported PDF');
    };

    /* ── Render element ── */
    const renderElement = (el: SlideElement) => {
        const isSelected = selected === el.id;
        const isEditing = editing === el.id;
        const base: React.CSSProperties = {
            position: 'absolute', left: el.x, top: el.y, width: el.w, height: el.h,
            cursor: isEditing ? 'text' : 'move',
            outline: isSelected ? `2px solid ${B.orange}` : 'none',
            outlineOffset: 2,
            userSelect: isEditing ? 'text' : 'none',
        };

        if (el.type === 'divider') {
            return (
                <div key={el.id} style={{ ...base, backgroundColor: el.bg || B.orange, borderRadius: 2 }}
                    onClick={e => e.stopPropagation()} onPointerDown={e => onPointerDown(e, el)} />
            );
        }

        if (el.type === 'box' || el.type === 'circle') {
            return (
                <div key={el.id} style={{ ...base, backgroundColor: el.bg || B.lgray, borderRadius: el.borderRadius || 0 }}
                    onClick={e => e.stopPropagation()} onPointerDown={e => onPointerDown(e, el)}>
                    {isSelected && (
                        <div style={{ position: 'absolute', right: -4, bottom: -4, width: 10, height: 10, background: B.orange, borderRadius: 2, cursor: 'se-resize' }}
                            onPointerDown={e => { e.stopPropagation(); setResizing({ id: el.id, ow: el.w, oh: el.h, sx: e.clientX, sy: e.clientY }); }} />
                    )}
                </div>
            );
        }

        if (el.type === 'table' && el.rows) {
            return (
                <div key={el.id} style={{ ...base, overflow: 'hidden' }} onClick={e => e.stopPropagation()} onPointerDown={e => onPointerDown(e, el)}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: el.fontSize || 13, fontFamily: "'Lora', Georgia, serif" }}>
                        <tbody>
                            {el.rows.map((row, ri) => (
                                <tr key={ri}>
                                    {row.map((cell, ci) => (
                                        <td key={ci}
                                            contentEditable={isEditing}
                                            suppressContentEditableWarning
                                            onBlur={ev => {
                                                const newRows = el.rows!.map(r => [...r]);
                                                newRows[ri][ci] = ev.currentTarget.textContent || '';
                                                updateElement(el.id, { rows: newRows });
                                            }}
                                            style={{
                                                padding: '6px 10px', border: `1px solid ${B.lgray}`,
                                                backgroundColor: ri === 0 ? B.dark : (ri % 2 === 0 ? '#f5f4ef' : 'transparent'),
                                                color: ri === 0 ? B.light : (el.color || B.dark),
                                                fontWeight: ri === 0 ? 600 : 400,
                                                fontFamily: ri === 0 ? "'Poppins', Arial, sans-serif" : undefined,
                                            }}
                                        >{cell}</td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    {isSelected && (
                        <div className="flex gap-1" style={{ position: 'absolute', bottom: -24, left: 0 }}>
                            <button onClick={() => { const newRows = [...el.rows!, el.rows![0].map(() => '')]; updateElement(el.id, { rows: newRows, h: el.h + 30 }); }}
                                style={{ fontSize: 10, padding: '2px 6px', background: B.dark, color: B.light, borderRadius: 4 }}>+ Row</button>
                            <button onClick={() => { const newRows = el.rows!.map(r => [...r, '']); updateElement(el.id, { rows: newRows, w: el.w + 80 }); }}
                                style={{ fontSize: 10, padding: '2px 6px', background: B.dark, color: B.light, borderRadius: 4 }}>+ Col</button>
                        </div>
                    )}
                </div>
            );
        }

        // Text elements (title, subtitle, body, bullet, image)
        return (
            <div key={el.id} style={base} onClick={e => e.stopPropagation()} onPointerDown={e => onPointerDown(e, el)}
                onDoubleClick={() => setEditing(el.id)}>
                {isEditing ? (
                    <textarea value={el.text || ''} autoFocus
                        onChange={e => updateElement(el.id, { text: e.target.value })}
                        onBlur={() => setEditing(null)}
                        onKeyDown={e => { if (e.key === 'Escape') setEditing(null); }}
                        style={{
                            width: '100%', height: '100%', resize: 'none', border: 'none', outline: 'none',
                            background: 'rgba(255,255,255,0.05)', color: el.color || B.dark,
                            fontSize: el.fontSize || 16, fontWeight: el.fontWeight || '400',
                            textAlign: el.textAlign || 'left', padding: 0, margin: 0,
                            fontFamily: (el.fontSize && el.fontSize >= 20) ? "'Poppins', Arial, sans-serif" : "'Lora', Georgia, serif",
                            whiteSpace: 'pre-wrap', lineHeight: 1.5,
                        }}
                    />
                ) : (
                    <div style={{
                        width: '100%', height: '100%', color: el.color || B.dark,
                        fontSize: el.fontSize || 16, fontWeight: el.fontWeight || '400',
                        textAlign: el.textAlign || 'left', lineHeight: 1.5,
                        fontFamily: (el.fontSize && el.fontSize >= 20) ? "'Poppins', Arial, sans-serif" : "'Lora', Georgia, serif",
                        whiteSpace: 'pre-wrap', overflow: 'hidden',
                    }}>{el.text}</div>
                )}
                {isSelected && (
                    <div style={{ position: 'absolute', right: -4, bottom: -4, width: 10, height: 10, background: B.orange, borderRadius: 2, cursor: 'se-resize' }}
                        onPointerDown={e => { e.stopPropagation(); setResizing({ id: el.id, ow: el.w, oh: el.h, sx: e.clientX, sy: e.clientY }); }} />
                )}
            </div>
        );
    };

    /* ── Properties panel for selected element ── */
    const selectedEl = slide.elements.find(e => e.id === selected);

    const toolbarBtns: { icon: React.ReactNode; label: string; type: SlideElement['type'] }[] = [
        { icon: <Type className="w-3.5 h-3.5" />, label: 'Title', type: 'title' },
        { icon: <AlignLeft className="w-3.5 h-3.5" />, label: 'Body', type: 'body' },
        { icon: <Minus className="w-3.5 h-3.5" />, label: 'Bullets', type: 'bullet' },
        { icon: <Square className="w-3.5 h-3.5" />, label: 'Box', type: 'box' },
        { icon: <Circle className="w-3.5 h-3.5" />, label: 'Circle', type: 'circle' },
        { icon: <Image className="w-3.5 h-3.5" />, label: 'Image', type: 'image' },
        { icon: <Table className="w-3.5 h-3.5" />, label: 'Table', type: 'table' },
        { icon: <GripVertical className="w-3.5 h-3.5" />, label: 'Divider', type: 'divider' },
    ];

    return (
        <div className="flex flex-col min-h-dvh bg-background">
            <Header />
            <div className="flex-1 overflow-hidden flex flex-col"
                onPointerMove={onPointerMove} onPointerUp={onPointerUp}>

                {/* Top bar */}
                <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-muted/30">
                    <div className="flex items-center gap-3">
                        <button onClick={() => navigate('/tools')} className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 font-heading">
                            <ArrowLeft className="w-3.5 h-3.5" /> Tools
                        </button>
                        <span className="text-sm font-semibold font-heading text-foreground">Research Slide Maker</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <button onClick={resetAll} className="px-3 py-1.5 text-xs font-heading rounded-lg bg-muted hover:bg-muted/80 transition-colors flex items-center gap-1">
                            <RotateCcw className="w-3 h-3" /> Reset
                        </button>
                        <button onClick={exportPPTX} className="px-3 py-1.5 text-xs font-heading rounded-lg bg-muted hover:bg-muted/80 transition-colors flex items-center gap-1">
                            <Download className="w-3 h-3" /> PPTX
                        </button>
                        <button onClick={exportPDF} className="px-3 py-1.5 text-xs font-heading rounded-lg bg-brand-orange text-white hover:opacity-90 transition-colors flex items-center gap-1">
                            <FileText className="w-3 h-3" /> Export PDF
                        </button>
                    </div>
                </div>

                <div className="flex flex-1 overflow-hidden">
                    {/* ── Left: Slide Thumbnails ── */}
                    <div className="w-[140px] border-r border-border bg-muted/20 overflow-y-auto p-2 flex flex-col gap-2">
                        {slides.map((s, i) => (
                            <button key={s.id} onClick={() => { setCurrentIdx(i); setSelected(null); setEditing(null); }}
                                className={`relative group rounded-lg border-2 transition-all ${i === currentIdx ? 'border-brand-orange shadow-md' : 'border-border hover:border-muted-foreground/30'}`}>
                                <div className="text-[8px] font-heading text-muted-foreground px-1 pt-1 truncate">{i + 1}. {s.label}</div>
                                <div className="aspect-video rounded-b-md" style={{ background: isViterbiBg(s.bg) ? VITERBI.bg : s.bg, position: 'relative', overflow: 'hidden' }}>
                                    {/* Viterbi theme mini overlay */}
                                    {isViterbiBg(s.bg) && (
                                        <>
                                            <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: `${VITERBI.footerRatio * 100}%`, display: 'flex', flexDirection: 'column' }}>
                                                <div style={{ height: 1, background: VITERBI.gold }} />
                                                <div style={{ flex: 1, background: VITERBI.cardinal }} />
                                            </div>
                                            <div style={{ position: 'absolute', top: 2, right: 3, width: 6, height: 6, border: `1px solid ${VITERBI.gold}`, borderRadius: '50%' }} />
                                        </>
                                    )}
                                    {/* Mini preview — simplified */}
                                    {s.elements.slice(0, 4).map(el => (
                                        <div key={el.id} style={{
                                            position: 'absolute',
                                            left: `${(el.x / SW) * 100}%`, top: `${(el.y / SH) * 100}%`,
                                            width: `${(el.w / SW) * 100}%`, height: `${(el.h / SH) * 100}%`,
                                            backgroundColor: el.bg || (el.type === 'divider' ? el.bg : 'transparent'),
                                            borderRadius: el.borderRadius ? el.borderRadius * 0.13 : 0,
                                            overflow: 'hidden',
                                        }}>
                                            {el.text && <div style={{ fontSize: 2, color: el.color || B.dark, lineHeight: 1.2 }}>{el.text.slice(0, 20)}</div>}
                                        </div>
                                    ))}
                                </div>
                            </button>
                        ))}
                        <button onClick={addSlide} className="flex items-center justify-center gap-1 py-2 rounded-lg border border-dashed border-border hover:border-brand-orange text-xs text-muted-foreground hover:text-brand-orange transition-colors font-heading">
                            <Plus className="w-3 h-3" /> Add
                        </button>
                    </div>

                    {/* ── Center: Canvas ── */}
                    <div className="flex-1 flex flex-col items-center justify-center bg-[#2a2a2a] p-4 overflow-auto">
                        {/* Toolbar */}
                        <div className="flex items-center gap-1 mb-3 bg-background/90 backdrop-blur rounded-lg border border-border px-2 py-1.5 shadow-sm">
                            {toolbarBtns.map(t => (
                                <button key={t.type} onClick={() => addElement(t.type)} title={t.label}
                                    className="flex flex-col items-center gap-0.5 px-2 py-1 rounded hover:bg-muted transition-colors text-muted-foreground hover:text-foreground">
                                    {t.icon}
                                    <span className="text-[9px] font-heading">{t.label}</span>
                                </button>
                            ))}
                            <div className="w-px h-6 bg-border mx-1" />
                            {selected && (
                                <>
                                    <button onClick={duplicateSelected} title="Duplicate" className="p-1.5 rounded hover:bg-muted text-muted-foreground hover:text-foreground transition-colors">
                                        <Copy className="w-3.5 h-3.5" />
                                    </button>
                                    <button onClick={deleteSelected} title="Delete" className="p-1.5 rounded hover:bg-red-100 dark:hover:bg-red-900/30 text-muted-foreground hover:text-red-500 transition-colors">
                                        <Trash2 className="w-3.5 h-3.5" />
                                    </button>
                                </>
                            )}
                        </div>

                        {/* Slide Canvas */}
                        <div ref={slideRef} className="relative shadow-2xl" onClick={e => { if (e.target === e.currentTarget) { setSelected(null); setEditing(null); } }}
                            style={{ width: '100%', maxWidth: 960, aspectRatio: '16/9', background: isViterbiBg(slide.bg) ? VITERBI.bg : slide.bg, borderRadius: 4, overflow: 'hidden', position: 'relative' }}>
                            {/* Viterbi theme background overlay */}
                            {isViterbiBg(slide.bg) && (
                                <>
                                    {/* Gold shield watermark — top right */}
                                    <div style={{ position: 'absolute', top: 16, right: 20, width: 52, height: 52, pointerEvents: 'none', zIndex: 0 }}>
                                        <svg viewBox="0 0 60 60" width="52" height="52" style={{ opacity: 0.35 }}>
                                            <circle cx="30" cy="30" r="27" fill="none" stroke={VITERBI.gold} strokeWidth="2.5" />
                                            <text x="30" y="34" textAnchor="middle" fill={VITERBI.gold} fontSize="14" fontWeight="bold" fontFamily="serif">USC</text>
                                        </svg>
                                    </div>
                                    {/* Footer */}
                                    <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: `${VITERBI.footerRatio * 100}%`, display: 'flex', flexDirection: 'column', pointerEvents: 'none', zIndex: 0 }}>
                                        {/* Gold accent line */}
                                        <div style={{ height: VITERBI.goldLineH, background: VITERBI.gold, flexShrink: 0 }} />
                                        {/* Cardinal band */}
                                        <div style={{ flex: 1, background: VITERBI.cardinal, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 30px' }}>
                                            <div style={{ color: VITERBI.white }}>
                                                <div style={{ fontSize: 15, fontWeight: 600, letterSpacing: 0.3 }}>
                                                    <span style={{ fontWeight: 700 }}>USC</span> Viterbi
                                                </div>
                                                <div style={{ fontSize: 9, opacity: 0.9, marginTop: 1 }}>School of Engineering</div>
                                            </div>
                                            <div style={{ color: VITERBI.white, fontSize: 11, fontStyle: 'italic', opacity: 0.9 }}>
                                                University of Southern California
                                            </div>
                                        </div>
                                    </div>
                                </>
                            )}
                            {slide.elements.map(renderElement)}
                        </div>

                        {/* Slide nav */}
                        <div className="flex items-center gap-3 mt-3">
                            <button onClick={() => { setCurrentIdx(Math.max(0, currentIdx - 1)); setSelected(null); }} disabled={currentIdx === 0}
                                className="p-1.5 rounded-lg bg-background/80 border border-border disabled:opacity-30 hover:bg-background transition-colors">
                                <ChevronLeft className="w-4 h-4" />
                            </button>
                            <span className="text-xs text-white/70 font-heading">{currentIdx + 1} / {slides.length}</span>
                            <button onClick={() => { setCurrentIdx(Math.min(slides.length - 1, currentIdx + 1)); setSelected(null); }} disabled={currentIdx === slides.length - 1}
                                className="p-1.5 rounded-lg bg-background/80 border border-border disabled:opacity-30 hover:bg-background transition-colors">
                                <ChevronRight className="w-4 h-4" />
                            </button>
                            <button onClick={deleteSlide} disabled={slides.length <= 1} title="Delete slide"
                                className="p-1.5 rounded-lg bg-background/80 border border-border disabled:opacity-30 hover:bg-red-100 dark:hover:bg-red-900/30 text-muted-foreground hover:text-red-500 transition-colors ml-2">
                                <Trash2 className="w-3.5 h-3.5" />
                            </button>
                        </div>
                    </div>

                    {/* ── Right: Properties Panel ── */}
                    <div className="w-[220px] border-l border-border bg-muted/20 overflow-y-auto p-3">
                        <h3 className="text-xs font-semibold font-heading text-muted-foreground uppercase tracking-wider mb-3">Properties</h3>

                        {/* Slide props */}
                        <div className="mb-4">
                            <label className="text-[10px] font-heading text-muted-foreground uppercase block mb-1">Slide Label</label>
                            <input value={slide.label} onChange={e => setSlides(p => p.map((s, i) => i === currentIdx ? { ...s, label: e.target.value } : s))}
                                className="w-full px-2 py-1 text-xs rounded border border-border bg-background font-heading" />
                            <label className="text-[10px] font-heading text-muted-foreground uppercase block mb-1 mt-2">Slide BG</label>
                            <div className="flex gap-1.5 flex-wrap">
                                {[B.light, B.dark, B.lgray, '#ffffff'].map(c => (
                                    <button key={c} onClick={() => setSlides(p => p.map((s, i) => i === currentIdx ? { ...s, bg: c } : s))}
                                        className={`w-6 h-6 rounded border-2 ${slide.bg === c ? 'border-brand-orange' : 'border-border'}`}
                                        style={{ background: c }} />
                                ))}
                                {/* Viterbi theme button */}
                                <button
                                    onClick={() => setSlides(p => p.map((s, i) => i === currentIdx ? { ...s, bg: VITERBI_BG } : s))}
                                    className={`h-6 px-1.5 rounded border-2 text-[8px] font-heading font-bold flex items-center gap-0.5 ${slide.bg === VITERBI_BG ? 'border-brand-orange' : 'border-border'}`}
                                    style={{ background: `linear-gradient(to bottom, ${VITERBI.bg} 60%, ${VITERBI.gold} 60%, ${VITERBI.gold} 63%, ${VITERBI.cardinal} 63%)` }}
                                    title="USC Viterbi Theme"
                                >
                                    <span style={{ color: VITERBI.cardinal }}>V</span>
                                </button>
                            </div>
                        </div>

                        {selectedEl && (
                            <div className="border-t border-border pt-3">
                                <h4 className="text-[10px] font-heading text-muted-foreground uppercase mb-2">Element: {selectedEl.type}</h4>

                                {selectedEl.text !== undefined && (
                                    <div className="mb-2">
                                        <label className="text-[10px] font-heading text-muted-foreground block mb-0.5">Text</label>
                                        <textarea value={selectedEl.text} rows={3}
                                            onChange={e => updateElement(selectedEl.id, { text: e.target.value })}
                                            className="w-full px-2 py-1 text-xs rounded border border-border bg-background resize-none" />
                                    </div>
                                )}

                                {selectedEl.fontSize !== undefined && (
                                    <div className="mb-2">
                                        <label className="text-[10px] font-heading text-muted-foreground block mb-0.5">Font Size</label>
                                        <input type="number" value={selectedEl.fontSize}
                                            onChange={e => updateElement(selectedEl.id, { fontSize: +e.target.value })}
                                            className="w-full px-2 py-1 text-xs rounded border border-border bg-background" />
                                    </div>
                                )}

                                <div className="mb-2">
                                    <label className="text-[10px] font-heading text-muted-foreground block mb-0.5">Color</label>
                                    <div className="flex gap-1 flex-wrap">
                                        {[B.dark, B.light, B.orange, B.blue, B.green, B.clay, B.mid, B.lgray].map(c => (
                                            <button key={c} onClick={() => updateElement(selectedEl.id, selectedEl.bg !== undefined && !selectedEl.text ? { bg: c } : { color: c })}
                                                className={`w-5 h-5 rounded border ${(selectedEl.color === c || selectedEl.bg === c) ? 'border-brand-orange border-2' : 'border-border'}`}
                                                style={{ background: c }} />
                                        ))}
                                    </div>
                                </div>

                                {selectedEl.textAlign !== undefined && (
                                    <div className="mb-2">
                                        <label className="text-[10px] font-heading text-muted-foreground block mb-0.5">Align</label>
                                        <div className="flex gap-1">
                                            {(['left', 'center', 'right'] as const).map(a => (
                                                <button key={a} onClick={() => updateElement(selectedEl.id, { textAlign: a })}
                                                    className={`px-2 py-0.5 text-[10px] rounded font-heading ${selectedEl.textAlign === a ? 'bg-brand-orange text-white' : 'bg-muted text-muted-foreground'}`}>{a}</button>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                <div className="grid grid-cols-2 gap-1.5 mt-2">
                                    <div>
                                        <label className="text-[10px] font-heading text-muted-foreground block">X</label>
                                        <input type="number" value={selectedEl.x} onChange={e => updateElement(selectedEl.id, { x: +e.target.value })}
                                            className="w-full px-1.5 py-0.5 text-[10px] rounded border border-border bg-background" />
                                    </div>
                                    <div>
                                        <label className="text-[10px] font-heading text-muted-foreground block">Y</label>
                                        <input type="number" value={selectedEl.y} onChange={e => updateElement(selectedEl.id, { y: +e.target.value })}
                                            className="w-full px-1.5 py-0.5 text-[10px] rounded border border-border bg-background" />
                                    </div>
                                    <div>
                                        <label className="text-[10px] font-heading text-muted-foreground block">W</label>
                                        <input type="number" value={selectedEl.w} onChange={e => updateElement(selectedEl.id, { w: +e.target.value })}
                                            className="w-full px-1.5 py-0.5 text-[10px] rounded border border-border bg-background" />
                                    </div>
                                    <div>
                                        <label className="text-[10px] font-heading text-muted-foreground block">H</label>
                                        <input type="number" value={selectedEl.h} onChange={e => updateElement(selectedEl.id, { h: +e.target.value })}
                                            className="w-full px-1.5 py-0.5 text-[10px] rounded border border-border bg-background" />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {toast && (
                <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-foreground text-background px-5 py-2.5 rounded-md text-xs font-heading z-[3000] pointer-events-none shadow-lg animate-in fade-in slide-in-from-bottom-2 duration-200">
                    {toast}
                </div>
            )}
        </div>
    );
};
