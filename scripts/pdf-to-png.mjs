import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { createCanvas } from 'canvas';
import { getDocument } from 'pdfjs-dist/legacy/build/pdf.mjs';

const PDF_PATH = 'viterbi.pdf';
const OUT_PATH = 'public/templates/viterbi-bg.png';
const SCALE = 3; // 3x for high-res

async function main() {
    const data = new Uint8Array(readFileSync(PDF_PATH));
    const doc = await getDocument({ data, useSystemFonts: true }).promise;
    const page = await doc.getPage(1);
    const viewport = page.getViewport({ scale: SCALE });

    const canvas = createCanvas(viewport.width, viewport.height);
    const ctx = canvas.getContext('2d');

    await page.render({
        canvasContext: ctx,
        viewport,
    }).promise;

    mkdirSync('public/templates', { recursive: true });
    const buf = canvas.toBuffer('image/png');
    writeFileSync(OUT_PATH, buf);
    console.log(`Saved ${OUT_PATH} (${viewport.width}x${viewport.height})`);
}

main().catch(console.error);
