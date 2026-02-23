const fs = require('fs');
const content = fs.readFileSync('build-stderr.txt', 'utf-8');
const lines = content.split('\n');
const out = lines.map((l, i) => `${i}: ${l}`).join('\n');
fs.writeFileSync('build-errors-readable.txt', out, 'utf-8');
console.log(`Wrote ${lines.length} lines to build-errors-readable.txt`);
