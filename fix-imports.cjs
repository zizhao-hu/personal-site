const fs = require('fs');
const path = require('path');

function walkDir(dir) {
    const results = [];
    const list = fs.readdirSync(dir);
    list.forEach(file => {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);
        if (stat.isDirectory()) {
            results.push(...walkDir(filePath));
        } else if (filePath.endsWith('.tsx') || filePath.endsWith('.ts')) {
            results.push(filePath);
        }
    });
    return results;
}

const files = walkDir(path.join('src', 'app'));
let count = 0;
for (const file of files) {
    let content = fs.readFileSync(file, 'utf-8');
    if (content.includes('@/pages/')) {
        content = content.replace(/@\/pages\//g, '@/views/');
        fs.writeFileSync(file, content, 'utf-8');
        count++;
        console.log('Fixed:', file);
    }
}
console.log(`\nFixed ${count} files`);
