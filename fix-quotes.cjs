const fs = require('fs');
const path = require('path');

function walk(dir) {
    const results = [];
    fs.readdirSync(dir).forEach(file => {
        const filePath = path.join(dir, file);
        if (fs.statSync(filePath).isDirectory()) {
            results.push(...walk(filePath));
        } else if (filePath.endsWith('.tsx') || filePath.endsWith('.ts')) {
            results.push(filePath);
        }
    });
    return results;
}

// Fix unescaped single quotes inside single-quoted strings in metadata
// The pattern: 'some text with Hu's inside single quotes'
// We need to convert them to double-quoted strings or escape the quotes
const files = walk(path.join('src', 'app'));
let count = 0;

for (const f of files) {
    let content = fs.readFileSync(f, 'utf-8');
    let changed = false;

    // Replace description values that use single quotes and contain possessives
    // Pattern: description: 'text with unescaped apostrophe'
    content = content.replace(
        /description:\s*'([^']*Hu)'s\s+([^']*)',/g,
        (match, before, after) => {
            changed = true;
            return `description: "Zizhao Hu's ${after}",`;
        }
    );

    // More general: fix any metadata string value using single quotes containing apostrophes
    // Replace 'something's something' with "something's something"
    content = content.replace(
        /:\s*'([^']*)'s\s+([^']*)',/g,
        (match, before, after) => {
            changed = true;
            return `: "${before}'s ${after}",`;
        }
    );

    if (changed) {
        fs.writeFileSync(f, content, 'utf-8');
        count++;
        console.log('Fixed:', f);
    }
}

console.log(`\nFixed ${count} files`);
