const { execSync } = require('child_process');
try {
    const result = execSync('npx next build', { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] });
    require('fs').writeFileSync('build-stdout.txt', result, 'utf-8');
} catch (e) {
    require('fs').writeFileSync('build-stdout.txt', e.stdout || '', 'utf-8');
    require('fs').writeFileSync('build-stderr.txt', e.stderr || '', 'utf-8');
    console.log('Build failed, check build-stdout.txt and build-stderr.txt');
}
