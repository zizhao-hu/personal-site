#!/bin/bash
# Autonomous Claude Code Runner

# Load nvm and set PATH to use WSL node/claude
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Override PATH to prefer nvm node over Windows
export PATH="$HOME/.nvm/versions/node/v20.20.0/bin:$PATH"

cd /mnt/c/Users/zizha/development/personal-site

echo "=== Starting Autonomous Claude Code Execution ==="
echo "Time: $(date)"
echo "Claude: $(which claude)"
echo "Working directory: $(pwd)"
echo ""

# Combine rules and task
PROMPT=$(cat ~/ClaudeNightsWatch/rules.md ~/ClaudeNightsWatch/task.md)

echo "=== Executing with Claude CLI ==="
echo ""

# Use --print for non-interactive output
claude --dangerously-skip-permissions --print -p "$PROMPT"

echo ""
echo "=== Execution Complete ==="
echo "Time: $(date)"
