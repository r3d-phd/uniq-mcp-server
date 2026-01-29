#!/bin/bash
# UniQ-MCP GitHub Setup Script
# Run this on your local machine to push to GitHub

set -e

echo "=============================================="
echo "  UniQ-MCP GitHub Repository Setup"
echo "=============================================="

cd ~/Downloads/uniq-mcp-server

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git branch -m main
fi

# Create .gitignore if not exists
if [ ! -f ".gitignore" ]; then
    echo "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
ENV/

# Environment
.env
*.env.local

# IDE
.idea/
.vscode/
*.swp
*.swo

# ChromaDB data
chroma_data/

# Logs
*.log

# OS
.DS_Store
Thumbs.db
EOF
fi

# Add all files
echo "Adding files..."
git add -A

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "No changes to commit"
else
    echo "Committing changes..."
    git commit -m "UniQ-MCP v13: Quantum Circuit Synthesis Server

Features:
- 9 MCP tools for quantum circuit operations
- Airlock integration for local GPU inference (Mistral 7B)
- ChromaDB episodic memory for learning
- Curriculum-based learning system
- MQT QCEC circuit verification
- LaTeX export for publications
- Comprehensive benchmark suite
- Research usage guide"
fi

# Check if remote exists
if git remote get-url origin &>/dev/null; then
    echo "Remote 'origin' already exists"
    echo "Pushing to existing remote..."
    git push -u origin main
else
    echo "Creating GitHub repository..."
    gh repo create uniq-mcp-server --private --source=. --push \
        --description "UniQ-MCP v13: Quantum Circuit Synthesis Server with SOAR Framework, Episodic Memory, and Local GPU Inference"
fi

echo ""
echo "=============================================="
echo "  ✓ GitHub repository setup complete!"
echo "=============================================="
echo ""
echo "Repository URL: https://github.com/$(gh api user -q .login)/uniq-mcp-server"
echo ""
