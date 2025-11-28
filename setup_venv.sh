#!/bin/bash
# setup_venv.sh - Automated setup script using Python venv

set -e  # Exit on any error

echo "🚀 Setting up Medulloblastoma G3/G4 Classification Environment (venv)"
echo "=================================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python $required_version+ required, found $python_version"
    echo "Please install Python $required_version or higher"
    exit 1
fi

echo "✅ Python version: $python_version (>= $required_version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import medulloblastoma
from medulloblastoma.dataset import download_data
from medulloblastoma.features import load_data
from medulloblastoma.plots import plot_umap_binary
print('✅ All modules imported successfully!')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To activate the environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
echo "To get started:"
echo "  jupyter lab notebooks/medulloblastoma-analysis.ipynb"
echo ""
echo "For more information, see README.md"
