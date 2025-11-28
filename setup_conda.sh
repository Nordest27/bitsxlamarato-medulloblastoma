#!/bin/bash
# setup_conda.sh - Automated setup script using conda

set -e  # Exit on any error

echo "🚀 Setting up Medulloblastoma G3/G4 Classification Environment (conda)"
echo "======================================================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Found conda: $(conda --version)"

# Create conda environment
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo "🔧 Activating environment..."
conda activate medulloblastoma

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
echo "  conda activate medulloblastoma"
echo ""
echo "To get started:"
echo "  jupyter lab notebooks/medulloblastoma-analysis.ipynb"
echo ""
echo "For more information, see README.md"
