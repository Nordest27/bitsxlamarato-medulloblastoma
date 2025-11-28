# Getting Started

This guide will help you install and set up the Medulloblastoma G3/G4 Classification tool in just a few minutes.

## Prerequisites

Before you begin, make sure you have:

- **Python 3.10+** installed ([Download Python](https://python.org/downloads/))
- **4GB+ RAM** (8GB+ recommended for large datasets)
- **~2GB disk space** for data and models
- **Git** for cloning the repository ([Download Git](https://git-scm.com/downloads))

## Installation Methods

Choose the installation method that works best for your setup:

=== "pip + venv (Recommended)"

    This is the most straightforward method and works on all platforms.

    ```bash
    # Clone the repository
    git clone https://github.com/bsc-health/bitsxlamarato-medulloblastoma.git
    cd bitsxlamarato-medulloblastoma

    # Create virtual environment
    python -m venv venv

    # Activate virtual environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    ```

=== "conda"

    If you prefer conda for package management:

    ```bash
    # Clone the repository
    git clone https://github.com/bsc-health/bitsxlamarato-medulloblastoma.git
    cd bitsxlamarato-medulloblastoma

    # Create conda environment
    conda env create -f environment.yml
    conda activate medulloblastoma
    ```

=== "mamba (Fastest)"

    For the fastest installation with mamba:

    ```bash
    # Clone the repository
    git clone https://github.com/bsc-health/bitsxlamarato-medulloblastoma.git
    cd bitsxlamarato-medulloblastoma

    # Create mamba environment
    mamba env create -f environment.yml
    mamba activate medulloblastoma
    ```

=== "Automated Setup"

    Use our automated setup scripts:

    **Windows:**
    ```batch
    # Run the Windows setup script
    setup_windows.bat
    ```

    **macOS/Linux:**
    ```bash
    # Make script executable and run
    chmod +x setup_venv.sh
    ./setup_venv.sh
    ```

## Verify Installation

Test that everything is working correctly:

```python
# Run this Python code to verify installation
python -c "
from medulloblastoma.dataset import download_data
from medulloblastoma.features import load_data
from medulloblastoma.plots import plot_umap_binary
print('✅ Installation successful!')
"
```

If you see "✅ Installation successful!", you're ready to go!

## Quick Start Example

Let's run through a complete example to make sure everything works:

### 1. Download Data

```python
from medulloblastoma.dataset import download_data, prepare_data

# Download the GSE85217 dataset (this may take a few minutes)
data_file = download_data(save_path="data/raw/")
print(f"Downloaded data to: {data_file}")

# Prepare the data for analysis
expr_path, meta_path = prepare_data(
    expression_file=data_file,
    metadata_path="data/raw/GSE85217_metadata.csv",  # You'll need to provide this
    save_path="data/raw/"
)
```

### 2. Preprocess Data

```python
from medulloblastoma.features import main as preprocess_pipeline

# Run the complete preprocessing pipeline
preprocess_pipeline(
    data_path="data/raw/cavalli.csv",
    metadata_path="data/raw/cavalli_subgroups.csv",
    save_path="data/processed/",
    outlier_method='auto'  # Automatically choose the best method
)
```

### 3. Visualize Results

```python
from medulloblastoma.plots import plot_umap_binary
import pandas as pd

# Load processed data
data = pd.read_csv("data/processed/cavalli_maha.csv", index_col=0)
metadata = pd.read_csv("data/raw/cavalli_subgroups.csv", index_col=0).squeeze()

# Create UMAP visualization
colors_dict = {'Group3': 'red', 'Group4': 'blue'}
plot_umap_binary(
    data=data,
    clinical=metadata,
    colors_dict=colors_dict,
    title="Medulloblastoma G3/G4 Subtypes",
    save_fig=True
)
```

## Using the Jupyter Notebook

For a complete walkthrough, open the main analysis notebook:

```bash
# Launch Jupyter Lab
jupyter lab notebooks/medulloblastoma-analysis.ipynb
```

The notebook includes:

- ✅ Data download and preparation
- ✅ Exploratory data analysis  
- ✅ Preprocessing pipeline
- ✅ UMAP visualization
- 🔲 Model training (ready for implementation)

## Development Installation

If you plan to contribute to the project:

```bash
# Clone your fork of the repository
git clone https://github.com/yourusername/bitsxlamarato-medulloblastoma.git
cd bitsxlamarato-medulloblastoma

# Install in development mode with all dependencies
pip install -e ".[dev,docs]"

# Run tests to make sure everything works
pytest tests/

# Format code (optional)
black medulloblastoma/ tests/
isort medulloblastoma/ tests/
```

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'medulloblastoma'"

Make sure you've installed the package:
```bash
pip install -e .
```

#### "Permission denied" on Windows

Run the command prompt as Administrator, or use:
```bash
pip install --user -r requirements.txt
```

#### Memory errors with large datasets

Try using the fast statistical outlier detection method:
```python
from medulloblastoma.features import main as preprocess_pipeline

preprocess_pipeline(
    data_path="data/raw/cavalli.csv",
    metadata_path="data/raw/cavalli_subgroups.csv", 
    save_path="data/processed/",
    outlier_method='statistical'  # Use faster method
)
```

#### Slow preprocessing

The preprocessing includes optimizations that automatically select the best method based on your dataset size. For very large datasets, it will automatically use faster statistical methods instead of the slower Mahalanobis distance approach.

### Getting Help

If you're still having issues:

1. Check the [GitHub Issues](https://github.com/bsc-health/bitsxlamarato-medulloblastoma/issues)
2. Create a new issue with:
   - Your operating system
   - Python version (`python --version`)
   - Error messages
   - Steps to reproduce

## Next Steps

Now that you have the tool installed:

- Explore the [API Reference](api/dataset.md) for detailed function documentation
- Check out [Basic Usage Examples](examples/basic-usage.md) for common workflows
- Read about [Advanced Usage](examples/advanced.md) for customization options
- Consider [Contributing](contributing.md) to the project

Happy analyzing! 🧬🔬
