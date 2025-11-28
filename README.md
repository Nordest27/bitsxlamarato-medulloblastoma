# Medulloblastoma G3/G4 Subgroup Classification

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

</div>

> *"Ataquem bé al tumor infantil medul·loblastoma!" - Advancing precision medicine for childhood brain cancer.*

An AI-powered tool for classifying medulloblastoma G3/G4 subtypes from gene expression data, developed during the BitsxLaMarató hackathon. This project provides explainable machine learning models to assist in pediatric brain tumor diagnosis and treatment planning.

## 🎯 Project Overview

Medulloblastoma is the most common malignant pediatric brain tumor. Accurate subtype classification (particularly G3 vs G4) is crucial for treatment decisions and prognosis. This project develops machine learning models that:

- **Classify patients** into G3/G4 subtypes using transcriptomics data
- **Generate probability scores** for subtype assignment with confidence intervals
- **Identify key genes** driving classification decisions for biological insight
- **Provide visualizations** for data exploration and result interpretation

### Key Features

- 🧬 **Large Dataset**: Utilizes GSE85217 with 763 medulloblastoma patients
- 🔬 **Gene-Level Analysis**: Direct analysis without dimensionality reduction to "metagenes"
- 📊 **Advanced Preprocessing**: Optimized outlier detection and feature selection
- 🎨 **Interactive Visualizations**: UMAP plots with continuous and discrete coloring
- ⚡ **High Performance**: Optimized algorithms for fast processing
- 📖 **Well Documented**: Comprehensive API documentation and examples

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- 4GB+ RAM (8GB+ recommended)
- ~2GB disk space for data and models

### Installation Options

#### Option 1: Using pip + venv (Recommended)

```bash
# Clone the repository
git clone https://github.com/bsc-health/bitsxlamarato-medulloblastoma.git
cd bitsxlamarato-medulloblastoma

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using conda

```bash
# Clone the repository
git clone https://github.com/bsc-health/bitsxlamarato-medulloblastoma.git
cd bitsxlamarato-medulloblastoma

# Create conda environment
conda env create -f environment.yml
conda activate medulloblastoma
```

#### Option 3: Using mamba (Fastest)

```bash
# Clone the repository
git clone https://github.com/bsc-health/bitsxlamarato-medulloblastoma.git
cd bitsxlamarato-medulloblastoma

# Create mamba environment
mamba env create -f environment.yml
mamba activate medulloblastoma
```

### Verify Installation

```python
# Test basic functionality
python -c "
from medulloblastoma.dataset import download_data
from medulloblastoma.features import load_data
from medulloblastoma.plots import plot_umap_binary
print('✅ Installation successful!')
"
```

## 📝 Usage

### 1. Data Download and Preparation

```python
from medulloblastoma.dataset import download_data, prepare_data

# Download GSE85217 dataset from GEO
data_file = download_data(save_path="data/raw/")

# Process and structure the data
expr_path, meta_path = prepare_data(
    expression_file=data_file,
    metadata_path="data/raw/GSE85217_metadata.csv",
    save_path="data/raw/"
)
```

### 2. Data Preprocessing

```python
from medulloblastoma.features import main as preprocess_pipeline

# Run complete preprocessing pipeline
preprocess_pipeline(
    data_path="data/raw/cavalli.csv",
    metadata_path="data/raw/cavalli_subgroups.csv",
    save_path="data/processed/",
    per=0.2,        # Filter genes with >20% zero expression
    cutoff=0.1,     # Filter genes with variance < 0.1
    alpha=0.05,     # Outlier detection significance level
    outlier_method='auto'  # Auto-select best method
)
```

### 3. Data Visualization

```python
from medulloblastoma.plots import plot_umap_binary, plot_umap_spectrum
import pandas as pd

# Load processed data
data = pd.read_csv("data/processed/cavalli_maha.csv", index_col=0)
metadata = pd.read_csv("data/raw/cavalli_subgroups.csv", index_col=0).squeeze()

# Create discrete UMAP plot for G3/G4 subtypes
colors_dict = {'G3': 'red', 'G4': 'blue'}
plot_umap_binary(
    data=data,
    clinical=metadata,
    colors_dict=colors_dict,
    title="Medulloblastoma G3/G4 Subtypes",
    save_fig=True
)

# Create continuous UMAP plot for probability scores
# (scores would come from trained model)
plot_umap_spectrum(
    data=data,
    clinical=probability_scores,  # pandas Series with values 0-1
    colormap='RdBu',
    title="G3/G4 Classification Scores"
)
```

### 4. Complete Workflow (Jupyter Notebook)

For a complete analysis workflow, see the main notebook:

```bash
jupyter lab notebooks/medulloblastoma-analysis.ipynb
```

This notebook includes:
- Data download and preparation
- Exploratory data analysis
- Preprocessing with quality control
- UMAP visualization
- Model training framework (ready for implementation)

## 📁 Project Structure

```
bitsxlamarato-medulloblastoma/
├── README.md                          # This file
├── pyproject.toml                     # Project metadata and dependencies
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment specification
├── LICENSE                           # MIT license
├── Makefile                          # Development commands
├──
├── data/                             # Data directory
│   ├── raw/                          # Original, immutable data
│   ├── interim/                      # Intermediate transformed data
│   ├── processed/                    # Final, model-ready data
│   └── external/                     # Third-party data
├──
├── notebooks/                        # Jupyter notebooks
│   └── medulloblastoma-analysis.ipynb # Main analysis workflow
├──
├── medulloblastoma/                  # Source code package
│   ├── __init__.py                   # Package initialization
│   ├── config.py                     # Project configuration
│   ├── dataset.py                    # Data download and preparation
│   ├── features.py                   # Data preprocessing and feature engineering
│   ├── plots.py                      # Visualization functions
│   └── modeling/                     # Model components
│       ├── __init__.py
│       ├── train.py                  # Model training
│       └── predict.py                # Model prediction
├──
├── models/                           # Trained model artifacts
├── reports/                          # Analysis reports and figures
└── tests/                           # Unit tests
```

## 🔧 API Reference

### Dataset Management

```python
from medulloblastoma.dataset import download_data, prepare_data

# Download GSE85217 data
download_data(save_path="data/raw/", remove_gz=True, timeout=300)

# Prepare data for analysis
prepare_data(expression_file="...", metadata_path="...", save_path="...")
```

### Data Preprocessing

```python
from medulloblastoma.features import (
    load_data, preprocess, maha_outliers, 
    fast_statistical_outliers, main as preprocess_pipeline
)

# Load and preprocess data
data, metadata = load_data(data_path="...", metadata_path="...")
processed_data = preprocess(data, per=0.2, cutoff=0.1)
clean_data = maha_outliers(processed_data, save_path="...", alpha=0.05)
```

### Visualization

```python
from medulloblastoma.plots import plot_umap_binary, plot_umap_spectrum

# Discrete coloring for categorical data
plot_umap_binary(data, clinical, colors_dict, title="Subtypes")

# Continuous coloring for scores/probabilities
plot_umap_spectrum(data, clinical, colormap='viridis', title="Scores")
```

## 🧪 Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=medulloblastoma tests/
```

### Code Quality

```bash
# Format code
black medulloblastoma/ tests/

# Sort imports
isort medulloblastoma/ tests/

# Lint code
flake8 medulloblastoma/ tests/
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Serve documentation locally
mkdocs serve
```

## 📊 Performance

### Preprocessing Performance

| Dataset Size | Method | Time | Speedup |
|-------------|--------|------|---------|
| < 1k features | Optimized MinCovDet | 30-60s | 3-5x |
| 1k-5k features | Fast Statistical | 1-3min | 10-15x |
| > 5k features | Auto-select | 2-5min | 20-50x |

### System Requirements

- **Minimum**: 4GB RAM, 2GB disk space
- **Recommended**: 8GB+ RAM, 5GB disk space
- **GPU**: Optional, speeds up large dataset processing

### Dataset Information

- **Source**: GSE85217 from Gene Expression Omnibus (GEO)
- **Platform**: Affymetrix microarrays
- **Samples**: 763 medulloblastoma patients
- **Features**: ~54,000 gene probes
- **Labels**: WHO subgroup classifications (WNT, SHH, G3, G4)

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Format code (`black`, `isort`)
7. Submit a pull request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/bitsxlamarato-medulloblastoma.git
cd bitsxlamarato-medulloblastoma

# Install in development mode with all dependencies
pip install -e ".[dev,docs]"

# Set up pre-commit hooks (optional)
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BitsxLaMarató Hackathon** for the challenge and platform
- **Barcelona Supercomputing Center** for computational resources
- **La Marató de TV3** for funding pediatric cancer research
- **GEO Database** for providing the GSE85217 dataset
- **Contributors** to the open-source packages used in this project

## 📞 Contact

- **Jose Estragues** - [jose.estragues@bsc.es]
- **Guillermo Prol** - [guillermo.prolcastelo@bsc.es]
- **Project Link**: [https://github.com/bsc-health/bitsxlamarato-medulloblastoma]

---

<div align="center">
<i>Developed with ❤️ for advancing pediatric cancer research</i>
</div>

