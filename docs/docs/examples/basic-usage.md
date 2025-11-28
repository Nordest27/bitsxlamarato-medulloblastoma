# Basic Usage Examples

This guide shows you how to use the core functionality of the Medulloblastoma G3/G4 Classification tool.

## Complete Workflow Example

Here's a complete example that takes you from raw data to visualization:

```python
import pandas as pd
import numpy as np
from medulloblastoma.dataset import download_data, prepare_data
from medulloblastoma.features import main as preprocess_pipeline, load_data
from medulloblastoma.plots import plot_umap_binary, plot_umap_spectrum

# Step 1: Download and prepare data
print("🔽 Downloading GSE85217 dataset...")
data_file = download_data(save_path="data/raw/")

print("🔧 Preparing data structure...")
expr_path, meta_path = prepare_data(
    expression_file=data_file,
    metadata_path="data/raw/GSE85217_metadata.csv",
    save_path="data/raw/"
)

# Step 2: Preprocess data
print("⚙️ Running preprocessing pipeline...")
preprocess_pipeline(
    data_path="data/raw/cavalli.csv",
    metadata_path="data/raw/cavalli_subgroups.csv", 
    save_path="data/processed/",
    per=0.2,        # Filter genes with >20% zero expression
    cutoff=0.1,     # Filter genes with variance < 0.1
    alpha=0.05,     # Outlier detection significance
    outlier_method='auto'  # Auto-select best method
)

# Step 3: Load processed data
print("📊 Loading processed data...")
data = pd.read_csv("data/processed/cavalli_maha.csv", index_col=0)
metadata = pd.read_csv("data/raw/cavalli_subgroups.csv", index_col=0).squeeze()

# Filter to G3/G4 only
metadata = metadata.map({'Group3': 'G3', 'Group4': 'G4'})
g3g4_metadata = metadata[metadata.isin(['G3', 'G4'])]
g3g4_data = data.loc[g3g4_metadata.index]

print(f"Final dataset: {g3g4_data.shape[0]} samples × {g3g4_data.shape[1]} genes")
print(f"Sample distribution: {g3g4_metadata.value_counts().to_dict()}")

# Step 4: Create visualizations
print("🎨 Creating UMAP visualization...")
colors_dict = {'G3': 'red', 'G4': 'blue'}

plot_umap_binary(
    data=g3g4_data,
    clinical=g3g4_metadata,
    colors_dict=colors_dict,
    title="Medulloblastoma G3/G4 Subtypes",
    save_fig=True,
    save_as="medulloblastoma_g3g4_umap",
    seed=42
)

print("✅ Analysis complete! Check the generated UMAP plot.")
```

## Individual Module Usage

### Dataset Management

```python
from medulloblastoma.dataset import download_data, prepare_data

# Download with custom settings
data_file = download_data(
    save_path="my_data/raw/",
    remove_gz=True,      # Delete compressed file after extraction
    timeout=600          # 10 minute timeout for large downloads
)

# Prepare data with custom paths
expr_path, meta_path = prepare_data(
    expression_file="my_data/raw/expression.txt",
    metadata_path="my_data/raw/metadata.csv",
    save_path="my_data/processed/"
)
```

### Data Preprocessing

```python
from medulloblastoma.features import load_data, preprocess, maha_outliers, fast_statistical_outliers

# Load data
data, metadata = load_data(
    data_path="data/processed/cavalli.csv",
    metadata_path="data/raw/cavalli_subgroups.csv"
)

# Preprocess with custom parameters
processed_data = preprocess(
    data=data.T,        # Transpose for function (genes as rows)
    per=0.3,           # Allow up to 30% zero expression
    cutoff=0.05        # Lower variance threshold
)

# Choose outlier detection method
if processed_data.shape[1] > 1000:  # Many features
    clean_data = fast_statistical_outliers(
        data=processed_data,
        save_path="data/processed/",
        alpha=0.05,
        method='iqr'   # Use IQR method for speed
    )
else:  # Fewer features
    clean_data = maha_outliers(
        data=processed_data,
        save_path="data/processed/",
        alpha=0.05,
        use_fast_method=True  # Use optimized MinCovDet
    )
```

### Visualization

```python
from medulloblastoma.plots import plot_umap_binary, plot_umap_spectrum

# Discrete coloring for categorical data
colors_dict = {
    'G3': '#FF6B6B',     # Coral red
    'G4': '#4ECDC4',     # Teal  
    'WNT': '#45B7D1',    # Blue
    'SHH': '#96CEB4'     # Green
}

plot_umap_binary(
    data=expression_data,
    clinical=subgroup_labels,
    colors_dict=colors_dict,
    n_components=2,      # 2D plot
    seed=42,            # Reproducible results
    title="Medulloblastoma Subgroups",
    marker_size=15,
    save_fig=True,
    save_as="subgroups_umap"
)

# Continuous coloring for scores/probabilities
# (This would typically come from a trained model)
probability_scores = pd.Series(
    np.random.uniform(0, 1, len(expression_data)),
    index=expression_data.index,
    name='G3_probability'
)

plot_umap_spectrum(
    data=expression_data,
    clinical=probability_scores,
    colormap='RdBu',           # Red-Blue colormap
    color_range=(0, 1),        # Full probability range
    colorbar_title="G3 Probability",
    title="G3/G4 Classification Scores",
    n_components=2,
    seed=42,
    save_fig=True,
    save_as="scores_umap"
)
```

## Performance Optimization Examples

### Choosing the Right Outlier Detection Method

```python
from medulloblastoma.features import main as preprocess_pipeline

# For small datasets (< 1000 features): Use Mahalanobis distance
preprocess_pipeline(
    data_path="small_dataset.csv",
    metadata_path="metadata.csv",
    save_path="results/",
    outlier_method='mahalanobis'  # Most accurate but slower
)

# For medium datasets (1000-5000 features): Use optimized Mahalanobis
preprocess_pipeline(
    data_path="medium_dataset.csv", 
    metadata_path="metadata.csv",
    save_path="results/",
    outlier_method='mahalanobis',
    alpha=0.05
)

# For large datasets (> 5000 features): Use statistical methods
preprocess_pipeline(
    data_path="large_dataset.csv",
    metadata_path="metadata.csv", 
    save_path="results/",
    outlier_method='statistical'  # Much faster
)

# Let the algorithm decide automatically
preprocess_pipeline(
    data_path="any_dataset.csv",
    metadata_path="metadata.csv",
    save_path="results/", 
    outlier_method='auto'  # Recommended
)
```

### Memory-Efficient Processing

```python
import pandas as pd
from medulloblastoma.features import preprocess

# For very large datasets, process in chunks
def process_large_dataset(data_path, chunk_size=1000):
    """Process large datasets in chunks to save memory."""
    
    # Read data info without loading full dataset
    data_info = pd.read_csv(data_path, nrows=1)
    total_genes = len(data_info.columns) - 1  # Subtract index column
    
    processed_chunks = []
    
    for start_col in range(1, total_genes + 1, chunk_size):
        end_col = min(start_col + chunk_size, total_genes + 1)
        
        print(f"Processing genes {start_col}-{end_col-1} of {total_genes}")
        
        # Load chunk
        cols_to_read = [0] + list(range(start_col, end_col))  # Include index
        chunk = pd.read_csv(data_path, usecols=cols_to_read, index_col=0)
        
        # Preprocess chunk
        processed_chunk = preprocess(chunk.T, per=0.2, cutoff=0.1)
        processed_chunks.append(processed_chunk)
        
        # Clear memory
        del chunk
    
    # Combine processed chunks
    final_data = pd.concat(processed_chunks, axis=1)
    return final_data

# Use for datasets > 10,000 genes
# processed_data = process_large_dataset("huge_dataset.csv")
```

## Error Handling Examples

```python
from medulloblastoma.dataset import download_data
from medulloblastoma.features import load_data
import os

# Robust data download with error handling
def safe_download_data(save_path="data/raw/", max_retries=3):
    """Download data with retry logic."""
    
    for attempt in range(max_retries):
        try:
            data_file = download_data(save_path=save_path, timeout=300)
            print(f"✅ Download successful: {data_file}")
            return data_file
            
        except Exception as e:
            print(f"❌ Download attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to download after {max_retries} attempts")
            
            print("🔄 Retrying in 10 seconds...")
            import time
            time.sleep(10)

# Robust data loading with validation
def safe_load_data(data_path, metadata_path):
    """Load data with comprehensive validation."""
    
    # Check files exist
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    try:
        data, metadata = load_data(data_path, metadata_path)
        
        # Validate data quality
        if data.isnull().sum().sum() > 0:
            print(f"⚠️ Warning: Found {data.isnull().sum().sum()} missing values")
        
        if (data < 0).any().any():
            print("⚠️ Warning: Found negative expression values")
            
        print(f"✅ Loaded {data.shape[0]} samples × {data.shape[1]} genes")
        print(f"✅ Metadata: {len(metadata)} samples, {metadata.nunique()} groups")
        
        return data, metadata
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

# Example usage
try:
    data_file = safe_download_data()
    data, metadata = safe_load_data("data/processed/cavalli.csv", "data/raw/cavalli_subgroups.csv")
except Exception as e:
    print(f"Failed to load data: {e}")
```

## Working with Different Data Formats

```python
import pandas as pd
from medulloblastoma.features import load_data

# Handle different file formats
def load_any_format(data_path, metadata_path):
    """Load data from various file formats."""
    
    # Detect file format and load accordingly
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path, index_col=0)
    elif data_path.endswith('.tsv') or data_path.endswith('.txt'):
        data = pd.read_csv(data_path, sep='\t', index_col=0)
    elif data_path.endswith('.xlsx'):
        data = pd.read_excel(data_path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    if metadata_path.endswith('.csv'):
        metadata = pd.read_csv(metadata_path, index_col=0).squeeze()
    elif metadata_path.endswith('.tsv') or metadata_path.endswith('.txt'):
        metadata = pd.read_csv(metadata_path, sep='\t', index_col=0).squeeze()
    elif metadata_path.endswith('.xlsx'):
        metadata = pd.read_excel(metadata_path, index_col=0).squeeze()
    else:
        raise ValueError(f"Unsupported file format: {metadata_path}")
    
    return data, metadata

# Example usage with different formats
# data, metadata = load_any_format("data.xlsx", "metadata.tsv")
```

This covers the most common usage patterns. For more advanced examples, see the [Advanced Usage](advanced.md) guide!
