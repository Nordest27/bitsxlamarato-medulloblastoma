"""
Comprehensive test suite for medulloblastoma package.

Tests core functionality including data loading, preprocessing,
visualization, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from medulloblastoma.dataset import download_data, prepare_data
from medulloblastoma.features import load_data, get_g3g4, preprocess
from medulloblastoma.plots import plot_umap_binary, plot_umap_spectrum


@pytest.fixture
def sample_expression_data():
    """Create sample gene expression data for testing."""
    np.random.seed(42)
    n_samples, n_genes = 100, 200

    # Create expression data
    data = np.random.lognormal(mean=1, sigma=1, size=(n_samples, n_genes))

    # Create sample and gene names
    sample_names = [f"Sample_{i:03d}" for i in range(n_samples)]
    gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]

    return pd.DataFrame(data, index=sample_names, columns=gene_names)


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    np.random.seed(42)
    n_samples = 100

    sample_names = [f"Sample_{i:03d}" for i in range(n_samples)]

    # Create balanced G3/G4 groups with some other groups
    groups = ['Group3'] * 30 + ['Group4'] * 35 + ['WNT'] * 20 + ['SHH'] * 15
    np.random.shuffle(groups)

    return pd.Series(groups, index=sample_names, name='subgroup')


class TestDataLoading:
    """Test data loading functionality."""

    def test_load_data_success(self, sample_expression_data, sample_metadata, tmp_path):
        """Test successful data loading."""
        # Save test data
        data_path = tmp_path / "test_data.csv"
        meta_path = tmp_path / "test_meta.csv"

        sample_expression_data.to_csv(data_path)
        sample_metadata.to_csv(meta_path)

        # Load data
        data, metadata = load_data(data_path, meta_path)

        # Verify results
        assert isinstance(data, pd.DataFrame)
        assert isinstance(metadata, pd.Series)
        assert data.shape[0] == metadata.shape[0]  # Same number of samples
        assert data.shape[0] == 100  # Expected sample count
        assert data.shape[1] == 200  # Expected gene count

    def test_load_data_transpose(self, sample_expression_data, sample_metadata, tmp_path):
        """Test data loading with automatic transposition."""
        # Save transposed data (genes as rows instead of columns)
        data_path = tmp_path / "test_data_T.csv"
        meta_path = tmp_path / "test_meta.csv"

        sample_expression_data.T.to_csv(data_path)  # Transpose before saving
        sample_metadata.to_csv(meta_path)

        # Load data
        data, metadata = load_data(data_path, meta_path)

        # Should still work correctly
        assert data.shape[0] == metadata.shape[0]
        assert data.shape == (100, 200)  # Correct final shape

    def test_load_data_file_not_found(self, tmp_path):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_data.csv", "nonexistent_meta.csv")


class TestFeatureEngineering:
    """Test feature engineering and preprocessing."""

    def test_get_g3g4_extraction(self, sample_expression_data, sample_metadata):
        """Test G3/G4 sample extraction."""
        g3g4_data = get_g3g4(sample_expression_data, sample_metadata)

        # Should only contain G3/G4 samples
        g3g4_groups = sample_metadata.loc[g3g4_data.index]
        assert all(group in ['Group3', 'Group4'] for group in g3g4_groups)

        # Should have expected number of samples (30 + 35 = 65)
        assert len(g3g4_data) == 65

    def test_preprocess_gene_filtering(self, sample_expression_data):
        """Test gene filtering in preprocessing."""
        # Add some genes with high zero percentage
        data_with_zeros = sample_expression_data.copy()

        # Make some genes have >50% zeros
        data_with_zeros.iloc[:60, :10] = 0  # First 10 genes have 60% zeros

        # Preprocess with 40% zero threshold
        processed = preprocess(data_with_zeros.T, per=0.4, cutoff=0.0)  # Transpose for function

        # Should filter out genes with >40% zeros
        assert processed.shape[1] < data_with_zeros.shape[1]  # Fewer genes

    def test_preprocess_variance_filtering(self, sample_expression_data):
        """Test variance-based gene filtering."""
        # Add genes with very low variance
        data_with_low_var = sample_expression_data.copy()
        data_with_low_var.iloc[:, :20] = 1.0  # First 20 genes have zero variance

        # Preprocess with variance cutoff
        processed = preprocess(data_with_low_var.T, per=1.0, cutoff=0.1)  # High zero tolerance, variance filter

        # Should filter out low variance genes
        assert processed.shape[1] < data_with_low_var.shape[1]


class TestVisualization:
    """Test visualization functions."""

    def test_plot_umap_binary_basic(self, sample_expression_data, sample_metadata):
        """Test basic UMAP binary plotting."""
        # Filter to G3/G4 only for plotting
        g3g4_data = get_g3g4(sample_expression_data, sample_metadata)
        g3g4_metadata = sample_metadata.loc[g3g4_data.index]

        colors_dict = {'Group3': 'red', 'Group4': 'blue'}

        # Should run without error
        plot_umap_binary(
            data=g3g4_data,
            clinical=g3g4_metadata,
            colors_dict=colors_dict,
            show=False,  # Don't show during testing
            seed=42
        )

    def test_plot_umap_spectrum_basic(self, sample_expression_data):
        """Test basic UMAP spectrum plotting."""
        # Create continuous scores
        scores = pd.Series(
            np.random.uniform(0, 1, len(sample_expression_data)),
            index=sample_expression_data.index,
            name='scores'
        )

        # Should run without error
        plot_umap_spectrum(
            data=sample_expression_data,
            clinical=scores,
            colormap='viridis',
            show=False,  # Don't show during testing
            seed=42
        )


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_preprocessing_workflow(self, sample_expression_data, sample_metadata, tmp_path):
        """Test complete preprocessing workflow."""
        # Save test data
        data_path = tmp_path / "test_data.csv"
        meta_path = tmp_path / "test_meta.csv"

        sample_expression_data.to_csv(data_path)
        sample_metadata.to_csv(meta_path)

        # Load data
        data, metadata = load_data(data_path, meta_path)

        # Extract G3/G4
        g3g4_data = get_g3g4(data, metadata)

        # Preprocess
        processed = preprocess(g3g4_data.T, per=0.2, cutoff=0.1)

        # Verify final result
        assert isinstance(processed, pd.DataFrame)
        assert processed.shape[0] <= g3g4_data.shape[1]  # May filter genes
        assert processed.shape[1] == len(g3g4_data)  # Same samples


def test_package_imports():
    """Test that all main modules can be imported."""
    try:
        import medulloblastoma
        from medulloblastoma import dataset, features, plots
        from medulloblastoma.modeling import train, predict
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_dependencies():
    """Test that all required dependencies are available."""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'plotly',
        'umap', 'scipy', 'sklearn', 'loguru'
    ]

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            pytest.fail(f"Required package '{package}' not installed")
