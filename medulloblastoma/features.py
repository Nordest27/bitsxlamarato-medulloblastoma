"""
Feature engineering and preprocessing module for medulloblastoma gene expression data.

This module provides comprehensive preprocessing functionality for medulloblastoma
gene expression datasets, including data loading, filtering, outlier detection,
and quality control visualization.

Key functionalities:
- Data loading and validation
- Gene filtering based on expression patterns
- Statistical and Mahalanobis-based outlier detection
- Data visualization and quality control plots
- Optimized processing for large datasets
"""

import os
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.covariance import MinCovDet
from pickle import dump
from loguru import logger

warnings.filterwarnings('ignore')
plt.style.use('ggplot')


def load_data(
    data_path: Union[str, Path], 
    metadata_path: Union[str, Path]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and validate gene expression data and corresponding metadata.
    
    This function loads gene expression data and metadata, ensures proper
    dimensionality alignment, and validates that samples match between datasets.
    
    Parameters
    ----------
    data_path : str or Path
        Path to CSV file containing gene expression data.
        Expected format: genes as rows, samples as columns (or vice versa)
    metadata_path : str or Path
        Path to CSV file containing sample metadata.
        Expected format: samples as rows, metadata as columns
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Tuple containing:
        - Gene expression DataFrame (samples × genes)
        - Metadata Series with sample information
        
    Raises
    ------
    FileNotFoundError
        If input files don't exist
    AssertionError
        If data and metadata dimensions don't match
        
    Examples
    --------
    >>> data, metadata = load_data(
    ...     "data/processed/cavalli_maha.csv",
    ...     "data/raw/cavalli_subgroups.csv"
    ... )
    >>> print(f"Data shape: {data.shape}")
    >>> print(f"Metadata shape: {metadata.shape}")
    """
    # Load data files
    data_path = Path(data_path)
    metadata_path = Path(metadata_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path, index_col=0)
    
    logger.info(f"Loading metadata from {metadata_path}")
    metadata = pd.read_csv(metadata_path, index_col=0).squeeze()
    
    # Check and fix dimensionality if needed
    logger.info(f"Initial data shape: {data.shape}, metadata shape: {metadata.shape}")
    
    if data.shape[0] != metadata.shape[0]:
        logger.info("Transposing data to match metadata dimensions")
        data = data.T
        
    # Validate final dimensions
    assert data.shape[0] == metadata.shape[0], (
        f"Data and metadata dimensions don't match: "
        f"data={data.shape[0]}, metadata={metadata.shape[0]}"
    )
    
    # Align indices
    data.index = metadata.index
    
    logger.success(f"Successfully loaded data: {data.shape} samples × {data.shape[1]} features")
    return data, metadata


def get_g3g4(data: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """
    Extract samples belonging to Group 3 and Group 4 medulloblastoma subtypes.
    
    Filters the input data to retain only samples classified as Group3 or Group4
    medulloblastoma subtypes, which are the focus of this classification task.
    
    Parameters
    ----------
    data : pd.DataFrame
        Gene expression data with samples as rows and genes as columns
    groups : pd.Series
        Sample group/subtype annotations. Should contain values like
        'Group3', 'Group4', 'G3', 'G4', etc.
        
    Returns
    -------
    pd.DataFrame
        Filtered data containing only G3/G4 samples
        
    Examples
    --------
    >>> # Filter for G3/G4 samples
    >>> g3g4_data = get_g3g4(expression_data, subgroup_labels)
    >>> print(f"G3/G4 subset: {g3g4_data.shape}")
    >>> print(subgroup_labels.value_counts())
    """
    # Identify G3/G4 samples (handle different naming conventions)
    g3g4_patterns = ['Group3', 'Group4', 'G3', 'G4', 'group3', 'group4']
    clinical_g3g4 = groups[groups.isin(g3g4_patterns)]
    
    if len(clinical_g3g4) == 0:
        logger.warning("No G3/G4 samples found. Check group labels format.")
        logger.info(f"Available groups: {groups.unique()}")
    
    # Filter data to G3/G4 samples only
    data_g3g4 = data.loc[clinical_g3g4.index]
    
    logger.info(f"Extracted {len(data_g3g4)} G3/G4 samples from {len(data)} total samples")
    return data_g3g4

def plot_original_distribution(data,save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(data.values.flatten(),bins=100,color="orange",ec="k",label="cavalli")
    ax.set_xlabel("Gene Expression",fontsize=12)
    ax.set_ylabel("Counts",fontsize=12)
    # plt.title(
    #     """
    #     Distribution of Gene Expression in Cavalli
    #     """,
    #     fontsize=14)
    # plt.legend()
    plt.savefig(os.path.join(save_path, 'original_distribution.png'), dpi=600)
    plt.savefig(os.path.join(save_path, 'original_distribution.svg'), dpi=600)
    plt.savefig(os.path.join(save_path, 'original_distribution.pdf'), dpi=600)
    plt.clf()
    # Creating var histogram:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(np.var(data, axis=1).values, bins=100, color="green", ec="k")
    # ax.vlines(0.1,0,4000,"r",lw=2)
    ax.set_xlabel("Variance of Genes' Expression", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    # ax.set_title("Distribution of Variance of Genes' Expression\nCavalli",fontsize=14)
    plt.savefig(os.path.join(save_path, 'original_variance_distribution.png'), dpi=600)
    plt.savefig(os.path.join(save_path, 'original_variance_distribution.svg'), dpi=600)
    plt.savefig(os.path.join(save_path, 'original_variance_distribution.pdf'), dpi=600)
    plt.clf(); fig.clf()
    # Creating mean histogram
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(np.mean(data, axis=1).values, bins=100, color="cornflowerblue", ec="k")
    # ax.vlines(0.5,0,1800,"r",lw=2)
    ax.set_xlabel("Mean of Genes' Expression", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    # ax.set_title("Distribution of Mean of Genes' Expression\nCavalli",fontsize=14)
    plt.savefig(os.path.join(save_path, 'original_mean_distribution.png'), dpi=600)
    plt.savefig(os.path.join(save_path, 'original_mean_distribution.svg'), dpi=600)
    plt.savefig(os.path.join(save_path, 'original_mean_distribution.pdf'), dpi=600)
    plt.clf();fig.clf()

def preprocess(data,per=0.2,cutoff=0.1):
    # Keep genes with 0 expression in at least per% of the patients
    data = data[(data==0).sum(axis=1)/data.shape[1]<=per]
    # Filter genes by var
    data = data.iloc[(np.var(data, axis=1).values >= cutoff)]
    return data

def maha_outliers(data, save_path, alpha=0.05, use_fast_method=True, support_fraction=None):
    # Mahalanobis
    nrows, ncols = data.shape
    print(f"Data shape: {nrows} samples x {ncols} features")

    # Define cutoff
    cutoff = scipy.stats.chi2.ppf(1-alpha, ncols)  # Use ncols (features) for chi-square degrees of freedom
    with open(os.path.join(save_path, "alpha.txt"), 'w') as f:
        f.write(str(alpha))
    with open(os.path.join(save_path, "cutoff.txt"), 'w') as f:
        f.write(str(cutoff))

    if use_fast_method and ncols > 50:  # Use fast method for high-dimensional data
        print("Using optimized fast outlier detection method")
        # Fast method: Use smaller support fraction for speed
        if support_fraction is None:
            support_fraction = min(0.5, (nrows + ncols + 1.) / (2. * nrows))

        # Use random state for reproducibility and faster convergence
        mcd = MinCovDet(support_fraction=support_fraction,
                       random_state=42,
                       assume_centered=False)

        print("Running optimized MinCovDet...")
        output_c = mcd.fit(data)
        print("Done MinCovDet")

    else:
        # Original method for smaller datasets
        print("Running standard MinCovDet...")
        output_c = MinCovDet(random_state=42).fit(data)
        print("Done MinCovDet")

    # Save the fitted model
    dump(output_c, open(os.path.join(save_path, 'MCDoutput_cavalli.pkl'), 'wb'))

    # Get Mahalanobis distances
    md_c = output_c.dist_

    # Identify outliers - vectorized operation
    outlier_mask = md_c > cutoff
    names_outliers_MH_c = np.where(outlier_mask)[0]

    print(f"Number of outliers: {len(names_outliers_MH_c)}")
    print(f"Outlier percentage: {len(names_outliers_MH_c)/nrows*100:.2f}%")
    print("")

    # Drop outliers from dataset - use boolean indexing for speed
    data_maha = data[~outlier_mask]
    print(f"Shape of output dataset: {data_maha.shape}")
    print("")

    # Clear memory
    del output_c

    # Plot resulting distribution
    plt.figure(figsize=(7,5))
    plt.hist(data.values.flatten(), bins=100, color="green", ec="k", label="before", alpha=0.7)
    plt.hist(data_maha.values.flatten(), bins=100, color="cornflowerblue", ec="k", label="after", alpha=0.7)
    plt.xlabel("Gene Expression",fontsize=14)
    plt.ylabel("Counts",fontsize=14)
    # plt.title(
    #     """
    #     Distribution of Gene Expression,
    #     for all genes and patients,
    #     before and after Mahalanobis outlier detection
    #     Cavalli
    #     """,
    #     fontsize=14)
    plt.legend()
    # Save the figure
    plt.savefig(os.path.join(save_path, 'distribution_after_mahalanobis.png'), dpi=600)
    plt.savefig(os.path.join(save_path, 'distribution_after_mahalanobis.svg'), dpi=600)
    plt.savefig(os.path.join(save_path, 'distribution_after_mahalanobis.pdf'), dpi=600)

    return data_maha


def fast_statistical_outliers(data, save_path, alpha=0.05, method='iqr'):
    """
    Ultra-fast outlier detection using statistical methods instead of MinCovDet.
    Suitable for very large datasets where MinCovDet is too slow.

    Parameters:
    - method: 'iqr' (Interquartile Range) or 'zscore' (Z-score based)
    """
    nrows, ncols = data.shape
    print(f"Using fast statistical outlier detection on {nrows} samples x {ncols} features")

    with open(os.path.join(save_path, "alpha.txt"), 'w') as f:
        f.write(str(alpha))

    if method == 'iqr':
        # IQR-based outlier detection (very fast)
        print("Using IQR-based outlier detection...")
        # Calculate outliers based on multivariate IQR
        outlier_scores = np.zeros(nrows)
        for i in range(ncols):
            col_data = data.iloc[:, i]
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_scores += ((col_data < lower_bound) | (col_data > upper_bound)).astype(int)

        # Define outliers as samples with outlier_scores in top alpha percentile
        cutoff_score = np.percentile(outlier_scores, (1 - alpha) * 100)
        outlier_mask = outlier_scores > cutoff_score

    else:  # zscore method
        print("Using Z-score based outlier detection...")
        # Modified Z-score using median (more robust)
        median = np.median(data, axis=0)
        mad = np.median(np.abs(data - median), axis=0)
        # Avoid division by zero
        mad = np.where(mad == 0, np.finfo(float).eps, mad)
        modified_z_scores = 0.6745 * (data - median) / mad

        # Calculate aggregate outlier score per sample
        outlier_scores = np.sum(np.abs(modified_z_scores) > 3.5, axis=1)
        cutoff_score = np.percentile(outlier_scores, (1 - alpha) * 100)
        outlier_mask = outlier_scores > cutoff_score

    with open(os.path.join(save_path, "cutoff.txt"), 'w') as f:
        f.write(str(cutoff_score))

    names_outliers = np.where(outlier_mask)[0]
    print(f"Number of outliers: {len(names_outliers)}")
    print(f"Outlier percentage: {len(names_outliers)/nrows*100:.2f}%")

    # Remove outliers
    data_clean = data[~outlier_mask]
    print(f"Shape of output dataset: {data_clean.shape}")

    # Plot results
    plt.figure(figsize=(7,5))
    plt.hist(data.values.flatten(), bins=100, color="green", ec="k", label="before", alpha=0.7)
    plt.hist(data_clean.values.flatten(), bins=100, color="cornflowerblue", ec="k", label="after", alpha=0.7)
    plt.xlabel("Gene Expression", fontsize=12)
    plt.ylabel("Counts", fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'distribution_after_statistical_outliers.png'), dpi=600)
    plt.savefig(os.path.join(save_path, 'distribution_after_statistical_outliers.svg'), dpi=600)
    plt.savefig(os.path.join(save_path, 'distribution_after_statistical_outliers.pdf'), dpi=600)
    plt.clf()

    return data_clean


def main(data_path,metadata_path,save_path,per,cutoff,alpha,outlier_method='auto'):
    print('Path to save data:',save_path)
    os.makedirs(save_path, exist_ok=True)
    data, groups = load_data(data_path=data_path,metadata_path=metadata_path)
    print('Shape of original data:',data.shape)
    plot_original_distribution(data,save_path)
    # Check for null or missing data:
    print('null in data:',data.isnull().sum().sum())
    print('na in data:',data.isna().sum().sum())
    # Check all values in the df are real:
    print('Are all values in the data real?','Yes' if data.size==np.sum(np.isreal(data)) else 'No')
    # Get data for groups 3 and 4, before any preprocessing
    data_g3g4 = get_g3g4(data=data,groups=groups)
    data_g3g4.to_csv(os.path.join(save_path, 'g3g4_noprepro.csv'))
    # Preprocess data
    # Here the data must reflect samples as columns:
    data_preprocessed = preprocess(data=data.T,per=per,cutoff=cutoff)

    # Choose outlier detection method based on data size and user preference
    nrows, ncols = data_preprocessed.shape

    if outlier_method == 'auto':
        # Auto-select method based on dataset size
        if ncols > 1000 or nrows > 5000:
            print(f"Large dataset detected ({nrows}x{ncols}). Using fast statistical method.")
            outlier_method = 'statistical'
        else:
            outlier_method = 'mahalanobis'

    if outlier_method == 'statistical':
        print("Using fast statistical outlier detection...")
        data_clean = fast_statistical_outliers(data=data_preprocessed, save_path=save_path, alpha=alpha, method='iqr')
        suffix = '_statistical'
    else:
        print("Using Mahalanobis distance outlier detection...")
        data_clean = maha_outliers(data=data_preprocessed, save_path=save_path, alpha=alpha, use_fast_method=True)
        suffix = '_maha'

    data_clean.T.to_csv(os.path.join(save_path, f'cavalli{suffix}.csv'))
    # Get data for groups 3 and 4, after preprocessing
    data_g3g4_clean = get_g3g4(data=data_clean.T,groups=groups)
    data_g3g4_clean.to_csv(os.path.join(save_path, f'g3g4{suffix}.csv'))
    print('done preprocessing')



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--metadata_path', type=str, help='path to metadata')
    parser.add_argument('--save_path', type=str, help='path to save results')
    parser.add_argument('--per', type=float, help='Filter out genes when they have this percentage of samples with 0 expression',default=0.2)
    parser.add_argument('--cutoff', type=float, help='Filter out genes with variance below this value',default=0.1)
    parser.add_argument('--alpha', type=float, help='alpha for outlier detection',default=0.05)
    parser.add_argument('--outlier_method', type=str, choices=['auto', 'mahalanobis', 'statistical'],
                       help='Method for outlier detection: auto (choose based on data size), mahalanobis (MinCovDet), or statistical (fast IQR/zscore)',
                       default='auto')
    args = parser.parse_args()
    plt.style.use('ggplot')
    main(data_path=args.data_path,metadata_path=args.metadata_path,save_path=args.save_path,per=args.per,cutoff=args.cutoff,alpha=args.alpha,outlier_method=args.outlier_method)
