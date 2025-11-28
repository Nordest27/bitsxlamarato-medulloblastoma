"""
Dataset management functions for medulloblastoma gene expression data.

This module provides functions to download and prepare the GSE85217 dataset
from the Gene Expression Omnibus (GEO) database, which contains gene expression
profiles for 763 medulloblastoma patients.
"""

import os
import gzip
from pathlib import Path
from typing import Union, Optional
import requests
import pandas as pd
import numpy as np
from loguru import logger


def download_data(
    save_path: Union[str, Path] = "data/raw/",
    remove_gz: bool = True,
    timeout: int = 300
) -> str:
    """
    Download GSE85217 medulloblastoma gene expression dataset from GEO.

    Downloads the microarray gene expression data for 763 medulloblastoma patients
    from the Gene Expression Omnibus (GEO) database. The dataset contains expression
    profiles used for subtype classification.

    Parameters
    ----------
    save_path : str or Path, optional
        Directory path where the downloaded file will be saved, by default "data/raw/"
    remove_gz : bool, optional
        Whether to remove the compressed .gz file after extraction, by default True
    timeout : int, optional
        Request timeout in seconds, by default 300

    Returns
    -------
    str
        Path to the extracted data file

    Raises
    ------
    requests.RequestException
        If the download request fails
    IOError
        If file writing or extraction fails

    Examples
    --------
    >>> # Download to default location
    >>> filepath = download_data()
    >>> print(f"Data saved to: {filepath}")

    >>> # Download to custom location and keep compressed file
    >>> filepath = download_data(save_path="my_data/", remove_gz=False)
    """
    # GEO download URL for GSE85217 dataset
    url = ("https://www.ncbi.nlm.nih.gov/geo/download/"
           "?acc=GSE85217&format=file&file="
           "GSE85217%5FM%5Fexp%5F763%5FMB%5FSubtypeStudy%5FTaylorLab%2Etxt%2Egz")

    # Ensure save directory exists
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Define file paths
    filename = "GSE85217_M_exp_763_MB_SubtypeStudy_TaylorLab.txt.gz"
    gz_path = save_path / filename
    txt_path = save_path / filename.replace('.gz', '')

    try:
        # Download the compressed file
        logger.info(f"Downloading GSE85217 dataset from GEO...")
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        # Write to file with progress indication
        total_size = int(response.headers.get('content-length', 0))
        with open(gz_path, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end='', flush=True)

        print(f"\nFile downloaded successfully: {gz_path}")

        # Extract the compressed file
        logger.info("Extracting compressed file...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(txt_path, 'wb') as f_out:
                f_out.write(f_in.read())

        logger.success(f"File extracted to: {txt_path}")

        # Remove compressed file if requested
        if remove_gz:
            gz_path.unlink()
            logger.info("Removed compressed .gz file")

        return str(txt_path)

    except requests.RequestException as e:
        logger.error(f"Failed to download file: {e}")
        raise
    except IOError as e:
        logger.error(f"File I/O error: {e}")
        raise

def prepare_data(
    expression_file: Union[str, Path] = "data/raw/GSE85217_M_exp_763_MB_SubtypeStudy_TaylorLab.txt",
    metadata_path: Union[str, Path] = "data/raw/GSE85217_metadata.csv",
    save_path: Union[str, Path] = "data/raw/"
) -> tuple[str, str]:
    """
    Process and prepare medulloblastoma gene expression data for analysis.

    Reads the raw GSE85217 gene expression file, separates gene correspondence
    information from expression data, and creates clean datasets ready for
    machine learning analysis.

    Parameters
    ----------
    expression_file : str or Path, optional
        Path to the raw gene expression file downloaded from GEO,
        by default "data/raw/GSE85217_M_exp_763_MB_SubtypeStudy_TaylorLab.txt"
    metadata_path : str or Path, optional
        Path to the metadata CSV file containing sample information,
        by default "data/raw/GSE85217_metadata.csv"
    save_path : str or Path, optional
        Directory where processed files will be saved, by default "data/raw/"

    Returns
    -------
    tuple[str, str]
        Tuple containing paths to:
        - Processed expression data CSV file
        - Processed subgroup metadata CSV file

    Raises
    ------
    FileNotFoundError
        If input files don't exist
    pd.errors.EmptyDataError
        If files are empty or corrupted

    Examples
    --------
    >>> # Process with default paths
    >>> expr_path, meta_path = prepare_data()
    >>> print(f"Expression data: {expr_path}")
    >>> print(f"Metadata: {meta_path}")

    >>> # Process with custom paths
    >>> expr_path, meta_path = prepare_data(
    ...     expression_file="my_data/raw_expression.txt",
    ...     metadata_path="my_data/sample_info.csv",
    ...     save_path="my_data/processed/"
    ... )
    """
    # Convert paths to Path objects for consistency
    expression_file = Path(expression_file)
    metadata_path = Path(metadata_path)
    save_path = Path(save_path)

    # Ensure save directory exists
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load raw gene expression data
        logger.info(f"Loading gene expression data from {expression_file}")
        if not expression_file.exists():
            raise FileNotFoundError(f"Expression file not found: {expression_file}")

        data_direct = pd.read_table(expression_file, sep="\t", index_col=0)
        logger.info(f"Loaded expression data with shape: {data_direct.shape}")

        # Extract gene correspondence information (first 4 columns)
        columns_genes = data_direct.columns[:4]
        gene_correspondence = data_direct[columns_genes]
        logger.info("Extracting gene correspondence information...")

        # Remove gene correspondence from expression data
        data_direct = data_direct.drop(columns=columns_genes)
        logger.info(f"Expression data after removing gene info: {data_direct.shape}")

        # Load metadata
        logger.info(f"Loading metadata from {metadata_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        metadata = pd.read_csv(metadata_path, index_col=0)
        logger.info(f"Loaded metadata with shape: {metadata.shape}")

        # Match sample names between expression data and metadata
        logger.info("Matching sample names between expression data and metadata...")

        # Find metadata indices that match data column names
        matching_indices = []
        for col in data_direct.columns:
            matching_meta = metadata[metadata['title'] == col]
            if len(matching_meta) == 1:
                matching_indices.append(matching_meta.index[0])
            else:
                logger.warning(f"Could not find unique match for sample: {col}")

        # Update column names with metadata indices
        if len(matching_indices) == len(data_direct.columns):
            data_direct.columns = matching_indices
            logger.success("Successfully matched all samples with metadata")
        else:
            logger.warning(f"Only matched {len(matching_indices)}/{len(data_direct.columns)} samples")

        # Extract subgroup information
        logger.info("Extracting subgroup information...")
        subgroups = metadata['subgroup:ch1'].copy()
        subgroups.name = 'Sample_characteristics_ch1'

        # Define output file paths
        expr_output = save_path / 'cavalli.csv'
        gene_corr_output = save_path / 'cavalli_gene_correspondence.csv'
        subgroups_output = save_path / 'cavalli_subgroups.csv'

        # Save processed data
        logger.info(f"Saving processed data to {save_path}")
        data_direct.to_csv(expr_output)
        gene_correspondence.to_csv(gene_corr_output)
        subgroups.to_csv(subgroups_output)

        logger.success("Data preparation completed successfully")
        logger.info(f"Expression data saved to: {expr_output}")
        logger.info(f"Gene correspondence saved to: {gene_corr_output}")
        logger.info(f"Subgroup data saved to: {subgroups_output}")

        return str(expr_output), str(subgroups_output)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty or corrupted data file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data preparation: {e}")
        raise

