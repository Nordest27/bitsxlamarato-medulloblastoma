"""
Medulloblastoma G3/G4 Subgroup Classification Package

An AI-powered tool for classifying medulloblastoma G3/G4 subtypes from
gene expression data, developed during the BitsxLaMarató hackathon.
"""

__version__ = "0.1.0"
__author__ = "Jose Estragues, Guillermo Prol"
__email__ = "jose.estragues@bsc.es, guillermo.prolcastelo@bsc.es"

# Import configuration
from medulloblastoma import config

# Import main functions for easy access
from medulloblastoma.dataset import download_data, prepare_data
from medulloblastoma.features import load_data, preprocess, maha_outliers, fast_statistical_outliers
from medulloblastoma.plots import plot_umap_binary, plot_umap_spectrum

# Define what gets imported with "from medulloblastoma import *"
__all__ = [
    "download_data",
    "prepare_data",
    "load_data",
    "preprocess",
    "maha_outliers",
    "fast_statistical_outliers",
    "plot_umap_binary",
    "plot_umap_spectrum",
    "config"
]

